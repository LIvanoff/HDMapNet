import torch
import numpy as np
from scipy.spatial.ckdtree import cKDTree as kdtree
from libs.utils.tools import cart2polar, polar2cat, nb_process_label, mp_logger, whether_aug, vis_range_view
from libs.utils.laserscan import LaserScan, SemLaserScan
from libs.utils.tools import (create_eval_log, load_arch_cfg, find_free_port,
                              load_data_cfg, load_pretrained, recover_uint8_trick)
from data.rasterize import preprocess_map
from .lidar import get_lidar_data
from .image import normalize_img, img_transform
from .utils import label_onehot_encoding
from model.voxel import pad_or_trim_to_np


def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)


def polar2cat(input_xyz_polar):
    x = input_xyz_polar[0] * np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0] * np.sin(input_xyz_polar[1])
    return np.stack((x, y, input_xyz_polar[2]), axis=0)

def SemKITTI2train(label):
    if isinstance(label, list):
        return [SemKITTI2train_single(a) for a in label]
    else:
        return SemKITTI2train_single(label)

def SemKITTI2train_single(label):
    return label - 1 # uint8 trick

def train2SemKITTI(input_label):
    # delete 0 label (uses uint8 trick : 0 - 1 = 255 )
    return input_label + 1

class OursSemanticDataset_v2:
    def __init__(self, grid_size, rotate_aug=False, flip_aug=False, ignore_label=255, return_test=False,
                fixed_volume_space=False, max_volume_space=[50, np.pi, 1.5], min_volume_space=[3, -np.pi, -3]
                ):
        self.grid_size = np.asarray(grid_size)
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space

    def prepocess_polarseg_data(self, data):
        if len(data) == 2:
            xyz, labels = data
        elif len(data) == 3:
            xyz, labels, sig = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        else:
            raise Exception('Return invalid data tuple')

        # convert coordinate into polar coordinates
        xyz_pol = cart2polar(xyz)

        max_bound_r = np.percentile(xyz_pol[:, 0], 100, axis=0)
        min_bound_r = np.percentile(xyz_pol[:, 0], 0, axis=0)
        max_bound = np.max(xyz_pol[:, 1:], axis=0)
        min_bound = np.min(xyz_pol[:, 1:], axis=0)
        max_bound = np.concatenate(([max_bound_r], max_bound))
        min_bound = np.concatenate(([min_bound_r], min_bound))
        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)

        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size
        intervals = crop_range / (cur_grid_size - 1)

        if (intervals == 0).any(): print("Zero interval!")
        grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

        # process voxel position
        voxel_position = np.zeros(self.grid_size, dtype=np.float32)
        dim_array = np.ones(len(self.grid_size) + 1, int)
        dim_array[0] = -1
        voxel_position = np.indices(self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)
        # voxel_position = polar2cat(voxel_position)

        # process labels
        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair.astype(np.int64))
        # data_tuple = (voxel_position,processed_label)

        # prepare visiblity feature
        # find max distance index in each angle,height pair
        valid_label = np.zeros_like(processed_label, dtype=bool)
        valid_label[grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2]] = True
        valid_label = valid_label[::-1]
        max_distance_index = np.argmax(valid_label, axis=0)
        max_distance = max_bound[0] - intervals[0] * (max_distance_index)
        distance_feature = np.expand_dims(max_distance, axis=2) - np.transpose(voxel_position[0], (1, 2, 0))
        distance_feature = np.transpose(distance_feature, (1, 2, 0))
        # convert to boolean feature
        distance_feature = (distance_feature > 0) * -1.
        distance_feature[grid_ind[:, 2], grid_ind[:, 0], grid_ind[:, 1]] = 1.

        data_tuple = (distance_feature.astype(np.float32),)  # , processed_label)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_xyz = np.concatenate((return_xyz, xyz_pol, xyz[:, :2]), axis=1)

        if len(data) == 2:
            return_fea = return_xyz
        elif len(data) == 3:
            return_fea = np.concatenate((return_xyz, sig[..., np.newaxis]), axis=1)

        data_tuple += (grid_ind, return_fea,)
        return data_tuple

    def get_lidar_data(self, lidar_data):
        num_points = lidar_data.shape[0]

        xyz = lidar_data[:, :3]
        labels = np.expand_dims(np.zeros_like(lidar_data[:, 0], dtype=int), axis=1)
        intensity = lidar_data[:, 3]

        # lidar_data = pad_or_trim_to_np(lidar_data, [81920, 5]).astype('float32')
        lidar_mask = np.zeros(num_points).astype('float32')
        # lidar_mask[num_points:] *= 0.0

        data_tuple = (xyz, labels, intensity)
        dataset = self.prepocess_polarseg_data(data_tuple)
        dataset += (lidar_data, lidar_mask)
        return dataset

    def preprocess_lidar_data(self, lidar_data):
        dataset = self.get_lidar_data(lidar_data)
        return dataset

class OursSemanticDataset_v1:
    def __init__(self, ARCH, DATA):

        self.split = 'test'
        self.polar = ARCH['polar']
        self.range = ARCH['range']
        self.dataset = ARCH['dataset']
        self.color_map = DATA['color_map']
        self.max_volume_space = [50, np.pi, 3]
        self.min_volume_space = [0, -np.pi, -5]
        self.ignore_label = 0
        self.knn = True
        self.neighbor = 7
        self.cal_valid = []

        self.labels = DATA['labels_16']
        self.learning_map = DATA['learning_map']
        self.nclasses = len(self.labels)

    def range_dataset(self, scan_points):
        self.sensor_img_means = torch.tensor(self.range['sensor_img_means'],
                                             dtype=torch.float32)
        self.sensor_img_stds = torch.tensor(self.range['sensor_img_stds'],
                                            dtype=torch.float32)
        scan = SemLaserScan(sem_color_dict=self.color_map,
                            train=False,
                            project=True,
                            H=self.range['sensor_img_H'],
                            W=self.range['sensor_img_W'],
                            fov_up=self.range['sensor_fov_up'],
                            fov_down=self.range['sensor_fov_down'],
                            proj_version=self.range['proj'],
                            hres=self.range['hres'],
                            factor=self.range['factor'],
                            points_to_drop=None,
                            flip=self.range['flip'],
                            trans=self.range['trans'],
                            rot=self.range['rot'])

        # open and obtain scan
        scan.open_scan(scan_points)
        # if self.split != 'test':
        #     scan.set_label_nuscenes(label[scan.valid].reshape((-1)))

        scan.max_points = self.range['max_points']
        unproj_n_points = scan.points.shape[0]
        unproj_xyz = torch.full((scan.max_points, 3), -1.0, dtype=torch.float)
        unproj_xyz[:unproj_n_points] = torch.from_numpy(scan.points)
        unproj_range = torch.full([scan.max_points], -1.0, dtype=torch.float)
        unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)
        unproj_remissions = torch.full([scan.max_points], -1.0, dtype=torch.float)
        unproj_remissions[:unproj_n_points] = torch.from_numpy(scan.remissions)

        unproj_labels = []

        # get points and labels
        proj_range = torch.from_numpy(scan.proj_range).clone()
        proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
        proj_remission = torch.from_numpy(scan.proj_remission).clone()
        proj_mask = torch.from_numpy(scan.proj_mask)
        proj_labels = []
        proj_color = []
        new_w = scan.sizeWH[0]

        proj_x = torch.full([scan.max_points], 0, dtype=torch.float)
        proj_x[:unproj_n_points] = torch.from_numpy(2 * (scan.proj_x/(new_w-1) - 0.5))
        proj_y = torch.full([scan.max_points], 0, dtype=torch.float)
        proj_y[:unproj_n_points] = torch.from_numpy(2 * (scan.proj_y/(scan.sizeWH[1]-1) - 0.5))
        proj_yx = torch.stack([proj_y, proj_x], dim=1)[None, :, :]

        proj = torch.cat([proj_range.unsqueeze(0).clone(),
                          proj_xyz.clone().permute(2, 0, 1),
                          proj_remission.unsqueeze(0).clone()])
        proj = (proj - self.sensor_img_means[:, None, None]
                ) / self.sensor_img_stds[:, None, None]
        proj = proj * proj_mask.float()

        proj_idx = torch.from_numpy(scan.proj_idx).clone()

        points2pixel = np.concatenate((scan.proj_y.reshape(-1, 1), scan.proj_x.reshape(-1, 1)), axis=1)
        points2pixel = torch.from_numpy(points2pixel).long()
        full_p2p = torch.full((scan.max_points, points2pixel.shape[-1]), -1, dtype=torch.long)
        full_p2p[:unproj_n_points] = points2pixel

        points2pixel = proj_yx[0, :unproj_n_points, :].float()

        data_tuple = (proj, proj_yx, unproj_xyz)

        return data_tuple, points2pixel, proj_idx, scan.valid, scan.max_points

    def polar_dataset(self, scan, labels):
        'Generates one sample of data'

        # put in attribute
        xyz = scan[:, 0:3]  # get xyz
        remissions = scan[:, 3]  # get remission
        # remissions = np.squeeze(remissions)
        num_pt = xyz.shape[0]

        # if self.points_to_drop is not None:
        # labels = np.delete(labels, self.points_to_drop, axis=0)

        # convert coordinate into polar coordinates
        xyz_pol = cart2polar(xyz)

        if self.polar['fixed_volume_space']:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)
        else:
            max_bound_r = np.percentile(xyz_pol[:, 0], 100, axis=0)
            min_bound_r = np.percentile(xyz_pol[:, 0], 0, axis=0)
            max_bound = np.max(xyz_pol[:, 1:], axis=0)
            min_bound = np.min(xyz_pol[:, 1:], axis=0)
            max_bound = np.concatenate(([max_bound_r], max_bound))
            min_bound = np.concatenate(([min_bound_r], min_bound))

        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = np.array(self.polar['grid_size'])
        intervals = crop_range / (cur_grid_size - 1)

        if (intervals == 0).any():
            mp_logger("Zero interval!")
        grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)
        pxpypz = (np.clip(xyz_pol, min_bound, max_bound) - min_bound) / crop_range
        pxpypz = 2 * (pxpypz - 0.5)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_xyz = np.concatenate((return_xyz, xyz_pol, xyz[:, :2]), axis=1)

        return_fea = np.concatenate((return_xyz, remissions[..., np.newaxis]), axis=1)

        # order in decreasing z
        pixel2point = np.full(shape=self.polar['grid_size'], fill_value=-1, dtype=np.float32)
        indices = np.arange(xyz_pol.shape[0])
        order = np.argsort(xyz_pol[:, 2])
        grid_ind_order = grid_ind[order].copy()
        indices = indices[order]
        pixel2point[grid_ind_order[:, 0], grid_ind_order[:, 1], grid_ind_order[:, 2]] = indices

        pixel2point = torch.from_numpy(pixel2point)

        return_fea = torch.from_numpy(return_fea)
        grid_ind = torch.from_numpy(grid_ind)

        full_return_fea = torch.full((self.max_points, return_fea.shape[-1]), -1.0, dtype=torch.float)
        full_return_fea[:return_fea.shape[0]] = return_fea

        full_grid_ind = torch.full((self.max_points, grid_ind.shape[-1]), -1, dtype=torch.long)
        full_grid_ind[:grid_ind.shape[0]] = grid_ind

        full_pxpypz = torch.full((self.max_points, pxpypz.shape[-1]), 0, dtype=torch.float)
        full_pxpypz[:pxpypz.shape[0]] = torch.from_numpy(pxpypz)
        full_pxpypz = full_pxpypz[None, None, :, :]

        if self.polar['return_test']:
            data_tuple = (full_grid_ind, full_return_fea, full_pxpypz, num_pt)
        else:
            data_tuple = (full_grid_ind, full_return_fea, full_pxpypz, num_pt)
        return data_tuple, torch.from_numpy(pxpypz).float(), pixel2point

    def get_semantic_label(self, range_data, polar_data, r2p_flow_matrix, p2r_flow_matrix, knns_full):
        in_vol, pxpy_range, points = range_data
        train_grid, pt_fea, pxpypz_polar, num_pt = polar_data

        in_vol = in_vol.cuda(non_blocking=True)
        train_grid_2d = train_grid[:, :, :2].cuda(non_blocking=True)
        pt_fea = pt_fea.cuda(non_blocking=True)

        r2p_matrix = r2p_flow_matrix.cuda(non_blocking=True)
        p2r_matrix = p2r_flow_matrix[:, :, :, :2].cuda(non_blocking=True)

        pxpy_range = torch.flip(pxpy_range.cuda(non_blocking=True),
                                dims=[-1])  # because for F.grid_sample, i,j,k index w,h,d (i.e., reverse order)
        pxpypz_polar = torch.flip(pxpypz_polar.cuda(non_blocking=True), dims=[-1])
        points = points.cuda(non_blocking=True)
        knns = knns_full.cuda(non_blocking=True)

        fusion_pred, range_pred, polar_pred, range_x, polar_x = self.model(
            in_vol, pt_fea, train_grid_2d, num_pt, r2p_matrix, p2r_matrix, pxpy_range, pxpypz_polar, points, knns)

        return fusion_pred, range_pred, polar_pred, range_x, polar_x

    def add_sem_lab(self, scan):
        label = np.expand_dims(np.zeros_like(scan[:, 0], dtype=int), axis=1)
        range_data, range_point2pixel, range_pixel2point, valid, max_points = self.range_dataset(scan)

        self.cal_valid.append(valid.sum())

        if valid.sum() == 0:
            scan = np.zeros_like(scan)[:10000]
        else:
            scan = scan[valid]
        # when training due to the order of index changed, we need to re-cal knn
        if self.knn:
            tree = kdtree(scan[:, :3])
            _, knns = tree.query(scan[:, :3], k=self.neighbor)
        self.valid = valid
        self.max_points = max_points

        polar_data, polar_point2pixel, polar_pixel2point = self.polar_dataset(scan, label)

        r2p_flow_matrix = self.r2p_flow_matrix(polar_pixel2point, range_point2pixel)
        p2r_flow_matrix = self.p2r_flow_matrix(range_pixel2point, polar_point2pixel)

        knns_full = torch.full((max_points, self.neighbor), 0, dtype=torch.long)
        knns_full[:knns.shape[0]] = torch.from_numpy(knns).long()

        return range_data, polar_data, r2p_flow_matrix, p2r_flow_matrix, knns_full
        # fusion_pred, range_pred, polar_pred, range_x, polar_x = self.get_semantic_label(range_data, polar_data, r2p_flow_matrix, p2r_flow_matrix, knns_full)
        '''return'''


    def p2r_flow_matrix(self, range_idx, polar_idx):
        """
        range_idx: [H, W] indicates the location of each range pixel on point clouds
        polar_idx: [N, 3] indicates the location of each points on polar grids
        """
        H, W = range_idx.shape
        N, K = polar_idx.shape
        flow_matrix = torch.full(size=(H, W, K), fill_value=-10, dtype=torch.float)
        if self.valid.sum() == 0:
            return flow_matrix

        valid_idx = torch.nonzero(range_idx+1).transpose(0, 1)
        valid_value = range_idx[valid_idx[0], valid_idx[1]].long()
        flow_matrix[valid_idx[0], valid_idx[1], :] = polar_idx[valid_value, :]
        return flow_matrix

    def r2p_flow_matrix(self, polar_idx, range_idx):
        """
        polar_idx: [H, W, C] indicates the location of each range pixel on point clouds
        range_idx: [N, 2] indicates the location of each points on polar grids
        """
        H, W, C = polar_idx.shape
        N, K = range_idx.shape
        flow_matrix = torch.full(size=(H, W, C, K), fill_value=-10, dtype=torch.float) # smaller than -1 to trigger the zero padding of grid_sample
        if self.valid.sum() == 0:
            return flow_matrix

        valid_idx = torch.nonzero(polar_idx+1).transpose(0, 1)
        valid_value = polar_idx[valid_idx[0], valid_idx[1], valid_idx[2]].long()
        flow_matrix[valid_idx[0], valid_idx[1], valid_idx[2], :] = range_idx[valid_value, ]
        return flow_matrix
    
    # def get_lidar(self, rec):
    #     lidar_data = get_lidar_data(self.nusc, rec, nsweeps=3, min_distance=2.2)
    #     lidar_data = lidar_data.transpose(1, 0)
    #     if lidar_data.shape[0] > 81920:
    #         lidar_data = lidar_data[:81920, :]
    #     # print(f'lidar_data.shape = {lidar_data.shape}')


    #     range_data, polar_data, r2p_flow_matrix, p2r_flow_matrix, knns_full = self.add_sem_lab(lidar_data)
    #     _, _, _, num_pt = polar_data
    #     lidar_data = pad_or_trim_to_np(lidar_data, [81920, 5]).astype('float32')
    #     lidar_mask = np.ones(81920).astype('float32')
    #     lidar_mask[num_pt:] *= 0.0
    #     return lidar_data, lidar_mask, range_data, polar_data, r2p_flow_matrix, p2r_flow_matrix, knns_full
    
    def get_lidar(self, lidar_data, semantic=True):
        if lidar_data.shape[0] > 35000:
            lidar_data = lidar_data[:35000, :]
            num_pt = 35000

        if semantic:
            range_data, polar_data, r2p_flow_matrix, p2r_flow_matrix, knns_full = self.add_sem_lab(lidar_data)
            _, _, _, num_pt = polar_data
        else:
            num_pt = lidar_data.shape[0]

        lidar_data = pad_or_trim_to_np(lidar_data, [35000, 5]).astype('float32')
        lidar_mask = np.ones(35000).astype('float32')
        lidar_mask[num_pt:] *= 0.0
        if semantic:
            return lidar_data, lidar_mask, range_data, polar_data, r2p_flow_matrix, p2r_flow_matrix, knns_full
        else:
            return lidar_data, lidar_mask