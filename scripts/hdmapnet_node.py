#!/usr/bin/env python3

import os
import cv2
import sys
import logging
import argparse
import time

from sensor_msgs.msg import PointCloud2
from hdmap_msgs.msg import HDMap, Vector
from sensor_msgs.msg import CompressedImage

import torch
import rospy
import tf2_ros
import numpy as np

np.float = float
np.int = int
np.bool = bool

import ros_numpy
from PIL import Image
import message_filters
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# from data.dataset import semantic_dataset
from data.const import NUM_CLASSES
from model import get_model
from data.preprocess import OursSemanticDataset_v2, train2SemKITTI
from postprocess.vectorize import vectorize
from model.voxel import pad_or_trim_to_np

from PolarSeg.network.BEV_Unet import BEV_Unet
from PolarSeg.network.ptBEV import ptBEVnet
from PolarSeg.dataloader.dataset_nuscenes import map_name_from_segmentation_class_to_segmentation_index
from PolarSeg.dataloader.dataset import SemKITTI_label_name


class HDMapNetNode:
    def __init__(self) -> None:
        rospy.init_node("HDMapNetNode")
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)

        # tasks
        self.make_hdmap = rospy.get_param("~hdmap", "false")
        self.make_segmentation = rospy.get_param("~semantic_segmentation", "true")
        self.publish_joint_pc = rospy.get_param("~publish_joint_pc", "false")

        
        self.model_type = rospy.get_param("~model_type", 'HDMapNet_lidar')
        self.xbound = rospy.get_param("~xbound", "[-30.0, 30.0, 0.15]")
        self.ybound = rospy.get_param("~ybound", "[-15.0, 15.0, 0.15]")
        self.zbound = rospy.get_param("~zbound", "[-10.0, 10.0, 20.0]")
        self.dbound = rospy.get_param("~dbound", "[4.0, 45.0, 1.0]")
        modelf = rospy.get_param("~modelf", "checkpoints/best_score.pt")
        seg_model = rospy.get_param("~seg_model", "~/Polarseg/pretrained_weight/Nuscenes_PolarSeg.pt")
        self.num_lidar_feats = rospy.get_param("~num_lidar_feats", 4)    

        # ARCH = load_arch_cfg(seg_arch_cfg)
        # DATA = load_data_cfg(seg_data_cfg)
        # color_dict = DATA['color_map']
        # self.color_map_semantic = np.array([color_dict[key] for key in sorted(color_dict.keys())])
        # self.color_map_semantic = self.color_map_semantic[:, [2, 1, 0]] # bgr -> rgb
        grid_size = [480,360,32]
        compression_model = grid_size[2]
        self.data_wrapper = OursSemanticDataset_v2(grid_size = grid_size, ignore_label = 0, fixed_volume_space = True, return_test= True,max_volume_space=[50,np.pi,3], min_volume_space=[0,-np.pi,-5])
        data_conf = {
            'num_channels': NUM_CLASSES + 1,
            'xbound': eval(self.xbound),
            'ybound': eval(self.ybound),
            'zbound': eval(self.zbound),
            'dbound': eval(self.dbound),
            # 'thickness': args.thickness,
            # 'angle_class': args.angle_class,
        }

        self.lidar_names = ["top", "right", "left"]
        self.frame_id = "base_link"

        output_topic = rospy.get_param("~output_topic", '/hdmap')
        self.save_dir = rospy.get_param("~save_dir", 'result')
        self.save_result = rospy.get_param("~save_result", True)

        if not os.path.exists(self.save_dir) and self.save_result:
            os.makedirs(self.save_dir)

        self.instance_seg = rospy.get_param("~instance_seg", True)
        self.embedding_dim = rospy.get_param("~embedding_dim", 16)
        self.delta_v = rospy.get_param("~delta_v", 0.5)
        self.delta_d = rospy.get_param("~delta_d", 3.0)

        self.direction_pred = rospy.get_param("~direction_pred", True)
        self.angle_class = rospy.get_param("~angle_class", 36)
        self.color_map_hdmap = {0: 'r', 1: 'b', 2: 'g'}
        image_path = rospy.get_param("~image_car")

        if self.make_hdmap:
            self.model = get_model(
                self.model_type, 
                data_conf, 
                self.instance_seg, 
                self.embedding_dim, 
                self.direction_pred,
                self.angle_class
                )
            self.model.load_state_dict(torch.load(modelf), strict=False)
            for _, param in self.model.named_parameters():
                param.requires_grad = False
            self.model.eval()
            self.model.cuda()


        # self.poincloud_sub = rospy.Subscriber(input_cloud, PointCloud2, self.callback)
        self.hdmapnet_pub = rospy.Publisher(output_topic, HDMap, queue_size=10)
        self.pc_publisher = rospy.Publisher('/joined_point_cloud', PointCloud2, queue_size=10) #10?

        # train_loader, val_loader = semantic_dataset(args.version, args.dataroot, data_conf, args.bsz, args.nworkers)
        self.car_img = Image.open(image_path)
        self.image = np.array([])

        # ARCH = load_arch_cfg('config/resnet_nuscenes.yaml')
        # DATA = load_data_cfg('config/nuscenes.yaml')
        if self.make_segmentation:
            visibility = True  # args.visibility
            pytorch_device = torch.device('cuda:0')
            fea_dim = 9
            circular_padding = True

            # prepare miou fun
            # unique_label_str = list(map_name_from_segmentation_class_to_segmentation_index)[1:]
            # unique_label = np.asarray([map_name_from_segmentation_class_to_segmentation_index[s] for s in unique_label_str]) - 1
            unique_label=np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
            unique_label_str=[SemKITTI_label_name[x] for x in unique_label+1]

            # prepare model
            Unet = BEV_Unet(n_class=len(unique_label), n_height=compression_model, input_batch_norm=True, dropout=0.5,
                                    circular_padding=circular_padding, use_vis_fea=visibility)
            self.ptBEVnet = ptBEVnet(Unet, pt_model='pointnet', grid_size=grid_size, fea_dim=fea_dim, max_pt_per_encode=256,
                                out_pt_fea_dim=512, kernal_size=1, pt_selection='random', fea_compre=compression_model)


            if os.path.exists(seg_model):
                self.ptBEVnet.load_state_dict(torch.load(seg_model))
                rospy.loginfo("Model loaded!")

            self.ptBEVnet.eval()
            self.ptBEVnet.cuda()

            for name, param in self.ptBEVnet.named_parameters():
                param.requires_grad = False


        all_subs = []
        self.tRs_base_link_lidar = []

        image_sub_right = message_filters.Subscriber("/camera/right/image_raw/compressed", CompressedImage)

        for lidar_name in self.lidar_names:
            all_subs.append(message_filters.Subscriber(f'/ouster_{lidar_name}/points', PointCloud2))
            self.tRs_base_link_lidar.append(self.get_tvec_rot_mat_for_lidar(f"os_sensor_{lidar_name}"))
        
        all_subs.append(message_filters.Subscriber(f'/lslidar_point_cloud', PointCloud2))
        self.tRs_base_link_lidar.append(self.get_tvec_rot_mat_for_lidar(f"laser_link"))
        ts = message_filters.ApproximateTimeSynchronizer(
            all_subs , # + [image_sub_right]
            queue_size=10,
            slop=0.11
        )
        ts.registerCallback(self.callback)
        print("Inizialization Done...")
        try:
            rospy.spin()
        except KeyboardInterrupt:
            pass
        
    def publish_joined_point_cloud(self, points, stamp, semantic_feat):
        # rospy.loginfo("publish_joined_point_cloud")
        # labels = self.color_map_semantic[semantic_feat.numpy()]
        unique, counts = np.unique(semantic_feat, return_counts=True)
        print(np.asarray((unique, counts)).T)
        print(semantic_feat)
        # points = points[semantic_feat == 15]
        data = np.zeros(points.shape[0], dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('intensity', np.uint8),
            # ('g', np.uint8),
            # ('b', np.uint8),
        ])
        # print(np.unique(semantic_feat.numpy()))
        # print(points.shape)
        data['x'] = points[:, 0]
        data['y'] = points[:, 1]
        data['z'] = points[:, 2]
        # data['r'] = labels[:, 0]
        # data['g'] = labels[:, 1]
        # data['b'] = labels[:, 2]
        data['intensity'] = semantic_feat.numpy()#[semantic_feat == 15]

        msg = ros_numpy.msgify(PointCloud2, data)
        msg.header.frame_id="base_link"
        msg.header.stamp = stamp
        self.pc_publisher.publish(msg)

    def image_callback(self, msg):
        self.image  = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='rgb8')


    def get_lidar(self, lidar_data):
        num_points = lidar_data.shape[0]
        lidar_data = pad_or_trim_to_np(lidar_data, [81920, 4]).astype('float32')
        lidar_mask = np.ones(81920).astype('float32')
        lidar_mask[num_points:] *= 0.0
        return lidar_data, lidar_mask

    def get_tvec_rot_mat_for_lidar(self, lidar_frame_id):
        tf_transform = self.tf_buffer.lookup_transform(
            self.frame_id, 
            lidar_frame_id, 
            rospy.Time(), 
            timeout=rospy.Duration(2)
        )
        translation = tf_transform.transform.translation
        rotation = tf_transform.transform.rotation

        t_btop = np.array([translation.x, translation.y, translation.z])
        R_btop = Rotation.from_quat([rotation.x,rotation.y, rotation.z, rotation.w])
        R_btop = R_btop.as_matrix().astype(float)

        return t_btop, R_btop
    
    def translate_and_rotate(self, points, t, R):
        point_base_link = points @ R.T + t

        return  point_base_link
    
    def transform_to_bin(self, *all_lidar_pc):
        points_list = []
        
        for idx in range(len(all_lidar_pc)):
            point_cloud = ros_numpy.numpify(all_lidar_pc[idx])

            n_used_features = 5
            points = np.zeros((*point_cloud.shape, n_used_features))

            points[..., 0] = point_cloud['x']
            points[..., 1] = point_cloud['y']
            points[..., 2] = point_cloud['z'] #-1.7
            points[..., 3] = point_cloud['intensity'] * 0.1464823
            points[..., 4] = 0 # for timestamp

            points = np.array(points, dtype=np.float32).reshape(-1, n_used_features)
            Rs = np.linalg.norm(points[:, :3], axis=1)
            # points = points[(Rs < 55) & (Rs > 3) & (points[:, 2] < 5)]
            # points = points[(points[:, 0] > 0.001)&((points[:, 0] < -0.001))]
            
            points[:, :3] = self.translate_and_rotate(
                points[:, :3], 
                *self.tRs_base_link_lidar[idx]
            )
            points = points[(points[:, 2] < 3.2)&(Rs < 50) & (Rs > 2)]
            # points[:, 3] *= 0.1464823
            points_list.append(points)

        points = np.concatenate(points_list)

        return points
    
    def vis_pred(self, segmentation, embedding, direction, stamp):
        for si in range(segmentation.shape[0]):
            # fix, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            coords, _, line_types = vectorize(segmentation[si], embedding[si], direction[si], self.angle_class)

            for coord, line_type in zip(coords, line_types):
                plt.plot(coord[:, 0], coord[:, 1], linewidth=5, c=self.color_map[line_type])

            # отрисовка HD-map
            plt.xlim((0, segmentation.shape[3]))
            plt.ylim((0, segmentation.shape[2]))
            plt.imshow(self.car_img, extent=[segmentation.shape[3]//2-15, segmentation.shape[3]//2+15, segmentation.shape[2]//2-12, segmentation.shape[2]//2+12])
            
            # отображение изображения
            # if self.image.shape[0] != 0:
            #     ax2.imshow(self.image)
            #     ax2.axis('off')
            #     ax2.set_title('/camera/right/image_raw/compressed')

            img_name = f'eval_{stamp}.jpg'
            print('saving', img_name)
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, img_name))
            plt.close()

    @torch.no_grad()
    def callback(self, *all_lidar_pc_and_image):
        stamp = rospy.Time.now()
        # if (stamp - all_lidar_pc[0].header.stamp) > rospy.Duration(0.2):
        #     print("PAST", stamp,  all_lidar_pc[0].header.stamp)
        #     return
        # print("all_lidar_pc: ", *[m.header for m in all_lidar_pc])
        # print("stamp", stamp)

        all_lidar_pc = all_lidar_pc_and_image[:-1]
        image_msg = all_lidar_pc_and_image[-1]

        points = self.transform_to_bin(*all_lidar_pc)
        # self.image_callback(image_msg)
        # print(points.shape)

        np.random.shuffle(points)
        # print("transform_to_bin time", time.time() - t1)
        t1 = time.time()
        
        val_vox_fea,val_grid,val_pt_fea, lidar_data, lidar_mask= self.data_wrapper.preprocess_lidar_data(points)
        lidar_data, lidar_mask = torch.Tensor(lidar_data).unsqueeze(0), torch.Tensor(lidar_mask).unsqueeze(0)

        if self.make_segmentation:
            val_vox_fea_ten = torch.from_numpy(val_vox_fea).unsqueeze(0).cuda()
            # val_vox_label = SemKITTI2train(val_vox_label)
            # val_pt_labs = SemKITTI2train(val_pt_labs)
            val_pt_fea_ten = [torch.from_numpy(val_pt_fea).type(torch.FloatTensor).cuda()]
            val_grid_ten = [torch.from_numpy(val_grid[:, :2]).cuda()]
            # print(f'val_vox_fea_ten: {val_vox_fea_ten.shape}')
            # print(f'val_pt_fea_ten {val_pt_fea_ten.shape}')
            # print(f'val_grid_ten: {val_grid_ten.shape}')

            predict_labels = self.ptBEVnet(val_pt_fea_ten, val_grid_ten, val_vox_fea_ten)
            predict_labels = torch.argmax(predict_labels, dim=1)
            predict_labels = predict_labels.cpu().detach().numpy()

            test_pred_label = predict_labels[0, val_grid[:, 0], val_grid[:, 1], val_grid[:, 2]]
            test_pred_label = train2SemKITTI(test_pred_label)
            points = lidar_data[0][:test_pred_label.shape[0]]
            test_pred_label = np.expand_dims(test_pred_label, axis=1)
            semantic_feat = torch.from_numpy(test_pred_label).type(torch.FloatTensor).squeeze()
            lidar_data[0,:test_pred_label.shape[0],4] = semantic_feat
        # rospy.loginfo(semantic_feat)


        if self.make_hdmap:
            with torch.no_grad():
                segmentation, embedding, direction = self.model(lidar_data.cuda(), lidar_mask.cuda())

                if self.save_result:
                    self.vis_pred(segmentation, embedding, direction, stamp)       
            torch.cuda.empty_cache()     
        
        if self.publish_joint_pc and self.make_segmentation:
            self.publish_joined_point_cloud(points.cpu().numpy(), stamp, semantic_feat)
        print("all", time.time() - t1)


if __name__ == '__main__':
    node = HDMapNetNode()
