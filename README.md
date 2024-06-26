# HDMapNet

**Repository fork with additional features for training and inference of the model**

**TODO**
- [x] Wandb logging;
- [x] Checkpoint saving (best score and last epoch);
- [ ] Docker container;
- [ ] Model config. 



### Preparation
1. Download  [nuScenes dataset](https://www.nuscenes.org/) and put it to `dataset/` folder.

2. Install dependencies (python==3.8)
- `pip install tqdm`
- `pip install numpy`
- `pip install matplotlib`
- `pip install nuscenes-devkit`

3. Install pytorch from `https://pytorch.org/get-started/locally/`

### Label
Run `python vis_label.py ` for demo of vectorized labels. The visualizations are in `dataset/nuScenes/samples/GT`.

### Training

Run `python train.py --instance_seg --direction_pred --version [v1.0-trainval or v1.0-mini] --logdir [output place]`. 

### Evaluation
Before running the evaluation code, you should get the `submission.json` file first, which can be generated by the following command.
```
python export_gt_to_json.py
```

Run `python evaluate.py --modelf [checkpoint path]` for evaluation. The script accepts vectorized or rasterized maps as input. For vectorized map, We firstly rasterize the vectors to map to do evaluation. For rasterized map, you should make sure the line width=1.

Below is the format for vectorized submission:

```
vectorized_submission {
    "meta": {
        "use_camera":   <bool>  -- Whether this submission uses camera data as an input.
        "use_lidar":    <bool>  -- Whether this submission uses lidar data as an input.
        "use_radar":    <bool>  -- Whether this submission uses radar data as an input.
        "use_external": <bool>  -- Whether this submission uses external data as an input.
        "vector":        true   -- Whether this submission uses vector format.
    },
    "results": {
        sample_token <str>: List[vectorized_line]  -- Maps each sample_token to a list of vectorized lines.
    }
}

vectorized_line {
    "pts":               List[<float, 2>]  -- Ordered points to define the vectorized line.
    "pts_num":           <int>,            -- Number of points in this line.
    "type":              <0, 1, 2>         -- Type of the line: 0: ped; 1: divider; 2: boundary
    "confidence_level":  <float>           -- Confidence level for prediction (used by Average Precision)
}
```

For rasterized submission, the format is:

```
rasterized_submisson {
    "meta": {
        "use_camera":   <bool>  -- Whether this submission uses camera data as an input.
        "use_lidar":    <bool>  -- Whether this submission uses lidar data as an input.
        "use_radar":    <bool>  -- Whether this submission uses radar data as an input.
        "use_external": <bool>  -- Whether this submission uses external data as an input.
        "vector":       false   -- Whether this submission uses vector format.
    },
    "results": {
        sample_token <str>: {  -- Maps each sample_token to a list of vectorized lines.
            "map": [<float, (C, H, W)>],         -- Raster map of prediction (C=0: ped; 1: divider 2: boundary). The value indicates the line idx (start from 1).
    	    "confidence_level": Array[float],    -- confidence_level[i] stands for confidence level for i^th line (start from 1). 
        }
    }
}
```

Run `python export_gt_to_json.py` to get a demo of vectorized submission. Run `python export_gt_to_json.py --raster` for rasterized submission.

Run `python export_pred_to_json.py --modelf [checkpoint]` to get submission file for trained model.

### Citation
If you found this paper or codebase useful, please cite our paper:
```
@misc{li2021hdmapnet,
      title={HDMapNet: An Online HD Map Construction and Evaluation Framework}, 
      author={Qi Li and Yue Wang and Yilun Wang and Hang Zhao},
      year={2021},
      eprint={2107.06307},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
