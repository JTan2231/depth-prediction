# depth-prediction
As of commit [16d4167](https://github.com/JTan2231/depth-prediction/commit/16d4167399e78193e0c4f95f4a0eae30e925de24), this is a somewhat working model.
Weights are in the `logs/run2/` directory, trained on the 2020 Waymo Open, [nuScenes](https://www.nuscenes.org/nuscenes), and [KITTI Odometry](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) datasets.
`display.py` contains code to load the model and display inputs and outputs, but it assumes a certain directory setup. You'll have to setup a file list of jpegs to use.
