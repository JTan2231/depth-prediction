# depth-prediction

This model takes as input two consecutive images from video, and predicts a depth map for the first as well as the transformation between the two perspectives.
A lot of the code for the loss function is taken from [here](https://github.com/google-research/google-research/tree/master/depth_from_video_in_the_wild).

The code uploaded here *should* work. I'm still in the process of training/uploading weights though.

`display.py` contains code to load the model and display inputs and outputs, but it assumes a certain directory setup. You'll have to setup a file list of jpegs to use.
`dataset.py` expects a folder of TFRecords of the format:
```
tfrecord_format = {
    'image': tf.io.FixedLenFeature([], tf.string),  
    'pose': tf.io.FixedLenFeature([16,], tf.float32)
}
```

## TODO
- Changing training directories, log directories, etc. from launch instead of having to change the code manually.
- Uploading weights
- Cleanup the directory
