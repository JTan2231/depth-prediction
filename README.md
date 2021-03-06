# depth-prediction

This model takes as input two consecutive images from video, and predicts a depth map for the first as well as the transformation between the two perspectives.
A lot of the code for the loss function is taken from [here](https://github.com/google-research/google-research/tree/master/depth_from_video_in_the_wild).

The primary motivation for this repository was to update the aforementioned link to a TensorFlow 2.0 repository, because I think the concept is _really_ cool. I plan on incorporating later advances in the topic in this repo as well.

The code uploaded here *should* work. I'm still in the process of training/uploading weights though.

`display.py` contains code to load the model and display inputs and outputs, but it assumes a certain directory setup. You'll have to setup a file list of jpegs to use.
`dataset.py` expects a folder of TFRecords of the format:
```
tfrecord_format = {
    'image': tf.io.FixedLenFeature([], tf.string),  
    'pose': tf.io.FixedLenFeature([16,], tf.float32)
}
```

You can see an example below, about 200k training steps in:

![cropped_depth](https://user-images.githubusercontent.com/37962780/158003658-bfd17b05-88b2-4cdf-8dd7-7ae751d8cdfe.gif)
![cropped_rgb](https://user-images.githubusercontent.com/37962780/158003665-d93d09a0-4df3-4398-a813-c94e46a4844f.gif)

You can get a similar output for the above using the weights currently in the releases section. I'm still working on completing training.



## TODO
- Changing training directories, log directories, etc. from launch instead of having to change the code manually.
- Uploading weights
- Cleanup the directory
- Complete training
