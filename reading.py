import tensorflow as tf

files = ['train0.tfrec']

dataset = tf.data.TFRecordDataset(files)

def read_tfrecord(example):
    tfrecord_format = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'pose': tf.io.FixedLenFeature([16,], tf.float32)
    }

    example = tf.io.parse_single_example(example, tfrecord_format)
    print(example['image'])
    print(example['pose'])

for data in dataset:
    read_tfrecord(data)
