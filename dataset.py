import os
import json
import pylab as pl
import numpy as np
import tensorflow as tf
import google.protobuf as protobuf

from random import shuffle, sample, seed
from natsort import natsorted
from dataset_pb2 import Frame, MatrixFloat

seed(11281999)

AUTOTUNE = tf.data.experimental.AUTOTUNE

IMAGENET_MEAN = [0.485, 0.486, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_tfrec_dataset(batch_size, sequence_length, resolution=None, validation=False):
    local_path = "/home/joey/python/tensorflow/projects/odometry/data/images/"

    def image_preprocess(example):
        tfrecord_format = {    
        'image': tf.io.FixedLenFeature([], tf.string),    
        'pose': tf.io.FixedLenFeature([16,], tf.float32)    
        }
            
        example = tf.io.parse_single_example(example, tfrecord_format)
        image = example['image']
        pose = example['pose']

        image = tf.io.decode_jpeg(image)    
        image = tf.image.convert_image_dtype(image, tf.float32)    
        if resolution is not None:    
            image = tf.image.resize(image, resolution)    

        image -= tf.reshape(IMAGENET_MEAN, (1, 1, 3))
        image /= tf.reshape(IMAGENET_STD, (1, 1, 3))

        # flattened 4x4 row-major matrix    
        pose = tf.reshape(pose, (4, 4))

        return image, pose

    def augment(image, pose):
        image = tf.image.random_saturation(image, 0.85, 1.15)
        image = tf.image.random_brightness(image, 0.1)
        image = tf.image.random_contrast(image, 0.85, 1.15)
        image = tf.image.random_hue(image, 0.1)

        return image, pose

    train_files = []
    all_files = [f"{local_path}{s}" for s in os.listdir(local_path)]
    for f in all_files:
        if "kitti" in f:
            if int(f.split('_')[-1].split('.')[0]) < 11:
                train_files.append(f)
        else:
            train_files.append(f)

    all_files = train_files
    dataset = tf.data.TFRecordDataset(all_files).map(image_preprocess, num_parallel_calls=AUTOTUNE).map(augment, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(sequence_length, drop_remainder=True).prefetch(AUTOTUNE)

    return dataset

def get_dataset(batch_size, sequence_length, validation_count=1000, buffer_size=1000, resolution=None, perm=None):
    # both nuScenes and Waymo
    train_dataset = get_tfrec_dataset(batch_size, sequence_length, resolution)
    validation_dataset = None

    train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True).repeat()

    return train_dataset, validation_dataset

