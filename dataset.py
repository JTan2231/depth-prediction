import os
import json
import pylab as pl
import numpy as np
import tensorflow as tf
import google.protobuf as protobuf

from random import shuffle, sample
from natsort import natsorted
from dataset_pb2 import Frame, MatrixFloat

AUTOTUNE = tf.data.experimental.AUTOTUNE

def map_decorator(function):
    def wrapper(value):
        return tf.py_function(function, [value], Tout=[tf.float32, tf.float32])

    return wrapper

# returns the front-camera image given the frame
# NOTE: images are taken at framerate of 10 FPS
def get_waymo_dataset(batch_size, sequence_length, resolution=None, validation=False):
    local_path = "/home/joey/python/tensorflow/projects/odometry/data/images/"
    external_path = "/media/joey/Initiate/waymo_dataset/Training/"

    def frame_preprocess(frame_proto):
        frame = Frame()
        frame.ParseFromString(frame_proto.numpy())
        image_proto = frame.images[0]
        image = tf.io.decode_jpeg(image_proto.image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        if resolution is not None:
            image = tf.image.resize(image, resolution)

        # flattened 4x4 row-major matrix
        pose = image_proto.pose.transform
        pose[11] = 0 # z is always zero to match Nuscenes dataset
        pose = tf.reshape(pose, (4, 4))

        return image, pose

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

        # flattened 4x4 row-major matrix    
        pose = tf.reshape(pose, (4, 4))

        return image, pose

    #local_files = [local_path + i + "/" + s for i in os.listdir(local_path) for s in os.listdir(local_path+i)]#[local_path + s for s in os.listdir(local_path)]
    #external_files = [external_path + i + "/" + s for i in os.listdir(external_path) for s in os.listdir(external_path+i) if i != "corrupted"]
    all_files = [f"{local_path}{s}" for s in os.listdir(local_path)]#local_files# + external_files
    shuffle(all_files)

    validation_files = all_files[-50:]
    all_files = all_files
    dataset = tf.data.TFRecordDataset(all_files).map(image_preprocess, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(sequence_length, drop_remainder=True).prefetch(AUTOTUNE)

    #validation_dataset = tf.data.TFRecordDataset(validation_files).map(map_decorator(frame_preprocess), num_parallel_calls=AUTOTUNE)
    #validation_dataset = validation_dataset.batch(sequence_length, drop_remainder=True).prefetch(AUTOTUNE)
    validation_dataset = None

    return dataset, validation_dataset

# quaternion is a list [x, y, z, w]
def quat_to_mat(quaternion):
    i, j, k, r = quaternion
    i = float(i)
    j = float(j)
    k = float(k)
    r = float(r)

    rotation_matrix = np.array([[1-2*(j*j+k*k), 2*(i*j-k*r), 2*(i*k+j*r)],
                                [2*(i*j+k*r), 1-2*(i*i+k*k), 2*(j*k-i*r)],
                                [2*(i*k-j*r), 2*(j*k+i*r), 1-2*(i*i+j*j)]])

    return rotation_matrix

# probably a cleaner way of doing this
# but i'm lazy as of this writing
def get_transformation_matrix(quaternion, translation):
    r = quat_to_mat(quaternion)
    t = translation

    transformation_matrix = [[r[0][0], r[0][1], r[0][2], float(t[0])],
                             [r[1][0], r[1][1], r[1][2], float(t[1])],
                             [r[2][0], r[2][1], r[2][2], float(t[2])],
                             [0.,      0.,      0.,      1.         ]]

    return transformation_matrix

# NOTE: images are taken at framerate of 20 FPS
def get_nuscenes_dataset(batch_size, sequence_length, resolution=None):
    trainval_path = "/home/joey/python/tensorflow/projects/odometry/data/nuscenes/trainval/"
    corrupted_files = ["data/nuscenes/trainval/CAM_FRONT/n015-2018-08-03-15-21-40+0800__CAM_FRONT__1533281283262460.jpg"]

    def image_preprocess(filename):
        image = tf.io.read_file(filename)
        image = tf.io.decode_jpeg(image)

        image = tf.image.convert_image_dtype(image, tf.float32)
        if resolution is not None:
            image = tf.image.resize(image, resolution)

        return image

    filenames = []
    ego_tokens = []

    with open(trainval_path+"labels/file_ego_pairs.txt") as f:
        filenames_ego_tokens = [s for s in f.read().split('\n') if len(s) > 0][::2]
        for pair in filenames_ego_tokens:
            split = pair.split(',')
            if split[0] in corrupted_files:
                print("Corrupted jpg caught.")
                continue

            filenames.append(split[0])
            ego_tokens.append(split[1])

    image_dataset = tf.data.Dataset.from_tensor_slices(filenames)
    image_dataset = image_dataset.map(image_preprocess).batch(sequence_length, drop_remainder=True)

    # get ego poses
    with open(trainval_path+"labels/extracted_poses.json", "r") as f:
        ego_poses = np.array([d['transformation'] for d in json.loads(f.read())]).astype(np.float32)

    ego_dataset = tf.data.Dataset.from_tensor_slices(ego_poses).batch(sequence_length, drop_remainder=True)

    dataset = tf.data.Dataset.zip((image_dataset, ego_dataset))
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset

def get_kitti_odometry_dataset(batch_size, sequence_length, resolution=None, validation=False, sequence10=False):
    sequences_path = "/home/joey/python/tensorflow/projects/odometry/data/kitti/dataset/sequences/"
    poses_path = "/home/joey/python/tensorflow/projects/odometry/data/kitti/dataset/poses/"

    def image_preprocess(filename):
        image = tf.io.read_file(filename)
        image = tf.io.decode_png(image)

        image = tf.image.convert_image_dtype(image, tf.float32)
        if resolution is not None:
            image = tf.image.resize(image, resolution)

        return image

    sequenced_files = []
    sequences = natsorted(os.listdir(sequences_path))[:11]
    if sequence10:
        images_path = sequences_path+"10/image_2/"
        images = [images_path+image for image in os.listdir(images_path)]
        while len(images) % sequence_length != 0:
            images = images[:-1]
    else:
        for sequence in sequences:
            images_path = sequences_path+sequence+"/image_2/"
            images = [images_path+image for image in os.listdir(images_path)]
            while len(images) % sequence_length != 0:
                images = images[:-1]

    sequenced_files.append(natsorted(images))

    if validation and not sequence10:
        validation_sequence_indices = sample(range(len(sequenced_files)), 4)
        validation_sequences = [sequenced_files[i] for i in validation_sequence_indices]

        training_sequence_indices = [i for i in range(len(sequenced_files)) if i not in validation_sequence_indices]
        training_sequences = [sequenced_files[i] for i in training_sequence_indices]

        validation_image_files = []
        training_image_files = []
        for sequence_files in validation_sequences:
            for image in sequence_files:
                validation_image_files.append(image)

        for sequence_files in training_sequences:
            for image in sequence_files:
                training_image_files.append(image)

        validation_image_dataset = tf.data.Dataset.from_tensor_slices(validation_image_files).map(image_preprocess)
        validation_image_dataset = validation_image_dataset.batch(sequence_length, drop_remainder=True)

        training_image_dataset = tf.data.Dataset.from_tensor_slices(training_image_files).map(image_preprocess)
        training_image_dataset = training_image_dataset.batch(sequence_length, drop_remainder=True)

        poses_files = natsorted([poses_path+p for p in os.listdir(poses_path)])
        validation_poses_files = [poses_files[i] for i in validation_sequence_indices]
        training_poses_files = [poses_files[i] for i in training_sequence_indices]

        validation_poses = []
        training_poses = []
        for poses_file in validation_poses_files:
            with open(poses_file) as f:
                contents = f.read().split('\n')[:-1]
                while len(contents) % sequence_length != 0:
                    contents = contents[:-1]

                for matrix in contents:
                    transformation_matrix = np.reshape([float(v) for v in matrix.split(' ')], (3, 4))
                    transformation_matrix = np.concatenate([transformation_matrix, [[0., 0., 0., 1.]]], axis=0).astype(np.float32)

                    validation_poses.append(transformation_matrix)

        for poses_file in training_poses_files:
            with open(poses_file) as f:
                contents = f.read().split('\n')[:-1]
                while len(contents) % sequence_length != 0:
                    contents = contents[:-1]

                for matrix in contents:
                    transformation_matrix = np.reshape([float(v) for v in matrix.split(' ')], (3, 4))
                    transformation_matrix = np.concatenate([transformation_matrix, [[0., 0., 0., 1.]]], axis=0).astype(np.float32)

                    training_poses.append(transformation_matrix)

        validation_poses_dataset = tf.data.Dataset.from_tensor_slices(validation_poses)
        validation_poses_dataset = validation_poses_dataset.batch(sequence_length, drop_remainder=True)

        training_poses_dataset = tf.data.Dataset.from_tensor_slices(training_poses)
        training_poses_dataset = training_poses_dataset.batch(sequence_length, drop_remainder=True)

        validation_dataset = tf.data.Dataset.zip((validation_image_dataset, validation_poses_dataset))
        training_dataset = tf.data.Dataset.zip((training_image_dataset, training_poses_dataset))

        return training_dataset, validation_dataset

    else:
        all_image_files = []
        for sequence_files in sequenced_files:
            for image in sequence_files:
                all_image_files.append(image)


        image_dataset = tf.data.Dataset.from_tensor_slices(all_image_files).map(image_preprocess)
        image_dataset = image_dataset.batch(sequence_length, drop_remainder=True)

        poses_files = natsorted([poses_path+p for p in os.listdir(poses_path)])
        if sequence10:
            poses_files = poses_files[-1:]

        poses = []
        for poses_file in poses_files:
            with open(poses_file) as f:
                contents = f.read().split('\n')[:-1]
                while len(contents) % sequence_length != 0:
                    contents = contents[:-1]

                for matrix in contents:
                    transformation_matrix = np.reshape([float(v) for v in matrix.split(' ')], (3, 4))
                    transformation_matrix = np.concatenate([transformation_matrix, [[0., 0., 0., 1.]]], axis=0).astype(np.float32)

                    poses.append(transformation_matrix)

        poses_dataset = tf.data.Dataset.from_tensor_slices(poses)
        poses_dataset = poses_dataset.batch(sequence_length, drop_remainder=True)

        dataset = tf.data.Dataset.zip((image_dataset, poses_dataset))

        return dataset

def get_dataset(batch_size, sequence_length, validation_count=1000, buffer_size=1000, resolution=None, perm=None):
    waymo_dataset, validation_dataset = get_waymo_dataset(batch_size, sequence_length, resolution)
    #nuscenes_dataset = get_nuscenes_dataset(batch_size, sequence_length, resolution)
    #kitti_dataset = get_kitti_odometry_dataset(batch_size, sequence_length, resolution)

    """names = ["waymo", "nuscenes", "kitti"]
    sets = { names[0]: waymo_dataset, names[1]: nuscenes_dataset, names[2]: kitti_dataset }

    if perm is not None:
        assert(len(perm) < 3)
        train_dataset = None
        for p in perm:
            if train_dataset is not None:
                train_dataset = train_dataset.concatenate(sets[names[p]])
            else:
                train_dataset = sets[names[p]]

        missing = None
        for i in range(3):
            if i not in perm:
                missing = i

        validation_dataset = sets[names[missing]]
        validation_name = names[missing]"""

    #total_dataset = nuscenes_dataset.concatenate(kitti_dataset).concatenate(waymo_dataset)
    train_dataset = waymo_dataset
    validation_dataset = None

    #train_dataset = total_dataset.skip(validation_count).shuffle(buffer_size).batch(batch_size, drop_remainder=True).repeat()
    #validation_dataset = total_dataset.take(validation_count).batch(batch_size, drop_remainder=True)

    train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True).repeat()
    #train_dataset = nuscenes_dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True).repeat()
    #validation_dataset = validation_dataset.batch(batch_size, drop_remainder=True)

    return train_dataset, validation_dataset#, validation_name

