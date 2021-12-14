import os
from pprint import pprint
import tensorflow as tf
from dataset_pb2 import Frame, MatrixFloat

AUTOTUNE = tf.data.experimental.AUTOTUNE

def map_decorator(function):
    def wrapper(value):
        return tf.py_function(function, [value], Tout=[tf.float32, tf.float32])

    return wrapper

local_path = "/home/joey/python/tensorflow/projects/odometry/data/"    
external_path = "/media/joey/Initiate/waymo_dataset/Training/"    

def frame_preprocess(frame_proto):    
    frame = Frame()    
    frame.ParseFromString(frame_proto.numpy())    
    image_proto = frame.images[0]    
    image = image_proto.image#tf.io.decode_jpeg(image_proto.image)    
    #image = tf.image.convert_image_dtype(image, tf.float32)    
    #if resolution is not None:    
    #    image = tf.image.resize(image, resolution)    

    # flattened 4x4 row-major matrix    
    pose = list(image_proto.pose.transform)
    #pose[11] = 0 # z is always zero to match Nuscenes dataset    
    #pose = tf.reshape(pose, (4, 4))    

    return image, pose    

local_files = [local_path + i + "/" + s for i in os.listdir(local_path) for s in os.listdir(local_path+i)]#[local_path + s for s in os.listdir(local_path)]             
#external_files = [external_path + i + "/" + s for i in os.listdir(external_path) for s in os.listdir(external_path+i) if i != "corrupted"]    
all_files = local_files

dataset = tf.data.TFRecordDataset(all_files)

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(image, pose):
    feature = {
        'image': _bytes_feature(image),
        'pose': _float_feature(pose)
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

    return example_proto.SerializeToString()

record_count = 0
written = 0
writer = tf.io.TFRecordWriter(f"data/images/train{record_count}.tfrec")
for i, data in enumerate(dataset):
    print(f"{written} images written     \r", end='')
    if (i+1) % 2000 == 0:
        record_count += 1
        del writer
        writer = tf.io.TFRecordWriter(f"data/images/train{record_count}.tfrec")

    image, pose = frame_preprocess(data)
    example = serialize_example(image, pose)

    writer.write(example)
    written += 1
