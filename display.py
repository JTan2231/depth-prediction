import os
import matplotlib.pyplot as pl
import tensorflow as tf
from random import randrange
from natsort import natsorted
#from mod_resnet18 import depth_prediction_resnet18unet, get_model
from effnet import get_model

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

IMAGENET_MEAN = tf.reshape([0.485, 0.486, 0.406], (1, 1, 3))
IMAGENET_STD = tf.reshape([0.229, 0.224, 0.225], (1, 1, 3))

RESOLUTION = (128, 416)

kitti_base = "./data/kitti/dataset/sequences/"
kitti_folders = [kitti_base+s+'/' for s in os.listdir(kitti_base)]
kitti_images = []
for folder in kitti_folders:
    kitti_images += natsorted([folder+"image_2/"+s for s in os.listdir(folder+"image_2/")])

nuscenes_base = "./data/nuscenes/sweeps/CAM_FRONT/"
nuscenes_images = natsorted([nuscenes_base+s for s in os.listdir(nuscenes_base)])

#all_images = nuscenes_images + kitti_images
all_images = kitti_images

def preprocess(filepath):
    image = tf.io.read_file(filepath)
    image = tf.io.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, RESOLUTION)

    image -= IMAGENET_MEAN
    image /= IMAGENET_STD

    return image

def deimagenet(image):
    image *= IMAGENET_STD
    image += IMAGENET_MEAN

    return image

def together(im1, im2):
    images = tf.concat([im1, im2], axis=-1)
    images = tf.expand_dims(images, axis=0)

    return images

inp = tf.keras.Input((RESOLUTION[0], RESOLUTION[1], 6))
model = get_model(inp, 1, RESOLUTION)
model.load_weights("weights/weights200.h5")

i = randrange(len(all_images)-1)
im1, im2 = preprocess(all_images[i]), preprocess(all_images[i+1])

images = together(im1, im2)

depth, _, _ = model(images)
depth = depth[0][0]

print(tf.math.reduce_mean(im2 - im1))

images = together(im2, im1)

depth2, _, _ = model(images)
depth2 = depth2[0][0]

# create figure
fig = pl.figure(figsize=(10, 7))
  
# setting values to rows and column variables
rows = 2
columns = 2
 
fig.add_subplot(rows, columns, 1)
  
pl.imshow(deimagenet(im1))
pl.axis('off')
pl.title("im1")
  
fig.add_subplot(rows, columns, 2)
  
pl.imshow(1. / depth)
pl.axis('off')
pl.title("disparity1")

fig.add_subplot(rows, columns, 3)
  
pl.imshow(deimagenet(im2))
pl.axis('off')
pl.title("im2")
  
fig.add_subplot(rows, columns, 4)
  
pl.imshow(1. / depth2)
pl.axis('off')
pl.title("disparity2")

pl.show()
