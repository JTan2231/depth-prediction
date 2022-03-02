import os
import sys
import matplotlib.pyplot as pl
import tensorflow as tf
from random import randrange
from natsort import natsorted
from effnet import get_model

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

IMAGENET_MEAN = tf.reshape([0.485, 0.486, 0.406], (1, 1, 3))
IMAGENET_STD = tf.reshape([0.229, 0.224, 0.225], (1, 1, 3))

RESOLUTION = (128, 416)

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

if __name__ == "__main__":
    if len(sys.argv) == 1:
        exit("Error: -w <weights> -i <image (jpg)>")

    opts = [opt[1:] for opt in sys.argv[1:] if opt.startswith("-")]
    args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

    arg_dict = dict()
    for o, a in zip(opts, args):
        arg_dict[o] = a

    weights = ''
    if arg_dict.get('w', '') == '':
        exit("Error: No weights. python display.py -w <weights> -i <image (jpg)>")
    else:
        weights = arg_dict['w']

    image = ''
    if arg_dict.get('i', '') == '':
        exit("Error: No image. python display.py -w <weights> -i <image (jpg)>")
    else:
        image = arg_dict['i']

    inp = tf.keras.Input((RESOLUTION[0], RESOLUTION[1], 6))
    model = get_model(inp, 1, RESOLUTION)
    model.load_weights(weights)

    im = preprocess(image)
    images = together(im, im)

    depth, _, _ = model(images)
    depth = depth[0][0]

    # create figure
    fig = pl.figure(figsize=(10, 4))
      
    # setting values to rows and column variables
    rows = 1
    columns = 2
     
    fig.add_subplot(rows, columns, 1)
      
    pl.imshow(deimagenet(im))
    pl.axis('off')
    pl.title("image")
      
    fig.add_subplot(rows, columns, 2)
      
    pl.imshow(1. / depth)
    pl.axis('off')
    pl.title("disparity")

    pl.show()
