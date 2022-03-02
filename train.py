import gc
import numpy as np
import matplotlib.pyplot as pl
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import tensorflow.keras as keras

from time import time

from effnet import get_model
from loss import rgb_loss_function, matrix_from_angles
from dataset import get_dataset
from tensorflow_graphics.geometry.transformation import euler


tf.random.set_seed(11281999)

DEBUGGING = False

LOGDIR = "logs/testing/3/"
BATCH_SIZE = 4
SEQUENCE_LENGTH = 2
BUFFER_SIZE = 1024
RESOLUTION = (128, 416)
FLIP_LR = True
res = tf.convert_to_tensor(RESOLUTION, dtype=tf.float32)

LOAD_WEIGHTS = False
WEIGHT_NUM = 25
WEIGHT_PATH = f"weights/weights{WEIGHT_NUM}.h5"

STARTING_STEP = WEIGHT_NUM * 1000 if LOAD_WEIGHTS else 0

if DEBUGGING:
    LOGDIR = "logs/testing/"

inp = keras.Input((RESOLUTION[0], RESOLUTION[1], 6), name="stacked_input_images")
model = get_model(inp, BATCH_SIZE, RESOLUTION)
if LOAD_WEIGHTS:
    model.load_weights(WEIGHT_PATH)
model.summary()
LR = 1e-4
opt = keras.optimizers.Adam(learning_rate=LR)

def switch_images(images):
    return tf.concat([images[...,-3:], images[...,0:3]], axis=-1)

# Increasing memory consumption w/tf.function ??
#@tf.function
def train_step(image_stack, model, batch_size, reso, writer, step):
    bs = tf.constant(batch_size)
    total_loss = 0.
    frames = min(SEQUENCE_LENGTH-1, (image_stack.shape[-1]//3)-1)
    for i in range(frames):
        j = i + 1

        images = tf.concat([image_stack[...,3*i:3*(i+1)], image_stack[...,3*j:3*(j+1)]], axis=-1)

        with tf.GradientTape() as tape:
            depths, transformation, intrinsics = model(images)
            depths2, transformation2, intrinsics2 = model(switch_images(images))

            intrinsics = 0.5 * (intrinsics + intrinsics2)

            rotation, translation = transformation
            rotation2, translation2 = transformation2

            translation = tf.tile(translation, [1, RESOLUTION[0], RESOLUTION[1], 1])
            translation2 = tf.tile(translation2, [1, RESOLUTION[0], RESOLUTION[1], 1])

            transformation, transformation2 = (rotation, translation), (rotation2, translation2)

            im1 = images[...,0:3]
            im2 = images[...,-3:]

            loss = 0.
            for depth, depth2 in zip(depths, depths2):
                depth, depth2 = tf.image.resize(depth, RESOLUTION), tf.image.resize(depth2, RESOLUTION)
                l, d, resampled = rgb_loss_function(im1, im2, depth, depth2, transformation, transformation2, intrinsics, writer, step)
                l2, x, y = rgb_loss_function(im2, im1, depth2, depth, transformation2, transformation, intrinsics, writer, step)

                loss += l + l2

            loss /= len(depths)

        gradients = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.trainable_variables))

        total_loss += loss

    return total_loss / frames, d, transformation, resampled#, pixel_x, pixel_y, intrinsics1

def preprocess_images(images):
    return tf.concat([images[:,n] for n in range(images.shape[1])], axis=-1)

transform_met = keras.metrics.Mean()

TRAIN_STEPS = 1000000
SAMPLE_FREQUENCY = 500
WEIGHT_FREQUENCY = 5000

train_dataset, validation_dataset = get_dataset(BATCH_SIZE,
                                                SEQUENCE_LENGTH,
                                                validation_count=1,
                                                buffer_size=BUFFER_SIZE,
                                                resolution=RESOLUTION)

it = iter(train_dataset)

def prep_data(images, poses):
    images = tf.concat([images[:,0:1,:,:,:], images[:,-1:,:,:,:]], axis=1)

    images = preprocess_images(images)
    labels = tf.zeros((4, 4), dtype=tf.float32)

    return images, labels

writer = tf.summary.create_file_writer(LOGDIR)

IMAGENET_MEAN = tf.reshape([0.485, 0.486, 0.406], (1, 1, 1, 3))
IMAGENET_STD = tf.reshape([0.229, 0.224, 0.225], (1, 1, 1, 3))

def get_abbr(number):
    out = str(number)
    if number > 1000:
        out = f"{number // 1000}"

    return out

def print_settings():
    print("---------------------------")
    print("DEBUGGING =", DEBUGGING)
    print("LOGDIR =", LOGDIR)
    print("STARTING_STEP =", STARTING_STEP)
    print("WEIGHT_NUM =", WEIGHT_NUM)
    print("BATCH_SIZE =", BATCH_SIZE)
    print("SEQUENCE_LENGTH =", SEQUENCE_LENGTH)
    print("LEARNING_RATE =", LR)
    print("BUFFER_SIZE =", BUFFER_SIZE)
    print("RESOLUTION =", RESOLUTION)
    print("FLIP_LR =", FLIP_LR)
    print("SAMPLE_FREQUENCY =", SAMPLE_FREQUENCY)
    print("WEIGHT_FREQUENCY =", WEIGHT_FREQUENCY)
    print("---------------------------")

fig = pl.figure(figsize=(10, 7))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

ax1.set_title("rgb1")
ax2.set_title("rgb2_resampled")
ax3.set_title("depth")
ax4.set_title("disparity")

i1 = ax1.imshow(np.zeros((*RESOLUTION, 3), dtype=np.float32))
i2 = ax2.imshow(np.zeros((*RESOLUTION, 3), dtype=np.float32))
i3 = ax3.imshow(np.zeros((*RESOLUTION, 3), dtype=np.float32), cmap=pl.get_cmap("inferno"), vmin=0, vmax=1)
i4 = ax4.imshow(np.zeros((*RESOLUTION, 1), dtype=np.float32), cmap=pl.get_cmap("inferno"), vmin=0, vmax=1)

print_settings()

if input("Continue? ") != "y":
    exit()

i = STARTING_STEP
while i < TRAIN_STEPS:
    t0 = time()
    images, poses = it.get_next()

    images, labels = prep_data(images, poses)
    if FLIP_LR:
        images = tf.image.random_flip_left_right(images)

    im1, im2 = images[0,:,:,:3], images[0,:,:,3:]
    im1 = (im1 * IMAGENET_STD) + IMAGENET_MEAN
    im2 = (im2 * IMAGENET_STD) + IMAGENET_MEAN
    
    # bad data
    if images.shape[0] != BATCH_SIZE:
        continue

    t1 = time()

    t0 = time()
    train_step_out = train_step(images, model, BATCH_SIZE, res, writer, i)
    t1 = time()

    loss, depth, transformation_prediction, image2_resample = train_step_out
    image2_resample = (image2_resample * IMAGENET_STD) + IMAGENET_MEAN
    transform_met(loss)

    if (i+1) % SAMPLE_FREQUENCY == 0:
        print(f"Sample {(i+1) // SAMPLE_FREQUENCY} of {TRAIN_STEPS // SAMPLE_FREQUENCY}:")

        interval = 0.5

        disp = 1. / depth[0]
        disp /= tf.math.reduce_mean(disp)
        disp /= tf.math.reduce_max(disp)

        depth_scaled = (depth[0] / tf.math.reduce_mean(depth[0])) / tf.math.reduce_max(depth[0])

        i1.set_data(tf.clip_by_value(im1[0], 0., 1.))
        i2.set_data(tf.clip_by_value(image2_resample[0], 0., 1.))
        i3.set_data(depth_scaled)
        i4.set_data(disp)

        pl.draw()
        pl.pause(interval)

    if (i+1) % WEIGHT_FREQUENCY == 0 and not DEBUGGING:
        model.save_weights(f"weights/weights{get_abbr(i+1)}.h5")

    print(f"Sequence {i} of {TRAIN_STEPS}, average loss: {transform_met.result()}      \r", end='')
    i += 1
