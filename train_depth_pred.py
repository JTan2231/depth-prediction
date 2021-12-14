import numpy as np
import matplotlib.pyplot as pl
import tensorflow as tf
import tensorflow.keras as keras

from time import time

#from net import DepthEgoNet
from mod_resnet18 import depth_prediction_resnet18unet, get_model
from loss import rgb_loss_function, matrix_from_angles
from dataset import get_dataset
from tensorflow_graphics.geometry.transformation import euler

#tf.debugging.enable_check_numerics()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass

BATCH_SIZE = 4
SEQUENCE_LENGTH = 4
BUFFER_SIZE = 1024
RESOLUTION = (128, 416)
res = tf.convert_to_tensor(RESOLUTION, dtype=tf.float32)

layer_count = 4
blocks_per_layer = 2
batch_input_shape = (BATCH_SIZE, RESOLUTION[0], RESOLUTION[1], 3)

inp = keras.Input((RESOLUTION[0], RESOLUTION[1], 6))
#out = depth_prediction_resnet18unet(inp)#DepthEgoNet(1, layer_count, blocks_per_layer, batch_input_shape, full_summary=True)
#model = keras.Model(inp, out)
model = get_model(inp)
opt = keras.optimizers.Adam(learning_rate=1e-4)

def repackage_transformation(net_out):
    euler = net_out[:,:3]
    #translation = tf.concat([net_out[:,3:], tf.tile([[1.]], [BATCH_SIZE, 1])], axis=-1)
    translation = tf.reshape(net_out[:,3:], (BATCH_SIZE, 3, 1))

    rotation_matrix = matrix_from_angles(euler)

    transformation_matrix = tf.concat([rotation_matrix, translation], axis=-1)
    transformation_matrix = tf.concat([transformation_matrix, tf.tile([[[0., 0., 0., 1.]]], [BATCH_SIZE, 1, 1])], axis=1)

    return transformation_matrix

# input shape: [batch_size, 3, 4]
def invert_transformation(transformation):
    rotation = transformation[:,:3,:3]
    translation = transformation[:,:3,3]

    rotation = tf.transpose(rotation, perm=[0, 2, 1])
    translation = -tf.reshape(translation, (transformation.shape[0], 3, 1))

    transformation_inverse = tf.concat([rotation, translation], axis=-1)
    transformation_inverse = tf.concat([transformation_inverse, tf.tile([[[0., 0., 0., 1.]]], [BATCH_SIZE, 1, 1])], axis=1)

    return transformation_inverse

def switch_images(images):
    return tf.concat([images[...,-3:], images[...,0:3]], axis=-1)

def train_step(images):
    bs = tf.constant(BATCH_SIZE)
    with tf.GradientTape() as tape:
        depth, transformation, foci, offsets = model(images)
        depth2, transformation2, foci2, offsets2 = model(switch_images(images))

        transformation = repackage_transformation(transformation)
        transformation2 = repackage_transformation(transformation2)

        foci, offsets = foci * res, offsets * res
        foci2, offsets2 = foci2 * res, offsets2 * res
    
        foci = tf.linalg.diag(foci)    
        foci2 = tf.linalg.diag(foci2)    
    
        intrinsic_mat = tf.concat([foci, tf.expand_dims(offsets, -1)], axis=2)    
        last_row = tf.tile([[[0.0, 0.0, 1.0]]], [bs, 1, 1])    
        intrinsics = tf.concat([intrinsic_mat, last_row], axis=1)

        intrinsic_mat = tf.concat([foci2, tf.expand_dims(offsets2, -1)], axis=2)    
        intrinsics2 = tf.concat([intrinsic_mat, last_row], axis=1)

        im1 = images[...,0:3]
        im2 = images[...,-3:]

        loss, depth, resampled = rgb_loss_function(im1, im2, depth, transformation, intrinsics)
        loss2, x, y = rgb_loss_function(im2, im1, depth2, transformation2, intrinsics2)

        loss += loss2

    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, depth, transformation, resampled#, pixel_x, pixel_y, intrinsics1

def eval_step(images):
    depth, transformation, intrinsics = model.call_no_build(images, training=True)

    #transformation = tf.concat([transformation, [[1.,]]], -1)
    transformation = tf.reshape(transformation, (BATCH_SIZE, 3, 4))


    intrinsics = tf.convert_to_tensor([[[intrinsics[0,0],               0, intrinsics[0,1]],
                                        [              0, intrinsics[0,2], intrinsics[0,3]],
                                        [              0,               0,              1]]], dtype=tf.float32)

    loss = loss_function(images, depth, confidence, transformation, intrinsics)

    return loss, transformation

def preprocess_images(images):
    return tf.concat([images[:,n] for n in range(images.shape[1])], axis=-1)

transform_met = keras.metrics.Mean()

to_string = lambda x: np.array2string(x.numpy(), floatmode='fixed', formatter={'float_kind':lambda x: "%.6f" % x})

def validate():
    transform_met.reset_states()

    validation_count = (i+1) // VALIDATION_FREQUENCY
    with open("depth_validation_log.txt", "a") as validation_log:
        for v, (images, poses) in enumerate(validation_dataset):
            images = preprocess_images(images)

            labels = poses[:,-1] - poses[:,0]
            if tf.math.reduce_any(tf.math.abs(labels) > 5.).numpy():
                continue

            loss, transformation = eval_step(images)

            transform_met(loss)

            transformation_string = to_string(transformation)
            labels_string = to_string(labels)

            validation_log.write(transformation_string + "\n")
            validation_log.write(labels_string + "\n")
            validation_log.write(f"{v} of {VALIDATION_COUNT} average loss: {transform_met.result().numpy()}\n\n")

            print(f"Validation {validation_count} of {TRAIN_STEPS//VALIDATION_FREQUENCY}. Batch {v} of {VALIDATION_COUNT}, average loss: {transform_met.result()}    \r", end='')

    model.save_weights("depth_weights.h5")
    transform_met.reset_states()

TRAIN_STEPS = 1000000
SAMPLE_FREQUENCY = 10000
VALIDATION_FREQUENCY = 25000000
VALIDATION_COUNT = 1

train_dataset, validation_dataset = get_dataset(BATCH_SIZE,
                                                SEQUENCE_LENGTH,
                                                validation_count=VALIDATION_COUNT,
                                                buffer_size=BUFFER_SIZE,
                                                resolution=RESOLUTION)
it = iter(train_dataset)

def prep_data(images, poses):
    images = tf.concat([images[:,0:1,:,:,:], images[:,-1:,:,:,:]], axis=1)

    images = preprocess_images(images)
    labels = euler.from_rotation_matrix(poses[:,-1,:3,:3]) - euler.from_rotation_matrix(poses[:,0,:3,:3])
    labels = tf.concat([labels, poses[:,-1,:3,-1] - poses[:,0,:3,-1]], axis=-1)

    return images, labels

i = 0
while i < TRAIN_STEPS:
    #if (i+1) % VALIDATION_FREQUENCY == 0:
    #    validate()

    #if (i+1) % (TRAIN_STEPS//5) == 0:
    #    keras.backend.set_value(opt.learning_rate, opt.learning_rate*0.5)

    t0 = time()
    images, poses = it.get_next()

    images, labels = prep_data(images, poses)
    
    # bad data
    if tf.math.reduce_any(tf.math.abs(labels) > 25.).numpy():
        continue
    # bad data
    if tf.math.reduce_all(tf.math.abs(labels[:,-3:-1]) < 1.0).numpy():
        continue

    t1 = time()

    t0 = time()
    train_step_out = train_step(images)
    t1 = time()

    if train_step_out is None:
        print("CHECK")
        continue

    #transformation_prediction, depth_estimation, confidences, loss = train_step_out
    loss, depth, transformation_prediction, image2_resample = train_step_out#, pixel_x, pixel_y, intrinsics = train_step_out
    if loss > 0.:
        transform_met(loss)

    if (i+1) % SAMPLE_FREQUENCY == 0:
        print(f"Sample {(i+1) // SAMPLE_FREQUENCY} of {TRAIN_STEPS // SAMPLE_FREQUENCY}:")
        print(transformation_prediction)
        print(labels)

        pl.imshow(images[0,:,:,:3])
        pl.draw()
        pl.pause(3)

        pl.imshow(images[0,:,:,3:])
        pl.draw()
        pl.pause(3)

        disparity = 1. / depth
        disparity_mean = tf.math.reduce_mean(disparity, axis=[1,2], keepdims=True)
        disparity = disparity / disparity_mean

        pl.imshow(image2_resample[0])
        pl.draw()
        pl.pause(3)

        pl.imshow(disparity[0,:,:], cmap=pl.get_cmap("inferno"))
        pl.draw()
        pl.pause(3)

        pl.imshow(depth[0], cmap=pl.get_cmap("inferno"))
        pl.draw()
        pl.pause(0.1)

        model.save_weights("depth_prediction_weights.h5")

        #pl.imshow(confidence[0,:,:,0], cmap=pl.get_cmap("inferno"))
        #pl.draw()
        #pl.pause(0.1)

    print(f"Sequence {i} of {TRAIN_STEPS}, average loss: {transform_met.result()}      \r", end='')
    i += 1
