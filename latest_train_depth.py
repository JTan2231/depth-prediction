import pylab as pl
import tensorflow as tf

from loss import rgb_loss_function, matrix_from_angles
from unused_loss import loss_function
from dataset import get_dataset
from latest_depth_net import get_model
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

model = get_model()
#model.load_weights("depth_prediction_weights.h5")
opt = tf.keras.optimizers.Adam(learning_rate=1e-4)

def repackage_transformation(net_out):    
    euler = net_out[:,:3]    
    #translation = tf.concat([net_out[:,3:], tf.tile([[1.]], [BATCH_SIZE, 1])], axis=-1)    
    translation = tf.reshape(net_out[:,3:], (net_out.shape[0], 3, 1))
    
    rotation_matrix = matrix_from_angles(euler)    
    
    transformation_matrix = tf.concat([rotation_matrix, translation], axis=-1)    
    transformation_matrix = tf.concat([transformation_matrix, tf.tile([[[0., 0., 0., 1.]]], [net_out.shape[0], 1, 1])], axis=1)    
    
    return transformation_matrix

def repackage_intrinsics(foci, offset):
    foci = tf.linalg.diag(foci)
    intrinsics = tf.concat([foci, tf.expand_dims(offset, -1)], axis=2)
    intrinsics = tf.concat([intrinsics, tf.tile([[[0.0, 0.0, 1.0]]], [foci.shape[0], 1, 1])], axis=1)

    return intrinsics

@tf.function(input_signature=[tf.TensorSpec((BATCH_SIZE, 128, 256, 6), tf.float32)])
def train_step(images):
    im1, im2 = images[...,:3], images[...,3:]

    images_reversed = tf.concat([im2, im1], axis=-1)

    res = tf.cast([images.shape[1], images.shape[2]], tf.float32)

    with tf.GradientTape() as tape:
        d1, t1, f1, o1 = model([im1, images], training=True)
        d2, t2, f2, o2 = model([im2, images_reversed], training=True)

        f1 *= res
        o1 = (o1 + 0.5) * res
        #f2 *= res
        #o2 = (o2 + 0.5) * res

        i1, i2 = repackage_intrinsics(f1, o1), repackage_intrinsics(f2, o2)
        t1, t2 = repackage_transformation(t1), repackage_transformation(t2)

        bundle1 = loss_function(im1, im2, d1, d2, t1, i1)
        #bundle2 = rgb_loss_function(im2, im1, d2, d1, t2, i2)

        loss = bundle1[0]# + bundle2[0]

    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))

    return bundle1, t1

TRAIN_STEPS = 500000
SAMPLE_FREQUENCY = 5000
VALIDATION_FREQUENCY = 25000000
VALIDATION_COUNT = 1

train_dataset, validation_dataset = get_dataset(BATCH_SIZE,
                                                2,
                                                validation_count=VALIDATION_COUNT,
                                                buffer_size=1,
                                                resolution=(128, 256))
it = iter(train_dataset)

transform_met = tf.keras.metrics.Mean()

i = 0
while i < TRAIN_STEPS:
    #if (i+1) % VALIDATION_FREQUENCY == 0:
    #    validate()

    #if (i+1) % (TRAIN_STEPS//5) == 0:
    #    keras.backend.set_value(opt.learning_rate, opt.learning_rate*0.5)

    images, poses = it.get_next()

    images = tf.concat([images[:,0], images[:,1]], axis=-1)

    #labels = euler.from_rotation_matrix(poses[:,-1,:3,:3]) - euler.from_rotation_matrix(poses[:,0,:3,:3])
    #labels = tf.concat([labels, poses[:,-1,:3,-1] - poses[:,0,:3,-1]], axis=-1)

    # bad data
    #if tf.math.reduce_any(tf.math.abs(labels) > 25.).numpy():
    #    continue
    # bad data
    #if tf.math.reduce_all(tf.math.abs(labels[:,-3:-1]) < 1.0).numpy():
    #    continue

    a, transformation_prediction = train_step(images)

    loss, depth, resampled_rgb = a
    transform_met(loss)

    if (i+1) % SAMPLE_FREQUENCY == 0:
        print(f"Sample {(i+1) // SAMPLE_FREQUENCY} of {TRAIN_STEPS // SAMPLE_FREQUENCY}:")
        print(transformation_prediction)

        pl.imshow(images[0,:,:,:3])
        pl.draw()
        pl.pause(3)

        pl.imshow(images[0,:,:,3:])
        pl.draw()
        pl.pause(3)

        pl.imshow(resampled_rgb[0])
        pl.draw()
        pl.pause(3)

        disparity = 1. / depth
        disparity_mean = tf.math.reduce_mean(disparity, axis=[1,2], keepdims=True)
        disparity = disparity / disparity_mean

        pl.imshow(disparity[0,:,:,0], cmap=pl.get_cmap("inferno"))
        pl.draw()
        pl.pause(3)

        pl.imshow(depth[0,:,:,0], cmap=pl.get_cmap("inferno"))
        pl.draw()
        pl.pause(0.1)

        model.save_weights("depth_prediction_weights.h5")

    print(f"Sequence {i} of {TRAIN_STEPS}, average loss: {transform_met.result()}      \r", end='')
    i += 1
