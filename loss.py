import tensorflow as tf
import geometry

from tensorflow_graphics.geometry.transformation import rotation_matrix_3d, euler
from unused_loss import weighted_ssim_loss, depth_smoothness, disparity_from_depth, transform_depth_image

def matrix_from_angles(rot):
    rank = tf.rank(rot)
    # Swap the two last dimensions
    perm = tf.concat([tf.range(rank - 1), [rank], [rank - 1]], axis=0)

    return tf.transpose(rotation_matrix_3d.from_euler(-rot), perm)

@tf.function
def rgb_loss_function(rgb1, rgb2, depth1, transformation, intrinsics):
    intrinsics_inv = tf.linalg.inv(intrinsics)

    resampled_rgb = geometry.inverse_warp(rgb1, depth1, transformation, intrinsics, intrinsics_inv)[0]

    #resampled_depth2 = geometry.inverse_warp(depth2, depth1, transformation, intrinsics, intrinsics_inv)[0]
    transformed_depth1 = transform_depth_image(depth1, transformation[:,:3,:3], transformation[:,:3,3], intrinsics)[2]
    transformed_depth1 = tf.expand_dims(transformed_depth1, axis=-1)
    #transformed_depth1 = tf.reshape(transformed_depth1, (rgb1.shape[0], rgb1.shape[1], rgb1.shape[2], transformed_depth1.shape[0]))
    #depth_loss = tf.math.abs(resampled_depth2 - transformed_depth1)
    #depth_loss = tf.math.reduce_mean(depth_loss)

    rgb_loss = tf.math.reduce_mean(tf.math.abs(resampled_rgb - rgb2))
    ssim_loss = 1 - tf.image.ssim(resampled_rgb, rgb2, 1., filter_size=7)#weighted_ssim_loss(rgb2, resampled_rgb, depth1, depth2, transformation, intrinsics)#1 - tf.image.ssim(resampled_rgb, rgb2, 1.)
    #ssim_loss = weighted_ssim_loss(rgb2, resampled_rgb, depth1, depth2, transformation, intrinsics)#1 - tf.image.ssim(resampled_rgb, rgb2, 1.)
    ssim_loss = tf.math.reduce_mean(ssim_loss)
    smoothness_loss = depth_smoothness(disparity_from_depth(depth1), rgb1)

    #brightness_loss = tf.math.square(a - 1) + tf.math.square(b)
    #brightness_loss = tf.math.reduce_mean(brightness_loss)

    image_loss_scalar = 0.85

    total_loss = 3.0*ssim_loss + (1-image_loss_scalar)*rgb_loss# + depth_loss
    #total_loss += tf.math.reduce_mean(smoothness_loss)# + depth_loss

    return total_loss, depth1, resampled_rgb
