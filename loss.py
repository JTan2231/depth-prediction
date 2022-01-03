import tensorflow as tf
import geometry

from tensorflow_graphics.geometry.transformation import rotation_matrix_3d, euler
from unused_loss import weighted_ssim, depth_smoothness, disparity_from_depth, transform_depth_image, resample, clamp_and_mask

def matrix_from_angles(rot):
    rank = tf.rank(rot)
    # Swap the two last dimensions
    perm = tf.concat([tf.range(rank - 1), [rank], [rank - 1]], axis=0)

    return tf.transpose(rotation_matrix_3d.from_euler(-rot), perm)


def combine(rot_mat1, trans_vec1, rot_mat2, trans_vec2):
    r2r1 = tf.matmul(rot_mat2, rot_mat1)
    r2t1 = tf.matmul(rot_mat2, tf.expand_dims(trans_vec1, -1))
    r2t1 = tf.squeeze(r2t1, axis=-1)

    return r2r1, r2t1 + trans_vec2

def _expand_dims_twice(x, dim):
    return tf.expand_dims(tf.expand_dims(x, dim), dim)

def motion_field_consistency_loss(frame1transformed_pixelxy, mask,
                                  rotation1, translation1,
                                  rotation2, translation2):
  translation2resampled = resample(translation2, tf.stop_gradient(frame1transformed_pixelxy[...,0]), tf.stop_gradient(frame1transformed_pixelxy[...,1]))
  rotation1field = tf.broadcast_to(_expand_dims_twice(rotation1, -2), tf.shape(translation1))
  rotation2field = tf.broadcast_to(_expand_dims_twice(rotation2, -2), tf.shape(translation2))
  rotation1matrix = matrix_from_angles(rotation1field)
  rotation2matrix = matrix_from_angles(rotation2field)

  rot_unit, trans_zero = combine(
      rotation2matrix, translation2resampled,
      rotation1matrix, translation1)
  eye = tf.eye(3, batch_shape=tf.shape(rot_unit)[:-2])

  # We normalize the product of rotations by the product of their norms, to make
  # the loss agnostic of their magnitudes, only wanting them to be opposite in
  # directions. Otherwise the loss has a tendency to drive the rotations to
  # zero.
  rot_error = tf.reduce_mean(tf.square(rot_unit - eye), axis=(3, 4))
  rot1_scale = tf.reduce_mean(tf.square(rotation1matrix - eye), axis=(3, 4))
  rot2_scale = tf.reduce_mean(tf.square(rotation2matrix - eye), axis=(3, 4))
  rot_error /= (1e-24 + rot1_scale + rot2_scale)
  rotation_error = tf.reduce_mean(rot_error)

  def norm(x):
    return tf.reduce_sum(tf.square(x), axis=-1)

  # Here again, we normalize by the magnitudes, for the same reason.
  translation_error = tf.reduce_mean(
      norm(trans_zero) /
      (1e-24 + norm(translation1) + norm(translation2)) * mask)

  return (rotation_error*0.001 + translation_error*0.01)

def _weighted_average(x, w, epsilon=1.0):
    weighted_sum = tf.reduce_sum(x * w, axis=(1, 2), keepdims=True)
    sum_of_weights = tf.reduce_sum(w, axis=(1, 2), keepdims=True)

    return weighted_sum / (sum_of_weights + epsilon)

def _smoothness(motion_map):
    norm = tf.reduce_mean(
        tf.square(motion_map), axis=[1, 2, 3], keepdims=True) * 3.0
    motion_map /= tf.sqrt(norm + 1e-12)
    return _smoothness_helper(motion_map)


def _smoothness_helper(motion_map):
    motion_map_dx = motion_map - tf.roll(motion_map, 1, 1)
    motion_map_dy = motion_map - tf.roll(motion_map, 1, 2)
    sm_loss = tf.sqrt(1e-24 + tf.square(motion_map_dx) + tf.square(motion_map_dy))
    tf.summary.image('motion_sm', sm_loss)
    return tf.reduce_mean(sm_loss)

IMAGENET_MEAN = tf.reshape([0.485, 0.486, 0.406], (1, 1, 1, 3))
IMAGENET_STD = tf.reshape([0.229, 0.224, 0.225], (1, 1, 1, 3))
def deimagenet(images):
    images *= IMAGENET_STD
    images += IMAGENET_MEAN

    return images

#@tf.function
def rgb_loss_function(rgb1, rgb2, depth1, depth2, trans1, trans2, intrinsics, writer, step):
    with writer.as_default():
        tf.summary.text("intrinsics", str(intrinsics).split(',')[0].split('(')[-1].strip('\n'), step+1)
        writer.flush()

    rot1, transl1 = trans1
    rot2, transl2 = trans1

    pixel_x, pixel_y, depth1_warped = transform_depth_image(depth1, matrix_from_angles(rot1), transl1, intrinsics)

    pixel_xy = tf.concat([tf.expand_dims(pixel_x, -1), tf.expand_dims(pixel_y, -1)], axis=-1)
    
    mask = clamp_and_mask(pixel_x, pixel_y, depth1_warped)
    valid_mask = tf.expand_dims(mask, -1)
    mask = valid_mask[...,0]

    motion_loss = motion_field_consistency_loss(pixel_xy, mask, rot1, transl1, rot2, transl2)
    
    resampled_rgb = resample(rgb1, pixel_x, pixel_y)
    resampled_depth2 = resample(depth2, pixel_x, pixel_y)[...,0]
    
    depth1_closer = tf.cast(
            tf.logical_and(tf.cast(mask, tf.bool), tf.less(depth1_warped, resampled_depth2)),
            tf.float32)

    depth_error_second_moment = _weighted_average(
            tf.square(resampled_depth2 - depth1_warped),
            depth1_closer) + 1e-4
    depth_prox_weight = depth_error_second_moment / (
            tf.square(resampled_depth2 - depth1_warped) + depth_error_second_moment) * mask

    depth_prox_weight = tf.stop_gradient(depth_prox_weight)

    ssim_loss, avg_weight = weighted_ssim(resampled_rgb, rgb2, depth_prox_weight, c1=float('inf'), c2=9e-6)
    ssim_loss = tf.math.reduce_mean(ssim_loss * avg_weight) * 1.5

    rgb_loss = tf.math.abs(resampled_rgb - rgb2) * tf.expand_dims(depth1_closer, -1)
    rgb_loss = tf.math.reduce_mean(rgb_loss) * 0.85

    depth_loss = tf.math.abs(resampled_depth2 - depth1_warped) * depth1_closer
    depth_loss = tf.math.reduce_mean(depth_loss) * 0.01

    disp1 = disparity_from_depth(depth1)
    depth_smoothness_loss = tf.math.reduce_mean(depth_smoothness(disp1, rgb1)) * 0.01
    trans_smoothness_loss = _smoothness(transl1) * 0.001
    trans_smoothness_loss += _smoothness(transl2) * 0.001

    total_loss = ssim_loss + rgb_loss + depth_loss
    total_loss += depth_smoothness_loss# + trans_smoothness_loss
    total_loss += motion_loss

    interval = 500
    with writer.as_default():
        if (step+1) % interval == 0:
            tf.summary.image("resampled_rgb", deimagenet(resampled_rgb), (step+1)//interval)
            tf.summary.image("rgb1", deimagenet(rgb1), (step+1)//interval)
            tf.summary.image("rgb2", deimagenet(rgb2), (step+1)//interval)
            tf.summary.image("depth1", depth1 / tf.reduce_max(depth1), (step+1)//interval)
            tf.summary.image("depth2", depth2 / tf.reduce_max(depth1), (step+1)//interval)

        tf.summary.scalar("total_loss", total_loss, step+1)
        tf.summary.scalar("ssim_loss", ssim_loss, step+1)
        tf.summary.scalar("rgb_loss", rgb_loss, step+1)
        tf.summary.scalar("depth_loss", depth_loss, step+1)
        tf.summary.scalar("depth_smoothness_loss", depth_smoothness_loss, step+1)
        tf.summary.scalar("trans_smoothness_loss", trans_smoothness_loss, step+1)
        tf.summary.scalar("motion_loss", motion_loss, step+1)

        writer.flush()

    return total_loss, depth1, resampled_rgb
