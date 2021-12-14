import geometry
import pylab as pl
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow_graphics.geometry.transformation import rotation_matrix_3d, euler

def matrix_from_angles(rot):
    rank = tf.rank(rot)
    # Swap the two last dimensions
    perm = tf.concat([tf.range(rank - 1), [rank], [rank - 1]], axis=0)
    return tf.transpose(rotation_matrix_3d.from_euler(-rot), perm)

def inverse_euler(angles):
    sin_angles = tf.sin(angles)
    cos_angles = tf.cos(angles)
    sz, sy, sx = tf.unstack(-sin_angles, axis=-1)
    cz, _, cx = tf.unstack(cos_angles, axis=-1)
    y = tf.asin((cx * sy * cz) + (sx * sz))
    x = -tf.asin((sx * sy * cz) - (cx * sz)) / tf.cos(y)
    z = -tf.asin((cx * sy * sz) - (sx * cz)) / tf.cos(y)
    return tf.stack([x, y, z], axis=-1)

def invert_rot_and_trans(rot, trans):
    rot = euler.from_rotation_matrix(rot)
    inv_rot = inverse_euler(rot)  # inv_rot = -rot  for small angles
    inv_rot_mat = matrix_from_angles(inv_rot)
    print("inverse rotation matrix shape:", inv_rot_mat.shape)
    inv_trans = -tf.matmul(inv_rot_mat, tf.expand_dims(trans, -1))
    inv_trans = tf.squeeze(inv_trans, -1)
    return inv_rot_mat, inv_trans

# depth.shape == [batch_size, resolution[0], resolution[1]]
# transformation.shape == [batch_size, 3, 4]
# intrinsics.shape == [batch_size, 3, 3]
#
# Transforms the depth image using the transformation and intrinsics matrices
def transform_depth_image(depth, rotation, translation, intrinsics):
    translation = tf.expand_dims(tf.expand_dims(translation, 1), 1)
    depth = depth[:,:,:,0]

    _, height, width = tf.unstack(tf.shape(depth))
    grid = tf.squeeze(tf.stack(tf.meshgrid(tf.range(width), tf.range(height), (1,))), axis=3)
    grid = tf.cast(grid, tf.float32)

    intrinsics_inv = tf.linalg.inv(intrinsics)

    projected_rotation = tf.einsum("bij,bjk,bkl->bil", intrinsics, rotation, intrinsics_inv)
    projected_coordinates = tf.einsum("bij,jhw,bhw->bihw", projected_rotation, grid, depth)

    projected_translation = tf.einsum("bij,bhwj->bihw", intrinsics, translation)
    projected_coordinates += projected_translation

    x, y, z = tf.unstack(projected_coordinates, axis=1)

    #z = tf.clip_by_value(z, 0.00001, 100000.)

    return x / z, y / z, z

def clamp_and_mask(pixel_x, pixel_y, z):
    _, height, width = tf.unstack(tf.shape(pixel_x))

    def _tensor(x):
        return tf.cast(tf.convert_to_tensor(x), tf.float32)

    x_not_underflow = pixel_x >= 0.0
    y_not_underflow = pixel_y >= 0.0

    x_not_overflow = pixel_x < _tensor(width - 1)
    y_not_overflow = pixel_y < _tensor(height - 1)

    z_positive = z > 0.0

    x_not_nan = tf.math.logical_not(tf.math.is_nan(pixel_x))
    y_not_nan = tf.math.logical_not(tf.math.is_nan(pixel_y))

    not_nan = tf.math.logical_and(x_not_nan, y_not_nan)
    not_nan_mask = tf.cast(not_nan, tf.float32)

    pixel_x *= not_nan_mask
    pixel_y *= not_nan_mask

    pixel_x = tf.clip_by_value(pixel_x, 0.0, _tensor(width - 1))
    pixel_y = tf.clip_by_value(pixel_y, 0.0, _tensor(height - 1))

    mask_stack = tf.stack([x_not_underflow, y_not_underflow,
                           x_not_overflow, y_not_overflow,
                           z_positive, not_nan], axis=0)
    mask = tf.cast(tf.reduce_all(mask_stack, axis=0), tf.float32)
    
    return mask

# resample the values of data with respect to the given warp coordinates
def resample(data, warp_x, warp_y):
    warp_shape = warp_x.shape

    # Compute the four points closest to warp with integer value.
    warp_floor_x = tf.math.floor(warp_x)
    warp_floor_y = tf.math.floor(warp_y)

    # Compute the weight for each point.
    right_warp_weight = warp_x - warp_floor_x
    down_warp_weight = warp_y - warp_floor_y

    warp_floor_x = tf.cast(warp_floor_x, tf.int32)
    warp_floor_y = tf.cast(warp_floor_y, tf.int32)
    warp_ceil_x = tf.cast(tf.math.ceil(warp_x), tf.int32)
    warp_ceil_y = tf.cast(tf.math.ceil(warp_y), tf.int32)

    left_warp_weight = tf.math.subtract(tf.convert_to_tensor(1.0, right_warp_weight.dtype), right_warp_weight)
    up_warp_weight = tf.math.subtract(tf.convert_to_tensor(1.0, down_warp_weight.dtype), down_warp_weight)

    # Extend warps from [batch_size, dim_0, ... , dim_n, 2] to
    # [batch_size, dim_0, ... , dim_n, 3] with the first element in last
    # dimension being the batch index.

    # A shape like warp_shape but with all sizes except the first set to 1:
    warp_batch_shape = tf.concat([warp_shape[0:1], tf.ones_like(warp_shape[1:])], 0)

    warp_batch = tf.reshape(tf.range(warp_shape[0], dtype=tf.int32), warp_batch_shape)

    # Broadcast to match shape:
    warp_batch += tf.zeros_like(warp_y, dtype=tf.int32)
    left_warp_weight = tf.expand_dims(left_warp_weight, axis=-1)
    down_warp_weight = tf.expand_dims(down_warp_weight, axis=-1)
    up_warp_weight = tf.expand_dims(up_warp_weight, axis=-1)
    right_warp_weight = tf.expand_dims(right_warp_weight, axis=-1)

    up_left_warp = tf.stack([warp_batch, warp_floor_y, warp_floor_x], axis=-1)
    up_right_warp = tf.stack([warp_batch, warp_floor_y, warp_ceil_x], axis=-1)
    down_left_warp = tf.stack([warp_batch, warp_ceil_y, warp_floor_x], axis=-1)
    down_right_warp = tf.stack([warp_batch, warp_ceil_y, warp_ceil_x], axis=-1)

    # gather data then take weighted average to get resample result.
    result = (
        (tf.gather_nd(data, up_left_warp) * left_warp_weight +
         tf.gather_nd(data, up_right_warp) * right_warp_weight) * up_warp_weight +
        (tf.gather_nd(data, down_left_warp) * left_warp_weight +
         tf.gather_nd(data, down_right_warp) * right_warp_weight) *
        down_warp_weight)
    result_shape = (warp_x.get_shape().as_list() + data.get_shape().as_list()[-1:])
    result = tf.reshape(result, result_shape)

    return result

def gradient_x(img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]

def gradient_y(img):
    return img[:, :-1, :, :] - img[:, 1:, :, :]

def depth_smoothness(depth, img):
    #depth = tf.expand_dims(depth, -1)
    depth_dx = gradient_x(depth)
    depth_dy = gradient_y(depth)
    image_dx = gradient_x(img)
    image_dy = gradient_y(img)
    weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_dx), 3, keepdims=True))
    weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_dy), 3, keepdims=True))
    smoothness_x = depth_dx * weights_x
    smoothness_y = depth_dy * weights_y

    return tf.reduce_mean(abs(smoothness_x)) + tf.reduce_mean(abs(smoothness_y))

def disparity_from_depth(depth):
    disparity = 1. / (depth+0.01)
    disparity_mean = tf.math.reduce_mean(disparity, axis=[1, 2], keepdims=True)
    disparity /= disparity_mean

    return disparity

def weighted_average(x, w, epsilon=1.0):
    weighted_sum = tf.reduce_sum(x * w, axis=(1, 2), keepdims=True)
    sum_of_weights = tf.reduce_sum(w, axis=(1, 2), keepdims=True)

    return weighted_sum / (sum_of_weights + epsilon)

def avg_pool3x3(x):
    return tf.nn.avg_pool(x, [1, 3, 3, 1], [1, 1, 1, 1], 'SAME')

def weighted_ssim(x, y, weight, c1=0.01**2, c2=0.03**2, weight_epsilon=0.01):
    if c1 == float('inf') and c2 == float('inf'):
        raise ValueError('Both c1 and c2 are infinite, SSIM loss is zero. This is '
                         'likely unintended.')

    average_pooled_weight = avg_pool3x3(weight)
    weight_plus_epsilon = weight + weight_epsilon
    inverse_average_pooled_weight = 1.0 / (average_pooled_weight + weight_epsilon)

    def weighted_avg_pool3x3(z):
        weighted_avg = avg_pool3x3(z * weight_plus_epsilon)
        return weighted_avg * inverse_average_pooled_weight

    mu_x = weighted_avg_pool3x3(x)
    mu_y = weighted_avg_pool3x3(y)
    sigma_x = weighted_avg_pool3x3(x**2) - mu_x**2
    sigma_y = weighted_avg_pool3x3(y**2) - mu_y**2
    sigma_xy = weighted_avg_pool3x3(x * y) - mu_x * mu_y

    if c1 == float('inf'):
        ssim_n = (2 * sigma_xy + c2)
        ssim_d = (sigma_x + sigma_y + c2)
    elif c2 == float('inf'):
        ssim_n = 2 * mu_x * mu_y + c1
        ssim_d = mu_x**2 + mu_y**2 + c1
    else:
        ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        ssim_d = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)

    result = ssim_n / ssim_d

    return tf.clip_by_value((1 - result) / 2, 0, 1), average_pooled_weight

def weighted_ssim_loss(rgb, rgb_resampled, depth1, depth2, transformation, intrinsics):
    rotation = transformation[:,:3,:3]
    translation = transformation[:,:3,3]

    pixel_x, pixel_y, depth1_warped = transform_depth_image(depth1, rotation, translation, intrinsics)
    mask = clamp_and_mask(pixel_x, pixel_y, depth1_warped)
    mask = tf.expand_dims(mask, -1)

    depth2_resampled = resample(depth2, pixel_x, pixel_y)

    depth1_warped = tf.expand_dims(depth1_warped, axis=-1)
    depth1_warped *= tf.cast(tf.math.is_finite(depth1_warped), tf.float32)
    depth_error_second_moment = weighted_average(tf.math.square(depth2_resampled - depth1_warped), mask) + 1e-4
    depth_proximity_weight  = depth_error_second_moment / (tf.math.square(depth2_resampled - depth1_warped) + depth_error_second_moment)# * mask
    depth_proximity_weight = tf.stop_gradient(depth_proximity_weight)

    ssim_loss, avg_weight = weighted_ssim(rgb_resampled, rgb, depth_proximity_weight, c1=float('inf'), c2=9e-6)

    ssim_loss = tf.math.reduce_mean(tf.math.multiply_no_nan(ssim_loss, avg_weight))

    return ssim_loss

# rgb_images.shape = [batch_size, resolution[0], resolution[1], 6]
# depth_images.shape = [batch_size, resolution[0], resolution[1], 2]
# confidence_masks.shape = [batch_size, resolution[0], resolution[1], 2]
# transformation.shape = [batch_size, 3, 4]
# intrinsics.shape = [batch_size, 3, 3]
#
# Transforms the point cloud of (rgb1, depth1),
# resamples rgb2 with respect to the transformed point cloud,
# and compares the result.
def loss_function(rgb1, rgb2, depth1, depth2, transformation, intrinsics):
    #depth1, depth2 = (tf.clip_by_value(depth1, 0.01, 100.), tf.clip_by_value(depth2, 0.01, 100.))

    #rgb1, rgb2 = (rgb_images[:,:,:,:3], rgb_images[:,:,:,3:])
    #depth1, depth2 = (depth_images[:,:,:,0], depth_images[:,:,:,1])
    rotation, translation = (transformation[:,:3,:3], transformation[:,:3,3])

    pixel_x, pixel_y, depth1_warped = transform_depth_image(depth1, rotation, translation, intrinsics)

    mask = clamp_and_mask(pixel_x, pixel_y, depth1_warped)
    mask = tf.expand_dims(mask, -1)

    rgb2_resampled = resample(rgb1, pixel_x, pixel_y)
    depth2_resampled = resample(depth2, pixel_x, pixel_y)

    #ssim_loss = 1 - tf.image.ssim(rgb2_resampled, rgb1, 1.0) * mask
    rgb_loss = tf.math.abs(rgb2_resampled - rgb2) * mask
    #rgb_loss = tf.math.reduce_mean(rgb_loss)
    #rgb_loss *= mask
    depth1_warped = tf.expand_dims(depth1_warped, axis=-1)
    depth_loss = tf.math.abs(depth2_resampled - depth1_warped)# * mask
    depth_loss = tf.math.reduce_mean(depth_loss)
    smoothness_loss = (depth_smoothness(disparity_from_depth(depth1), rgb1))

    depth_error_second_moment = weighted_average(tf.math.square(depth2_resampled - depth1_warped), mask) + 1e-4
    depth_proximity_weight  = depth_error_second_moment / (tf.math.square(depth2_resampled - depth1_warped) + depth_error_second_moment)# * mask
    depth_proximity_weight = tf.stop_gradient(depth_proximity_weight)

    ssim_loss, avg_weight = weighted_ssim(rgb2_resampled, rgb2, depth_proximity_weight, c1=float('inf'), c2=9e-6)
    #ssim_loss = (ssim_loss*avg_weight + rgb_loss) / conf
    rgb_ssim_weight = 0.15
    #ssim_loss = tf.math.reduce_mean(((0.5*(1-rgb_ssim_weight)*ssim_loss*avg_weight + (rgb_ssim_weight)*rgb_loss) / conf) + tf.math.log(conf))

    ssim_loss = tf.math.reduce_mean((((1-rgb_ssim_weight)*ssim_loss*avg_weight + (rgb_ssim_weight)*rgb_loss)))
    ssim_loss = 1 - tf.image.ssim(rgb2_resampled, rgb2, 1., filter_size=3) + tf.math.reduce_mean(rgb_loss)

    total_loss = ssim_loss# + depth_loss + smoothness_loss

    return total_loss, depth1, rgb2_resampled
    #return total_loss, rgb2_resampled, pixel_x, pixel_y, ssim_loss, tf.math.reduce_mean(rgb_loss), smoothness_loss, depth_loss
