import tensorflow as tf

def _meshgrid_abs(height, width):
  """Meshgrid in the absolute coordinates."""
  x_t = tf.matmul(
      tf.ones(shape=tf.stack([height, 1])),
      tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
  y_t = tf.matmul(
      tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
      tf.ones(shape=tf.stack([1, width])))
  x_t = (x_t + 1.0) * 0.5 * tf.cast(width - 1, tf.float32)
  y_t = (y_t + 1.0) * 0.5 * tf.cast(height - 1, tf.float32)
  x_t_flat = tf.reshape(x_t, (1, -1))
  y_t_flat = tf.reshape(y_t, (1, -1))
  ones = tf.ones_like(x_t_flat)
  grid = tf.concat([x_t_flat, y_t_flat, ones], axis=0)

  return grid

def _pixel2cam(depth, pixel_coords, intrinsic_mat_inv):
  """Transform coordinates in the pixel frame to the camera frame."""
  cam_coords = tf.matmul(intrinsic_mat_inv, pixel_coords) * depth

  return cam_coords

def _cam2pixel(cam_coords, proj_c2p):
  """Transform coordinates in the camera frame to the pixel frame."""
  pcoords = tf.matmul(proj_c2p, cam_coords)
  x = tf.slice(pcoords, [0, 0, 0], [-1, 1, -1])
  y = tf.slice(pcoords, [0, 1, 0], [-1, 1, -1])
  z = tf.slice(pcoords, [0, 2, 0], [-1, 1, -1])
  # Not tested if adding a small number is necessary
  x_norm = x / (z + 1e-10)
  y_norm = y / (z + 1e-10)
  pixel_coords = tf.concat([x_norm, y_norm], axis=1)

  return pixel_coords

def _bilinear_sampler(im, x, y, name='blinear_sampler'):
    """Perform bilinear sampling on im given list of x, y coordinates.
    Implements the differentiable sampling mechanism with bilinear kernel
    in https://arxiv.org/abs/1506.02025.
    x,y are tensors specifying normalized coordinates [-1, 1] to be sampled on im.
    For example, (-1, -1) in (x, y) corresponds to pixel location (0, 0) in im,
    and (1, 1) in (x, y) corresponds to the bottom right pixel in im.
    Args:
    im: Batch of images with shape [B, h, w, channels].
    x: Tensor of normalized x coordinates in [-1, 1], with shape [B, h, w, 1].
    y: Tensor of normalized y coordinates in [-1, 1], with shape [B, h, w, 1].
    name: Name scope for ops.
    Returns:
    Sampled image with shape [B, h, w, channels].
    Principled mask with shape [B, h, w, 1], dtype:float32.  A value of 1.0
      in the mask indicates that the corresponding coordinate in the sampled
      image is valid.
    """
    x = tf.reshape(x, [-1])
    y = tf.reshape(y, [-1])

    # Constants.
    batch_size = tf.shape(im)[0]
    #_, height, width, channels = im.get_shape().as_list()
    shape = tf.shape(im)
    _, height, width, channels = shape[0], shape[1], shape[2], shape[3]

    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    height_f = tf.cast(height, 'float32')
    width_f = tf.cast(width, 'float32')
    zero = tf.constant(0, dtype=tf.int32)
    max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
    max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

    # Scale indices from [-1, 1] to [0, width - 1] or [0, height - 1].
    x = (x + 1.0) * (width_f - 1.0) / 2.0
    y = (y + 1.0) * (height_f - 1.0) / 2.0

    # Compute the coordinates of the 4 pixels to sample from.
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    mask = tf.logical_and(
        tf.logical_and(x0 >= zero, x1 <= max_x),
        tf.logical_and(y0 >= zero, y1 <= max_y))
    mask = tf.cast(mask, tf.float32)

    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)
    dim2 = width
    dim1 = width * height

    # Create base index.
    base = tf.range(batch_size) * dim1
    base = tf.reshape(base, [-1, 1])
    base = tf.tile(base, [1, height * width])
    base = tf.reshape(base, [-1])

    base_y0 = base + y0 * dim2
    base_y1 = base + y1 * dim2
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    # Use indices to lookup pixels in the flat image and restore channels dim.
    im_flat = tf.reshape(im, tf.stack([-1, channels]))
    im_flat = tf.cast(im_flat, tf.float32)
    pixel_a = tf.gather(im_flat, idx_a)
    pixel_b = tf.gather(im_flat, idx_b)
    pixel_c = tf.gather(im_flat, idx_c)
    pixel_d = tf.gather(im_flat, idx_d)

    x1_f = tf.cast(x1, tf.float32)
    y1_f = tf.cast(y1, tf.float32)

    # And finally calculate interpolated values.
    wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
    wb = tf.expand_dims((x1_f - x) * (1.0 - (y1_f - y)), 1)
    wc = tf.expand_dims(((1.0 - (x1_f - x)) * (y1_f - y)), 1)
    wd = tf.expand_dims(((1.0 - (x1_f - x)) * (1.0 - (y1_f - y))), 1)

    output = tf.add_n([wa * pixel_a, wb * pixel_b, wc * pixel_c, wd * pixel_d])
    output = tf.reshape(output, tf.stack([batch_size, height, width, channels]))
    mask = tf.reshape(mask, tf.stack([batch_size, height, width, 1]))
    return output, mask

def _spatial_transformer(img, coords):
  """A wrapper over binlinear_sampler(), taking absolute coords as input."""
  img_height = tf.cast(tf.shape(img)[1], tf.float32)
  img_width = tf.cast(tf.shape(img)[2], tf.float32)
  px = coords[:, :, :, :1]
  py = coords[:, :, :, 1:]
  # Normalize coordinates to [-1, 1] to send to _bilinear_sampler.
  px = px / (img_width - 1) * 2.0 - 1.0
  py = py / (img_height - 1) * 2.0 - 1.0
  output_img, mask = _bilinear_sampler(img, px, py)

  return output_img, mask

def inverse_warp(img, depth, egomotion_mat, intrinsic_mat,
                 intrinsic_mat_inv):

  dims = tf.shape(img)
  batch_size, img_height, img_width = dims[0], dims[1], dims[2]
  depth = tf.reshape(depth, [batch_size, 1, img_height * img_width])
  grid = _meshgrid_abs(img_height, img_width)
  grid = tf.tile(tf.expand_dims(grid, 0), [batch_size, 1, 1])
  cam_coords = _pixel2cam(depth, grid, intrinsic_mat_inv)
  ones = tf.ones([batch_size, 1, img_height * img_width])
  cam_coords_hom = tf.concat([cam_coords, ones], axis=1)

  # Get projection matrix for target camera frame to source pixel frame
  hom_filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
  hom_filler = tf.tile(hom_filler, [batch_size, 1, 1])
  intrinsic_mat_hom = tf.concat(
      [intrinsic_mat, tf.zeros([batch_size, 3, 1])], axis=2)
  intrinsic_mat_hom = tf.concat([intrinsic_mat_hom, hom_filler], axis=1)
  proj_target_cam_to_source_pixel = tf.matmul(intrinsic_mat_hom, egomotion_mat)
  source_pixel_coords = _cam2pixel(cam_coords_hom,
                                   proj_target_cam_to_source_pixel)
  source_pixel_coords = tf.reshape(source_pixel_coords,
                                   [batch_size, 2, img_height, img_width])
  source_pixel_coords = tf.transpose(source_pixel_coords, perm=[0, 2, 3, 1])
  projected_img, mask = _spatial_transformer(img, source_pixel_coords)

  return projected_img, mask

"""def inverse_warp(image, depth, egomotion, intrinsics, intrinsics_inv):
  depth = depth[...,0]
  height, width = depth.shape[1:3]
  grid = tf.squeeze(tf.stack(tf.meshgrid(tf.range(width), tf.range(height), (1,))), axis=3)

  grid = tf.cast(grid, tf.float32)
  rot_mat = egomotion[:,:3,:3]#transform_utils.matrix_from_angles(rotation_angles)
  # We have to treat separately the case of a per-image rotation vector and a
  # per-image rotation field, because the broadcasting capabilities of einsum
  # are limited.
  # The calculation here is identical to the one in inverse_warp above.
  # Howeverwe use einsum for better clarity. Under the hood, einsum performs
  # the reshaping and invocation of BatchMatMul, instead of doing it manually,
  # as in inverse_warp.
  projected_rotation = tf.einsum('bij,bjk,bkl->bil', intrinsics, rot_mat,
                                 intrinsics_inv)
  pcoords = tf.einsum('bij,jhw,bhw->bihw', projected_rotation, grid, depth)

  print(intrinsics)
  print(egomotion[:,:3,3])
  projected_translation = tf.einsum('bij,bhwj->bihw', intrinsics, tf.expand_dims(tf.expand_dims(egomotion[:,:3,3], 1), 1))

  pcoords += projected_translation
  x, y, z = tf.unstack(pcoords, axis=1)

  return x / z, y / z, z"""
