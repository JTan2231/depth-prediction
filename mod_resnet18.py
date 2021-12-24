import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

def conv2d(filters,    
           kernel_size=3,    
           dilation_rate=1,    
           padding='same',    
           strides=1,
           activation='linear'):    
    
    return layers.Conv2D(filters,    
                         kernel_size,    
                         padding=padding,    
                         strides=strides,
                         dilation_rate=dilation_rate,    
                         activation=activation)

def conv2dt(filters,    
            kernel_size=3,    
            dilation_rate=1,    
            padding='same',    
            strides=1,
            activation='linear'):    
    
    return layers.Conv2DTranspose(filters,    
                                  kernel_size,    
                                  padding=padding,    
                                  strides=strides,
                                  dilation_rate=dilation_rate,    
                                  activation=activation)

def concat_pad(decoder_layer, encoder_layer, padding_mode):
    decoder_layer = tf.image.resize(decoder_layer, (encoder_layer.shape[1], encoder_layer.shape[2]))
    concat = tf.concat([decoder_layer, encoder_layer], axis=3)
    
    return tf.pad(concat, [[0, 0,], [1, 1], [1, 1], [0, 0]], mode=padding_mode)

def residual_block_first(input_tensor, in_channels, out_channels, kernel_size=3, strides=1):
    if in_channels == out_channels:
        if strides == 1:
            shortcut = input_tensor
        else:
            shortcut = tf.nn.max_pool(input_tensor, [1, strides, strides, 1], [1, strides, strides, 1], 'VALID')
    else:
        shortcut = conv2d(out_channels, strides=strides)(input_tensor)

    conv = conv2d(out_channels, strides=strides)(input_tensor)
    conv = layers.BatchNormalization()(conv)
    conv = tf.nn.relu(conv)

    conv = conv2d(out_channels)(conv)
    conv = layers.BatchNormalization()(conv)

    conv = conv + shortcut
    conv = tf.nn.relu(conv)

    return conv

def residual_block(input_tensor, out_channels):
    shortcut = input_tensor

    conv = conv2d(out_channels)(input_tensor)
    conv = layers.BatchNormalization()(conv)
    conv = tf.nn.relu(conv)

    conv = conv2d(out_channels)(conv)
    conv = layers.BatchNormalization()(conv)

    conv = conv + shortcut
    conv = tf.nn.relu(conv)

    return conv

def encoder_resnet(input_tensor):
    conv = conv2d(64, kernel_size=7, strides=2)(input_tensor)
    conv = layers.BatchNormalization()(conv)
    econv1 = tf.nn.relu(conv)
    conv = tf.nn.max_pool(econv1, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

    conv = residual_block(conv, 64)
    econv2 = residual_block(conv, 64)

    conv = residual_block_first(econv2, 64, 128, strides=2)
    econv3 = residual_block(conv, 128)

    conv = residual_block_first(econv3, 128, 256, strides=2)
    econv4 = residual_block(conv, 256)

    conv = residual_block_first(econv4, 256, 512, strides=2)
    econv5 = residual_block(conv, 512)

    return econv5, (econv4, econv3, econv2, econv1)

def depth_prediction_resnet18unet(input_tensor, res):
    padding_mode = 'REFLECT'

    bottleneck, (econv4, econv3, econv2, econv1) = encoder_resnet(input_tensor)

    upconv5 = conv2dt(256, [3, 3], padding='VALID', strides=2)(bottleneck)
    iconv5 = conv2d(256)(concat_pad(upconv5, econv4, padding_mode))

    upconv4 = conv2dt(128, [3, 3], padding='VALID', strides=2)(iconv5)
    iconv4 = conv2d(128)(concat_pad(upconv4, econv3, padding_mode))

    upconv3 = conv2dt(64, [3, 3], padding='VALID', strides=2)(iconv4)
    iconv3 = conv2d(64)(concat_pad(upconv3, econv2, padding_mode))

    upconv2 = conv2dt(32, [3, 3], padding='VALID', strides=2)(iconv3)
    iconv2 = conv2d(32)(concat_pad(upconv2, econv1, padding_mode))

    upconv1 = conv2dt(16, [3, 3], padding='VALID', strides=2)(iconv2)
    iconv1 = conv2d(16)(upconv1)

    depth_input = tf.image.resize(iconv1, res)

    return conv2d(1, activation='softplus')(depth_input)

# https://github.com/google-research/google-research/blob/ce4e9e70127b1560f73616fba657f01a2b388aee/depth_from_video_in_the_wild/motion_prediction_net.py#L202
def refine_motion_field(motion_field, layer):
    _, h, w, _ = tf.unstack(tf.shape(layer))
    upsampled_motion_field = tf.image.resize(motion_field, [h, w])

    conv_input = tf.concat([upsampled_motion_field, layer], axis=3)
    conv_output = conv2d(max(4, layer.shape.as_list()[-1]), kernel_size=3)(conv_input)
    conv_input = conv2d(max(4, layer.shape.as_list()[-1]), kernel_size=3)(conv_input)
    conv_output2 = conv2d(max(4, layer.shape.as_list()[-1]), kernel_size=3)(conv_input)
    conv_output = tf.concat([conv_output, conv_output2], axis=-1)

    return upsampled_motion_field + conv2d(motion_field.shape.as_list()[-1], kernel_size=1)(conv_output)

def egonet(input_tensor, batch_size):
    conv1 = conv2d(16, strides=2, activation='relu')(input_tensor)
    conv2 = conv2d(32, strides=2, activation='relu')(conv1)
    conv3 = conv2d(64, strides=2, activation='relu')(conv2)
    conv4 = conv2d(128, strides=2, activation='relu')(conv3)
    conv5 = conv2d(256, strides=2, activation='relu')(conv4)
    conv6 = conv2d(512, strides=2, activation='relu')(conv5)
    #conv7 = conv2d(768, strides=2, activation='relu')(conv6)

    #bottleneck = tf.reduce_mean(conv7, axis=[1, 2], keepdims=True)
    bottleneck = tf.reduce_mean(conv6, axis=[1, 2], keepdims=True)
    background_motion = conv2d(6, kernel_size=1)(bottleneck)

    rotation = background_motion[:,0,0,:3] * tf.constant(0.001)
    translation = background_motion[...,3:]

    #residual_translation = refine_motion_field(translation, conv7)
    residual_translation = refine_motion_field(translation, conv6)
    residual_translation = refine_motion_field(residual_translation, conv5)
    residual_translation = refine_motion_field(residual_translation, conv4)
    residual_translation = refine_motion_field(residual_translation, conv3)
    residual_translation = refine_motion_field(residual_translation, conv2)
    residual_translation = refine_motion_field(residual_translation, conv1)
    residual_translation = refine_motion_field(residual_translation, input_tensor)

    translation *= tf.constant(0.001)
    residual_translation *= tf.constant(0.001)

    #conv = layers.GlobalAveragePooling2D()(conv)
    #transformation = layers.Dense(6)(conv) * tf.constant(0.001)

    foci = layers.Dense(2, activation='softplus')(bottleneck)[:,0,0,:]
    offset = (layers.Dense(2)(bottleneck) + tf.constant(0.5))[:,0,0,:]

    reso = tf.cast(input_tensor.shape[1:3], tf.float32)
    foci, offsets = foci * reso, offset * reso

    foci = tf.linalg.diag(foci)

    intrinsic_mat = tf.concat([foci, tf.expand_dims(offsets, -1)], axis=-1)
    last_row = tf.tile([[[0.0, 0.0, 1.0]]], [batch_size, 1, 1])
    intrinsics = tf.concat([intrinsic_mat, last_row], axis=-2)

    return (rotation, translation), residual_translation, intrinsics

def get_model(input_tensor, batch_size, res):
    # NOTE: Input shape == (BATCH_SIZE, RES[0], RES[1], 6)
    #       where the channels refer to two images (first, second)
    depth = depth_prediction_resnet18unet(input_tensor[...,:3], res)
    transformation, res_trans, intrinsics = egonet(input_tensor, batch_size)

    return keras.Model(input_tensor, (depth, transformation, res_trans, intrinsics))
