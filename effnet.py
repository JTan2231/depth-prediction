import string
import random
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from mod_resnet18 import depth_prediction_resnet18unet

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

def mb_conv_block(input_tensor, input_filters, output_filters, kernel_size=3, strides=2, expand_ratio=3, activation='swish', se=True, se_ratio=0.25):
    filters = input_filters * expand_ratio

    if strides == 1:
        pad = "SAME"
    else:
        pad = "VALID"

    if expand_ratio != 1:
        x = layers.Conv2D(filters, 1, padding='same', use_bias=False)(input_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
    else:
        x = input_tensor

    x = layers.DepthwiseConv2D(kernel_size, strides, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)

    if se:
        reduced_filters = max(1, int(input_filters * se_ratio))
        se_ten = layers.GlobalAveragePooling2D()(x)
        se_ten = layers.Reshape((1, 1, filters))(se_ten)
        se_ten = layers.Conv2D(reduced_filters, 1, activation=activation, padding='same')(se_ten)
        se_ten = layers.Conv2D(filters, 1, activation='sigmoid', padding='same')(se_ten)

        x = x * se_ten

    x = layers.Conv2D(output_filters, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if input_filters == output_filters:
        if strides == 1:
            shortcut = tf.image.resize(input_tensor, x.shape[1:3])
        else:
            shortcut = tf.image.resize(input_tensor, x.shape[1:3])#tf.nn.max_pool(input_tensor, [1, strides, strides, 1], [1, strides, strides, 1], 'VALID')
    else:
        shortcut = layers.Conv2D(output_filters, kernel_size, strides=strides, padding='same', use_bias=False)(input_tensor)

    x = x + shortcut
    x = tf.nn.swish(x)

    return x

def encoder_effnet(input_tensor):
    conv = layers.Conv2D(32, 7, strides=2)(input_tensor)
    conv = layers.BatchNormalization()(conv)
    econv1 = tf.nn.swish(conv)
    #conv = tf.nn.max_pool(econv1, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

    conv = mb_conv_block(conv, 16, 24, strides=1)
    econv2 = mb_conv_block(conv, 24, 40, strides=1)

    conv = mb_conv_block(econv2, 40, 80)
    econv3 = mb_conv_block(conv, 80, 112, strides=1, kernel_size=5)

    conv = mb_conv_block(econv3, 112, 192, kernel_size=5)
    econv4 = mb_conv_block(conv, 192, 320, strides=1, kernel_size=5)

    conv = mb_conv_block(econv4, 320, 512)
    econv5 = mb_conv_block(conv, 512, 512, strides=1)

    return econv5, (econv4, econv3, econv2, econv1)

def depth_prediction_effnetunet(input_tensor, res):
    padding_mode = 'REFLECT'

    bottleneck, (econv4, econv3, econv2, econv1) = encoder_effnet(input_tensor)

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

    depth_output = conv2d(1, activation='softplus')(depth_input)
    depth_output1 = conv2d(1, activation='softplus')(iconv1)
    depth_output2 = conv2d(1, activation='softplus')(iconv2)
    #depth_output3 = conv2d(1, activation='softplus')(iconv3)
    #depth_output4 = conv2d(1, activation='softplus')(iconv4)
    #depth_output5 = conv2d(1, activation='softplus')(iconv5)

    return depth_output, depth_output1, depth_output2#, depth_output3#, depth_output4, depth_output5

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

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

class Var(layers.Layer):
    def __init__(self, value_min, name=None):
        super().__init__(name=name)

        self.value_min = value_min
        self.value = tf.Variable(0.01, constraint=self.constraint, name=id_generator(10))

    def constraint(self, x):
        return tf.nn.relu(x - self.value_min) + self.value_min

    def call(self, input_tensor):
        return input_tensor * self.value

def egonet(input_tensor, batch_size):
    conv1 = mb_conv_block(input_tensor, 6, 32)
    conv2 = mb_conv_block(conv1, 32, 64)
    conv3 = mb_conv_block(conv2, 64, 128)
    conv4 = mb_conv_block(conv3, 128, 196)
    conv5 = mb_conv_block(conv4, 196, 256)
    conv6 = mb_conv_block(conv5, 256, 384)
    conv7 = mb_conv_block(conv6, 384, 512)

    bottleneck = tf.reduce_mean(conv7, axis=[1, 2], keepdims=True)
    background_motion = conv2d(6, kernel_size=1)(bottleneck)

    rotation = background_motion[:,0,0,:3]
    rotation = Var(0.001, name=id_generator(10))(rotation)
    translation = background_motion[...,3:]

    #residual_translation = refine_motion_field(translation, conv7)
    #residual_translation = refine_motion_field(residual_translation, conv6)
    #residual_translation = refine_motion_field(residual_translation, conv5)
    #residual_translation = refine_motion_field(residual_translation, conv4)
    #residual_translation = refine_motion_field(residual_translation, conv3)
    #residual_translation = refine_motion_field(residual_translation, conv2)
    #residual_translation = refine_motion_field(residual_translation, conv1)
    #residual_translation = refine_motion_field(residual_translation, input_tensor)

    translation_scale = Var(0.001)

    translation = translation_scale(translation)
    #residual_translation = translation_scale(residual_translation)

    foci = layers.Dense(2, activation='softplus')(bottleneck)[:,0,0,:]
    offset = (layers.Dense(2)(bottleneck) + tf.constant(0.5))[:,0,0,:]

    reso = tf.cast(input_tensor.shape[1:3], tf.float32)
    foci, offsets = foci * reso, offset * reso

    foci = tf.linalg.diag(foci)

    intrinsic_mat = tf.concat([foci, tf.expand_dims(offsets, -1)], axis=-1)
    last_row = tf.tile([[[0.0, 0.0, 1.0]]], [batch_size, 1, 1])
    intrinsics = tf.concat([intrinsic_mat, last_row], axis=-2)

    return (rotation, translation), intrinsics

def get_model(input_tensor, batch_size, res):
    # NOTE: Input shape == (BATCH_SIZE, RES[0], RES[1], 6)
    #       where the channels refer to two images (first, second)
    depth = keras.Model(input_tensor, depth_prediction_effnetunet(input_tensor[...,:3], res), name="depth_net")
    ego = keras.Model(input_tensor, egonet(input_tensor, batch_size), name="ego_net")

    return keras.Model(input_tensor, (depth(input_tensor), *(ego(input_tensor))))
