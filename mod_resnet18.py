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

def depth_prediction_resnet18unet(input_tensor):
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

    depth_input = tf.image.resize(iconv1, (128, 416))

    return conv2d(1, activation='softplus')(depth_input)

def egonet(input_tensor):
    conv = conv2d(16, strides=2, activation='relu')(input_tensor)
    conv = conv2d(32, strides=2, activation='relu')(conv)
    conv = conv2d(64, strides=2, activation='relu')(conv)
    conv = conv2d(128, strides=2, activation='relu')(conv)
    conv = conv2d(256, strides=2, activation='relu')(conv)
    conv = conv2d(512, strides=2, activation='relu')(conv)
    #conv = conv2d(1024, strides=2, activation='relu')(conv)

    conv = layers.GlobalAveragePooling2D()(conv)
    transformation = layers.Dense(6)(conv) * 0.001

    foci = layers.Dense(2, activation='softplus')(conv)
    offset = layers.Dense(2)(conv) + 0.5

    return transformation, foci, offset

def get_model(input_tensor):
    depth = depth_prediction_resnet18unet(input_tensor)
    transformation, foci, offset = egonet(input_tensor)

    return keras.Model(input_tensor, (depth, transformation, foci, offset))
