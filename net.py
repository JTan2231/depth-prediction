import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

kernel_reg = "l1_l2"
activity_reg = "l2"

def conv2d(filters,
           kernel_size=3,
           dilation_rate=1,
           padding='same',
           activation='linear',
           kernel_regularizer=kernel_reg,
           activity_regularizer=activity_reg):

    return layers.Conv2D(filters,
                         kernel_size,
                         padding=padding,
                         activation=activation,
                         kernel_regularizer=kernel_reg,
                         activity_regularizer=activity_reg)

class ResBlock(layers.Layer):
    def __init__(self, filters, dropout_rate=0.2, upscale=False):
        super().__init__()

        self.upscale = upscale

        if upscale:
            self.scale = conv2d(filters, kernel_size=1)
            self.bn_scale = layers.BatchNormalization()

        self.neck = conv2d(filters//2, kernel_size=1)
        self.conv = conv2d(filters//2)
        self.out = conv2d(filters, kernel_size=1)

        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.bn3 = layers.BatchNormalization()
        self.dropout = layers.Dropout(rate=dropout_rate)

    def call(self, input_tensor, training=True):
        init = input_tensor
        if self.upscale:
            init = self.scale(init)
            init = self.bn_scale(init, training=training)

        conv = self.neck(init)
        conv = self.bn1(conv, training=training)
        conv = tf.nn.swish(conv)

        conv = self.conv(conv)
        conv = self.bn2(conv, training=training)
        conv = tf.nn.swish(conv)

        conv = self.out(conv)
        conv = self.bn3(conv, training=training)
        conv = tf.nn.swish(conv)

        return self.dropout(conv + init, training=training)

class ResLayer(layers.Layer):
    def __init__(self, blocks, filters, dropout_rate=0.2, upscale=False):
        super().__init__()

        if upscale:
            self.blocks = [ResBlock(filters, dropout_rate, upscale=True),] + [ResBlock(filters, dropout_rate) for b in range(blocks-1)]
        else:
            self.blocks = [ResBlock(filters, dropout_rate) for b in range(blocks-1)]


    def call(self, input_tensor):
        conv = input_tensor

        for block in self.blocks:
            conv = block(conv)

        return conv

# in addition to the number of layers the model will have,
# layers corresponds to the amount of times the feature maps will be downsampled
# as well as how many times the dimensionality of those feature maps will double
class ResNet(keras.Model):
    def __init__(self, layer_count, blocks_per_layer, batch_input_shape, dropout_rate=0.2, global_pool=False, summary=False):
        super().__init__()

        self.layer_count = layer_count
        self.global_pool = global_pool

        self.initial_conv = layers.Conv2D(32, 5, dilation_rate=3, padding='same', activation='swish')
        self.conv_layers = [ResLayer(blocks_per_layer, 32*(2**(l+1)), dropout_rate, upscale=True) for l in range(1, layer_count+1, 1)]
        self.pooling_layers = [layers.AveragePooling2D() for l in range(layer_count)]
        if global_pool:
            self.pooling_layers += [layers.GlobalAveragePooling2D(),]

        # build model graph
        #inp = keras.Input(shape=batch_input_shape[1:])
        #self.call(inp)[1].mark_used()
        #self.build(input_shape=batch_input_shape)
        #if summary:
        #    self.summary()

    def call(self, input_tensor, unet=True, training=True):
        conv = self.initial_conv(input_tensor)

        if unet:
            residuals = tf.TensorArray(tf.float32, size=self.layer_count, infer_shape=False, clear_after_read=False)
            for l in range(self.layer_count):
                conv = self.pooling_layers[l](conv)
                conv = self.conv_layers[l](conv, training=training)
                residuals = residuals.write(l, conv)

            if self.global_pool:
                conv = self.pooling_layers[-1](conv)

            return conv, residuals
        else:
            for l in range(self.layer_count):
                conv = self.conv_layers[l](conv, training=training)
                conv = self.pooling_layers[l](conv)

            if self.global_pool:
                conv = self.pooling_layers[-1](conv)

            return conv

class ResNetTranspose(keras.Model):
    def __init__(self, output_channels, layer_count, blocks_per_layer, batch_input_shape, dropout_rate=0.2, summary=False):
        super().__init__()

        self.layer_count = layer_count

        self.conv_layers = [ResLayer(blocks_per_layer, 32*(2**(l+1)), dropout_rate, upscale=True) for l in range(layer_count, 0, -1)]
        self.final_conv = layers.Conv2D(output_channels, 1, activation='softplus', name="depth")
        self.conf_conv = layers.Conv2D(output_channels, 1, activation='sigmoid', name="confidence")
        # learned confidence masked to be element-wise multiplied
        # with the depth estimation
        # for handling occlusion and moving objects within the scene
        #self.confidence_mask = layers.Conv2D(output_channels, 1, activation='sigmoid', name="confidence_mask")
        self.pooling_layers = [layers.UpSampling2D() for l in range(layer_count)]

        #inp = keras.Input(shape=batch_input_shape[1:])
        #self.call(inp)
        #self.build(input_shape=batch_input_shape)
        #if summary:
        #    self.summary()

    # residuals is a tf.TensorArray containing the residual feature maps
    # from encoding
    def call(self, input_tensor, residuals, training=True):
        conv = input_tensor
        prev = None

        for l, r in zip(range(self.layer_count), range(self.layer_count-1, -1, -1)):
            res = residuals.read(r)
            if prev is None:
                conv = tf.concat([conv, res], axis=-1)
            else:
                conv = self.resize_like(conv, res)
                prev = self.resize_like(prev, res)
                conv = tf.concat([conv, res, prev], axis=-1)

            conv = self.conv_layers[l](conv, training=training)
            pooled = self.pooling_layers[l](conv)
            prev = self.resize_like(conv, pooled)
            conv = pooled

        #depth, confidence = (self.final_conv(conv), self.confidence_mask(conv))
        depth = self.final_conv(conv)
        #confidence = self.conf_conv(conv)
        #depth = tf.clip_by_value(depth, 0.01, 100.)
        #depth = 100*depth+0.1
        #confidence += 0.1

        return depth#, confidence

    def resize_like(self, inputs, ref):
        i_h, i_w = inputs.get_shape()[1], inputs.get_shape()[2]
        r_h, r_w = ref.get_shape()[1], ref.get_shape()[2]
        if i_h == r_h and i_w == r_w:
            return inputs
        else:
            return tf.image.resize(inputs, [r_h, r_w], method='nearest')

class ResUNet(keras.Model):
    def __init__(self, output_channels, layer_count, blocks_per_layer, batch_input_shape, dropout_rate=0.2):
        super().__init__()

        self.encoder = ResNet(layer_count, blocks_per_layer, batch_input_shape, dropout_rate=dropout_rate)
        self.decoder = ResNetTranspose(output_channels, layer_count, blocks_per_layer, batch_input_shape, dropout_rate=dropout_rate)

    def call(self, input_tensor, reconstruction=True, training=True):
        if reconstruction:
            conv, residuals = self.encoder(input_tensor, training=training)
            return self.decoder(conv, residuals, training=training)
        else:
            return self.encoder(input_tensor, unet=False, training=training)

class DepthEgoNet(keras.Model):
    def __init__(self, output_channels, layer_count, blocks_per_layer, batch_input_shape, dropout_rate=0.2, summary=True, full_summary=False):
        super().__init__()

        dec_inp_shape = [batch_input_shape[0],] + list((np.array(batch_input_shape[1:-1])/(2**layer_count)).astype(np.int32)) + [int(32*(2**layer_count)),]
        dec_inp_shape = tuple(dec_inp_shape)

        self.ego_net = ResNet(layer_count, blocks_per_layer, batch_input_shape, dropout_rate=dropout_rate, summary=full_summary)

        self.encoder = ResNet(layer_count, blocks_per_layer, batch_input_shape, dropout_rate=dropout_rate, summary=full_summary)
        #self.extra_pool = layers.AveragePooling2D(strides=(3, 2))
        self.global_pool = layers.GlobalAveragePooling2D()

        self.reshape = layers.Reshape((3, 3))

        self.translation = layers.Dense(3)
        self.rotation = layers.Dense(3)
        self.translation_scale = tf.Variable(tf.tile([[1.]], (1, 3)), trainable=True, name='translation_scalars')
        self.rotation_scale = tf.Variable(tf.tile([[0.01]], (1, 3)), trainable=True, name='rotation_scalars')

        self.brightness_a = tf.Variable(tf.tile([[1.]], (1, 1)), trainable=True, name='brightness_a')
        self.brightness_b = tf.Variable(tf.tile([[0.1]], (1, 1)), trainable=True, name='brightness_b')

        #self.translation = layers.Dense(3)
        #self.rotation = layers.Dense(9)
        self.foci = layers.Dense(2, activation='softplus')
        self.offset = layers.Dense(2)

        self.decoder = ResNetTranspose(output_channels, layer_count, blocks_per_layer, dec_inp_shape, dropout_rate=dropout_rate, summary=full_summary)

        #inp = keras.Input(shape=batch_input_shape[1:])
        #self.call(inp)
        #self.build(input_shape=batch_input_shape)
        #if summary or full_summary:
        #    self.summary()

    # NOTE: dummy function for getting the model to build
    #       do not use for computation, use call_no_build
    def call(self, input_tensor):
        img_enc = self.encoder(input_tensor, unet=False)

        #ego_enc = self.ego_net(input_tensor, unet=False)

        #flat_enc = self.extra_pool(img_enc)
        flat_enc = self.global_pool(img_enc)

        transformation = self.transformation(flat_enc)

        #translation = self.translation(flat_enc)
        #rotation = self.rotation(flat_enc)
        #rotation = self.reshape(rotation)
        #translation = tf.expand_dims(translation, axis=-1)
        #transformation = tf.concat([rotation, translation], axis=-1)

        intrinsics = self.intrinsics(flat_enc)

        depth = self.decoder(img_enc) + 0.01

        return depth, transformation, intrinsics

    def depth_prediction(self, image, training):
        img_enc, residuals = self.encoder(image, training=training)
        depth = self.decoder(img_enc, residuals, training=training)
        #depth_confidence = self.decoder(img_enc, residuals, training=training)
        #depth, confidence = self.decoder(img_enc, residuals, training=training)
        depth = tf.squeeze(depth, axis=-1) + 1e-5
        #confidence = tf.squeeze(confidence, axis=-1) + 1e-5 + 1

        return depth#, confidence

    def ego_intrinsics_prediction(self, image1, image2, training):
        input_tensor = tf.concat([image1, image2], axis=-1)
        resolution = tf.convert_to_tensor([[input_tensor.shape[2], input_tensor.shape[1]]], dtype=tf.float32)

        img_enc = self.ego_net(input_tensor, unet=False, training=training)

        flat_enc = self.global_pool(img_enc)

        translation = self.translation(flat_enc) * (tf.nn.relu(self.translation_scale-1e-4)+1e-4)
        rotation = self.rotation(flat_enc) * (tf.nn.relu(self.rotation_scale-1e-4)+1e-4)
        transformation = tf.concat([rotation, translation], axis=-1)

        foci, offsets = (self.foci(flat_enc)*resolution, (self.offset(flat_enc)+0.5)*resolution)

        foci = tf.linalg.diag(foci)

        intrinsic_mat = tf.concat([foci, tf.expand_dims(offsets, -1)], axis=2)
        batch_size = tf.shape(input_tensor)[0]
        last_row = tf.tile([[[0.0, 0.0, 1.0]]], [batch_size, 1, 1])
        intrinsics = tf.concat([intrinsic_mat, last_row], axis=1)

        return transformation, intrinsics

    def photometric_adjustment(self, image):
        return self.brightness_a*image + self.brightness_b

    # input_tensor.shape == [batch_size, resolution[0], resolution[1], 6]
    # represents two consecutive images stacked channel-wise
    #
    # Returns a depth map corresponding to the first image of the stack
    def call_no_build(self, image1, image2, training=True):
        depth = self.depth_prediction(image1, training)
        transformation, intrinsics = self.ego_intrinsics_prediction(image1, image2, training)

        return depth, transformation, intrinsics
