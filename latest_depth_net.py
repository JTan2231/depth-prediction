import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

kernel_reg = "l1"
activity_reg = "l1_l2"

DIM1 = 128
DIM2 = 256

def conv2d(filters,
           kernel_size=3,
           dilation_rate=1,
           padding='same',
           activation='linear',
           kernel_regularizer=kernel_reg,
           activity_regularizer=activity_reg,
           name='conv2d'):

    return layers.Conv2D(filters,
                         kernel_size,
                         padding=padding,
                         dilation_rate=dilation_rate,
                         activation=activation,
                         #kernel_regularizer=kernel_regularizer,
                         #activity_regularizer=activity_regularizer,
                         name=name)

class ResBlock(layers.Layer):
    def __init__(self, filters, dilation_rate=1, dropout_rate=0.0, upscale=True, name=None):
        super().__init__(name=name)

        self.upscale = upscale

        if upscale:
            self.scale = conv2d(filters, kernel_size=1)
            self.bn_scale = layers.BatchNormalization()

        self.neck = conv2d(filters//2, kernel_size=1)
        self.conv = conv2d(filters//2, dilation_rate=dilation_rate)
        self.out = conv2d(filters, kernel_size=1)

        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
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

        conv = self.out(conv)
        conv = tf.nn.swish(conv)

        return self.dropout(conv + init, training=training)

def resize_like(image, target):
    ishape = tf.shape(image)
    tshape = tf.shape(target)

    return tf.image.resize(image, [tshape[1], tshape[2]])

def get_layer_outputs(model, layer_names, input_data):
    outputs = [model.get_layer(layer_name).output for layer_name in layer_names]
    #layers_fn = keras.backend.function(model.input, outputs)

    return outputs#layers_fn(input_data)

class Scalar(layers.Layer):
    def __init__(self, initializer=1.0, constraint=0.01):
        super().__init__()

        self.value = initializer
        self.rot = initializer# / 10
        self.constraint = constraint

    def build(self, input_shape):
        self.translation = tf.Variable(initial_value=[self.value, self.value, self.value], dtype=tf.float32, trainable=True, name="translation_scalar")
        self.rotation = tf.Variable(initial_value=[self.rot, self.rot, self.rot], dtype=tf.float32, trainable=True, name="rotation_scalar")

    def call(self, input_tensor):
        r = input_tensor[:,:3] * self.rotation
        t = input_tensor[:,3:] * self.translation

        return tf.nn.relu(tf.concat([r, t], axis=-1) - self.constraint) + self.constraint

def get_sequential(names=True):
    if names:
        return keras.Sequential([ResBlock(32),
                                 layers.AveragePooling2D(),
                                 ResBlock(64, name='conv1/relu'),
                                 layers.AveragePooling2D(),
                                 ResBlock(64, name='pool2_relu'),
                                 layers.AveragePooling2D(),
                                 ResBlock(128, name='pool3_relu'),
                                 layers.AveragePooling2D(),
                                 ResBlock(256, name='pool4_relu'),
                                 layers.AveragePooling2D(),
                                 ResBlock(512, name='relu')])
    else:
        return keras.Sequential([ResBlock(32),
                                 layers.AveragePooling2D(),
                                 ResBlock(64),
                                 layers.AveragePooling2D(),
                                 ResBlock(64),
                                 layers.AveragePooling2D(),
                                 ResBlock(128),
                                 layers.AveragePooling2D(),
                                 ResBlock(256),
                                 layers.AveragePooling2D(),
                                 ResBlock(512, name='top_activation')])

def get_resnet18():
    return keras.Sequential([conv2d(64, 7, strides=2, activation='relu'),
                             layers.MaxPooling2D(3),
                             ResBlock(64),
                             ResBlock(64),
                             ResBlock(128),
                             ResBlock(256),
                             ResBlock(512),
def get_model():
    #depth_backbone = keras.applications.DenseNet121(weights='imagenet', include_top=False, pooling=None, input_shape=(DIM1, DIM2, 3))
    #ego_backbone = keras.applications.EfficientNetB0(weights=None, include_top=False, pooling='avg', input_shape=(DIM1, DIM2, 6))

    heads = ["relu",
             "pool4_relu",
             "pool3_relu",
             "pool2_relu",
             "conv1/relu"]

    dcnn = [ResBlock(256),
            ResBlock(128),
            ResBlock(64),
            ResBlock(64),
            ResBlock(32)]

    depth_backbone = get_sequential()
    depth_backbone.build((None, DIM1, DIM2, 3))

    ego_backbone = get_sequential(names=False)
    ego_backbone.build((None, DIM1, DIM2, 6))

    upsamplers = [layers.Conv2DTranspose(256, 2, strides=2),
                  layers.Conv2DTranspose(128, 2, strides=2),
                  layers.Conv2DTranspose(64, 2, strides=2),
                  layers.Conv2DTranspose(64, 2, strides=2),
                  layers.Conv2DTranspose(32, 2, strides=2)]

    depth = layers.Conv2D(3, 1, padding='same', activation='softplus', name='depth')

    inp = keras.Input((DIM1, DIM2, 6))

    depth_inp = keras.Input((DIM1, DIM2, 3))

    image1 = inp[...,0:3]
    image2 = inp[...,3:]

    heads = get_layer_outputs(depth_backbone, heads, depth_inp)

    ego_encoding = get_layer_outputs(ego_backbone, ['top_activation'], inp)[0]
    ego_encoding = layers.Reshape((-1,))(ego_encoding)#layers.GlobalAveragePooling2D()(ego_encoding)

    #ego_enc1 = get_layer_outputs(ego_backbone, ["relu"], image1)[0]
    #ego_enc2 = get_layer_outputs(ego_backbone, ["relu"], image2)[0]

    #ego_enc1 = layers.Reshape((-1,))(ego_enc1)
    #ego_enc2 = layers.Reshape((-1,))(ego_enc2)

    #ego_encoding = layers.Concatenate(axis=-1)((ego_enc1, ego_enc2))

    transformation = layers.Dense(6, name='transformation')(ego_encoding)
    transformation = Scalar(0.1)(transformation)

    foci = layers.Dense(2, activation='softplus', name='foci')(ego_encoding)# * tf.cast(tf.shape(image1)[1:3], tf.float32)
    offset = layers.Dense(2, name='offset')(ego_encoding)# * tf.cast(tf.shape(image1)[1:3], tf.float32)

    #intrinsics = layers.Concatenate(axis=-1, name='intrinsics')((foci, offset))

    conv = dcnn[0](heads[0])
    conv = upsamplers[0](conv)

    convs = []
    for i in range(1, len(dcnn), 1):
        conv = tf.concat([conv, heads[i]], axis=-1)
        conv = dcnn[i](conv)
        convs.append(conv)
        conv = upsamplers[i](conv)

    final_conv = tf.concat([resize_like(x, conv) for x in convs] + [conv], axis=-1)
    final_conv = depth(final_conv)
    final_conv *= tf.cast(tf.math.is_finite(final_conv), tf.float32)

    return keras.Model([depth_backbone.input, ego_backbone.input], [final_conv, transformation, foci, offset])
