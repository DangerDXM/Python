import tensorflow as tf


def conv_block(x, filters, kernel=(1, 1), stride=(1, 1)):
    x = tf.keras.layers.Conv2D(filters, kernel, strides=stride, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.nn.relu6)(x)
    return x


def depthwise_res_block(x, filters, kernel, stride, t, resdiual=False):
    input_tensor = x
    exp_channels = x.shape[-1] * t  # 扩展维度

    x = conv_block(x, exp_channels, (1, 1), (1, 1))

    x = tf.keras.layers.DepthwiseConv2D(kernel, padding='same', strides=stride)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.nn.relu6)(x)

    x = tf.keras.layers.Conv2D(filters, (1, 1), padding='same', strides=(1, 1))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if resdiual:
        x = tf.keras.layers.add([x, input_tensor])

    return x


def inverted_residual_layers(x, filters, stride, t, n):
    x = depthwise_res_block(x, filters, (3, 3), stride, t, False)

    for i in range(1, n):
        x = depthwise_res_block(x, filters, (3, 3), (1, 1), t, True)
    return x


def MovblieNetV2 (classes):

    img_input = tf.keras.layers.Input(shape=(224, 224, 3))

    x = conv_block(img_input, 32, (3, 3), (2, 2))

    x = tf.keras.layers.DepthwiseConv2D((3, 3), padding='same', strides=(1,1))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.nn.relu6)(x)

    x = inverted_residual_layers(x, 16, (1, 1), 1, 1)
    x = inverted_residual_layers(x, 24, (2, 2), 6, 3)
    x = inverted_residual_layers(x, 64, (2, 2), 6, 4)
    x = inverted_residual_layers(x, 96, (1, 1), 6, 3)
    x = inverted_residual_layers(x, 160, (2, 2), 6, 3)
    x = inverted_residual_layers(x, 320, (1, 1), 6, 1)

    x = conv_block(x, 1280, (1, 1), (2, 2))

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Reshape((1, 1, 1280))(x)
    x = tf.keras.layers.Conv2D(classes, (1, 1), padding='same')(x)
    x = tf.keras.layers.Reshape((classes,))(x)
    x = tf.keras.layers.Activation('softmax')(x)

    model = tf.keras.Model(img_input, x)
    return model

