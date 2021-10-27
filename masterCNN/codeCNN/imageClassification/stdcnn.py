import tensorflow.keras as keras
from codeCNN.imageClassification import dataProcessing_cifar10 as data
from numpy import *


class CNN(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CNN, self).__init__(**kwargs)
        # function of first convolution layer is to decrease input channels
        self.conv1 = keras.layers.Conv2D(32, kernel_size=3, strides=1, padding='same')
        self.bn1 = keras.layers.BatchNormalization()
        self.relu1 = keras.layers.Activation('relu')

        self.conv2 = keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same')
        self.bn2 = keras.layers.BatchNormalization()
        self.relu2 = keras.layers.Activation('relu')
        self.pool2 = keras.layers.MaxPool2D((2, 2))

    def build(self, input_shape):
        # super(CNN, self).build(input_shape)
        # self.kernel = self.add_weight("kernel", shape=[int(input_shape[-1]), 32, 32, 3])
        self.kernel = self.add_variable("kernel", shape=[int(input_shape[-1]), 32, 32, 3])
        super().build(input_shape)

    def call(self, inputs):
        x_ = self.conv1(inputs)
        x_ = self.bn1(x_)
        x_ = self.relu1(x_)

        x_ = self.conv2(x_)
        x_ = self.bn2(x_)
        x_ = self.relu2(x_)
        x_ = self.pool2(x_)
        return x_

    def get_config(self):
        config = {
            "conv1": self.conv1,
            "bn1": self.bn1,
            "relu1": self.relu1,
            "conv2": self.conv2,
            "bn2": self.bn2,
            "relu2": self.relu2,
            "pool2": self.pool2
        }
        base_config = super(CNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


model = keras.models.Sequential()
model.add(keras.layers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal', input_shape=[32, 32, 3]))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(CNN())
model.add(CNN())
model.add(keras.layers.GlobalAveragePooling2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(90, activation='relu', kernel_initializer="he_normal"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()
model.compile(loss="sparse_categorical_crossentropy",
              optimizer='Nadam',
              metrics=["accuracy"])

saved_folder = '../model/h5/std.h5'  # 'E:\Program\Python\masterCNN\model\keras_model.h5'

# checkpoint and early stop
# checkpoint = keras.callbacks.ModelCheckpoint(saved_folder)
earlyStop = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

validationSplit = 40000
x_train, y_train, x_valid, y_valid = data.trainImg[:validationSplit], data.trainLab[:validationSplit], \
                                     data.trainImg[validationSplit:], data.trainLab[validationSplit:]

history = model.fit(x_train, y_train, epochs=2, batch_size=64,
                    validation_data=(x_valid, y_valid),
                    callbacks=[earlyStop])

model.save(filepath=saved_folder)
print('Standard model has been saved in direction codeCNN...')
model.evaluate(data.testImg, data.testLab)

_custom_objects = {"CNN":  CNN}

modelLoad = keras.models.load_model(saved_folder, custom_objects=_custom_objects)
x_predict = data.testImg[0]
x_predict = reshape(array(x_predict), [-1, 32, 32, 3])
y = modelLoad.predict(x_predict)
y = y.reshape(-1, )
print('number of dim: {} \n array.ndimtype y: {}\n y: {}'.format(y.ndim, type(y), data.className[y.argmax()]))
