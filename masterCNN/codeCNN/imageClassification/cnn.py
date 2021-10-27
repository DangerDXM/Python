from codeCNN.imageClassification import dataProcessing_cifar100 as data

from numpy import *
import numpy as np
import tensorflow.keras as keras

import os
from matplotlib import pyplot as plt
import time

'''

A standard method for datasets

class Standardization(keras.layers.Layer):
    def adapt(self, data_sample):
        self.means_ = np.mean(data_sample, axis=0, keepdims=True)
        self.stds_ = np.std(data_sample, axis=0, keepdims=True)

    def call(self, inputs):
        return (inputs - self.means_) / (self.stds_ + keras.backend.epsilon())
std_layer = Standardization()
std_layer.adapt(data.trainImg)
'''
'''
Total params: 2,486,782
Trainable params: 2,483,454
Non-trainable params: 3,328

Total params: 2,335,518
Trainable params: 2,331,102
Non-trainable params: 4,416

Total params: 2,209,962
Trainable params: 2,205,546
Non-trainable params: 4,416
'''
class baseUnit(keras.layers.Layer):
    def __init__(self, filters_, pooling=False, strides=1, k_size=3, activation='elu', **kwargs):
        # super().__init__(**kwargs)
        super(baseUnit, self).__init__(**kwargs)
        self.filters = filters_
        self.pooling = pooling
        self.k_size = k_size
        self.strides = strides
        self.activation = activation

        # self.avgpool = keras.layers.AveragePooling2D(pool_size=2, strides=2)
        # self.input_conv = keras.layers.SeparableConv2D(self.filters, kernel_size=self.k_size,
        #                                                depthwise_initializer="he_normal",
        #                                                depthwise_regularizer=keras.regularizers.l2(1e-2),
        #                                                kernel_initializer="he_normal",
        #                                                kernel_regularizer=keras.regularizers.l2(1e-2),
        #                                                strides=self.strides, padding='same')
        # self.input_bn = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        # self.input_elu = self.elu1 = keras.layers.Activation(self.activation)

        # function of first convolution layer is to decrease input channels

        self.conv1 = keras.layers.Conv2D(self.filters, kernel_size=1,
                                         kernel_initializer="he_normal",
                                         kernel_regularizer=keras.regularizers.l2(1e-4),
                                         strides=1, padding='same')
        self.bn1 = keras.layers.BatchNormalization()
        self.elu1 = keras.layers.Activation(self.activation)
        # self.bn1 = keras.layers.BatchNormalization()
        # two processing branches, first the standard convolution and second the DSConv.
        self.conv2 = keras.layers.Conv2D(int(self.filters / 2), kernel_size=self.k_size,
                                         kernel_initializer="he_normal",
                                         kernel_regularizer=keras.regularizers.l2(1e-4),
                                         strides=self.strides, padding='same')
        self.bn2 = keras.layers.BatchNormalization()
        self.elu2 = keras.layers.Activation(self.activation)
        # self.bn2 = keras.layers.BatchNormalization()

        self.s_conv2 = keras.layers.SeparableConv2D(int(self.filters / 2), kernel_size=self.k_size,
                                                    depthwise_initializer="he_normal",
                                                    depthwise_regularizer=keras.regularizers.l2(1e-4),
                                                    kernel_initializer="he_normal",
                                                    kernel_regularizer=keras.regularizers.l2(1e-4),
                                                    strides=self.strides, padding='same')
        self.bn2_ = keras.layers.BatchNormalization()
        self.elu2_ = keras.layers.Activation(self.activation)
        # self.bn2_ = keras.layers.BatchNormalization()
        self.pool = keras.layers.MaxPool2D((2, 2))

        self.conv3 = keras.layers.Conv2D(int(self.filters * 0.5), kernel_size=1,
                                         kernel_initializer="he_normal",
                                         kernel_regularizer=keras.regularizers.l2(1e-4),
                                         strides=1, padding='same')
        self.bn3 = keras.layers.BatchNormalization()
        self.elu3 = keras.layers.Activation(self.activation)

    def build(self, input_shape):
        super(baseUnit, self).build(input_shape)
        self.kernel = self.add_weight("kernel", shape=[int(input_shape[-1]), 32, 32, 3])

    def call(self, inputs):
        x_ = self.conv1(inputs)
        x_ = self.bn1(x_)
        x_ = self.elu1(x_)
        # x_ = self.bn1(x_)

        x1_ = self.conv2(x_)
        x1_ = self.bn2(x1_)
        x1_ = self.elu2(x1_)
        # x1 = self.bn2(x1)

        x2_ = self.s_conv2(x_)
        x2_ = self.bn2_(x2_)
        x2_ = self.elu2_(x2_)
        # x2 = self.bn2_(x2)

        x_ = keras.layers.concatenate([x1_, x2_])
        x_ = self.conv3(x_)
        x_ = self.bn3(x_)
        if self.filters > 64:
            x_ = self.elu3(x_)

        if self.pooling:
            x_ = self.pool(x_)
        # x_ = keras.layers.Add()([x_, inputs])

        '''
        if self.pooling:
            x_i = self.avgpool(inputs)
            x_i = self.input_conv(x_i)
            x_i = self.input_bn(x_i)
            x_i = self.input_elu(x_i)
            x_ = keras.layers.Add()([x_, x_i])
        else:
            x_i = self.input_conv(inputs)
            x_i = self.input_bn(x_i)
            x_i = self.input_elu(x_i)
            x_ = keras.layers.Add()([x_, x_i])
        '''
        return x_

    def get_config(self):
        config = super(baseUnit, self).get_config().copy()
        # config["avgpool"] = self.avgpool,
        config["filters"] = self.filters,
        config["k_size"] = self.k_size,
        config["strides"] = self.strides,
        config["activation"] = self.activation,
        config["conv1"] = self.conv1,
        config["bn1"] = self.bn1,
        config["elu1"] = self.elu1,
        config["conv2"] = self.conv2,
        config["bn2"] = self.bn2,
        config["elu2"] = self.elu2,
        config["s_conv2"] = self.s_conv2,
        config["bn2_"] = self.bn2_,
        config["elu2_"] = self.elu2_,
        config["conv3"] = self.conv3,
        config["bn3"] = self.bn3,
        config["elu3"] = self.elu3
        '''
        config.update({
            "filters": self.filters,
            "k_size": self.k_size,
            "strides": self.strides,
            "activation": self.activation,
            "conv1": self.conv1,
            "bn1": self.bn1,
            "elu1": self.elu1,
            "conv2": self.conv2,
            "bn2": self.bn2,
            "elu2": self.elu2,
            "s_conv2": self.s_conv2,
            "bn2_": self.bn2_,
            "elu2_": self.elu2_})  # "pool": self.pool
        '''
        return config


model_new = keras.models.Sequential()
model_new.add(keras.layers.Conv2D(32, 3, padding='same',
                                  kernel_initializer="he_normal",
                                  kernel_regularizer=keras.regularizers.l2(1e-4),
                                  input_shape=[32, 32, 3]))
model_new.add(keras.layers.BatchNormalization())
model_new.add(keras.layers.Activation('elu'))
# model_new.add(keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5))
# for filters in [64] * 2 + [128] * 1 + [256] * 1:
filters = [32, 64, 64, 128, 128, 256]
for i in range(len(filters)):
    # act = 'elu'
    # if filters[i] < 100:
    #     act = None
    if i == 2 or i == 4:
        model_new.add(baseUnit(filters[i], pooling=True))
        # model_new.add(keras.layers.Dropout(0.5))
    else:
        model_new.add(baseUnit(filters[i]))

model_new.add(keras.layers.Conv2D(1000, 1, padding='same',
                                  kernel_initializer="he_normal",
                                  kernel_regularizer=keras.regularizers.l2(1e-4)))
model_new.add(keras.layers.GlobalMaxPool2D())
# model_new.add(keras.layers.Dense(1000, activation='elu', kernel_initializer="he_normal",
#                                  kernel_regularizer=keras.regularizers.l2(1e-5)))
# model_new.add(keras.layers.Dropout(0.5))
# # model_new.add(keras.layers.Dense(90, activation='elu', kernel_initializer="he_normal",
# #                                  kernel_regularizer=keras.regularizers.l2(1e-4)))
# # model_new.add(keras.layers.Dropout(0.5))
model_new.add(keras.layers.Dense(100, activation='softmax'))
model_new.summary()

# Nadam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.01)
model_new.compile(loss="sparse_categorical_crossentropy",
                  optimizer='Nadam',
                  metrics=["accuracy"])

saved_folder = '../model/h5/masterModel.h5'  # 'E:\Program\Python\masterCNN\model\keras_model.h5'
# checkpoint and early stop
checkpoint = keras.callbacks.ModelCheckpoint(saved_folder, save_best_only=True)
# es 的四个可选 monitor: loss,accuracy,val_loss,val_accuracy
earlyStop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10,
                                          mode='max', restore_best_weights=True)

root_logdir = os.path.join(os.curdir, "my_logs")


# def scheduler(epoch):
#     # if epoch < 10:
#     #     return 0.05
#
#     # if epoch < 30:
#     #     return 0.001
#     if epoch < 40:
#         return 0.0005
#     if epoch < 80:
#         return 0.0001
#     if epoch < 120:
#         return 0.00005
#     if epoch < 150:
#         return 0.00001
#     return 0.000001


# change_lr = keras.callbacks.LearningRateScheduler(scheduler)


def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

# validationSplit = 45000
x_train, y_train = data.trainImg, data.trainLab
#                                      data.trainImg[validationSplit:], data.trainLab[validationSplit:]

# data_gen = ImageDataGenerator(horizontal_flip=True,
#                               width_shift_range=0.125,
#                               height_shift_range=0.125,
#                               fill_mode='constant', cval=0.)
# data_gen.fit(x_train)
# data_gen.flow(x_train, y_train, batch_size=32),

history = model_new.fit(x_train, y_train,
                        epochs=200,
                        validation_split=0.2,
                        # validation_data=(x_valid, y_valid),
                        callbacks=[earlyStop, tensorboard_cb])

plt.plot(np.arange(len(history.history['loss'])), history.history['loss'], label='training')
plt.plot(np.arange(len(history.history['val_loss'])), history.history['val_loss'], label='validation')
# plt.title('Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc=0)
plt.savefig(os.path.join(run_logdir, 'loss.png'))
plt.show()


plt.plot(np.arange(len(history.history['accuracy'])), history.history['accuracy'], label='training')
plt.plot(np.arange(len(history.history['val_accuracy'])), history.history['val_accuracy'], label='validation')
# plt.title('Accuracy')
plt.xlabel('epoch')
plt.ylabel('top-1 accuracy')
plt.legend(loc=0)
plt.savefig(os.path.join(run_logdir, 'accuracy.png'))
plt.show()

predict = model_new.evaluate(data.testImg, data.testLab)


# model_new.save(filepath=saved_folder)
modelLoad = keras.models.load_model(saved_folder)
x_predict = data.testImg[0]
x_predict = reshape(array(x_predict), [-1, 32, 32, 3])
y = modelLoad.predict(x_predict)
y = y.reshape(-1, )
print('number of dim: {} \n array.ndimtype y: {}\n y: {}'.format(y.ndim, type(y), data.className[y.argmax()]))

