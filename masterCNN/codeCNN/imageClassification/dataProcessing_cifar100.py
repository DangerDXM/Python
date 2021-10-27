import os
import tensorflow as tf
from numpy import *
import pickle as pk


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pk.load(fo, encoding='bytes')
    return dict


path_tr = 'E:/Program/Python/masterCNN/dataSet/cifar-100-py/train'
path_t = 'E:/Program/Python/masterCNN/dataSet/cifar-100-py/test'
d_tr = unpickle(path_tr)
d_t = unpickle(path_t)
trainImg, trainLab, className = d_tr[b'data'], d_tr[b'fine_labels'], d_tr[b'filenames']
testImg, testLab = d_t[b'data'], d_t[b'fine_labels']

randnum = random.randint(0, 100)
random.seed(randnum)
random.shuffle(trainImg)
random.seed(randnum)
random.shuffle(trainLab)

trainImg, trainLab = reshape(trainImg, [-1, 32, 32, 3]), reshape(array(trainLab), [-1, 1])
testImg, testLab = reshape(testImg, [-1, 32, 32, 3]), reshape(array(testLab), [-1, 1])
for i in range(20):
    print(testLab[i])

r = trainImg[:, :, :, :1]
g = trainImg[:, :, :, 1:2]
b = trainImg[:, :, :, 2:3]
print(array(r).shape)
print('r均值与方差', mean(r), std(r, ddof=1))
print('g均值与方差', mean(g), std(g, ddof=1))
print('b均值与方差', mean(b), std(b, ddof=1))
'''
r均值与方差 121.93828919921874 68.38905170212391
g均值与方差 121.93687603515625 68.38590474405197
b均值与方差 121.933013125 68.3919150830306
'''


def color_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean_ = [121.938, 121.937, 121.933]
    std_ = [68.389, 68.386, 68.392]
    for i in range(3):
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean_[i]) / std_[i]
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean_[i]) / std_[i]
    return x_train, x_test


trainImg, testImg = color_preprocessing(trainImg, testImg)
