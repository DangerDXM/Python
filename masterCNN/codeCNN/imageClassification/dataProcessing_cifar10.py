import os
import tensorflow as tf
from numpy import *
import pickle as pk


# 拿到路径，拿到标签，再做batch
def unpickle(file):
    with open(file, 'rb') as fo:
        dict_item = pk.load(fo, encoding='bytes')
    return dict_item


def data_extract(pathList):
    img, label = [], []
    print(pathList, '\n', len(pathList))
    for item in pathList:
        dataDict = unpickle(os.path.join(cFile, item))
        img.extend(dataDict[b'data'])
        label.extend(dataDict[b'labels'])
    return img, label


'''
['batches.meta', 'data_batch_1', 'data_batch_2', 'data_batch_3', 
 'data_batch_4', 'data_batch_5', 'readme.html', 'test_batch']
 '''
cFile = '../dataSet/cifar-10-batches-py'
dataPath = os.listdir(cFile)
trainDir = dataPath[1:6]
testDir = [dataPath[7]]

trainImg, trainLabel = data_extract(trainDir)
randnum = random.randint(0, 100)
random.seed(randnum)
random.shuffle(trainImg)
random.seed(randnum)
random.shuffle(trainLabel)

testImg, testLabel = data_extract(testDir)

className = ['airplan', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

trainImg, trainLab = reshape(trainImg, [-1, 32, 32, 3]), reshape(array(trainLabel), [-1, 1])
testImg, testLab = reshape(testImg, [-1, 32, 32, 3]), reshape(array(testLabel), [-1, 1])


def color_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean_ = [125.307, 122.95, 113.865]
    std_ = [62.9932, 62.0887, 66.7048]
    # mean_ = [120.709, 120.709, 120.705]
    # std_ = [64.149, 64.147, 64.154]
    for i in range(3):
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean_[i]) / std_[i]
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean_[i]) / std_[i]
    return x_train, x_test


r = trainImg[:, :, :, :1]
g = trainImg[:, :, :, 1:2]
b = trainImg[:, :, :, 2:3]
print(array(r).shape)
print('r均值与方差', mean(r), std(r, ddof=1))
print('g均值与方差', mean(g), std(g, ddof=1))
print('b均值与方差', mean(b), std(b, ddof=1))

trainImg, testImg = color_preprocessing(trainImg, testImg)

print('train data dimension:', trainImg.shape)
print('train label dimension:', trainLab.shape)

print('test data dimension:', array(testImg).shape)
print('test label dimension:', array(testLab).shape)


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [100, 100])
    image /= 255.0  # normalize to [0,1] range
    # image = tf.reshape(image,[100*100*3])
    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)
