import numpy as np
import pickle
import glob
import os
import matplotlib.pyplot as plt
import keras.backend as K

import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Add, BatchNormalization, Activation
from keras.layers import AveragePooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler


def read_cifar10(path="./cifar-10/cifar-10-batches-py/"):
    traindatapath = path + "data*"
    testdatapath = path + "test*"
    # data_img = np.zeros(shape=(50000, 32, 32, 3), dtype=np.float32)
    train_data = []
    train_label = []
    test_data = []
    test_label = []

    # 训练数据
    for path in sorted(glob.glob(traindatapath)):
        print(path)
        path = open(path, 'rb')

        img_dic = pickle.load(path, encoding='bytes')

        # 查看数据情况
        # print(img_dic.keys())
        # print(img_dic[b"batch_label"])
        # print(img_dic[b"labels"])
        # print(img_dic[b"data"])
        # print(img_dic[b"filenames"])

        train_label.append(np.array(img_dic[b"labels"]))
        img = np.array(img_dic[b"data"])
        for i in range(img.shape[0]):
            tmp = np.array(img[i].reshape((3, 32, 32)))
            tmp = tmp.transpose((1, 2, 0))
            # 输出图像
            # plt.imshow(tmp)
            # plt.show()
            train_data.append(tmp)
    train_label = np.array(train_label).reshape((50000))
    train_data = np.array(train_data, dtype=np.float32)
    train_data /= 255


    # 测试数据
    for path in sorted(glob.glob(testdatapath)):
        print(path)
        path = open(path, 'rb')
        img_dic = pickle.load(path, encoding='bytes')
        test_label.append(np.array(img_dic[b"labels"]))
        img = np.array(img_dic[b"data"])
        for i in range(img.shape[0]):
            tmp = np.array(img[i].reshape((3, 32, 32)))
            tmp = tmp.transpose((1, 2, 0))
            # 输出图像
            # plt.imshow(tmp)
            # plt.show()
            test_data.append(tmp)
    test_label = np.array(test_label).reshape((10000))
    test_data = np.array(test_data, dtype=np.float32)
    test_data /= 255

    print(train_data.shape)
    print(train_label.shape)
    print(test_data.shape)
    print(test_label.shape)
    return train_data, train_label, test_data, test_label


'''
resnet50
'''

def identity_block(X, f, filters):
    F1, F2, F3 = filters
    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), padding='valid', kernel_initializer='he_normal', use_bias=True)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), padding='same', kernel_initializer='he_normal', use_bias=True)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), padding='valid', kernel_initializer='he_normal', use_bias=True)(X)
    X = BatchNormalization(axis=3)(X)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def convolution_block(X, f, filters, s=2):
    F1, F2, F3 = filters
    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), padding='valid', kernel_initializer='he_normal', use_bias=True)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), padding='same', kernel_initializer='he_normal', use_bias=True)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), padding='valid', kernel_initializer='he_normal', use_bias=True)(X)
    X = BatchNormalization(axis=3)(X)

    X_shortcut = Conv2D(F3, kernel_size=(1, 1), strides=(s, s), kernel_initializer='he_normal', use_bias=True)(X_shortcut)
    X_shortcut = BatchNormalization(axis=3)(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def ResNet50_model():
    input_shape = (32, 32, 3)
    inputs = Input(input_shape)

    X = ZeroPadding2D((3, 3))(inputs)

    X = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), kernel_initializer='he_normal')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolution_block(X, f=3, filters=[64, 64, 256], s=1)
    X = identity_block(X, 3, [64, 64, 256])
    X = identity_block(X, 3, [64, 64, 256])

    X = convolution_block(X, f=3, filters=[128, 128, 512], s=1)
    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])

    X = convolution_block(X, f=3, filters=[256, 256, 1024], s=1)
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])

    X = convolution_block(X, f=3, filters=[512, 512, 2048], s=1)
    X = identity_block(X, 3, [512, 512, 2048])
    X = identity_block(X, 3, [512, 512, 2048])

    X = AveragePooling2D((2, 2))(X)

    X = Flatten()(X)
    outputs = Dense(10, activation='softmax', kernel_initializer='he_normal')(X)

    model = Model(inputs, outputs)
    return model

'''
resnet50
'''



def train_resnet():
    # 做数据
    train_data, train_label, test_data, test_label = read_cifar10()

    train_label = keras.utils.to_categorical(train_label, num_classes=10, dtype=np.float32)
    test_label = keras.utils.to_categorical(test_label, num_classes=10, dtype=np.float32)

    # 加载模型
    # 设置学习率衰减
    lr = 1e-3

    def lr_scheduler(epoch):
        return lr * (0.1 ** (epoch // 60))

    reduce_lr = LearningRateScheduler(lr_scheduler)

    SGDop = SGD(lr=lr, momentum=0.9, nesterov=False)
    model = ResNet50_model()
    model.compile(optimizer=SGDop, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=train_data, y=train_label, batch_size=64, epochs=30, shuffle=True, callbacks=[reduce_lr])
    modelpath = "Resnet50.hdf5"
    loss, acc = model.evaluate(test_data, test_label)
    print(loss, acc)
    model.save(modelpath)
    print("model saved!!!")


def feature_map():

    t = 150
    _, _, test_data, _ = read_cifar10()
    test_data = test_data[t:t+9]
    test_data = np.array(test_data)
    # test_data = np.expand_dims(test_data, 0)
    print("test data shape:", test_data.shape)
    model = ResNet50_model()
    model.load_weights('Resnet50.hdf5')
    layer_1 = K.function([model.layers[0].input], [model.layers[2].output])
    # f1 = layer_1([test_data])[0]

    for i in range(9):
        tmp = test_data[i]
        tmp = np.array(tmp)
        tmp = np.expand_dims(tmp, 0)
        f1 = layer_1([tmp])[0]
        show_img = f1[:, :, :, 8]
        show_img.shape = [32, 32]
        plt.subplot(3, 3, i+1)
        plt.imshow(show_img, cmap='gray')
        plt.axis('off')
    # for i in range(16):
    #     show_img = f1[:, :, :, i]
    #     show_img.shape = [32, 32]
    #     plt.subplot(4, 4, i+1)
    #     plt.imshow(show_img, cmap='gray')
    #     plt.axis('off')
    plt.savefig('resnetfeature_map4.png')
    plt.show()


if __name__ == "__main__":
    # train_VGG()
    # train_resnet()
    feature_map()