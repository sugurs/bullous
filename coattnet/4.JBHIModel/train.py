#!/usr/bin/python
# coding:utf-8
from keras.layers import Input, Embedding, LSTM, Dense, add, Concatenate
from keras.models import Model
from keras.utils import plot_model
import os
import keras
import time
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import os.path
import glob
import shutil
import cPickle
from tqdm import tqdm
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.layers import Conv1D, Conv2D, Dense, MaxPool1D, concatenate, Flatten, \
    AveragePooling2D, Dropout, BatchNormalization, Activation
from keras import Input, Model
from keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler

def multi_input_model():
    """构建多输入模型"""
    basemodel = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    clinicInput = basemodel.input
    x = basemodel.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = BatchNormalization()(x)
    # x = Dense(19, activation='relu')(x)
    # x = Dropout(0.5)(x)

    metadataInput = Input(shape=(19,))
    Merged = concatenate([x, metadataInput])
    # x = Dense(1024, activation='relu')(Merged)
    # x = Dropout(0.5)(x)
    # x = Dense(1024, activation='relu')(x)
    # x = Dropout(0.5)(x)
    predictions = Dense(2, activation='softmax')(Merged)
    model = Model(input=[clinicInput, metadataInput], output=predictions)
    return model


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

    # 产生训练数据
    train_dir = [
        '/media/wz209/a29353b7-1090-433f-b452-b4ce827adb17/sugurs/PythonProjects/pem_multi/dataset/patch/1',
        '/media/wz209/a29353b7-1090-433f-b452-b4ce827adb17/sugurs/PythonProjects/pem_multi/dataset/patch/2',
        '/media/wz209/a29353b7-1090-433f-b452-b4ce827adb17/sugurs/PythonProjects/pem_multi/dataset/patch/4',
    ]
    test_dir = '/media/wz209/a29353b7-1090-433f-b452-b4ce827adb17/sugurs/PythonProjects/pem_multi/dataset/patch/3'

    train_images = []
    train_labels = []
    train_features = []

    test_images = []
    test_labels = []
    test_features = []

    # 读取TPC和LTPC病人的序列号
    with open('dict19.pkl', 'rb') as f:
        samples = cPickle.load(f)

    img_list = []
    for train in train_dir:
        for fpathe, dirs, fs in os.walk(train):
            for f in fs:
                file = os.path.join(fpathe, f)
                img_list.append(file)

    for file in tqdm(img_list):
        x = image.load_img(file, target_size=(299, 299))
        x = image.img_to_array(x)
        x /= 255
        x = x.tolist()
        train_images.append(x)

        lch_in_file = file.split('/')[-1].split('-')[0][:-1]
        for each in samples:
            if lch_in_file == each['lch']:
                train_features.append(each['feature19'])
                train_labels.append(each['label_bl'])
                break

    img_list = []
    for fpathe, dirs, fs in os.walk(test_dir):
        for f in fs:
            file = os.path.join(fpathe, f)
            img_list.append(file)
    for file in tqdm(img_list):
        x = image.load_img(file, target_size=(299, 299))
        x = image.img_to_array(x)
        x /= 255
        x = x.tolist()
        test_images.append(x)

        lch_in_file = file.split('/')[-1].split('-')[0][:-1]
        for each in samples:
            if lch_in_file == each['lch']:
                test_features.append(each['feature19'])
                test_labels.append(each['label_bl'])
                break

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    train_features = np.array(train_features)

    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    test_features = np.array(test_features)

    y = []
    for yy in train_labels.tolist():
        if yy == 0:
            y.append([1, 0])
        elif yy == 1:
            y.append([0, 1])
    train_labels = np.array(y)

    y = []
    for yy in test_labels.tolist():
        if yy == 0:
            y.append([1, 0])
        elif yy == 1:
            y.append([0, 1])
    test_labels = np.array(y)

    print train_images.shape, test_images.shape

    model = multi_input_model()
    # 保存模型图
    plot_model(model, './backup/3/Multi_input_model.png', show_shapes=True,)
    # exit(0)
    training_checkpoint_path = "./backup/3/weights-improvement-{epoch:02d}-{val_acc:.4f}.hdf5"
    MyModelCheckpoint = ModelCheckpoint(training_checkpoint_path, monitor='val_acc', verbose=1,
                                        save_best_only=True, save_weights_only=False, mode='max', period=1)

    MyTensorBoard = TensorBoard(log_dir='./backup/3/', histogram_freq=0, write_graph=True,
                                write_grads=False, write_images=False)

    # learning_rate = 0.001
    # decay_rate = 0.000001
    # momentum = 0.9
    # sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
    # model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'],)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'],)

    model.fit([train_images, train_features], train_labels,
              validation_data=([test_images, test_features], test_labels),
              epochs=100, batch_size=30, callbacks=[MyModelCheckpoint, MyTensorBoard])
