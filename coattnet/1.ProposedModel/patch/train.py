#!/usr/bin/python
# coding:utf-8
from keras.layers import Input, Embedding, LSTM, Dense, add, Concatenate
from keras.models import Model
from keras.utils import plot_model
import os
import tensorflow as tf
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
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.layers import multiply, dot, Flatten, RepeatVector, Activation, Conv1D, Dense, MaxPool1D, concatenate, Permute, Dropout, Reshape, Layer
from keras import Input, Model
from keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
from keras.layers.wrappers import TimeDistributed
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.layers import Activation, Dense, MaxPool1D, concatenate, Flatten, AveragePooling2D, Dropout, Conv2D, MaxPooling2D
from keras import Input, Model
from keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
import warnings


def multi_input_model():
    """构建多输入模型"""
    #--------------------------------Model 5-------------------------------------------------
    baseModel = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    baseModelOutput = baseModel.output
    ClinicPicInput = baseModel.input

    x = GlobalAveragePooling2D(name='avg_pool')(baseModelOutput)
    x = Dense(19, activation='relu')(x)
    x = Dropout(0.5)(x)

    img_dense_w = Activation('sigmoid')(x)
    img_dense_f = Activation('relu')(x)

    metadataInput = Input(shape=(19,), name='metadataInput')
    metadata_dense_w = Activation('sigmoid')(metadataInput)
    metadata_dense_f = Activation('relu')(metadataInput)

    refined_Image_f = multiply([metadata_dense_w, img_dense_f])
    refined_Metadata_f = multiply([img_dense_w, metadata_dense_f])

    Merged_Feature = concatenate([refined_Image_f, refined_Metadata_f])
    Merged_Feature = Activation('relu')(Merged_Feature)

    # x = Dense(512, activation='relu')(Merged_Feature)
    # x = Dropout(0.5)(x)
    # x = Dense(512, activation='relu')(x)
    # x = Dropout(0.5)(x)

    predictions = Dense(2, activation='softmax', name='output')(Merged_Feature)
    model = Model(input=[ClinicPicInput, metadataInput], output=predictions)

    return model

    #--------------------------------Model 4-------------------------------------------------
    # baseModel = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    # baseModelOutput = baseModel.output
    # ClinicPicInput = baseModel.input
    #
    # x = GlobalAveragePooling2D(name='avg_pool')(baseModelOutput)
    #
    # img_dense_w = Dense(19, activation="sigmoid")(x)
    # img_dense_w = Dropout(0.5)(img_dense_w)
    #
    # img_dense_f = Dense(512, activation="relu")(x)
    # img_dense_f = Dropout(0.5)(img_dense_f)
    #
    # metadataInput = Input(shape=(19,), name='metadataInput')
    #
    # metadata_dense_w = Dense(512, activation="sigmoid")(metadataInput)
    # metadata_dense_w = Dropout(0.5)(metadata_dense_w)
    #
    # metadata_dense_f = Dense(19, activation="relu")(metadataInput)
    # metadata_dense_f = Dropout(0.5)(metadata_dense_f)
    #
    # Refined_Image_Feature = multiply([metadata_dense_w, img_dense_f])
    # Refined_Image_Feature = Dense(19, activation='relu')(Refined_Image_Feature)
    # Refined_Image_Feature = Dropout(0.5)(Refined_Image_Feature)
    #
    # Refined_Metadata_Feature = multiply([img_dense_w, metadata_dense_f])
    # Refined_Metadata_Feature = Dense(19, activation='relu')(Refined_Metadata_Feature)
    # Refined_Metadata_Feature = Dropout(0.5)(Refined_Metadata_Feature)
    #
    # Merged_Feature = concatenate([Refined_Image_Feature, Refined_Metadata_Feature])
    #
    # x = Dense(512, activation='relu')(Merged_Feature)
    # x = Dropout(0.5)(x)
    # x = Dense(512, activation='relu')(x)
    # x = Dropout(0.5)(x)
    #
    # predictions = Dense(2, activation='softmax', name='output')(x)
    # model = Model(input=[ClinicPicInput, metadataInput], output=predictions)
    #
    # return model


if __name__ == '__main__':
    model = multi_input_model()
    print 'load model successfully！'
    plot_model(model, './Multi_input_model.png', show_shapes=True, )
    exit(0)


    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    warnings.filterwarnings("ignore")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    with open('dict19.pkl', 'rb') as f:
        samples = cPickle.load(f)

    targetSize = (299, 299)
    optimizeFuncLst = ['./backup/IncepResV2-SGD/', './backup/IncepResV2-Adam/']
    maxEpochs = 100
    batchSize = 30
    ifTestNetwork = False
    subsetSuffix = '-less/' if ifTestNetwork == True else '/'
    subsetPrefix = '/media/wz209/a29353b7-1090-433f-b452-b4ce827adb17/sugurs/PythonProjects/pem_multi/dataset/patch/'
    AllImagePathLst = []
    AllImageDataLst = []
    AllFeature19Lst = []
    AllLabelLst = []

    for index in range(1, 5):
        subsetDir = subsetPrefix + str(index) + subsetSuffix
        subsetImagePathLst = []
        for fpathe, dirs, fs in os.walk(subsetDir):
            for f in fs:
                file = os.path.join(fpathe, f)
                subsetImagePathLst.append(file)
        AllImagePathLst.append(subsetImagePathLst)
        print 'subset_dir %d' % index, subsetDir, 'num of pictures = ', len(subsetImagePathLst)

    for subsetImageLst in AllImagePathLst:
        subsetImageDataLst = []
        subsetFeature19Lst = []
        subsetLabelLst = []
        for file in tqdm(subsetImageLst):
            x = image.load_img(file, target_size=targetSize)
            x = image.img_to_array(x)
            x /= 255
            x = x.tolist()
            subsetImageDataLst.append(x)
            lch_in_file = file.split('/')[-1].split('-')[0][:-1]
            for each in samples:
                if lch_in_file == each['lch']:
                    subsetFeature19Lst.append(each['feature19'])
                    subsetLabelLst.append(each['label_bl'])
                    break
        AllImageDataLst.append(subsetImageDataLst)
        AllFeature19Lst.append(subsetFeature19Lst)
        AllLabelLst.append(subsetLabelLst)


    for optimizeFunc in optimizeFuncLst:
        print '-------------------------------------------------------------optimizeFunc', optimizeFunc
        for testIndex in range(1, 5):
            # if optimizeFunc != './backup/IncepResV2-Adam/' or testIndex != 4:
            #     continue
            test_dir = subsetPrefix + str(testIndex) + subsetSuffix
            savePath = optimizeFunc + str(testIndex) + '/'
            print '-------------------------------------------------------------test_dir', test_dir
            train_images = []
            train_labels = []
            train_features = []
            test_images = []
            test_labels = []
            test_features = []
            for i in range(1, 5):
                if i == testIndex:
                    test_images = AllImageDataLst[i-1]
                    test_features = AllFeature19Lst[i-1]
                    test_labels = AllLabelLst[i-1]
                    continue
                train_images += AllImageDataLst[i-1]
                train_features += AllFeature19Lst[i-1]
                train_labels += AllLabelLst[i-1]

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

            # --------------------------- Load the model -----------------------------------------------------
            model = multi_input_model()
            print 'load model successfully！'
            plot_model(model, savePath + 'Multi_input_model.png', show_shapes=True, )
            # exit(0)

            training_checkpoint_path = savePath + "weights-improvement-{epoch:02d}-{val_acc:.4f}.hdf5"
            MyModelCheckpoint = ModelCheckpoint(training_checkpoint_path, monitor='val_acc', verbose=1,
                                                save_best_only=True, save_weights_only=False, mode='max', period=1)

            MyTensorBoard = TensorBoard(log_dir=savePath, histogram_freq=0, write_graph=True,
                                        write_grads=False, write_images=False)
            if optimizeFunc == './backup/IncepResV2-SGD/':
                learning_rate = 0.001
                decay_rate = 0.000001
                momentum = 0.9
                sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
                model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'],)
            elif optimizeFunc == './backup/IncepResV2-Adam/':
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], )

            model.fit([train_images, train_features], train_labels,
                      validation_data=([test_images, test_features], test_labels), verbose=2,
                      epochs=maxEpochs, batch_size=batchSize, callbacks=[MyModelCheckpoint, MyTensorBoard])
            K.clear_session()
            tf.reset_default_graph()
            del model
