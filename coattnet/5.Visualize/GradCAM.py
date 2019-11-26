# -*- coding: utf-8 -*-
from keras.preprocessing.image import ImageDataGenerator
import os.path
from keras.models import load_model
from keras.optimizers import SGD
import glob
from keras.models import Model
import scipy.io as scio
import numpy as np
import cPickle
from keras.preprocessing import image
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import auc as aucCalc
import warnings
from keras.preprocessing.image import ImageDataGenerator
import os.path
from keras.models import load_model
from keras.optimizers import SGD
import glob
from keras.models import Model
import scipy.io as scio
import numpy as np
import cPickle
from keras.preprocessing import image
from tqdm import tqdm
from keras.applications.vgg16 import (
    VGG16, preprocess_input, decode_predictions)
from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Sequential, load_model
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
import sys
import cv2
import os
import shutil
from keras.models import Model


def get_nb_files(directory):
    nb = 0
    if not os.path.exists(directory):
        return 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            nb += len(glob.glob(os.path.join(r, dr+'/*')))
    return nb


def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))


def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                   tf.cast(op.inputs[0] > 0., dtype)


def deprocess_image(x):
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)
    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def compile_saliency_function(model, activation_layer):
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_output = layer_dict[activation_layer].output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])


def modify_backprop(model, name):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):
        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]
        # replace relu activation
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu
        new_model = model
    return new_model


def target_category_loss_output_shape(input_shape):
    return input_shape


def _compute_gradients(tensor, var_list):
    grads = tf.gradients(tensor, var_list)
    return [grad if grad is not None else tf.zeros_like(var)
            for var, grad in zip(var_list, grads)]

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def grad_cam(input_model, image, category_index, layer_name):
    nb_classes = 2
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    x = Lambda(target_layer, output_shape=target_category_loss_output_shape)(input_model.output)
    model = Model(inputs=input_model.input, outputs=x)
    # model.summary()
    loss = K.sum(model.output)
    conv_output = [l for l in model.layers if l.name == layer_name][0].output
    grads = normalize(_compute_gradients(loss, [conv_output])[0])
    gradient_function = K.function([model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.ones(output.shape[0: 2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]
    print('cam shape = {0}'.format(cam.shape))
    cam = cv2.resize(cam, (img_width, img_height))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    # Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    # 增强对比度，更好的可视化效果
    # newHeatmap = np.uint8(255*heatmap)
    # print(newHeatmap.shape)
    # for i in range(len(newHeatmap)):
    #     for j in range(len(newHeatmap[i])):
    #         if newHeatmap[i][j]>120:
    #             newHeatmap[i][j] = np.minimum(newHeatmap[i][j]+50, 255)
    #         elif j<120:
    #             newHeatmap[i][j] = np.maximum(newHeatmap[i][j]-50, 0)
    # cam = cv2.applyColorMap(newHeatmap, cv2.COLORMAP_JET)

    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    cam = 1.6 * np.float32(cam) + 0.4 * np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap


def plotROCCurve(fpr, tpr, auc):
    plt.plot(fpr, tpr, color='r', lw=2, label=' (AUC = %0.4f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig("ROC.png")
    # plt.show()


def testModelOnOneDataset(loaded_weights, test_data_dir, modelDirIndex):
    full_model = load_model(loaded_weights)
    # 读取TPC和LTPC病人的序列号
    with open('dict19.pkl', 'rb') as f:
        samples = cPickle.load(f)

    """
    对图片的评估
    """
    print '开始计算整体的以图片为单位的准确率'
    img_list = []
    test_images = []
    test_labels = []
    test_features = []
    for fpathe, dirs, fs in os.walk(test_data_dir):
        for f in fs:
            file = os.path.join(fpathe, f)
            img_list.append(file)
    for file in tqdm(img_list):
        x = image.load_img(file, target_size=(img_width, img_height))
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

    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    test_features = np.array(test_features)
    y = []
    for yy in test_labels.tolist():
        if yy == 0:
            y.append([1, 0])
        elif yy == 1:
            y.append([0, 1])
    test_labels = np.array(y)


    print '开始计算整体的以病人为单位的准确率'
    LTPC_lch = []
    TPC_lch = []
    all_patient_pred_list = []
    all_patient_truelabel_list = []
    all_patient_in_one_pred_list = []
    all_patient_in_one_truelabel_list = []

    for each in samples:
        if each['label_bl'] == 0:
            LTPC_lch.append(each['lch'])
        elif each['label_bl'] == 1:
            TPC_lch.append(each['lch'])

    # 先检查LTPC的临床编号
    correct_cnt_vote_ltpc = 0
    correct_cnt_prob_ltpc = 0
    all_cnt_ltpc = 0
    all_pics_ltpc = 0
    correct_pics_ltpc = 0

    for each in LTPC_lch:
        for person in samples:
            if person['lch'] == each:
                features19 = person['feature19']

        one_patient_pic_list = []
        one_patient_test_res_list = []
        for fpathe, dirs, fs in os.walk(test_data_dir):
            for f in fs:
                file = os.path.join(fpathe, f)
                lch_in_file = file.split('/')[-1].split('-')[0][:-1]
                if each == lch_in_file:
                    one_patient_pic_list.append(file)

        if len(one_patient_pic_list) == 0:
            continue

        all_cnt_ltpc += 1

        for i in one_patient_pic_list:
            raw_img = image.load_img(i, target_size=(img_width, img_height))
            x = image.img_to_array(raw_img)
            x /= 255
            x = x.tolist()
            image_input = np.array([x])
            all_pics_ltpc += 1
            result = full_model.predict(image_input).tolist()[0]
            one_patient_test_res_list.append(result)

            # ------------------------ Grad CAM ------------------------------
            predicted_class = np.argmax(result)
            lastFeatureMapName = 'conv_7b_ac'
            cam, heatmap = grad_cam(full_model, image_input, predicted_class, lastFeatureMapName)
            name = i.split('/')[-1].split('.')[0]
            name_former = './CAM_Results/%d' % modelDirIndex + '/' + name + ".jpg"
            name0 = './CAM_Results/%d' % modelDirIndex + '/' + name + "_gradcam.jpg"
            name1 = './CAM_Results/%d' % modelDirIndex + '/' + name + "_combine_heatmap.jpg"
            name2 = './CAM_Results/%d' % modelDirIndex + '/' + name + "_guided.jpg"
            if not os.path.exists(name_former):
                raw_img.save(name_former)
            if not os.path.exists(name1):
                cv2.imwrite(name1, cam)
            if not os.path.exists(name0):
                cv2.imwrite(name0, 255 * heatmap)
            register_gradient()
            guided_model = modify_backprop(full_model, 'GuidedBackProp')
            saliency_fn = compile_saliency_function(guided_model, lastFeatureMapName)
            saliency = saliency_fn([image_input, 0])
            gradcam = saliency[0] * heatmap[..., np.newaxis]
            if not os.path.exists(name2):
                cv2.imwrite(name2, deprocess_image(gradcam))

    # 先检查TPC的临床编号
    correct_cnt_vote_tpc = 0
    correct_cnt_prob_tpc = 0
    all_cnt_tpc = 0
    all_pics_tpc = 0
    correct_pics_tpc = 0

    for each in TPC_lch:
        one_patient_pic_list = []
        one_patient_test_res_list = []
        for fpathe, dirs, fs in os.walk(test_data_dir):
            for f in fs:
                file = os.path.join(fpathe, f)
                lch_in_file = file.split('/')[-1].split('-')[0][:-1]
                if each == lch_in_file:
                    one_patient_pic_list.append(file)

        if len(one_patient_pic_list) == 0:
            continue

        all_cnt_tpc += 1

        for i in one_patient_pic_list:
            raw_img = image.load_img(i, target_size=(img_width, img_height))
            x = image.img_to_array(raw_img)
            x /= 255
            x = x.tolist()
            image_input = np.array([x])
            all_pics_tpc += 1

            result = full_model.predict(image_input).tolist()[0]
            # print result
            one_patient_test_res_list.append(result)

            # ------------------------ Grad CAM ------------------------------
            predicted_class = np.argmax(result)
            lastFeatureMapName = 'conv_7b_ac'
            cam, heatmap = grad_cam(full_model, image_input, predicted_class, lastFeatureMapName)
            name = i.split('/')[-1].split('.')[0]
            name_former = './CAM_Results/%d' % modelDirIndex + '/' + name + ".jpg"
            name0 = './CAM_Results/%d' % modelDirIndex + '/' + name + "_gradcam.jpg"
            name1 = './CAM_Results/%d' % modelDirIndex + '/' + name + "_combine_heatmap.jpg"
            name2 = './CAM_Results/%d' % modelDirIndex + '/' + name + "_guided.jpg"
            if not os.path.exists(name_former):
                raw_img.save(name_former)
            if not os.path.exists(name1):
                cv2.imwrite(name1, cam)
            if not os.path.exists(name0):
                cv2.imwrite(name0, 255 * heatmap)
            register_gradient()
            guided_model = modify_backprop(full_model, 'GuidedBackProp')
            saliency_fn = compile_saliency_function(guided_model, lastFeatureMapName)
            saliency = saliency_fn([image_input, 0])
            gradcam = saliency[0] * heatmap[..., np.newaxis]
            if not os.path.exists(name2):
                cv2.imwrite(name2, deprocess_image(gradcam))




if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    warnings.filterwarnings("ignore")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # 找所有的hdf5
    hdf5_dir = ['/media/wz209/a29353b7-1090-433f-b452-b4ce827adb17/sugurs/PythonProjects/'
                'pem_multi/Expirements/5.Visualize/1',
                '/media/wz209/a29353b7-1090-433f-b452-b4ce827adb17/sugurs/PythonProjects/'
                'pem_multi/Expirements/5.Visualize/2',
                ]

    testDir = '/media/wz209/a29353b7-1090-433f-b452-b4ce827adb17/' \
              'sugurs/PythonProjects/pem_multi/dataset/GradCAM'

    # testDir = '/media/wz209/a29353b7-1090-433f-b452-b4ce827adb17/' \
    #           'sugurs/PythonProjects/pem_multi/dataset/patch/vis-less'

    modelDirIndex = [1, 2]
    findH5Count = 0

    for index in range(2):
        for fpathe, dirs, fs in os.walk(hdf5_dir[index]):
            for f in fs:
                file = os.path.join(fpathe, f)
                if file.endswith('.hdf5'):
                    findH5Count += 1
                    modelPath = file

                    if 'VGG' in modelPath:
                        img_width, img_height = 224, 224
                    else:
                        img_width, img_height = 299, 299

                    print 'Now is testing model #%d' % findH5Count, modelPath.replace(
                        '/media/wz209/a29353b7-1090-433f-b452-b4ce827adb17/sugurs/PythonProjects/pem_multi/Expirements/3.OnlyClinicPic/patch/backup/',
                        ''
                    )
                    testModelOnOneDataset(modelPath, testDir, modelDirIndex[index])