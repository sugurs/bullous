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
import numpy as np
from sklearn import datasets, manifold
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_nb_files(directory):
    nb = 0
    if not os.path.exists(directory):
        return 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            nb += len(glob.glob(os.path.join(r, dr+'/*')))
    return nb


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


def testModelOnOneDataset(loaded_weights, test_data_dir):
    # loaded_weights = '/media/wz209/a29353b7-1090-433f-b452-b4ce827adb17/sugurs/PythonProjects/pem_multi/Expirements/1.ProposedModel/patch/backup/IncepResV2-Adam/1/weights-improvement-15-0.8322.hdf5'
    # test_data_dir = '/media/wz209/a29353b7-1090-433f-b452-b4ce827adb17/sugurs/PythonProjects/pem_multi/dataset/patch/1'

    base_model = load_model(loaded_weights)
    partial_model = Model(input=base_model.input, output=base_model.get_layer('activation_208').output)
    # partial_model = Model(input=base_model.input, output=base_model.get_layer('activation_208').output)

    # print(partial_model.summary())

    # 读取TPC和LTPC病人的序列号
    with open('dict19.pkl', 'rb') as f:
        samples = cPickle.load(f)

    """
    对图片的评估
    """
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

    test_labels = np.array(test_labels)
    y_tsne = []
    for yy in test_labels.tolist():
        if yy == 0:
            y_tsne.append([1, 0])
        elif yy == 1:
            y_tsne.append([0, 1])

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
    all_cnt_ltpc = 0
    all_pics_ltpc = 0

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
            x = image.load_img(i, target_size=(img_width, img_height))
            x = image.img_to_array(x)
            x /= 255
            x = x.tolist()
            image_input = np.array([x])
            all_pics_ltpc += 1

            bingshi_input = np.array([features19])

            result = partial_model.predict([image_input, bingshi_input]).tolist()[0]
            one_patient_test_res_list.append(result)
        # 以图片为单位
        all_patient_pred_list += one_patient_test_res_list
        all_patient_truelabel_list += np.zeros(len(one_patient_test_res_list)).tolist()
        # 以病人为单位
        # print one_patient_test_res_list
        myArray = np.array(one_patient_test_res_list)
        # print myArray
        # print np.sum(myArray, axis=0)
        ret = (np.sum(myArray, axis=0)/len(one_patient_test_res_list)).tolist()
        all_patient_in_one_pred_list += [ret]
        all_patient_in_one_truelabel_list += [0]

    # 先检查TPC的临床编号
    all_cnt_tpc = 0
    all_pics_tpc = 0

    for each in TPC_lch:
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

        all_cnt_tpc += 1

        for i in one_patient_pic_list:
            x = image.load_img(i, target_size=(img_width, img_height))
            x = image.img_to_array(x)
            x /= 255
            x = x.tolist()
            image_input = np.array([x])
            all_pics_tpc += 1

            bingshi_input = np.array([features19])

            result = partial_model.predict([image_input, bingshi_input]).tolist()[0]
            one_patient_test_res_list.append(result)

        # 以图片为单位
        all_patient_pred_list += one_patient_test_res_list
        all_patient_truelabel_list += np.ones(len(one_patient_test_res_list)).tolist()

        # 以病人为单位
        myArray = np.array(one_patient_test_res_list)
        ret = (np.sum(myArray, axis=0)/len(one_patient_test_res_list)).tolist()
        all_patient_in_one_pred_list += [ret]
        all_patient_in_one_truelabel_list += [1]

    x_tsne = np.array(all_patient_pred_list)
    y_tsne = np.array(all_patient_truelabel_list)
    print x_tsne.shape
    print len(y_tsne)
    print('num of class is %d' % len(set(y_tsne)))
    # tsne = manifold.TSNE(n_components=2, init='random', random_state=0, perplexity=100)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    x_tsne = tsne.fit_transform(x_tsne)
    colors = ['r', 'b']
    target_names = range(2)

    x_min, x_max = np.min(x_tsne, 0), np.max(x_tsne, 0)
    x_tsne = (x_tsne - x_min) / (x_max - x_min)
    plt.figure()
    for (i, color, target) in zip(target_names, colors, target_names):
        plt.scatter(x_tsne[y_tsne == i, 0], x_tsne[y_tsne == i, 1], c=color, label=target, s=2, lw=1)
    plt.show()
    # plt.savefig('./2.png', dpi=330)

    # # **×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×

    x_tsne = np.array(all_patient_in_one_pred_list)
    y_tsne = np.array(all_patient_in_one_truelabel_list)
    print x_tsne.shape
    print len(y_tsne)
    print('num of class is %d' % len(set(y_tsne)))
    # tsne = manifold.TSNE(n_components=2, init='random', random_state=0, perplexity=100)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    x_tsne = tsne.fit_transform(x_tsne)
    colors = ['r', 'b']
    target_names = range(2)

    x_min, x_max = np.min(x_tsne, 0), np.max(x_tsne, 0)
    x_tsne = (x_tsne - x_min) / (x_max - x_min)
    plt.figure()
    for (i, color, target) in zip(target_names, colors, target_names):
        plt.scatter(x_tsne[y_tsne == i, 0], x_tsne[y_tsne == i, 1], c=color, label=target, s=2, lw=1)
    plt.show()
    # plt.savefig('./2.png', dpi=330)

    # # **×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×
    # x_tsne = np.array(all_patient_pred_list)
    # y_tsne = np.array(all_patient_truelabel_list)
    # print x_tsne.shape
    # print len(y_tsne)
    # print('num of class is %d' % len(set(y_tsne)))
    # tsne = manifold.TSNE(n_components=3, init='pca', random_state=0)
    # x_tsne = tsne.fit_transform(x_tsne)
    # colors = ['r', 'b']
    # target_names = range(2)
    #
    # x_min, x_max = np.min(x_tsne, 0), np.max(x_tsne, 0)
    # x_tsne = (x_tsne - x_min) / (x_max - x_min)
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # for (i, color, target) in zip(target_names, colors, target_names):
    #     ax.scatter(x_tsne[y_tsne == i, 0], x_tsne[y_tsne == i, 1], x_tsne[y_tsne == i, 2],
    #                 c=color, label=target, s=2, lw=1)
    # plt.show()
    # # plt.savefig('./4.png', dpi=330)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    warnings.filterwarnings("ignore")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # 找所有的hdf5
    # hdf5_dir = '/media/wz209/a29353b7-1090-433f-b452-b4ce827adb17/' \
    #            'sugurs/PythonProjects/pem_multi/Expirements/1.ProposedModel/patch/backup/IncepResV2-Adam/3'

    hdf5_dir = '/media/wz209/a29353b7-1090-433f-b452-b4ce827adb17/' \
               'sugurs/PythonProjects/pem_multi/Expirements/1.ProposedModel/patch/backup/IncepResV2-Adam/3'

    findH5Count = 0
    for fpathe, dirs, fs in os.walk(hdf5_dir):
        for f in fs:
            file = os.path.join(fpathe, f)
            if file.endswith('.hdf5'):
                findH5Count+=1
                modelPath = file

                if 'VGG' in modelPath:
                    img_width, img_height = 224, 224
                else:
                    img_width, img_height = 299, 299

                # testDir = '/media/wz209/a29353b7-1090-433f-b452-b4ce827adb17/sugurs/PythonProjects/pem_multi/dataset/patch/3'
                testDir = '/media/wz209/a29353b7-1090-433f-b452-b4ce827adb17/sugurs/PythonProjects/pem_multi/dataset/crop346'


                print 'Now is testing model #%d' % findH5Count, modelPath.replace(
                    '/media/wz209/a29353b7-1090-433f-b452-b4ce827adb17/sugurs/PythonProjects/pem_multi/Expirements/1.ProposedModel/patch/backup/',
                    ''
                )
                testModelOnOneDataset(modelPath, testDir)



