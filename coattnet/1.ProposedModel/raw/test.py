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
    full_model = load_model(loaded_weights)
    # print '#############################################loading modal：' + loaded_weights, '#############################################'
    # print(full_model.summary())

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

        lch_in_file = file.split('/')[-1][:-5]
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

    evaluate = full_model.evaluate([test_images, test_features], test_labels, batch_size=30, verbose=0)

    print '测试集整体LOSS:', ("%.4f" % evaluate[0]), '测试集平均ACC:', ("%.4f" % evaluate[1])

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
                lch_in_file = file.split('/')[-1][:-5]
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

            result = full_model.predict([image_input, bingshi_input]).tolist()[0]
            # print result
            one_patient_test_res_list.append(result)

        # 以图片为单位
        all_patient_pred_list += [a[0] for a in one_patient_test_res_list]
        all_patient_truelabel_list += np.zeros(len(one_patient_test_res_list)).tolist()
        # 以病人为单位
        all_patient_in_one_pred_list += [sum([a[0] for a in one_patient_test_res_list])/len(one_patient_test_res_list)]
        all_patient_in_one_truelabel_list += [0]


        # 按照投票策略
        correct_cnt = 0
        for res in one_patient_test_res_list:
            if res.index(max(res)) == 0:
                correct_cnt += 1
                correct_pics_ltpc += 1
            else:
                correct_cnt -= 1
        if correct_cnt > 0:
            correct_cnt_vote_ltpc += 1

        # 按照置信度策略
        res_all = [0, 0]
        for res in one_patient_test_res_list:
            res_all[0] += res[0]
            res_all[1] += res[1]
        # print res_all
        if res_all.index(max(res_all)) == 0:
            correct_cnt_prob_ltpc += 1

    print 'LTPC-patientsNum', all_cnt_ltpc, 'LTPC-picsNum', all_pics_ltpc

    if all_pics_ltpc > 0:
        print 'LTPC-picsAcc', correct_pics_ltpc / float(all_pics_ltpc)
    else:
        print 'LTPC-picsAcc', 'Null'

    if all_cnt_ltpc > 0:
        print 'LTPC-voteAcc', correct_cnt_vote_ltpc / float(all_cnt_ltpc)
    else:
        print 'LTPC-voteAcc', 'Null'

    if all_cnt_ltpc > 0:
        print 'LTPC-probAcc', correct_cnt_prob_ltpc / float(all_cnt_ltpc)
    else:
        print 'LTPC-probAcc', 'Null'

    # 先检查TPC的临床编号
    correct_cnt_vote_tpc = 0
    correct_cnt_prob_tpc = 0
    all_cnt_tpc = 0
    all_pics_tpc = 0
    correct_pics_tpc = 0

    for each in TPC_lch:
        for person in samples:
            if person['lch'] == each:
                features19 = person['feature19']

        one_patient_pic_list = []
        one_patient_test_res_list = []
        for fpathe, dirs, fs in os.walk(test_data_dir):
            for f in fs:
                file = os.path.join(fpathe, f)
                lch_in_file = file.split('/')[-1][:-5]
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

            result = full_model.predict([image_input, bingshi_input]).tolist()[0]
            # print result
            one_patient_test_res_list.append(result)

        # 以图片为单位
        all_patient_pred_list += [a[0] for a in one_patient_test_res_list]
        all_patient_truelabel_list += np.ones(len(one_patient_test_res_list)).tolist()
        # 以病人为单位
        all_patient_in_one_pred_list += [sum([a[0] for a in one_patient_test_res_list])/len(one_patient_test_res_list)]
        all_patient_in_one_truelabel_list += [1]

        # 按照投票策略
        correct_cnt = 0
        for res in one_patient_test_res_list:
            if res.index(max(res)) == 1:
                correct_cnt += 1
                correct_pics_tpc += 1
            else:
                correct_cnt -= 1
        if correct_cnt > 0:
            correct_cnt_vote_tpc += 1

        # 按照置信度策略
        res_all = [0, 0]
        for res in one_patient_test_res_list:
            res_all[0] += res[0]
            res_all[1] += res[1]
        if res_all.index(max(res_all)) == 1:
            correct_cnt_prob_tpc += 1

    print 'TPC-patientsNum', all_cnt_tpc, 'TPC-picsNum', all_pics_tpc
    print 'TPC-picsAcc', correct_pics_tpc / float(all_pics_tpc)
    print 'TPC-voteAcc', correct_cnt_vote_tpc / float(all_cnt_tpc)
    print 'TPC-probAcc', correct_cnt_prob_tpc / float(all_cnt_tpc)

    print 'ALL-patientsNum', all_cnt_tpc + all_cnt_ltpc, 'ALL-picsNum', all_pics_tpc + all_pics_ltpc
    print 'ALL-picsAcc', (correct_pics_tpc + correct_pics_ltpc) / float(all_pics_tpc + all_pics_ltpc)
    print 'ALL-voteAcc', (correct_cnt_vote_ltpc + correct_cnt_vote_tpc) / float(all_cnt_ltpc + all_cnt_tpc)
    print 'ALL-probAcc', (correct_cnt_prob_ltpc + correct_cnt_prob_tpc) / float(all_cnt_ltpc + all_cnt_tpc)

    # —————————————————————— auc for each picture ——————————————————————————————————
    fpr, tpr, thresholds = metrics.roc_curve(all_patient_truelabel_list, all_patient_pred_list, pos_label=0)  # bcc
    auc = aucCalc(fpr, tpr)
    print 'the total auc for image', auc
    # plotROCCurve(fpr, tpr, auc)
    # calc the best chosen TPR FPR
    yuedeng = []
    for i in range(len(fpr)):
        yuedeng.append(tpr[i] - fpr[i])
    yuedeng_index = yuedeng.index(max(yuedeng))
    print 'the best TPR FPR for image', tpr[yuedeng_index], fpr[yuedeng_index]

    # —————————————————————— auc for each person ——————————————————————————————————
    fpr, tpr, thresholds = metrics.roc_curve(all_patient_in_one_truelabel_list, all_patient_in_one_pred_list, pos_label=0)  # bcc
    auc = aucCalc(fpr, tpr)
    print 'the total auc for person', auc
    # plotROCCurve(fpr, tpr, auc)
    # calc the best chosen TPR FPR
    yuedeng = []
    for i in range(len(fpr)):
        yuedeng.append(tpr[i] - fpr[i])
    yuedeng_index = yuedeng.index(max(yuedeng))
    print 'the best TPR FPR for person', tpr[yuedeng_index], fpr[yuedeng_index]


    # #-----------------------------Test other 40 patients data-------------------
    # lch40 = np.load('lch40.npy')
    # lch40Feature = []
    # lch40Lable = []
    # for lch in lch40:
    #     for each in samples:
    #         if each['lch'] == lch:
    #             lch40Feature.append(each['feature19'])
    #             lch40Lable.append(each['label_bl'])
    # for i in range(1, 5):
    #     clf = joblib.load("./OnlyMetadataModel-%d.m" % i)
    #     acc = clf.score(X=lch40Feature, y=lch40Lable)
    #     print acc
    # # result1 = clf.predict_proba(test_features_set).tolist()
    # # # print result1
    # # result2 = clf.predict(test_features_set).tolist()
    # # print result1


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    warnings.filterwarnings("ignore")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # 找所有的hdf5
    hdf5_dir = '/media/wz209/a29353b7-1090-433f-b452-b4ce827adb17/sugurs/PythonProjects/pem_multi/Expirements/1.ProposedModel/raw/backup'
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

                testDir = '/media/wz209/a29353b7-1090-433f-b452-b4ce827adb17/sugurs/PythonProjects/pem_multi/dataset/raw/%s' % (file.split('/')[-2])
                print 'Now is testing model #%d' % findH5Count, modelPath.replace(
                    '/media/wz209/a29353b7-1090-433f-b452-b4ce827adb17/sugurs/PythonProjects/pem_multi/Expirements/1.ProposedModel/raw/backup/',
                    ''
                )
                testModelOnOneDataset(modelPath, testDir)



