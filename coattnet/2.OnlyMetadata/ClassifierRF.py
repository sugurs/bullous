#!/usr/bin/python
# coding:utf-8
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
from sklearn.externals import joblib
import cPickle
import os
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import auc
import warnings


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


warnings.filterwarnings("ignore")
acc_lst = []
class2_acc_lst = []
auc_lst = []
trueAllLst = []
scoreAllLst = []
feature_importances_ = np.zeros(19)

# 读取TPC和LTPC病人的序列号
with open('dict19.pkl', 'rb') as f:
    samples = cPickle.load(f)

for cnt in range(1, 5):
    prefix = '/media/wz209/a29353b7-1090-433f-b452-b4ce827adb17/sugurs/PythonProjects/pem_multi/dataset/patch/'
    testIndex = cnt

    #-------------------------generate train and test data-------------------------------------------------
    # sum = 0
    for i in range(1, 5):
        if i == testIndex:
            test_dir = prefix + str(i)
            img_list = []
            for root, dirs, files in os.walk(test_dir):
                for f in files:
                    file = os.path.join(root, f)
                    img_list.append(file.split('/')[-1].split('-')[0][:-1])
            test_lch_img_set = set(img_list)
            test_features_set = []
            test_label_set = []
            for lch in test_lch_img_set:
                for each in samples:
                    if each['lch'] == lch:
                        test_features_set.append(each['feature19'])
                        test_label_set.append(each['label_bl'])
            continue

        train_dir = prefix + str(i)
        img_list = []
        for root, dirs, files in os.walk(train_dir):
            for f in files:
                file = os.path.join(root, f)
                img_list.append(file.split('/')[-1].split('-')[0][:-1])
        train_lch_img_set = set(img_list)
        train_features_set = []
        train_label_set = []
        for lch in train_lch_img_set:
            for each in samples:
                if each['lch'] == lch:
                    train_features_set.append(each['feature19'])
                    train_label_set.append(each['label_bl'])

    # -------------------------start train and test-------------------------------------------------
    clf = RandomForestClassifier()
    clf.fit(X=np.array(train_features_set), y=np.array(train_label_set), sample_weight=None)
    name_list = [str(a) for a in range(0, 19)]
    index = [a for a in range(0, 19)]

    if testIndex == 3:
        for aaa in clf.feature_importances_:
            print aaa
    plt.bar(range(len(clf.feature_importances_)), clf.feature_importances_, color='rgby')
    feature_importances_+=np.array(clf.feature_importances_)
    # plt.ylim(ymax=80, ymin=0)
    plt.xticks(index, name_list)
    plt.ylabel("importance")  # X轴标签
    plt.ylabel("number")  # X轴标签
    # for rect in rects:
    #     height = rect.get_height()
    #     plt.text(rect.get_x() + rect.get_width() / 2, height, str(height), ha='center', va='bottom')
    plt.savefig('importance-%d.jpg' % testIndex)

    joblib.dump(clf, 'RF-%d.m' % testIndex)
    acc = clf.score(X=test_features_set, y=test_label_set)
    result1 = clf.predict_proba(test_features_set).tolist()
    # print result1
    result2 = clf.predict(test_features_set).tolist()
    # print result1
    allLTPC = 0
    allTPC = 0
    corrctLTPC = 0
    corrctTPC = 0
    for i in range(len(test_label_set)):
        true_label = test_label_set[i]
        pred_label = result2[i]
        if true_label == 0:
            allLTPC+=1
            if true_label == pred_label:
                corrctLTPC+=1
        elif true_label == 1:
            allTPC+=1
            if true_label == pred_label:
                corrctTPC+=1
    acc_lst.append(acc)
    class2_acc_lst.append([corrctLTPC/float(allLTPC), corrctTPC/float(allTPC)])

    # auc roc
    true_class = np.array(test_label_set)  # true_class为数据的真实标签
    pred_scores = np.array([a[0] for a in result1])  # scores为分类其预测的得分
    fpr, tpr, thresholds = metrics.roc_curve(true_class, pred_scores, pos_label=0)  # bcc
    AUC = auc(fpr, tpr)
    # tpr fpr
    yuedeng = []
    for i in range(len(fpr)):
        yuedeng.append(tpr[i] - fpr[i])
    yuedeng_index = yuedeng.index(max(yuedeng))
    # print 'the best TPR FPR in subset-%d'%testIndex, tpr[yuedeng_index], fpr[yuedeng_index]

    auc_lst.append(AUC)
    trueAllLst+=test_label_set
    scoreAllLst+=[a[0] for a in result1]

    true_class = np.array(test_label_set)  # true_class为数据的真实标签
    pred_scores = np.array([a[1] for a in result1])  # scores为分类其预测的得分
    fpr0, tpr0, thresholds0 = metrics.roc_curve(true_class, pred_scores, pos_label=1)  # bcc
    AUC0 = auc(fpr0, tpr0)
    # print AUC, AUC0

# print '-----------------------------Test test TestSet-------------------'
# print 'acc_lst', acc_lst
# print 'acc_lst average', sum(acc_lst) / len(acc_lst)
#
# a = []
# b = []
# for i in class2_acc_lst:
#     a.append(i[0])
#     b.append(i[1])
#
# print 'class2_acc_lst', class2_acc_lst
# print 'class2_acc_lst average', sum(a) / len(a), sum(b) / len(b)
#
# print 'auc_lst', auc_lst
# print 'auc_lst average', sum(auc_lst) / len(auc_lst)
#
# trueAllLst = np.array(trueAllLst)
# scoreAllLst = np.array(scoreAllLst)
# fpr, tpr, thresholds = metrics.roc_curve(trueAllLst, scoreAllLst, pos_label=0)  # bcc
# auc = auc(fpr, tpr)
# print 'total dataset auc', auc
# fig = plt.figure()
# plotROCCurve(fpr, tpr, auc)
#
# # calc the best chosen TPR FPR
# yuedeng = []
# for i in range(len(fpr)):
#     yuedeng.append(tpr[i] - fpr[i])
# yuedeng_index = yuedeng.index(max(yuedeng))
# print 'total dataset best TPR FPR', tpr[yuedeng_index], fpr[yuedeng_index]
#
# fig = plt.figure()
# kk = [a / 4.0 for a in feature_importances_]
# for aaa in kk:
#     print aaa
# name_list = [str(a) for a in range(0, 19)]
# index = [a for a in range(0, 19)]
# plt.bar(range(len(kk)), kk, color='rgby')
# plt.xticks(index, name_list)
# plt.ylabel("importance")  # X轴标签
# plt.ylabel("number")  # X轴标签
# plt.savefig('importance-all.jpg')
#
# #-----------------------------Test other 40 patients data-------------------
# print '-----------------------------Test other 40 patients data-------------------'
# lch40 = np.load('lch40.npy')
# lch40Feature = []
# lch40Lable = []
# for lch in lch40:
#     for each in samples:
#         if each['lch'] == lch:
#             lch40Feature.append(each['feature19'])
#             lch40Lable.append(each['label_bl'])
# for i in range(1, 5):
#     clf = joblib.load("./RF-%d.m" % i)
#     acc = clf.score(X=lch40Feature, y=lch40Lable)
#     print acc
# result1 = clf.predict_proba(test_features_set).tolist()
# # print result1
# result2 = clf.predict(test_features_set).tolist()
# print result1

