# encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import xlrd
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn import svm
import pickle
import os
import shutil
import cPickle
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import auc as aucCalc
import warnings
import pickle
import cPickle


info1 = ["乡镇医院或社区医院",
"县级医院",
"市医院",
"省部级医院",]

info2 = ["内科",
"外科",
"皮肤科",
"其他科室",]

info3 = ["经治医生等初级",
"主治等中级",
"副教授、教授等高级职称",
"其他",]

# 读取TPC和LTPC病人的序列号
with open('dict19.pkl', 'rb') as f:
    samples = cPickle.load(f)
LTPC_lch = []
TPC_lch = []
for each in samples:
    if each['label_bl'] == 0:
        LTPC_lch.append(each['lch'])
    elif each['label_bl'] == 1:
        TPC_lch.append(each['lch'])

lch40 = np.load('lch40.npy')
trueLabelLst = []
for lch in lch40:
    if lch in LTPC_lch:
        trueLabelLst.append(0)
    elif lch in TPC_lch:
        trueLabelLst.append(1)
    else:
        print 'xxxxxxxxxxxxxxxxxxxxxx'

workbook1 = xlrd.open_workbook('./人机大战-0.xls')
sheet1 = workbook1.sheet_by_index(0)

workbook2 = xlrd.open_workbook('./人机大战-1.xls')
sheet2 = workbook2.sheet_by_index(0)

rows1 = sheet1.nrows
cols1 = sheet1.ncols

rows2 = sheet2.nrows
cols2 = sheet2.ncols

if rows2 != rows1 or cols2 != cols1:
    print("请更新表格！")
    exit(-1)
# exit(0)

correct_lc_list = []
correct_bs_list = []
correct_lcbs_list = []
cnt_doc_num = [0,0,0,0]
all_file = []

for row in range(1, rows1):
    correct_lc = 0
    correct_bs = 0
    correct_lcbs = 0

    correct_lc_ltpc = 0
    uncorrect_lc_tpc = 0

    correct_bs_ltpc = 0
    uncorrect_bs_tpc = 0

    correct_lcbs_ltpc = 0
    uncorrect_lcbs_tpc = 0

    info1_value = sheet2.cell(row, 8).value
    info2_value = sheet2.cell(row, 9).value
    info3_value = sheet2.cell(row, 10).value
    # info1_value = 1
    # if info2_value == 3 and (info1_value == 3 or info1_value == 4):
    if info2_value == 3 and (info1_value == 3 or info1_value == 4):
        for i in range(1, 5):
            if info1_value == i:
                cnt_doc_num[i-1]+=1

        for col in range(11, cols1):
            count = col - 10
            trueLabel =  trueLabelLst[(count-1) % 40]
            score = sheet1.cell(row, col).value
            if count <= 40:
                if score == 5.0:
                    correct_lc += 1
                    if trueLabel == 0:
                        correct_lc_ltpc += 1
                else:
                    if trueLabel == 1:
                        uncorrect_lc_tpc += 1

            if count <= 80 and count > 40:
                if score == 5.0:
                    correct_bs += 1
                    if trueLabel == 0:
                        correct_bs_ltpc += 1
                else:
                    if trueLabel == 1:
                        uncorrect_bs_tpc += 1

            if count > 80:
                if score == 5.0:
                    correct_lcbs += 1
                    if trueLabel == 0:
                        correct_lcbs_ltpc += 1
                else:
                    if trueLabel == 1:
                        uncorrect_lcbs_tpc += 1

        # print("id:%d" % row, "纯图片：%.3f" % (correct_lc/40), "纯病史：%.3f" % (correct_bs/40), "图片+病史：%.3f" % (correct_lcbs/40))

        if correct_lcbs<correct_lc or correct_lcbs<correct_bs:
            continue

        correct_lc_list.append(correct_lc/40.0)
        correct_bs_list.append(correct_bs/40.0)
        correct_lcbs_list.append(correct_lcbs/40.0)

        each_person = {}
        each_person['tprlc'] = correct_lc_ltpc/20.0
        each_person['fprlc'] = uncorrect_lc_tpc/20.0
        each_person['tprbs'] = correct_bs_ltpc/20.0
        each_person['fprbs'] = uncorrect_bs_tpc/20.0
        each_person['tprlcbs'] = correct_lcbs_ltpc/20.0
        each_person['fprlcbs'] = uncorrect_lcbs_tpc/20.0
        all_file.append(each_person)

print "总答题人数%d" % (len(correct_lc_list))

print "纯图片：%.3f" % (sum(correct_lc_list)/len(correct_lc_list)), "纯病史：%.3f" % (sum(correct_bs_list)/len(correct_bs_list)),\
    "图片+病史：%.3f" % (sum(correct_lcbs_list)/len(correct_lcbs_list))

print cnt_doc_num, sum(cnt_doc_num)

print(all_file)
with open("doc34.pkl", "wb") as f:
    pickle.dump(all_file, f)

print '---------------------------plot roc curve--------------------------------------'

# 读取TPC和LTPC病人的序列号
with open('doc12.pkl', 'rb') as f:
    doc12 = cPickle.load(f)
with open('doc34.pkl', 'rb') as f:
    doc34 = cPickle.load(f)
with open('file_img_test_res.pkl', 'rb') as f:
    file_img_test_res = cPickle.load(f)
with open('file_patient_test_res.pkl', 'rb') as f:
    file_patient_test_res = cPickle.load(f)

fpr1, tpr1, thresholds1 = metrics.roc_curve(file_patient_test_res[0], file_patient_test_res[1], pos_label=0)
auc1 = aucCalc(fpr1, tpr1)
fpr2, tpr2, thresholds2 = metrics.roc_curve(file_img_test_res[0], file_img_test_res[1], pos_label=0)
auc2 = aucCalc(fpr2, tpr2)

plt.plot(fpr1, tpr1, color='r', lw=1, label="after decision-making, AUC = %0.3f±0.005" % auc1)
plt.plot(fpr2, tpr2, color='g', lw=1, label='before decision-making, AUC = %0.3f±0.013' % auc2)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')

# plot points
x=[]
y=[]
for each in doc34:
    x.append(each['fprlcbs'])
    y.append(each['tprlcbs'])
plt.scatter(x, y, c='deepskyblue', s=5, label='69 expert dermatologists')
avgx = sum(x) / len(x)
avgy = sum(y) / len(y)
plt.scatter([avgx], [avgy], c='deepskyblue', s=5, marker='^', label='Average of 69 expert dermatologists')

x=[]
y=[]
for each in doc12:
    x.append(each['fprlcbs'])
    y.append(each['tprlcbs'])
plt.scatter(x, y, c='limegreen', s=5, label='27 general dermatologists')
avgx = sum(x) / len(x)
avgy = sum(y) / len(y)
plt.scatter([avgx], [avgy], c='limegreen', s=5, marker='^', label='Average of 27 general dermatologists')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

font1 = {'weight': 'normal', 'size': 8, }
plt.legend(loc="lower right", prop=font1)

plt.savefig("./ROC0918.png", dpi=960)
# plt.show()

# calc the best chosen TPR FPR
yuedeng = []
for i in range(len(fpr1)):
    yuedeng.append(tpr1[i] - fpr1[i])
yuedeng_index = yuedeng.index(max(yuedeng))
print 'the best TPR FPR for person', tpr1[yuedeng_index], fpr1[yuedeng_index]

yuedeng = []
for i in range(len(fpr2)):
    yuedeng.append(tpr2[i] - fpr2[i])
yuedeng_index = yuedeng.index(max(yuedeng))
print 'the best TPR FPR for img', tpr2[yuedeng_index], fpr2[yuedeng_index]