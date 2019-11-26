# -*- coding: utf-8 -*-
import os.path
import shutil


patchSubsetPrefix = '/media/wz209/a29353b7-1090-433f-b452-b4ce827adb17/' \
               'sugurs/PythonProjects/pem_multi/dataset/patch/'

rawSubsetPrefix = '/media/wz209/a29353b7-1090-433f-b452-b4ce827adb17/' \
               'sugurs/PythonProjects/pem_multi/dataset/raw/'

rawImageOriginalPath = '/media/wz209/a29353b7-1090-433f-b452-b4ce827adb17/' \
                       'sugurs/PythonProjects/pem_multi/dataset/raw306'

allOriginalImageLst = []
for fpathe, dirs, fs in os.walk(rawImageOriginalPath):
    for f in fs:
        file = os.path.join(fpathe, f)
        allOriginalImageLst.append(file)

hasBeenFoundSet = []
for index in range(1, 5):
    patchSubsetDir = patchSubsetPrefix + str(index)
    rawSubsetDir = rawSubsetPrefix + str(index)
    for fpathe, dirs, fs in os.walk(patchSubsetDir):
        for f in fs:
            file = os.path.join(fpathe, f)
            patchLchWithabcd = file.split('/')[-1].split('-')[0]
            #——————————————————————————— go to find raw images ——————————————————————————————
            for eachRawImages in allOriginalImageLst:
                rawLchWithabcd = eachRawImages.split('/')[-1][:-4]
                if patchLchWithabcd == rawLchWithabcd:
                    oldImageFilePath = eachRawImages
                    newImageFilePath = rawSubsetDir +  '/%s.jpg' % rawLchWithabcd
                    shutil.copyfile(oldImageFilePath, newImageFilePath)





