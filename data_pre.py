import sys
sys.path.append("F:/Riku_Ka/AudioTestCode")
import shutil
import os
from pydub  import AudioSegment
import numpy as np
from keras import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import adam
import codecs
import json

myDIR = "F:/Riku_Ka/Training_new/random test data set/MFCC Depression"
#myDIR = "F:/Riku_Ka/Training/MFCC for Health Data for Training"




MFCCfiles = os.listdir(myDIR)

for f in MFCCfiles:
    tempFile = myDIR + '/' + f
    tempMatrix = np.loadtxt(tempFile)
    n,m = tempMatrix.shape
    print(n,m)
    newCol = np.ones((n,1))
    newMatrix = np.hstack((newCol,tempMatrix))
    newFile = myDIR + '/1_' + f
    np.savetxt(newFile,newMatrix)

'''
# if there are cp932 error, run the following codes.
for f in MFCCfiles:
    tempFile = myDIR + '/' + f
    tempData = []
    of = open(tempFile, mode='r', encoding='cp932', errors='ignore')
    while True:
        line = of.readline()
        if not line:
            break
        tempData.append([float(i) for i in line.split()])
    of.close()

    tempMatrix = np.array(tempData)    
    n,m = tempMatrix.shape
    print(n,m)
    newCol = np.zeros((n,1))
    newMatrix = np.hstack((newCol,tempMatrix))
    newFile = myDIR + '/' + 'h_' + f
    np.savetxt(newFile,newMatrix)
'''

