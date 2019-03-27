#training the NN
import sys
#sys.path.append("F:/Riku_Ka/AudioTestCode")
import shutil
import os
from pydub  import AudioSegment
import numpy as np
from keras import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import adam
from keras.models import load_model
import time
from keras.models import model_from_json
import h5py

modelStructureFILE = "C:/Users/fhc_workPC1/Documents/Looker/Dementia-Health-with-score/Model_for Dementia-Health-with-score.json"
modelWeightFILE = "C:/Users/fhc_workPC1/Documents/Looker/Dementia-Health-with-score/Model_for Dementia-Health-with-score.h5"

reloadModel = model_from_json(open(modelStructureFILE).read())
reloadModel.load_weights(modelWeightFILE)

reloadModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#produce test data set.
testDIR = "C:/Users/fhc_workPC1/Documents/Looker/Dementia-Health-with-score/test data set"

MFCCfiles = os.listdir(testDIR)
for f in MFCCfiles:
    tempFile = testDIR + '/' + f
    print(tempFile)
    tempMatrix = np.loadtxt(tempFile)
    testData = tempMatrix[:,1:]
    testLabels = tempMatrix[:,0]
    
    scores = reloadModel.evaluate(x=testData, y=testLabels)
    print("The accuracy for file {0} is {1}".format(f,scores[1]*100))