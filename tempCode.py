#training the NN
import sys
sys.path.append("F:/Riku_Ka/AudioTestCode")
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


startTime = time.time()

#produce training data set.
trainingDIR = "F:/Riku_Ka/Training_Dementia/Training Data Set"
modelDIR = "F:/Riku_Ka/Training_Dementia"
modelFILE = "Model_for Dementia1.json"
weightFILE = "Model_for Dementia1.h5"

trainingMatrix = np.zeros(shape=(1,14))

MFCCfiles = os.listdir(trainingDIR)
for f in MFCCfiles:
    tempFile = trainingDIR + '/' + f
    print(tempFile)
    tempMatrix = np.loadtxt(tempFile)
    print(tempMatrix.shape)
    trainingMatrix = np.vstack((trainingMatrix,tempMatrix))

trainingDate = trainingMatrix[:,1:]
print("The shape of training data matrix is: ")
print(trainingDate.shape)
trainingLabels = trainingMatrix[:,0]

print("Time point 1 is " + str(time.time() - startTime))


np.random.seed(10)

myModel = Sequential()

myModel.add(Dense(units=64,input_dim=13, activation='relu'))
myModel.add(Dense(units=8, activation='relu'))
myModel.add(Dense(1, activation='sigmoid'))

myModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

myModel.fit(x=trainingDate, y=trainingLabels, epochs=150, batch_size=32 )

# Save the trained model
myModel_json = myModel.to_json()
with open(modelDIR + '/' + modelFILE, 'w') as json_file:
    json_file.write(myModel_json)
myModel.save_weights(modelDIR + '/' + weightFILE)
print("The trained model has been saved.")



print("Time point 3 is " + str(time.time() - startTime))

#produce test data set.
testDIR = "F:/Riku_Ka/Training_Dementia/Test Data Set"

MFCCfiles = os.listdir(testDIR)
for f in MFCCfiles:
    tempFile = testDIR + '/' + f
    print(tempFile)
    tempMatrix = np.loadtxt(tempFile)
    testData = tempMatrix[:,1:]
    testLabels = tempMatrix[:,0]
    
    scores = myModel.evaluate(x=testData, y=testLabels)
    print("The accuracy for file {0} is {1}".format(f,scores[1]*100))
    

print("Time point 2 is " + str(time.time() - startTime))


#Select files according to special WORDS
import os, random, shutil
resDIR = "F:/Riku_Ka/Training_Dementia/MFCC with Labels for Health"
dstDIR = "F:/Riku_Ka/Training_Dementia/Test Data Set"
import sys
sys.path.append("F:/Riku_Ka/AudioTestCode")
import shutil
import os
from pydub  import AudioSegment
import numpy as np

import FHC_Audio

sourceDIR = "F:/Riku_Ka/PROMPT-UnzipAudioFile/音声データ_003"
dstDIR = "F:/Riku_Ka/Training_Dementia/MFCC for Dementia-Pt only"
subDIRs = os.listdir(sourceDIR)
#listHealth = ['BH', 'KH', 'OH', 'PKH', 'POH', 'PTH', 'TH']
#listDepression = ['AD', 'BD', 'GD', 'KD', 'MD', 'ND', 'OD', 'PKD', 'POD', 'PTD', 'SD', 'TD']
listDementias = ['AC','GC','KC','OC','PKC','SC','TC']
print(listDementias)

for dirs in subDIRs:
    tempDIR = sourceDIR + '/' + dirs
    files = os.listdir(tempDIR)
    for f in files:
        if any(x in f for x in listDementias) and f.find('Pt') != -1:
            tempFILE = tempDIR + '/' + f
            print("For the file {}: ".format(f))
            try:
                featureMFCC = FHC_Audio.getMFCC4SingleFile(tempFILE, 13)
                featureFILE = dstDIR + '/' + os.path.splitext(f)[0] +"_MFCC.txt"
                np.savetxt(featureFILE, featureMFCC)
            except ValueError:
                print("There is a ValueError.")

#select data randomly for TEST and TRAINING respectively
import os, random, shutil
sourceDIR = "F:/Riku_Ka/PROMPT-UnzipAudioFile/音声データ_003"
dstDIR = "F:/Riku_Ka/Training_Dementia/MFCC for Dementia-Pt only"
files = os.listdir(sourceDIR)
numOfTest = int(len(files)/3)
trainingSamples = random.sample(files, numOfTest)
for sample in trainingSamples:
    shutil.move(resDIR + '/' + sample, dstDIR + '/' + sample)

#Give the labels to file
resDIR = "F:/Riku_Ka/Training_Dementia/MFCC for Dementia-Pt only"
dstDIR = "F:/Riku_Ka/Training_Dementia/MFCC with Labels for Dementia"
#myDIR = "F:/Riku_Ka/Training/MFCC for Health Data for Training"

MFCCfiles = os.listdir(resDIR)

for f in MFCCfiles:
    tempFile = resDIR + '/' + f
    tempMatrix = np.loadtxt(tempFile)
    n,m = tempMatrix.shape
    print("The size of {0} is {1} * {2}".format(f,n,m))
    newCol = np.ones((n,1))
    newMatrix = np.hstack((newCol,tempMatrix))
    newFile = dstDIR + '/1_' + f
    np.savetxt(newFile,newMatrix)

