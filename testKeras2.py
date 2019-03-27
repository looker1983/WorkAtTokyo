import sys
sys.path.append("F:/Riku_Ka/AudioTestCode")
import shutil
import os
from pydub  import AudioSegment
import numpy as np
from keras import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import adam
import time

startTime = time.time()

#produce training data set.
trainingDIR = "F:/Riku_Ka/Training/training data with labels"
trainingMatrix = np.zeros(shape=(1,14))
MFCCfiles = os.listdir(trainingDIR)
for f in MFCCfiles:
    tempFile = trainingDIR + '/' + f
    tempMatrix = np.loadtxt(tempFile)
    print(tempMatrix.shape)
    trainingMatrix = np.vstack((trainingMatrix,tempMatrix))
trainingDate = trainingMatrix[:,1:]
print("The shape of training data matrix is: ")
print(trainingDate.shape)
trainingLabels = trainingMatrix[:,0]

print("Time point 1 is " + str(time.time() - startTime))

#training the model
myModel = Sequential()
myModel.add(Dense(units=64,input_dim=13, activation='relu'))
myModel.add(Dense(units=8, activation='relu'))
myModel.add(Dense(1, activation='sigmoid'))

myModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
myModel.fit(x=trainingDate, y=trainingLabels, epochs=2, batch_size=10 )

print("Time point 2 is " + str(time.time() - startTime))

#produce test data set and evaluate the model.
testDIR = "F:/Riku_Ka/Training/test data with labels"

MFCCfiles = os.listdir(testDIR)
for f in MFCCfiles:
    tempFile = testDIR + '/' + f
    tempMatrix = np.loadtxt(tempFile)
    testData = tempMatrix[:,1:]
    testLabels = tempMatrix[:,0]
    print(testLabels)
    
    scores = myModel.evaluate(x=testData, y=testLabels)
    print("{0},{1}".format(myModel.metrics_names[1],scores[1]*100))
    

print("Time point 3 is " + str(time.time() - startTime))

############################################
