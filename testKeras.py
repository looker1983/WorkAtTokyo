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
myDIR = "F:/Riku_Ka/Training/training data with labels"

combinedMatrix = np.zeros(shape=(1,14))

MFCCfiles = os.listdir(myDIR)
for f in MFCCfiles:
    tempFile = myDIR + '/' + f
    tempMatrix = np.loadtxt(tempFile)
    print(tempMatrix.shape)
    combinedMatrix = np.vstack((combinedMatrix,tempMatrix))

trainingDate = combinedMatrix[:,1:]
print("The shape of training data matrix is: ")
print(trainingDate.shape)
trainingLabels = combinedMatrix[:,0]

print("Time point 1 is " + str(time.time() - startTime))

#produce test data set.
myDIR = "F:/Riku_Ka/Training/test data with labels"

combinedMatrix = np.zeros(shape=(1,14))

MFCCfiles = os.listdir(myDIR)
for f in MFCCfiles:
    tempFile = myDIR + '/' + f
    tempMatrix = np.loadtxt(tempFile)
    print(tempMatrix.shape)
    combinedMatrix = np.vstack((combinedMatrix,tempMatrix))

testDate = combinedMatrix[:,1:]
print("The shape of test data matrix is: ")
print(trainingDate.shape)
testLabels = combinedMatrix[:,0]

print("Time point 2 is " + str(time.time() - startTime))

############################################


np.random.seed(10)

myModel = Sequential()

myModel.add(Dense(units=64,input_dim=13, activation='relu'))
myModel.add(Dense(units=8, activation='relu'))
myModel.add(Dense(1, activation='sigmoid'))

myModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

myModel.fit(x=trainingDate, y=trainingLabels, epochs=2, batch_size=10 )

scores = myModel.evaluate(x=testDate, y=testLabels)
print("{0},{1}".format(myModel.metrics_names[1],scores[1]*100))


print("Time point 3 is " + str(time.time() - startTime))



