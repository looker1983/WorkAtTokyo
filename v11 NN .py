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

print("Loading data....")
startTime = time.time()

#produce training data set.
train_npy_FILE = "D:/TestWithMFCC39/NN for 0Health-1Dementia/npy/totalNPY for train.npy"
testDIR = "D:/TestWithMFCC39/NN for 0Health-1Dementia/npy/npy for test"
modelDIR = "D:/TestWithMFCC39/NN for 0Health-1Dementia/npy"
modelFILE = "NN Modelfor 0Health-1Dementia with 39MFCC with npy form by 333 samples.json"
weightFILE = "NN Modelfor 0Health-1Dementia with 39MFCC with npy form by 333 samples.h5"

trainDS = np.load(train_npy_FILE)
print("The shape of train dataset is {}".format(trainDS.shape))
trainDATA = trainDS[:,1:]
trainLABLE = trainDS[:,0]

print("Start training!")
print("{} secondes have passed. ".format(str(time.time() - startTime)))


np.random.seed(10)

myModel = Sequential()

myModel.add(Dense(units=64,input_dim=39, activation='relu'))
myModel.add(Dense(units=8, activation='relu'))
myModel.add(Dense(1, activation='sigmoid'))

myModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

myModel.fit(x=trainDATA, y=trainLABLE, epochs=50, batch_size=64)

# Save the trained model
myModel_json = myModel.to_json()
with open(modelDIR + '/' + modelFILE, 'w') as json_file:
    json_file.write(myModel_json)
myModel.save_weights(modelDIR + '/' + weightFILE)
print("The trained model has been saved.")


print("Start testing!")
print("Time point 3 is " + str(time.time() - startTime))

#produce test data set.


NPYfiles = os.listdir(testDIR)
for f in NPYfiles:
    tempFile = testDIR + '/' + f
    tempMatrix = np.load(tempFile)
    print("The shape of {0} is {1}.".format(f, tempMatrix.shape))
    testData = tempMatrix[:,1:]
    testLabels = tempMatrix[:,0]
    
    scores = myModel.evaluate(x=testData, y=testLabels)
    print(scores)
    print("The accuracy for file {0} is {1}".format(f,scores[1]*100))

totalTestNPY = "D:/TestWithMFCC39/NN for 0Health-1Dementia/npy/totalNPY for test.npy"
totalMatrix = np.load(totalTestNPY)
print("The shape of totalTestNPY is {}.".format(tempMatrix.shape))
testTotalData = totalMatrix[:,1:]
testTotalLabels = totalMatrix[:,0]
scores = myModel.evaluate(x=testTotalData, y=testTotalLabels)
print(scores)
print("The accuracy for file {0} is {1}".format(f,scores[1]*100))

print("Time point 2 is " + str(time.time() - startTime))
print("Testing ends!")