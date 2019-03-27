#training the NN
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
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.layers import Conv1D, Reshape, GlobalAveragePooling1D, MaxPooling1D
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint



startTime = time.time()

#produce training data set.
trainingDIR = "C:/TestWithMFCC39/very small data set for 0Health-1Dementia/training"
testDIR = "C:/TestWithMFCC39/very small data set for 0Health-1Dementia/test"

dataDIR = "C:/TestWithMFCC39/MFCC39 without lable"
modelDIR = "C:/TestWithMFCC39/MFCC39 without lable"
modelFILE = "Model_for 0Health-1Depression with 39MFCC.json"
weightFILE = "Model_for 0Health-1Depression with 39MFCC.h5"

trainingMatrix = np.zeros(shape=(1,40))

#preparing the training data sets
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

x_train, x_test, y_train, y_test = train_test_split(np.asarray(trainingDate), np.asarray(trainingLabels), test_size=0.33, shuffle= True)
print("The shape of x_train is {}.".format(x_train.shape))
print("The shape of x_test is {}.".format(x_test.shape))
print("The shape of y_train is {}.".format(y_train.shape))
print("The shape of y_test is {}.".format(y_test.shape))
# for health and depression
num_classes = 2
# for 10 seconds, the duaration time for each frame is about 0.031 second 
TIME_PERIODS = 300
# the demention size for each line of MFCC feature
INPUT_SHAPE = (39,)

# Convert class vectors to binary class matrices. This uses 1 hot encoding.
y_train_binary = keras.utils.to_categorical(y_train, num_classes)
print(y_train_binary)
y_test_binary = keras.utils.to_categorical(y_test, num_classes)
print(y_test_binary)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
print(x_train.shape)
print(x_test.shape)

print("Time point 1 is " + str(time.time() - startTime))



np.random.seed(10)

myModel = Sequential()

myModel = Sequential()
myModel.add(Conv1D(67, (30), input_shape=(39,1), activation='relu'))
myModel.add(Flatten())
myModel.add(Dense(64, activation='softmax'))
myModel.add(Dense(num_classes, activation='softmax'))

myModel.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
print(myModel.summary())


BATCH_SIZE = 400
EPOCHS = 20
myModel.fit(x_train, y_train_binary,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=1,
          validation_data=(x_test, y_test_binary))








# Save the trained model
myModel_json = myModel.to_json()
with open(modelDIR + '/' + modelFILE, 'w') as json_file:
    json_file.write(myModel_json)
myModel.save_weights(modelDIR + '/' + weightFILE)
print("The trained model has been saved.")



print("Time point 3 is " + str(time.time() - startTime))
'''
#produce test data set.
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
'''