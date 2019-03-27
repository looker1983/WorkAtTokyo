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

print("Loading data....")
startTime = time.time()

#produce training data set.
train_npy_FILE = "D:/TestWithMFCC39/NN for 0Health-1Dementia/npy/totalNPY for train.npy"
testDIR = "D:/TestWithMFCC39/NN for 0Health-1Dementia/npy/npy for test"
modelDIR = "D:/TestWithMFCC39/NN for 0Health-1Dementia/npy"
modelFILE = "1D-CNN Modelfor 0Health-1Dementia with 39MFCC with npy form by 333 samples.json"
weightFILE = "1D-CNN Modelfor 0Health-1Dementia with 39MFCC with npy form by 333 samples.h5"

trainDS = np.load(train_npy_FILE)
print("The shape of train dataset is {}".format(trainDS.shape))
x_train = trainDS[:,1:]
y_train = trainDS[:,0]

print("{} secondes have passed. ".format(str(time.time() - startTime)))
print("Now, start training!")


# for health and dementia
num_classes = 2
# for 10 seconds, the duaration time for each frame is about 0.031 second 
TIME_PERIODS = 300
# the demention size for each line of MFCC feature
INPUT_SHAPE = (39,)

# Convert class vectors to binary class matrices. This uses 1 hot encoding.
y_train_binary = keras.utils.to_categorical(y_train, num_classes)
print("y_train_binary is {}.".format(y_train_binary))

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
print("The shape of y_train is {}.".format(y_train.shape))

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


BATCH_SIZE = 64
EPOCHS = 1
myModel.fit(x_train, y_train_binary,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=1
          )
print(myModel.summary())


NPYfiles = os.listdir(testDIR)
for f in NPYfiles:
    tempFile = testDIR + '/' + f
    tempMatrix = np.load(tempFile)
    print("The shape of {0} is {1}.".format(f, tempMatrix.shape))
    testData = tempMatrix[:,1:]
    testLabels = tempMatrix[:,0]
    testData = testData.reshape(testData.shape[0], testData.shape[1], 1)
    
    scores = myModel.evaluate(x=testData, y=testLabels)
    print(scores)
    print("The accuracy for file {0} is {1}".format(f,scores[1]*100))


'''
# test the model
NPYfiles = os.listdir(testDIR)
for f in NPYfiles:
    tempFile = testDIR + '/' + f
    tempMatrix = np.load(tempFile)
    print("The shape of {0} is {1}.".format(f, tempMatrix.shape))
    x_test = tempMatrix[:,1:]
    y_test = tempMatrix[:,0]
    y_test_binary = keras.utils.to_categorical(y_test, num_classes)
    print("y_test_binary is {}.".format(y_test_binary))
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    print("The shape of x_test is {}.".format(x_test.shape))

    myPrediction = myModel.predict_classes(x_test)
    print(myPrediction)
'''    
# Save the trained model
myModel_json = myModel.to_json()
with open(modelDIR + '/' + modelFILE, 'w') as json_file:
    json_file.write(myModel_json)
myModel.save_weights(modelDIR + '/' + weightFILE)
print("The trained model has been saved.")




print("Time point 3 is " + str(time.time() - startTime))