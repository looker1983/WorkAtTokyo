import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn.functional as F
from prettytable import PrettyTable
import time

# predefine some const
EPOCH = 20
BATCH_SIZE = 200
LEARNING_RATE = 0.001
train_npy_FILE = "D:/TestWithMFCC39/NN for 0Health-1Dementia/npy/totalNPY for train.npy"
testDIR = "D:/TestWithMFCC39/NN for 0Health-1Dementia/npy/npy for test"
modelDIR = "D:/TestWithMFCC39/NN for 0Health-1Dementia/npy"
modelFILE = "1D-CNN Modelfor 0Health-1Dementia with 39MFCC of npy form by 333 samples.pth"


#print the seperating line
def print_line(char,string):
    print(char*33,string,char*32)

timePoint1 = time.time()

# define the Conv. neural network
class myCNN1D(nn.Module):

    def num_flat_features(self,x):
        size=x.size()[1:] # all dimensions except the batch dimension
        num_features=1
        for s in size:
            num_features*=s
        return num_features

    def __init__(self):
        super(myCNN1D, self).__init__()
        self.layer1_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=2, kernel_size=5, stride=1, padding=0, dilation=1, groups=1,
                      bias=True),
            nn.ReLU(),
            # nn.MaxPool1d(2)
        )
        self.layer2_conv = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=3, kernel_size=6, stride=1, padding=0, dilation=1, groups=1,
                      bias=True),
            nn.ReLU()
        )
        self.layer3_fc1 = nn.Linear(10, 100)
        self.layer4_output = nn.Linear(100, 1)

    def forward(self, x):
        out = self.layer1_conv(x)
        out = self.layer2_conv(out)
        out = out.view(-1, self.num_flat_features(out))
        out = self.layer3_fc1(out)
        out = self.layer4_output(out)
        return out




# build a CNN
myConv = myCNN1D()

# Define Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(myConv.parameters(), lr=LEARNING_RATE)


# prepare the training dataset
trainDS = np.load(train_npy_FILE)
#print("The shape of trainDS is {}".format(trainDS.shape))
trainDS = trainDS[:,np.newaxis]
print("The shape of trainDS is {}".format(trainDS.shape))

x_train = trainDS[:,:,1:]
y_train = trainDS[:,:,0]
y_train = torch.from_numpy(y_train).float()
x_train = torch.from_numpy(x_train).float()

trainDataSet = Data.TensorDataset(x_train, y_train)

trainLoader = Data.DataLoader(
    dataset=trainDataSet,  # torch TensorDataset format
    batch_size=BATCH_SIZE,  # mini batch size
    shuffle=True,  # shuffle the data
    #num_workers=1,  # 多线程来读数据
)

timePoint2 = time.time()
print("{} seconds have passed!".format(timePoint2 - timePoint1))

# Train the Model
print_line('*','Training Begin')
for epoch in range(EPOCH):
    for i, (x_train, y_train) in enumerate(trainLoader):
        x_train = Variable(x_train)
        y_train = Variable(y_train)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = myConv(x_train)

        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}] Loss: {:.8f}'.format(epoch + 1, EPOCH, loss.data))

torch.save(myConv, modelDIR+'/'+modelFILE)
reload_model = torch.load(modelDIR+'/'+modelFILE)

timePoint3 = time.time()
print("{} seconds have passed!".format(timePoint3 - timePoint2))

# Test the Model
print_line('*', 'Testing Begin')

testFILES = os.listdir(testDIR)

for f in testFILES:
    if f[0] == '.':
        continue
    correct = 0
    try:
        tempFILE = testDIR + '/' + f
        tempMatrix = np.load(tempFILE)
        testMatrix = tempMatrix[:, np.newaxis]

        x_test = testMatrix[:, :, 1:]
        y_test = testMatrix[:, :, 0]
        y_test = torch.from_numpy(y_test).float()
        x_test = torch.from_numpy(x_test).float()

        testDataSet = Data.TensorDataset(x_test, y_test)

        testLoader = Data.DataLoader(
            dataset=testDataSet,  # torch TensorDataset format
            batch_size=BATCH_SIZE,  # mini batch size
            shuffle=True  # shuffle the data
            # num_workers=1,  # 多线程来读数据
        )

        
        for x_test, y_test in testLoader:
            x_test = Variable(x_test)
            outputs = reload_model(x_test)
            outputs = outputs.data.squeeze()

        is_sick = 0
        in_one_file_total = 0
        for output in outputs:
            in_one_file_total += 1
            if (output - 0) ** 2 - (output - 1) ** 2 < 0:
                predicted = 0
            else:
                predicted = 1
                is_sick += 1
            if (predicted - y_test[0])**2<0.00001:
                correct+=1    
        print('This guy may be {} percentage in sick.'.format(100*is_sick / in_one_file_total))
    
    except ZeroDivisionError: 
        print("division by zero: in_one_file_total of file {} is equal to 0.".format(f))
    except TypeError:
        print("There is a TypeError with file {}.".format(f))

    try:
        print('Test Accuracy of file {0} is {1}%.'.format(f, 100 * correct / in_one_file_total))
    except ZeroDivisionError: 
        print("division by zero: in_one_file_total is equal to 0.")
    

    

timePoint4 = time.time()
print("{} seconds have passed!".format(timePoint4 - timePoint3))
print("Test Ends")






