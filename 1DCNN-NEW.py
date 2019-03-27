import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import os

data = np.loadtxt('c:/odom.txt')
print(data.shape)
data = data[:,np.newaxis]
print(data.shape)
input = torch.FloatTensor(data)
input=Variable(input)

x=torch.nn.Conv1d(in_channels=1,out_channels=4,kernel_size=2,groups=1)

out=x(input)
print(out)

print("Test ends!")

# predefine some const
EPOCH = 10
BATCH_SIZE = 200
LEARNING_RATE = 0.001
trainDIR = "C:/testData/very small test data set/training data set"
testDIR = "C:/testData/very small test data set/test data set"

# Prepare the training dataset
trainMatrix = np.zeros(shape=(1,40,1))
trainFILES = os.listdir(trainDIR)

print("Prepare the training dataset:")
for f in trainFILES:
    tempFILE = trainDIR + '/' + f
    tempMatrix = np.loadtxt(tempFILE)
    print("The shape of file {0} is {1}.".format(tempFILE, tempMatrix.shape))
    tempMatrix = tempMatrix[:,np.newaxis]
    print("The shape of tempMatrix for {0} is {1}.".format(tempFILE, tempMatrix.shape))
    trainMatrix = np.vstack((trainMatrix,tempMatrix))
    DataForTorch = torch.from_numpy(trainMatrix)
    x_train = DataForTorch[:,1:]
    y_train = DataForTorch[:,0]

trainDataSet = Data.TensorDataset(x_train,y_train)

# prepare the test dataset
testMatrix = np.zeros(shape=(1,40))
testFILES = os.listdir(testDIR)

print("Prepare the test dataset:")
for f in testFILES:
    tempFILE = testDIR + '/' + f
    tempMatrix = np.loadtxt(tempFILE)
    print("The shape of file {0} is {1}.".format(tempFILE, tempMatrix.shape))
    testMatrix = np.vstack((testMatrix,tempMatrix))
    DataForTorch = torch.from_numpy(testMatrix)
    x_test = DataForTorch[:,1:]
    y_test = DataForTorch[:,0]

testDataSet = Data.TensorDataset(x_test,y_test)

# load the train dataset
trainLoader = Data.DataLoader(
    dataset=trainDataSet,  # torch TensorDataset format
    batch_size=BATCH_SIZE,  # mini batch size
    shuffle=True,  # 要不要打乱数据 (打乱比较好)
    #num_workers=2,  # 多线程来读数据
)

# load the test dataset
testLoader = Data.DataLoader(
    dataset=trainDataSet,  # torch TensorDataset format
    batch_size=BATCH_SIZE,  # mini batch size
    shuffle=True,  # 要不要打乱数据 (打乱比较好)
    #num_workers=2,  # 多线程来读数据
)



class myCNN1D(nn.Module):
    def __init__(self):
        super(myCNN1D, self).__init__()
        self.layer1_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=2, kernel_size=5, stride=1, padding=0, dilation=1, groups=1, bias=True),
            nn.ReLU(),
        #nn.MaxPool1d(2)
        )
        self.layer2_conv = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=3, kernel_size=6, stride=1, padding=0, dilation=1, groups=1, bias=True),
            nn.ReLU()
        )
        self.layer3_fc1 = nn.Linear(20*1*6, 100)
        self.layer4_output = nn.Linear(100, 2)

    def forward(self, x):
        out = self.layer1_conv(x)
        out = self.layer2_conv(out)
        out = out.view(1, -1)
        out = self.layer3_fc1(out)
        out = self.layer4_output(out)
        return out

myConv = myCNN1D()
print(myConv)

# Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(myConv.parameters(), lr=LEARNING_RATE)

# Train the Model
for epoch in range(EPOCH):
    for i, (x_train, y_train) in enumerate(trainLoader):
        x_train = Variable(x_train)
        print("x_train is {}".format(x_train))
        print("The size of x_train is {}".format(x_train.size()))
        y_train = Variable(y_train)
        print("y_train is {}".format(y_train))
        print("The size of y_train is {}".format(y_train.size()))

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = myConv(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' %(epoch+1, EPOCH, i+1, len(trainLoader)//BATCH_SIZE, loss.data[0]))

# Test the Model
correct = 0
total = 0
for x_test , y_test in testLoader:
    x_train = Variable(x_train)
    outputs = myConv(y_test)
    _, predicted = torch.max(outputs.data, 1)
    total += x_test.size(0)
    correct += (predicted == x_test).sum()

print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))


'''
# test 1D CNN
input=torch.ones(200,1,4)
input=Variable(input)
x=torch.nn.Conv1d(in_channels=1,out_channels=4,kernel_size=2,groups=1)
out=x(input)
print(out)
print("Parameters are as following {}".format(list(x.parameters())))


data = np.loadtxt('c:/odom.txt')
data = data[:,np.newaxis]
input = torch.FloatTensor(data)
input=Variable(input)

x=torch.nn.Conv1d(in_channels=1,out_channels=4,kernel_size=2,groups=1)

out=x(input)
print(out)

print("Test ends!")
'''