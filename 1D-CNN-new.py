import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
import os

##################### Prepare the data start #############################

# obtaining training data
trainDIR = "C:/testData/very small test data set/training data set"
testDIR = "C:/testData/very small test data set/test data set"
BATCH_SIZE = 200

# prepare the train dataset
trainMatrix = np.zeros(shape=(1,40))
trainFILES = os.listdir(trainDIR)

for f in trainFILES:
    tempFILE = trainDIR + '/' + f
    tempMatrix = np.loadtxt(tempFILE)
    print("The shape of file {0} is {1}.".format(tempFILE, tempMatrix.shape))
    trainMatrix = np.vstack((trainMatrix,tempMatrix))
    DataForTorch = torch.from_numpy(trainMatrix)
    x_train = DataForTorch[:,1:]
    y_train = DataForTorch[:,0]

trainDataSet = Data.TensorDataset(x_train,y_train)

# prepare the test dataset
testMatrix = np.zeros(shape=(1,40))
testFILES = os.listdir(testDIR)

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
    num_workers=2,  # 多线程来读数据
)

# load the test dataset
testLoader = Data.DataLoader(
    dataset=trainDataSet,  # torch TensorDataset format
    batch_size=BATCH_SIZE,  # mini batch size
    shuffle=True,  # 要不要打乱数据 (打乱比较好)
    num_workers=2,  # 多线程来读数据
)
################# Prepare the data ends! ##########################

########### Define the neural network start #########################
class CNN1D(nn.Module):
#定义net的初始化函数,这个函数定义了该神经网络的基本结构
    def __init__(self):
        super(CNN1D, self).__init__() #复制并使用Net的父类的初始化方法,即先运行nn.Module的初始化函数
        # 定义conv1函数的是一维卷积核：输入为39维MFCC特征,输出为6张特征图, 卷积核为5x1的向量
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # 定义conv2函数的是一维卷积核：输入为35维向量,输出为训练全连接网络的26维特征数据, 卷积核为10x1矩阵
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=10, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.fc1   = nn.Linear(26, 50)# 定义fc1（fullconnect）全连接函数1为线性函数：y = Wx + b，并将26节点连接到50个节点上。
        self.fc2   = nn.Linear(50, 20)#定义fc2（fullconnect）全连接函数2为线性函数：y = Wx + b，并将50个节点连接到20个节点上。
        self.fc3   = nn.Linear(20, 2)#定义fc3（fullconnect）全连接函数3为线性函数：y = Wx + b，并将20个节点连接到2个节点上。
#定义该神经网络的向前传播函数，该函数必须定义，一旦定义成功，向后传播函数也会自动生成（autograd）
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))#输入x经过卷积conv1之后，经过激活函数ReLU，使用2x2的窗口进行最大池化Max pooling，然后更新到x。
        x = self.pool(F.relu(self.conv2(x)))#输入x经过卷积conv2之后，经过激活函数ReLU，使用2x2的窗口进行最大池化Max pooling，然后更新到x。
        x = x.view(-1, 26)#view函数将张量x变形成一维的向量形式，总特征数并不改变，为接下来的全连接作准备。
        x = F.relu(self.fc1(x))#输入x经过全连接1，再经过ReLU激活函数，然后更新x
        x = F.relu(self.fc2(x))#输入x经过全连接2，再经过ReLU激活函数，然后更新x
        x = self.fc3(x)#输入x经过全连接3，然后更新x
        return x
###########定义网络结束#########################
#新建一个之前定义的网路
myNN = CNN1D()

#################查看需要训练的网络参数的相关信息开始############
print(myNN)
params = list(myNN.parameters())

k=0
for i in params:
    l =1
    print("The structure of this layer is "+str(list(i.size())))
    for j in i.size():
        l *= j
    print("THe number of parameters is "+str(l))
    k = k+l

print("THe number of total parameters is ："+ str(k))
#################查看需要训练的网络参数的相关信息结束############

#######关于loss function 开始################
criterion = nn.CrossEntropyLoss() #交叉熵损失函数
optimizer = optim.SGD(myNN.parameters(), lr=0.001, momentum=0.9)  #使用SGD（随机梯度下降）优化，学习率为0.001，动量为0.9
#######关于loss function 结束################

###############训练过程开始##########################
for epoch in range(2): # 遍历数据集两次

    running_loss = 0.0
    #enumerate(sequence, [start=0])，i序号，data是数据
    for i, data in enumerate(trainLoader, 0): 
        # get the inputs
        inputs, labels = data   #data的结构是：[4x3x32x32的张量,长度4的张量]

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)  #把input数据从tensor转为variable

        # zero the parameter gradients
        optimizer.zero_grad() #将参数的grad值初始化为0

        # forward + backward + optimize
        outputs = myNN(inputs)
        loss = criterion(outputs, labels) #将output和labels使用叉熵计算损失
        loss.backward() #反向传播
        optimizer.step() #用SGD更新参数

        # 每2000批数据打印一次平均loss值
        running_loss += loss.data[0]  #loss本身为Variable类型，所以要使用data获取其Tensor，因为其为标量，所以取0
        if i % 200 == 199: # 每200批打印一次
            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
#############训练过程结束##################
#############测试过程开始##################
correct = 0
total = 0
for data in testLoader:
    MFCCdata, labels = data
    outputs = myNN(Variable(MFCCdata))
    #print outputs.data
    _, predicted = torch.max(outputs.data, 1)  #outputs.data是一个4x10张量，将每一行的最大的那一列的值和序号各自组成一个一维张量返回，第一个是值的张量，第二个是序号的张量。
    total += labels.size(0)
    correct += (predicted == labels).sum()   #两个一维张量逐行对比，相同的行记为1，不同的行记为0，再利用sum(),求总和，得到相同的个数。

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
#############测试过程结束##################
