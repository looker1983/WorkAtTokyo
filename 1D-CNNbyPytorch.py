import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# obtaining training data
trainingDIR = "C:/testData/very small test data set/training data set"
testDIR = "C:/testData/very small test data set/test data set"
BATCH_SIZE = 200

trainingMatrix = np.zeros(shape=(1,40))

trainingFILES = os.listdir(trainingDIR)
for f in trainingFILES:
    tempFILE = trainingDIR + '/' + f
    print(tempFILE)
    tempMatrix = np.loadtxt(tempFILE)
    print(tempMatrix.shape)
    trainingMatrix = np.vstack((trainingMatrix,tempMatrix))
    DataForTorch = torch.from_numpy(trainingMatrix)
    x = DataForTorch[:,1:]
    y = DataForTorch[:,0]


# 把 dataset 放入 DataLoader
y=y.type(torch.LongTensor)
x=x.type(torch.FloatTensor)

torch_dataset = Data.TensorDataset(x,y)

torch_loader = Data.DataLoader(
    dataset=torch_dataset,  # torch TensorDataset format
    batch_size=BATCH_SIZE,  # mini batch size
    shuffle=True,  # 要不要打乱数据 (打乱比较好)
    num_workers=2,  # 多线程来读数据
)

# define the Neural Network
class myCNN1D(nn.Module):
    def __init__(self):
        #使用super()方法调用基类的构造器，即nn.Module.__init__(self)
        super(myCNN1D,self).__init__()
        # 1 input image channel ,6 output channels,5x5 square convolution kernel
        self.conv1=nn.Conv1d(1,6,5)
        # 6 input channl,16 output channels,5x5 square convolution kernel
        self.conv2=nn.Conv1d(6,16,5)
        # an affine operation:y=Wx+b
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

    def forward(self,x):
        # x是网络的输入，然后将x前向传播，最后得到输出
        # 下面两句定义了两个2x2的池化层
        x=F.max_pool1d(F.relu(self.conv1(x)),(2,2))
        # if the size is square you can only specify a single number
        x=F.max_pool1d(F.relu(self.conv2(x)),2)
        x=self.fc3(x)
        return x

    def num_flat_features(self,x):
        size=x.size()[1:] # all dimensions except the batch dimension
        num_features=1
        for s in size:
            num_features*=s
        return num_features

NUM_EPOCH = 10
NUM_CLASSES = 2
LEARNING_RATE = 0.001

net_SGD         = myCNN1D()
net_Momentum    = myCNN1D()
net_RMSprop     = myCNN1D()
net_Adam        = myCNN1D()
nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]


opt_SGD         = torch.optim.SGD(net_SGD.parameters(), lr=LEARNING_RATE,momentum=0.001)
opt_Momentum    = torch.optim.SGD(net_Momentum.parameters(), lr=LEARNING_RATE, momentum=0.8)
opt_RMSprop     = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LEARNING_RATE, alpha=0.9)
opt_Adam        = torch.optim.Adam(net_Adam.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.99))
optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

myFirstNNbyTouch = myCNN1D()
print(myFirstNNbyTouch)


loss_func = torch.nn.CrossEntropyLoss()
losses_his = [[], [], [], []]



for net, opt, l_his in zip(nets, optimizers, losses_his):
    for epoch in range(NUM_EPOCH):
        for step, (x, y) in enumerate(torch_loader):
            x, y = Variable(x), Variable(y)
            y = y.squeeze(1)
            output = net(x)              # get output for every net
            loss = loss_func(output, y)  # compute loss for every net
            opt.zero_grad()                # clear gradients for next train
            loss.backward()                # backpropagation, compute gradients
            opt.step()                     # apply gradients
            if epoch%1==0:
                l_his.append(loss.data.numpy())     # loss recoder
                print("optimizers: {0}, Epoch: {1}, Step: {2}, loss: {3}".format(opt, epoch, step, float(loss)))


labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
for i, l_his in enumerate(losses_his):
    plt.plot(l_his, label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.xlim((0,1000))
    plt.ylim((0,4))
    plt.show()


'''
myInput = torch.randn(2,2,5)
print("myInput is {}".format(myInput))

myOutput = myModel(myInput)
print("myOutput is {}".format(myOutput))
print(myModel.weight)
print(myModel.bias)

print(torch.is_tensor(myOutput))
print(torch.is_storage(myOutput))

y = torch.rand(5,3)
one = torch.ones(5,3)

print("y + one = {}".format(y + one))
print("torch.add(y,one) = {}".format(y + one))
print("y + one = {}".format(y + one))
print("y + one = {}".format(y + one))
'''