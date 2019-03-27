import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import numpy as np
import torch.utils.data as Data

# Hyper Parameters
EPOCH = 10               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
TIME_STEP = 10000          # rnn time step / image height
INPUT_SIZE = 39        # rnn input size / image width
LR = 0.01               # learning rate
trainDIR = "C:/testData/very small test data set/training data set"
testDIR = "C:/testData/very small test data set/test data set"
modelDIR = "C:/testData/very small test data set"
modelFILE = "1D-RNNv5.pth"
modelPATH = modelDIR + '/' + modelFILE


# 制作训练集
trainFILES = os.listdir(trainDIR)

trainMatrix = np.zeros(shape=(1,TIME_STEP,40))
for f in trainFILES:
    if f[0]=='.':
        continue
    tempFILE = trainDIR + '/' + f
    tempMatrix = np.loadtxt(tempFILE)
    tempMatrix = tempMatrix[np.newaxis,0:TIME_STEP,:]
    trainMatrix = np.vstack((trainMatrix,tempMatrix))

x_train = trainMatrix[:,:,1:]
y_train = np.array([0,0,1])
y_train = torch.from_numpy(y_train).float()
x_train = torch.from_numpy(x_train).float()

trainDataSet = Data.TensorDataset(x_train, y_train)

train_loader = Data.DataLoader(
    dataset=trainDataSet,  # torch TensorDataset format
    batch_size=BATCH_SIZE,  # mini batch size
    shuffle=True,  # 要不要打乱数据
    # num_workers=1,  # 多线程来读数据
)





# 制作测试集合
testFILES = os.listdir(testDIR)

testMatrix = np.zeros(shape=(1,TIME_STEP,40))
for f in testFILES:
    if f[0]=='.':
        continue
    tempFILE = testDIR + '/' + f
    tempMatrix = np.loadtxt(tempFILE)
    tempMatrix = tempMatrix[np.newaxis,0:TIME_STEP,:]
    testMatrix = np.vstack((testMatrix,tempMatrix))

x_test = testMatrix[:,:,1:]
y_test = np.array([0,0,1])
x_test = torch.from_numpy(x_test).float()




class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=64,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(64, 2)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out


rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted
#
# training and testing
for epoch in range(EPOCH):
    print(epoch)
    for step, (b_x, b_y) in enumerate(train_loader):        # gives batch data
        b_x = b_x.view(-1, TIME_STEP, 39)              # reshape x to (batch, time_step, input_size)
        print(step)
        output = rnn(b_x)                               # rnn output
        b_y = b_y.type(torch.LongTensor)
        loss = loss_func(output, b_y)                   # cross entropy loss
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients

        if step % 5 == 0:
            test_output = rnn(x_test)                   # (samples, time_step, input_size)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == y_test).astype(int).sum()) / float(y_test.size)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

