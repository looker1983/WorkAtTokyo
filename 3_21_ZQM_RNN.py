import torch
from torch import nn
import os
import numpy as np
import torch.utils.data as Data

# Hyper Parameters
EPOCH = 100               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
TIME_STEP = 1000          # rnn time step / image height
INPUT_SIZE = 39        # rnn input size / image width
LR = 0.01               # learning rate
trainDIR = "./training data set"
testDIR = "./test data set"


# 制作训练集
trainFILES = os.listdir(trainDIR)

trainMatrix = np.zeros(shape=(1,TIME_STEP,40))
for f in trainFILES:
    if f[0]=='.':
        continue
    tempFILE = trainDIR + '/' + f
    tempMatrix = np.load(tempFILE)
    tempMatrix = tempMatrix[np.newaxis,0:TIME_STEP,:]
    trainMatrix = np.vstack((trainMatrix,tempMatrix))

x_train = trainMatrix[:,:,1:]
y_train = trainMatrix[:,0,0]
y_train = torch.from_numpy(y_train).float()
x_train = torch.from_numpy(x_train).float()

trainDataSet = Data.TensorDataset(x_train, y_train)

train_loader = Data.DataLoader(
    dataset=trainDataSet,  # torch TensorDataset format
    batch_size=BATCH_SIZE,  # mini batch size
    shuffle=True,  # 要不要打乱数据 (打乱比较好)
)

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

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# training
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):        # gives batch data
        b_x = b_x.view(-1, TIME_STEP, 39)              # reshape x to (batch, time_step, input_size)
        output = rnn(b_x)                               # rnn output
        b_y = b_y.type(torch.LongTensor)
        loss = loss_func(output, b_y)                   # cross entropy loss
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients
        print('Epoch [{}/{}] Loss: {:.8f}'.format(epoch + 1, EPOCH, loss.data))



