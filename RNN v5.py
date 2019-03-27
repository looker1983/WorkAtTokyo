import torch
from torch import nn
import os
import numpy as np
import torch.utils.data as Data
from torch.autograd import Variable

# Hyper Parameters
EPOCH = 1            # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
TIME_STEP = 500          # rnn time step / image height
INPUT_SIZE = 39        # rnn input size / image width
LR = 0.01               # learning rate
train_npy_FILE = "D:/TestWithMFCC39/NN for 0Health-1Dementia/npy/totalNPY for train.npy"
testDIR = "D:/TestWithMFCC39/NN for 0Health-1Dementia/npy/npy for test"
modelDIR = "D:/TestWithMFCC39/NN for 0Health-1Dementia/npy"
modelFILE = "1D-CNN Modelfor 0Health-1Dementia with 39MFCC of npy form by 333 samples.pth"

#print the seperating line
def print_line(char,string):
    print(char*33,string,char*32)

# prepare the training dataset
trainDS = np.load(train_npy_FILE)
print("The shape of trainDS is {}".format(trainDS.shape))
num_of_line = np.shape(trainDS)[0]
print("The number of trainDS's line is {}".format(num_of_line))
win_trainDS = np.zeros([1, TIME_STEP, INPUT_SIZE+1])
for i in range(int(num_of_line/1000)):
    window = trainDS[np.newaxis, i * 1000:(i + 1) * 1000, :]
    win_trainDS = np.vstack((win_trainDS, window))
print("The shape of win_trainDS is {}".format(win_trainDS.shape))


x_train = win_trainDS[:,:,1:]
y_train = win_trainDS[:,0,0]
y_train = torch.from_numpy(y_train).float()
x_train = torch.from_numpy(x_train).float()

trainDataSet = Data.TensorDataset(x_train, y_train)

trainLoader = Data.DataLoader(
    dataset=trainDataSet,  # torch TensorDataset format
    batch_size=BATCH_SIZE,  # mini batch size
    shuffle=True,  # shuffle the data
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
    for step, (b_x, b_y) in enumerate(trainLoader):        # gives batch data
        b_x = b_x.view(-1, TIME_STEP, 39)              # reshape x to (batch, time_step, input_size)
        output = rnn(b_x)                               # rnn output
        b_y = b_y.type(torch.LongTensor)
        loss = loss_func(output, b_y)                   # cross entropy loss
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients
        print('Epoch [{}/{}] Loss: {:.8f}'.format(epoch + 1, EPOCH, loss.data))

# Test the Model
print_line('*', 'Testing Begin')

testFILES = os.listdir(testDIR)

correct = 0
in_all_file_total = 0
for f in testFILES:
    if f[0] == '.':
        continue
    correct = 0
    try:
        in_all_file_total += 1
        is_sick = 0
        in_one_file_total = 0
        testDS = np.load(testDIR + '/' + f)
        print("The shape of testDS is {}".format(testDS.shape))
        num_of_line = np.shape(testDS)[0]
        print("The number of testDS's line is {}".format(num_of_line))
        win_testDS = np.zeros([1, TIME_STEP, INPUT_SIZE + 1])
        for i in range(int(num_of_line / 1000)):
            window = testDS[np.newaxis, i * 1000:(i + 1) * 1000, :]
            win_testDS = np.vstack((win_testDS, window))
        print("The shape of win_testDS is {}".format(win_testDS.shape))

        x_test = win_testDS[:, :, 1:]
        x_test = torch.from_numpy(x_test).float()
        label = win_testDS[1, 0, 0] #


        for b_x in x_test:
            in_one_file_total += 1
            b_x = b_x.view(-1, TIME_STEP, 39)  # reshape x to (batch, time_step, input_size)
            output = rnn(b_x)  # rnn output
            if output[0,0]<output[0,1]:
                is_sick += 1
        sick_prob = 100 * is_sick / in_one_file_total
        print('This guy may be {} percentage in sick.'.format(sick_prob))

        if sick_prob > 0.5:
            predicted = 1
        else:
            predicted = 0
        if (predicted - label)**2<0.0001:
            correct += 1

    except ZeroDivisionError:
        print("division by zero: in_one_file_total of file {} is equal to 0.".format(f))
    except TypeError:
        print("There is a TypeError with file {}.".format(f))

    try:
        print('Test Accuracy of for file {0} is {1}%.'.format(f, 100 * correct / in_one_file_total))
    except ZeroDivisionError:
        print("division by zero: in_one_file_total is equal to 0.")    
    