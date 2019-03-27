import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import os

myInput = torch.randn(200, 1, 39)
print("The size of myInput is {}".format(myInput.size()))
#print(myInput)

m1 = nn.Conv1d(1, 10, 5, stride=1)
print(m1)
# batch size: 20,  number of channels: 20, length of samples: 50 
myOutput1 = m1(myInput)
print("The size of myOutput1 is {}".format(myOutput1.size()))
#print(myOutput)
m2 = nn.Conv1d(10, 20, 5, stride=1)
myOutput2 = m2(myOutput1)
print("The size of myOutput2 is {}".format(myOutput2.size()))
#print(myOutput)

