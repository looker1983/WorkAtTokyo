import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import os



input=torch.ones(2,1,4)
input=Variable(input)
x=torch.nn.Conv1d(in_channels=1,out_channels=4,kernel_size=2,groups=1)
out=x(input)
print(out)
print("Parameters are as following {}".format(list(x.parameters())))

'''
myOut = torch.rand(4,3)
print(myOut)
print(myOut.size(0))

myOut = myOut.view(-1,4)
print(myOut)
'''