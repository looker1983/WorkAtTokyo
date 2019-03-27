import torch
import numpy as np

print("Hello world!")

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

x = torch.empty(5,3)
print(x)


y = torch.rand(5,3)
print(y)

result = torch.Tensor(5,3)
torch.add(x,y,out=result)
print("Result is {}".format(result))

a = torch.ones(5)
print("Matrix a is {}".format(a))
b = a.numpy()
print("Matrix b is {}".format(b))

a.add_(1)
print("Matrix new a is {}".format(a))
print("Matrix new b is {}".format(b))

a = torch.Tensor([[1,1,],[1,2],[2,3]])
print(a)
print(a.size())