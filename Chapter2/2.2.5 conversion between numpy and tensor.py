import torch
import numpy as np

#tensor to numpy
#numpy()————share memory
a = torch.ones(5)
b = a.numpy()
print(a,b)

a += 1
print(a, b)
b += 1
print(a, b)

#numpy to tensor
#tensor()————copy data,slower,more space
#from_numpy()————share memory
a = np.ones(5)
b = torch.from_numpy(a)
print(a, b)

a += 1
print(a, b)
b += 1
print(a, b)


c = torch.tensor(a)
a += 1
print(a, c)