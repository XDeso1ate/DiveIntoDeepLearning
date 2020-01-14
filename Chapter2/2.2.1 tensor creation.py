import torch

#create uninitialized 5*3 tensor
x = torch.empty(5,3)
print(x)

#create 5*3 random initialized tensor
y = torch.rand(5,3)
print(y)

z = torch.zeros(5,3,dtype=torch.long)
print(z)

#create tensor according to data
w = torch.tensor([5.5,3])
print(w)

#create tensor according to another tensor
u = torch.rand_like(x,dtype=torch.float)
print(u)

print(x.size())
print(x.shape)