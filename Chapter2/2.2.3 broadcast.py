import torch
#复制元素使tensor形状相同后再按照元素运算
x = torch.arange(1,3).view(1,2)
print(x)
y = torch.arange(1,4).view(3,1)
print(y)

print(x+y)