import torch
import torch.nn as nn
from torch.nn import init
'''
net = nn.Sequential(
    nn.Linear(4,3),
    nn.ReLU(),
    nn.Linear(3,1)
)
#print(net)
X = torch.rand(2,4)
Y = net(X).sum()
'''

##########访问模型参数##########
'''
print(type(net.named_parameters()))
for name,param in net.named_parameters():
    print(name,param.size())
'''
#如果一个Tensor是Parameter那么会被自动添加到模型的参数列表里
class MyModel(nn.Module):
    def __init__(self,**kwargs):
        super(MyModel,self).__init__(**kwargs)
        self.weight1 = nn.Parameter(torch.rand(20,20))
        self.weight2 = torch.rand(20,20)
    def forward(self,x):
        pass
n = MyModel()
'''
for name,param in n.named_parameters():
    print(name)
'''


##########初始化模型参数##########
'''
for name,param in net.named_parameters():
    if 'bias' in name:
        init.constant_(param,val=0)
        print(name,param.data)
'''
#只对某个特定参数初始化，可以调用Parameter类的initialize函数


##########自定义初始化方法##########
def init_wight_(tensor):
    with torch.no_grad():
        tensor.uniform_(-10,10)
        tensor *= (tensor.abs()>=5).float()
'''
for name,param in net.named_parameters():
    if 'weight' in name:
        #print(name, param.data)
        init_wight_(param)
        print(name,param.data)
'''
#改变参数的data同时不影响梯度
'''
for name, param in net.named_parameters():
    if 'bias' in name:
        param.data += 1
        print(name, param.data)
'''


##########共享模型参数##########
linear = nn.Linear(1,1,bias=False)
net = nn.Sequential(linear,linear)
print(net)
for name, param in net.named_parameters():
    init.constant_(param, val=3)
    print(name, param.data)
#内存中其实为一个对象
print(id(net[0]) == id(net[1]))
print(id(net[0].weight) == id(net[1].weight))
#模型参数里包含了梯度，所以在反向传播计算时，这些共享的参数的梯度是累加的
x = torch.ones(1, 1)
y = net(x).sum()
print(y)
y.backward()
print(net[0].weight.grad) # 单次梯度是3，两次所以就是6
