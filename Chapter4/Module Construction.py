import torch
import torch.nn as nn


##########继承MODULE类构造模型##########
class MLP(nn.Module):
    #声明两个全连接层
    def __init__(self,**kwargs):
        super(MLP,self).__init__(**kwargs)
        self.hidden = nn.Linear(784,256)
        self.activ = nn.ReLU()
        self.output = nn.Linear(256,10)
    #定义前向计算
    def forward(self,x):
        a = self.activ(self.hidden(x))
        return self.output(a)

#实例化MLP类
#X = torch.rand(2,784)
#net = MLP()
#print(net)
#print(net(X))



##########MODULE的子类##########
#Sequential类
class MySequential(nn.Module):
    from collections import OrderedDict
    def __init__(self,*args):
        super(MySequential,self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key,module in args[0].items():
                self.add_module(key,module)
        else:
            for idx,module in enumerate(args):
                self.add_module(str(idx),module)
    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input
#用MySequential类实现MLP类，并进行一次前向计算
'''X = torch.rand(2,784)
net = MySequential(
    nn.Linear(784,256),
    nn.ReLU(),
    nn.Linear(256,10),
)'''
#print(net)
#print(net(X))

#ModuleList类
'''
net = nn.ModuleList([nn.Linear(784,256),nn.ReLU()])
net.append(nn.Linear(256,10))
print(net[-1])
print(net)
'''

#ModuleDict类
'''
net = nn.ModuleDict({
    'linear': nn.Linear(784, 256),
    'act': nn.ReLU(),
})
net['output'] = nn.Linear(256, 10) # 添加
print(net['linear']) # 访问
print(net.output)
print(net)
'''


##########构造复杂的模型##########
class FancyMLP(nn.Module):
    def __init__(self,**kwargs):
        super(FancyMLP,self).__init__(**kwargs)
        self.rand_weight = torch.rand((20,20),requires_grad=True)
        self.linear = nn.Linear(20,20)
    def forward(self,x):
        x = self.linear(x)
        x = nn.functional.relu(
            torch.mm(x,self.rand_weight.data) + 1
        )
        #复用全连接层。等价于两个全连接层共享参数
        x = self.linear(x)
        while x.norm().item() > 1:
            x /= 2
        if x.norm().item() < 0.8:
            x *= 10
        return x.sum()
'''
X = torch.rand(2, 20)
net = FancyMLP()
print(net)
print(net(X))
'''

class NestMLP(nn.Module):
    def __init__(self,**kwargs):
        super(NestMLP,self).__init__(**kwargs)
        self.net = nn.Sequential(nn.Linear(40,30),nn.ReLU())
    def forward(self,x):
        return self.net(x)
'''
net = nn.Sequential(NestMLP(), nn.Linear(30, 20), FancyMLP())
X = torch.rand(2, 40)
print(net)
print(net(X))
'''