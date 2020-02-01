import torch
import torch.nn as nn
##########读写Tensor##########
#创建tensor变量并存在pt文件里
x = torch.ones(3)
'''
torch.save(x,'x.pt')
#将文件中数据读回内存
x2 = torch.load('x.pt')
'''
#print(x2)
#存储和读取tensor列表和字典
y = torch.zeros(4)
'''
torch.save([x,y],'xy.pt')
xy1 = torch.load('xy.pt')
print(xy1)

torch.save({'x':x,'y':y},'xy_dict.pt')
xy2 = torch.load('xy_dict.pt')
print(xy2)
'''


##########读写模型##########
#state_dict
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(3, 2)
        self.act = nn.ReLU()
        self.output = nn.Linear(2, 1)
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)
net = MLP()
#print(net.state_dict())
optimizer = torch.optim.SGD(net.parameters(), lr=0.001,
momentum=0.9)
#print(optimizer.state_dict())
#保存和加载模型
#……
