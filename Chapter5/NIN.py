import time
import torch
from torch import nn,optim
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def nin_block(in_channels,out_channels,kernel_size,stride,padding):
    blk = nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),
        nn.ReLU(),
        nn.Conv2d(out_channels,out_channels,kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
    )
    return blk

#NIN模型
net = nn.Sequential(
    nin_block(1,96,kernel_size=11,stride=4,padding=0),
    nn.MaxPool2d(kernel_size=3,stride=2),
    nin_block(96,256,kernel_size=5,stride=1,padding=2),
    nn.MaxPool2d(kernel_size=3,stride=2),
    nin_block(256, 384, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Dropout(0.5),
    nin_block(384, 10, kernel_size=3, stride=1, padding=1),
    d2l.GlobalAvgPool2d(),
    d2l.FlattenLayer()
)
'''
#每一层输出形状测试
X = torch.rand(1, 1, 224, 224)
for name, blk in net.named_children():
    X = blk(X)
    print(name, 'output shape: ', X.shape)
'''

#获取数据训练模型
batch_size = 128
#如出现“out of memory”的报错信息，可减小batch_size或resize
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size,resize=224)
lr, num_epochs = 0.002, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer,device, num_epochs)

'''
training on  cpu
epoch 1, loss 1.4068, train acc 0.545, test acc 0.751, time 3226.8 sec
epoch 2, loss 0.7333, train acc 0.781, test acc 0.778, time 3510.4 sec
epoch 3, loss 0.5988, train acc 0.810, test acc 0.831, time 3147.9 sec
epoch 4, loss 0.4213, train acc 0.846, test acc 0.852, time 3193.2 sec
epoch 5, loss 0.3864, train acc 0.856, test acc 0.859, time 3144.6 sec
'''