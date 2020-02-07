import time
import torch
from torch import nn,optim
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def vgg_block(num_convs,in_channels,out_channels):
    blk = []
    for i in range(num_convs):
        if i==0:
            blk.append(
                nn.Conv2d(in_channels,out_channels,
                        kernel_size=3,padding=1))
        else:
            blk.append(
                nn.Conv2d(out_channels,out_channels,
                          kernel_size=3,padding=1))
    blk.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*blk)

#VGG network
#5个卷积块，前两块单卷积层，后三块双卷积层
#第一块输入输出通道均为1，此后输出通道数翻倍直至512
conv_arch = ((1,1,64),(1,64,128),
             (2,128,256),(2,256,512),(2,512,512))
fc_features = 512 * 7 * 7 # c * w * h
fc_hidden_units = 4096
def vgg(conv_arch,fc_features,fc_hidden_units=4096):
    net = nn.Sequential()
    #Convolution layers
    for i,(num_convs,in_channels,out_channels) in enumerate(conv_arch):
        net.add_module("vgg_block_"+str(i+1),
                       vgg_block(num_convs,in_channels,out_channels))
    net.add_module("fc",
                   nn.Sequential(
                       d2l.FlattenLayer(),
                       nn.Linear(fc_features,fc_hidden_units),
                       nn.ReLU(),
                       nn.Dropout(0.5),
                       nn.Linear(fc_hidden_units,fc_hidden_units),
                       nn.ReLU(),
                       nn.Dropout(0.5),
                       nn.Linear(fc_hidden_units,10)
                   ))
    return net
'''
#构造单通道数据样本观察每一层输出形状
net = vgg(conv_arch, fc_features, fc_hidden_units)
X = torch.rand(1, 1, 224, 224)
# named_children获取一级子模块及其名字(named_modules会返回所有子模块,包括子模块的子模块)
for name, blk in net.named_children():
    X = blk(X)
    print(name, 'output shape: ', X.shape)
'''

#获取数据训练模型
ratio = 8
small_conv_arch = [(1, 1, 64//ratio), (1, 64//ratio, 128//ratio),
            (2, 128//ratio, 256//ratio),(2, 256//ratio, 512//ratio), (2, 512//ratio,512//ratio)]
net = vgg(small_conv_arch, fc_features // ratio, fc_hidden_units //ratio)
#print(net)

batch_size = 64
# 如出现“out of memory”的报错信息，可减小batch_size或resize
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size,resize=224)
lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer,device,num_epochs)

'''
epoch 1, loss 0.5111, train acc 0.813, test acc 0.875, time 1679.6 sec
epoch 2, loss 0.3344, train acc 0.879, test acc 0.892, time 1671.3 sec
epoch 3, loss 0.2876, train acc 0.894, test acc 0.899, time 1677.5 sec
epoch 4, loss 0.2631, train acc 0.905, test acc 0.900, time 1673.8 sec
epoch 5, loss 0.2446, train acc 0.911, test acc 0.908, time 1666.1 sec
'''
