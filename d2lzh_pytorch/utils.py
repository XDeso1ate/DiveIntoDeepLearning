import collections
import math
import os
import random
import sys
import tarfile
import time
import json
import zipfile
import IPython
from tqdm import tqdm
from PIL import Image
from collections import namedtuple

from IPython import display
from matplotlib import pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import numpy as np


# ###################### 3.2 ############################
def use_svg_display():
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5,2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # 最后一次可能不足一个batch
        yield  features.index_select(0, j), labels.index_select(0, j)

def linreg(X, w, b):
    return torch.mm(X, w) + b

def squared_loss(y_hat, y):
    # 注意这里返回的是向量, 另外, pytorch里的MSELoss并没有除以 2
    return ((y_hat - y.view(y_hat.size())) ** 2) / 2

def sgd(params, lr, batch_size):
    # 为了和原书保持一致，这里除以了batch_size，但是应该是不用除的，因为一般用PyTorch计算loss时就默认已经
    # 沿batch维求了平均了。
    for param in params:
        param.data -= lr * param.grad / batch_size # 注意这里更改param时用的param.data

# ###################### 3.5 ############################
def get_fashion_mnist_labels(labels):
    text_labels=['t-shirt','trouser','pullover','dress',
                 'coat','sandal','shirt','sneaker','bag','ankleboot']
    return [text_labels[int(label)] for label in labels]

def show_fashion_mnist(images, labels):
    use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _, axs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(axs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

def load_data_fashion_mnist(batch_size,root="~/Datasets/FashionMNIST"):
    transform = transforms.ToTensor();
    mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST',train=True,download=True,
                                       transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST',train=False,download=True,
                                       transform=transforms.ToTensor())
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4

    train_iter = torch.utils.data.DataLoader(mnist_train,batch_size=batch_size,
                                    shuffle=True,num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test,batch_size=batch_size,
                                    shuffle=False,num_workers = num_workers)
    return train_iter,test_iter

def train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,
             params=None,lr=None,optimizer = None):
    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum = 0.0,0.0
        n = 0
        for X,y in train_iter:
            y_hat = net(X)
            l = loss(y_hat,y).sum()
            
            #梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            if optimizer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step()
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
                % (epoch + 1, train_l_sum / n, train_acc_sum / n,test_acc))
        
#计算分类准确率
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n