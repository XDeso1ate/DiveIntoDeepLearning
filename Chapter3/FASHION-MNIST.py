import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
sys.path.append("..") # 为了了导⼊入上层⽬目录的d2lzh_pytorch
import d2lzh_pytorch as d2l

mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST',train=True,download=True,
                                       transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST',train=False,download=True,
                                       transform=transforms.ToTensor())

#trainset:6000 images * 10
#testset:1000 images * 10
#print(type(mnist_train))
#print(len(mnist_train),len(mnist_test))

#访问任意样本
#feature尺寸 C*H*W 1*28*28
#feature,label = mnist_train[0]
#print(feature.shape,label)

X,y = [],[]
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
d2l.show_fashion_mnist(X,d2l.get_fashion_mnist_labels(y))

