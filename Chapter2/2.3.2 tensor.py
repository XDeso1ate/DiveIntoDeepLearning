import torch

#requires_grad设置为True，将开始追踪在tensor上的操作
#完成计算后调用backward完成所有梯度计算，梯度将积累到grad属性中
#注意在 y.backward() 时，如果 y 是标量，则不需要为 backward() 传⼊任何参数；
#否则，需要传⼊一个与 y同形的 Tensor
#不想继续追踪用detach分离
#with torch.no_grad()将不想被追踪的操作代码块包裹起来
#tensor和function相互结合构成记录整个计算过程的DAG，每个tensor的grad_Fn属性即创建tensor
#的function

x = torch.ones(2,2,requires_grad=True)
print(x)
print(x.grad_fn)

y = x + 2
print(y)
print(y.grad_fn)

z = y * y * 3
out = z.mean()
print(z,out)

out.backward()
print(x.grad)
#grad在反向传播过程中是累加的，这意味着每一次运行反向传播，
#梯度都会累加之前的梯度，所以一般在反向传播之前需把梯度清零。
out3 = x.sum()
x.grad.data.zero_()
out3.backward()
print(x.grad)



a = torch.tensor(1.0, requires_grad=True)
y1 = a ** 2
with torch.no_grad():
    y2 = a ** 3
y3 = y1 + y2
print(a.requires_grad)
print(y1, y1.requires_grad) # True
print(y2, y2.requires_grad) # False
print(y3, y3.requires_grad) # True

#如果想要修改 tensor 的数值，但是又不希望被 autograd 记录（即不会影响反向传播）
#那么可以对 tensor.data 进行操作。
x = torch.ones(1,requires_grad=True)
print(x.data)
print(x.data.requires_grad)
y = 2 * x
x.data *= 100
y.backward()
print(x)
print(x.grad)