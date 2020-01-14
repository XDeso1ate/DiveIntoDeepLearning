import torch

#ADD
x = torch.rand(5,3)
y = torch.rand(5,3)
print(x+y)

#指定输出
res = torch.empty(5,3)
torch.add(x,y,out=res)
print(res)

#inplace
y.add_(x)
print(y)


#INDEX
#share memory with original data,
#once one is modified,the other is also modified
y = x[0,:]
y += 1
print(y)
print(x[0,:])

#OTHER SELECTION FUNCS
#index_select(input,dim,index)
#masked_select(input,mask)
#non_zero(input)
#gather(input,dim,index)


#SHAPE CHANGE
#the return tensor shares memory with original tensor
y = x.view(15)
z = x.view(-1,5)
print(x.size(),y.size(),z.size())
x+=1
print(x)
print(y)

#view after clone
#seperated memory
x_cp = x.clone().view(15)
x -= 1
print(x)
print(x_cp)

#matrix operations
#trace,diag,mm,bmm,dot,cross,inverse...