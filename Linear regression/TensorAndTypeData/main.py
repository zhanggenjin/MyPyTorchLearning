import torch

# 生成两行三列的随机张量
# 例如生成如下张量:(0-1之间的均匀分布)
# tensor([[0.0307, 0.8814, 0.3980],
#         [0.6931, 0.3275, 0.2393]])
x1 = torch.rand(2, 3)
print(x1)

# 生成三行四列的随机张量
# 例如生成如下张量:(0-1之间的正态分布)
# tensor([[-0.3822, -0.3799, -0.3756, -0.8449],
#         [ 1.9751, -0.4837, -0.4094,  0.5569],
#         [ 0.8702, -0.7357,  0.4227,  1.2921]])
x2 = torch.randn(3, 4)
print(x2)

# 生成两行三列的全零的张量
x3 = torch.zeros(2, 3)
print(x3)

# 生成2*3*4的全1的张量
x4 = torch.ones(2, 3, 4)
print(x4)

# 查看张量的大小
print(x4.size())
# 查看张量某一维度的大小
print(x4.size(0))   # x4第零维的大小
# shape不能查看某个维上的大小
print(x4.shape)

# 生成一个[6, 2]张量，列表创建
x5 = torch.tensor(data=[6, 2], dtype=torch.float32)
print(x5)
print(x5.type())
# 数据类型转化为int64类型
x5 = x5.type(torch.int64)
print(x5)

"""Tensor与ndarray类型的转化"""
import numpy as np

a = np.random.randn(2, 3)
print(a)
# 从ndarray创建Tensor
x6 = torch.from_numpy(a)
print(x6)
# 张量转化为ndarray
# x6 = x6.numpy()
# print(x6)

x7 = torch.rand(2, 3)
print(x7)
# 两个张量相加，对应元素相加
# 和x8 = x6.add(x7)一样，并不会改变x6
# 而x6.add_(x7),和x6 = x6.add(x7)一样x6将会改变
x8 = x6 + x7
print(x8)
# 每个元素加3
x9 = x6 + 3
print(x9)

# 改变张量形状
x6 = x6.view(3, 2)
print(x6)
# 展平，-1表示自动计算根据后面的参数来，1表示每个维1个元素，shape(n, 1)
x6 = x6.view(-1, 1)
print(x6)

# 求均值
mean = x6.mean()
print(mean)
# 求和
sum = x6.sum()
print(sum)
# 转化为标量，返回数值
print(sum.item())

"""张量的自动微分
将Torch.Tensor属性.requires_grad设置为True,
pytorch将开始跟踪对此张量的所有操作。
完成计算后，可以调用.backward()并自动计算所有梯度。
该张量的梯度将累加到.grad属性中。
"""
x10 = torch.ones(2, 2, requires_grad=True)
print(x10)
# 查看属性
print(x10.requires_grad)
# 张量的数据
print(x10.data)
# 张量的梯度
print(x10.grad)
# 张量的梯度函数
print(x10.grad_fn)

# 运算演示
y = x10 + 2
print(y)
print(y.grad_fn)
z = y*y + 3
out = z.mean()
print(z)
print(out)

# 算微分
# out = f(x)
# d(out)/dx
out.backward()
ward = x10.grad
print(ward)
# print(x10.data)

# 无法计算梯度,只有在这里面的时候才不能跟踪梯度
with torch.no_grad():
    print((x10**2).requires_grad)
# 可以跟踪计算
print((x10**2).requires_grad)
# detach()等价于torch.no_grad()
y1 = x10.detach()
print(y1)
print(y1.requires_grad)

a1 = torch.tensor([3, 2], dtype=torch.float32)
# 加下划线就地改变
print(a1.requires_grad_(True))
