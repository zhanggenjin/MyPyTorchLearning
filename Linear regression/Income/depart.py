# 模型的分解写法
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""读取数据集"""
data = pd.read_csv('./dataset/Income1.csv')
# data.info()
# print(data)

"""画成散点图，观察有何关系"""
# plt.scatter(data.Education, data.Income)
# plt.xlabel('Education')
# plt.ylabel('Income')
# plt.show()

# 引入nn模块
from torch import nn

"""数据预处理"""
# reshape(-1, 1)中的-1代表自动计算，1代表每个中的数据是一个
X = data.Education.values.reshape(-1, 1).astype(np.float32)
# 将x转化成Tensor类型
X = torch.from_numpy(X)
Y = data.Income.values.reshape(-1, 1).astype(np.float32)
Y = torch.from_numpy(Y)
# 创建随机的张量权重
w = torch.randn(1, requires_grad=True)
# 初始化偏置
b = torch.zeros(1, requires_grad=True)
# 模型公式：w*x + b
learning_rate = 0.0001
for epoch in range(5000):
    for x, y in zip(X, Y):
        # 手写线性模型
        y_pred = torch.matmul(x, w) + b
        loss = ((y - y_pred)**2).mean()
        # 梯度归零，否则梯度会累加
        if not w.grad is None:
            w.grad.data.zero_()
        if not b.grad is None:
            b.grad.data.zero_()
        # 计算梯度
        loss.backward()
        # 优化模型参数不需要跟踪梯度
        with torch.no_grad():
            w.data -= w.grad.data*learning_rate
            b.data -= b.grad.data*learning_rate

# 画出模型的曲线
plt.scatter(data.Education, data.Income)
plt.plot(X.numpy(), (X*w + b).data.numpy(), color='r')
plt.show()

