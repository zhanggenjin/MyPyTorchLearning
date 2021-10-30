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

"""创建模型"""
# Linear(1, 1):输入的特征是1，输出的特征是1
model = nn.Linear(1, 1)   # w*input + b 等价于 model(input)
# 损失函数
loss_fn = nn.MSELoss()
# 优化器
opt = torch.optim.SGD(params=model.parameters(), lr=0.0001)

# 模型的训练
for epoch in range(5000):
    for x, y in zip(X, Y):
        y_pred = model(x)           # 使用模型预测
        loss = loss_fn(y, y_pred)   # 根据模型预测结果计算损失
        opt.zero_grad()             # 把变量的梯度清零
        loss.backward()             # 求解梯度
        opt.step()                  # 优化模型参数

# 画出模型的曲线
plt.scatter(data.Education, data.Income)
plt.plot(X.numpy(), model(X).data.numpy(), color='r')
plt.show()
