from torch import nn
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
# import torch
from torch import nn


# 生成数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
# 生成特征
# features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# ...existing code...
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
labels = labels.reshape(-1, 1)  # 保证labels为二维张量
# ...existing code...

#  我们将features和labels作为API的参数传递，
# 并通过数据迭代器指定batch_size。 此外，
# 布尔值is_train表示是否希望数据迭代器对象在每个迭代周期内打乱数据。
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)  # 将特征和标签组合成数据集
    return data.DataLoader(dataset, batch_size, shuffle=is_train)  # 返回数据迭代器


batch_size = 10  # 设置批量大小
data_iter = load_array((features, labels), batch_size)  # 加载数据集
# 取出第一个批次的特征和数据
a = next(iter(data_iter))
print(a)

loss = nn.MSELoss()

# 在PyTorch中，全连接层在Linear类中定义。 
# 值得注意的是，我们将两个参数传递到nn.Linear中。 
# 第一个指定输入特征形状，即2，第二个指定输出特征形状，
# 输出特征形状为单个标量，因此为1。
net = nn.Sequential(nn.Linear(2,1))

# 正如我们在构造nn.Linear时指定输入和输出尺寸一样， 
# 现在我们能直接访问参数以设定它们的初始值。 
# 我们通过net[0]选择网络中的第一个图层， 
# 然后使用weight.data和bias.data方法访问参数。 
# 我们还可以使用替换方法normal_和fill_来重写参数值。
net[0].weight.data.normal_(0, 0.01)  # 初始化权重
net[0].bias.data.fill_(0)  # 初始化偏置
print(net[0].weight.data)  # 打印权重
print(net[0].bias.data)  # 打印偏置
trainer = torch.optim.SGD(net.parameters(), lr=0.03)  # 定义优化器



num_epochs = 3
for epoch in range(num_epochs):
  for X, y in data_iter:
    # 1.进入loss计算误差
    # 2.梯度清零
    # 3.反向传播
    l = loss(net(X), y)
    trainer.zero_grad()
    l.backward()
    trainer.step()
  # 打印损失
  l = loss(net(features), labels)
  print(f'epoch {epoch + 1}, loss {l.item():.4f}')

  # 下面我们比较生成数据集的真实参数和通过有限数据训练获得的模型参数。
  # 要访问参数，我们首先从net访问所需的层，然后读取该层的权重和偏置。 
  # 正如在从零开始实现中一样，
  # 我们估计得到的参数与生成数据的真实参数非常接近。
  w = net[0].weight.data
  print('w的误差估计：', true_w - w.reshape(true_w.shape))
  b = net[0].bias.data
  print('b的误差估计：', true_b - b)