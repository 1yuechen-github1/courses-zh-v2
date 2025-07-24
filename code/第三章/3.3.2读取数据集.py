import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l


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