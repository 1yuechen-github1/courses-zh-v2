import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l


# 生成数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
# 生成特征
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

