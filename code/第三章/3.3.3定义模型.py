from torch import nn


# 在PyTorch中，全连接层在Linear类中定义。 
# 值得注意的是，我们将两个参数传递到nn.Linear中。 
# 第一个指定输入特征形状，即2，第二个指定输出特征形状，
# 输出特征形状为单个标量，因此为1。
net = nn.Sequential(nn.Linear(2,1))