from torch import nn


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