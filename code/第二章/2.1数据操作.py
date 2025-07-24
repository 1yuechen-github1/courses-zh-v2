import torch

# 首先，我们可以使用 arange 创建一个行向量 x。这个行向量包含以0开始的前12个整数
x = torch.arange(12)
print(x)

# 使用shape 访问张量的形状
print(x.shape)


#要想改变一个张量的形状而不改变元素数量和元素值，可以调用reshape函数
X = x.reshape(3, 4)
print(X)

#有时，我们希望使用全0、全1、其他常量，或者从特定分布中随机采样的数字来初始化矩阵。
# 我们可以创建一个形状为（2,3,4）的张量，其中所有元素都设置为0
zero_tensor = torch.zeros((2, 3, 4))
zero_tensor = torch.ones((2, 3, 4))
print(zero_tensor)

# 构建一个符合高斯分布的随机采样
# 均值为0 标准差为1
gauss_tensor = torch.randn((2, 3, 4))
print(gauss_tensor)

# 两个张量的加减运算
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])
print(x + y)
print(x - y)
print(x * y)
print(x / y)