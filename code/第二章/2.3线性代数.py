import torch

# 标量的运算
# 仅包含一个数值被称为标量（scalar）。
x = torch.tensor(3)
y = torch.tensor(4)
print(x + y)
print(x * y)
print(x ** y)
print(x // y)
print(x % y)

# 向量的运算
x = torch.arange(4)
print(x)
print(x[3])
# 向量的长度
print(len(x))
# 向量的形状
print(x.shape)

# 矩阵
A = torch.arange(20).reshape(5, 4)
print(A)
# 矩阵的转置
print(A.T)
# 矩阵的转置
B = torch.tensor([[1,2,3],[2,0,4],[3,4,5]])
print(B)
B == B.T
print(B)

# 高维张量
X = torch.arange(24).reshape(2, 3, 4)
print(X)
# 张量的基本性质
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone() # 通过分配新内存，将A的一个副本分配给B
print(A, A+B)
# 张量 + 一个标量
a = 2
X = torch.arange(24).reshape(2, 3, 4)
print(X + a)
print((X * a).shape)
# 降维
x = torch.arange(4,dtype=torch.float32)
print(x,x.sum())
print(A.shape)
print(A.sum())
