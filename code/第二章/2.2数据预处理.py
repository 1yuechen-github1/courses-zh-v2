import os

import torch

# 举一个例子，我们首先创建一个人工数据集，并存储在CSV（逗号分隔值）文件
# ../data/house_tiny.csv中。
# 以其他格式存储的数据也可以通过类似的方式进行处理。
# 下面我们将数据集按行写入CSV文件中。
os.makedirs(os.path.join('..','data'),exist_ok=True)
data_path=os.path.join('..','data','house_tiny.csv')
with open(data_path,'w') as f:
    f.write('NumRooms,Alley,Price\n')
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')

# 要从创建的CSV文件中加载原始数据集，我们导入pandas包并调用read_csv函数。
# 该数据集有四行三列。
# 其中每行描述了房间数量（“NumRooms”）、巷子类型（“Alley”）和房屋价格（“Price”）。
import pandas as pd
data=pd.read_csv(data_path)
print(data)

# 注意，“NaN”项代表缺失值。 为了处理缺失的数据，典型的方法包括插值法和删除法，
# 其中插值法用一个替代值弥补缺失值，
# 而删除法则直接忽略缺失值。 在这里，我们将考虑插值法。
#
# 通过位置索引iloc，我们将data分成inputs和outputs， 其中前者为data的前两列，
# 而后者为data的最后一列。 对于inputs中缺少的数值，我们用同一列的均值替换“NaN”项。
inputs,outputs=data.iloc[:,0:2],data.iloc[:,2]
# inputs.mean()返回inputs中每个列的均值
# fillna函数将inputs中的所有nan替换为inputs中每一列的均值
# 只对数值列计算均值并填充
inputs['NumRooms']=inputs['NumRooms'].fillna(inputs['NumRooms'].mean())
print(inputs)

# 对于inputs中的类别值或离散值，
# 我们将“NaN”视为一个类别。
# 由于“巷子类型”（“Alley”）列只接受两种类型的类别值“Pave”和“NaN”，
# pandas可以自动将此列转换为两列“Alley_Pave”和“Alley_nan”。
# 巷子类型为“Pave”的行会将“Alley_Pave”的值设置为1，
# “Alley_nan”的值设置为0。
# 缺少巷子类型的行会将“Alley_Pave”和“Alley_nan”分别设置为0和1。
# 我们通过get_dummies函数来实现这一转换。
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

# 最后，我们将inpurts, outputs转换为张量格式。
x = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
print(x)
print(y)
