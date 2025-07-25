from torch.utils import data
from torchvision import datasets, transforms
import torchvision
import d2l
import datetime

batch_size = 256

def get_dataloader_workers():
    """使用4个进程来读取数据"""
    return 4

trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root='../data', train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root='../data', train=False, transform=trans, download=True)

# 开始读取数据
time_start = datetime.datetime.now()
print(f'开始读取数据: {time_start}')
train_iter = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True,
                              num_workers=get_dataloader_workers())
# 读取数据结束
time_end = datetime.datetime.now()
print(f'结束读取数据: {time_end}')
print(f'每个epoch在{len(train_iter)}个批次上的时间: {time_end - time_start}')   

