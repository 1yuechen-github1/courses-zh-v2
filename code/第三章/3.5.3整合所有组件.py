
# 现在我们定义load_data_fashion_mnist函数，
# 用于获取和读取Fashion-NIST数据集。 这个函数返回训练集和验证集的数据迭代器。 
# 此外，这个函数还接受一个可选参数resize，用来将图像大小调整为另一种形状。



# from matplotlib import transforms
import torchvision
from torch.utils import data
import numpy as np
from torchvision import transforms


def get_dataloader_workers():
    """使用4个进程来读取数据"""
    return 4

def load_data_fashion_mnist(batch_size, resize=None):
    """下载Fashion-MNIST数据集并返回数据迭代器"""
    # 把图像数据转化为 tensor张量
    # trans = transforms.ToTensor()

    
    
    if resize:
      # trans.insert(0, transforms.Resize(resize))

        # 正确写法
      trans = transforms.Compose([
          transforms.Resize(resize),  # 先调整尺寸
          transforms.ToTensor()       # 再转为张量
    ])


      
    trans = transforms.Compose([trans])
    minist_train = torchvision.datasets.FashionMNIST(
        root='../data', train=True, transform=trans, download=True)
    minist_test = torchvision.datasets.FashionMNIST(
        root='../data', train=False, transform=trans, download=True)
    return (data.DataLoader(minist_train, batch_size=batch_size, shuffle=True,
                              num_workers=get_dataloader_workers()),

            data.DataLoader(minist_test, batch_size=batch_size, shuffle=False,
                             num_workers=get_dataloader_workers()))


if __name__ == '__main__':
    # 测试load_data_fashion_mnist函数
    # 这里我们将图像大小调整为64x64

  train_iter, test_iter = load_data_fashion_mnist(32, resize=(64, 64))
  for X, y in train_iter:
      print(X.shape, y.shape)
      break
