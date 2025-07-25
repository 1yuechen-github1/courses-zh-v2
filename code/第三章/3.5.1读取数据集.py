import torch
from torchvision import transforms
import torchvision
import d2l
from torch.utils import data
from matplotlib import pyplot as plt

# 通过ToTensor实例将图像数据从PIL类型变换
# 成32位浮点数格式，
# 并除以255使得所有像素的数值均在0～1之间


trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root='../data', train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root='../data', train=False, transform=trans, download=True)


print(len(mnist_train), len(mnist_test))

print(mnist_train[0][0].shape)  # 查看第一个图像的形状

def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = [
        'T恤/上衣', '裤子', '套头衫', '连衣裙', '外套',
        '凉鞋', '衬衫', '运动鞋', '包', '短靴'
    ]
    return [text_labels[int(i)] for i in labels]

def show_images(imags, num_rows, num_cols, titles=None,scale=1.5):
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax,img) in enumerate(zip(axes, imags)):
        if torch.is_tensor(img):
            # 图片张量
            # ax.imshow(img.numpy())
            ax.imshow(img.numpy().squeeze(), cmap='gray')
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

# iter：创建一个迭代器，可以逐批次读取数据
# next：取出迭代器的第一个批次数据
# 返回结果是一个元组（X, y）， X是本批次的18张图像（[张量，形状为[18,1,28,28]]），y是对应的标签([形状为[18]])
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X, 2, 9, titles=get_fashion_mnist_labels(y))