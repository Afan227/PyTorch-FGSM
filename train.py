from torch.utils.tensorboard import SummaryWriter
import time
from model import *
# 准备数据集
import torch.utils.data
import torchvision.datasets
# 训练数据集
from torch import nn
from torch.utils.data import DataLoader
# 网安实验  神经网络模型的训练，用作对抗样本攻击的测试
transformss = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
train_data = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform= transformss,
                                          download=True)
# 测试数据集
test_data = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=transformss,
                                         download=True)     # ./ 表示运行程序脚本所在程序下, ../表示运行程序所在脚本的上一目录下
# 数据集对应标签['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog','horse', 'ship', 'truck']
# 测试数据集和训练数据集的长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用Dataloader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
wd = WangDi()
if torch.cuda.is_available():
    wd = wd.cuda()

# 损失函数
loss_fn = nn.CrossEntropyLoss()   # 损失函数选择交叉熵
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()
# 优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(wd.parameters(), lr=learning_rate)  # 梯度下降

# 设置训练网络的参数
# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 训练的轮数
epoch = 100

# 添加tensorboard
writer = SummaryWriter("./logs", )

start_time = time.time()
for i in range(epoch):
    print("------第 {} 轮训练开始------".format(i+1))
    #训练步骤开始
    wd.train()
    for data in train_dataloader:
        imgs , targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = wd(imgs)
        loss = loss_fn(outputs, targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step%100==0:
            end_time = time.time()
            print(end_time-start_time)
            start_time = time.time()
            print("训练次数：{}， Loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
    # 测试步骤开始
    wd.eval()
    total_test_loss = 0
    total_accu = 0
    with torch.no_grad():   # 无梯度值
        for data in test_dataloader:
            imgs , targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = wd(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss +loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accu = accuracy + total_accu

    print("整体测试数据集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的准确率：{}".format(total_accu/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("accuracy", total_accu/test_data_size, total_test_step)
    total_test_step = total_test_step + 1
    if (i+1)%20==0:
        torch.save(wd, "wd_{}.pth".format(i))
        print("模型已保存")

writer.close()


