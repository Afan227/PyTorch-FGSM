# 对抗样本攻击
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision
from torch import nn
from model import *
import torch.nn.functional as F
import matplotlib.pyplot as plt #导入matplotlib模块，并简写成plt

use_cuda = True
# 设置不同扰动大小
epsilons = [0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03,0.1]
# 获取预训练第100轮模型
pretrained_model = "wd_99.pth"
# 加载测试数据集
transformss = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
test_data = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=transformss,
                                         download=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
# 定义使用的设备
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
# 添加tensorboard
writer = SummaryWriter("./logs", )
# 损失函数
loss_fn = nn.CrossEntropyLoss()   # 损失函数选择交叉熵
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()
# 加载神经网络模型
wd = WangDi().to(device)   # 创建网络对象
# 加载已经预训练的模型，也即赋予网络模型已经训练好的参数
wd=torch.load(pretrained_model)
wd.eval()

# FGSM算法攻击代码
def fgsm_attack(image, epsilon, data_grad):
    # 收集数据梯度的元素符号,将梯度符号化
    sign_data_grad = data_grad.sign()
    # 通过调整输入图像的每个像素来创建扰动图像
    Counter_sample = image + epsilon * sign_data_grad
    # 将输入input张量每个元素的夹紧到区间 [min,max][min,max]，并返回结果到一个新张量
    # 通过打印张量的值得到范围
    Counter_sample = torch.clamp(Counter_sample, -1, 1)
    # 返回被扰动的图像
    return Counter_sample


def test(model, device, test_loader, epsilon):
    # 计算对抗样本的测试精度
    correct = 0
    adv_examples = []
    loss_all=0
    epoch=0
    # 循环遍历测试集中的所有示例
    for data, target in test_loader:
        # 把数据和标签发送到设备
        data, target = data.to(device), target.to(device)
        # 设置张量的requires_grad属性，这对于攻击很关键
        #print(data)
        data.requires_grad = True
        # 通过模型前向传递数据
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]  # 获得最大预测概率对应的索引

        # 如果初始预测是错误的，不打断攻击，继续
        if init_pred.item() != target.item():
            continue
        # 计算损失
        loss = loss_fn(output, target)
        loss_all=loss_all+loss.item()
        # 将所有现有的渐变归零
        model.zero_grad()
        # 计算后向传递模型的梯度
        loss.backward()
        # 收集datagrad
        data_grad = data.grad.data
        # 唤醒FGSM进行攻击
        #perturbed_data = fgsm_attack(data, epsilon, data_grad)
        perturbed_data = data
        # 重新分类受扰乱的图像
        output = model(perturbed_data)

        # 检查是否成功
        final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # 保存0 epsilon示例的特例
        writer.add_scalar("Epsilon:{}".format(epsilon),loss , epoch)
        epoch+=1
    # 计算这个epsilon的最终准确度
    final_acc = correct / float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))
    # 返回准确性和对抗性示例
    return final_acc, adv_examples, loss_all

accuracies = []
examples = []

# 对每个epsilon运行测试
for eps in epsilons:
    acc, ex, loss = test(wd, device, test_loader, eps)
    print(acc)
    accuracies.append(acc)
    examples.append(ex)
plt.plot(epsilons, accuracies)
plt.show()
writer.close()
