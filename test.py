import torch
import torchvision
from PIL import Image
from torch import nn

img_path = "./imgs/ship1.png"
img = Image.open(img_path)
# 本模型的输入为三通道，所以要将png格式的图片转换为RGB格式，为三通道
img = img.convert("RGB")

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)), torchvision.transforms.ToTensor()])

img = transform(img)
print(img)

class Dxq(nn.Module):
    def __init__(self):
        super(Dxq, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


model = torch.load("wd_99.pth", map_location=torch.device("cpu"))
img = torch.reshape(img, (1, 3, 32, 32))    # 1 batch_Size, 3 channels, 32*32大小
model.eval()
with torch.no_grad():    #节约内存
    output = model(img)
print(output)
print(output.argmax(1))

