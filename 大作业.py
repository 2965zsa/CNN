import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from PIL import Image

# 预处理：将两个步骤整合在一起
transform = transforms.Compose([
    transforms.ToTensor(),  # 转为Tensor，范围改为0-1
    transforms.Normalize((0.1307,), (0.3081))  # 数据归一化，即均值为0，标准差为1
])

# 训练数据集
train_data = MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, shuffle=True, batch_size=64)

# 测试数据集
test_data = MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, shuffle=False, batch_size=64)


# CNN模型
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 14x14
        x = self.pool(torch.relu(self.conv2(x)))  # 7x7
        x = self.pool(torch.relu(self.conv3(x)))  # 3x3
        x = x.view(-1, 128 * 3 * 3)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# CrossEntropyLoss
model = CNNModel()
criterion = nn.CrossEntropyLoss()  # 交叉熵损失，相当于Softmax+Log+NllLoss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# 模型训练
def train():
    for epoch in range(5):  # 训练5个epoch
        for index, (input, target) in enumerate(train_loader):
            optimizer.zero_grad()  # 梯度清零
            y_predict = model(input)  # 模型预测
            loss = criterion(y_predict, target)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            if index % 100 == 0:  # 每一百次打印一次损失
                print(f"Epoch [{epoch + 1}/5], Step [{index + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")


# 模型测试
def test():
    correct = 0  # 正确预测的个数
    total = 0  # 总数
    with torch.no_grad():  # 测试不用计算梯度
        for data in test_loader:
            input, target = data
            output = model(input)  # output输出10个预测取值，其中最大的即为预测的数
            _, predict = torch.max(output.data, dim=1)  # 返回一个元组，第一个为最大概率值，第二个为最大值的下标
            total += target.size(0)  # target是形状为(batch_size,1)的矩阵，使用size(0)取出该批的大小
            correct += (predict == target).sum().item()  # predict和target均为(batch_size,1)的矩阵，sum()求出相等的个数
        print("准确率为：%.2f" % (correct / total))


# 自定义手写数字识别测试
def test_mydata():
    # 确保图片路径正确，并且图片是灰度的
    image_path = 'test_6.png'  # 替换为你的图片路径
    if not os.path.exists(image_path):
        print(f"图片文件不存在：{image_path}")
        return

    image = Image.open(image_path)  # 读取自定义手写图片
    image = image.resize((28, 28))  # 裁剪尺寸为28*28
    image = image.convert('L')  # 转换为灰度图像
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),  # 随机旋转
        transforms.RandomAffine(0, translate=(0.1, 0.1)),  # 平移
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081))
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # 添加batch维度
    output = model(image)
    probability, predict = torch.max(output.data, dim=1)
    print("此手写图片值为:%d,其最大概率为:%.2f" % (predict[0], probability))
    plt.title('此手写图片值为：{}'.format((int(predict))), fontname="SimHei")
    plt.imshow(image.squeeze().numpy(), cmap='gray')
    plt.show()


# 创建模型目录
os.makedirs("model2", exist_ok=True)

# 主函数
if __name__ == '__main__':
    # 训练与测试
    for epoch in range(5):
        train()
        test()

    # 保存模型
    torch.save(model.state_dict(), "model2/model.pkl")
    print("模型已保存到 ./model2/model.pkl")

    # 自定义测试
    test_mydata()