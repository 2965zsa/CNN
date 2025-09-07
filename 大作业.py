import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image

# ----------------------------------
# 1. 数据增强
# ----------------------------------
train_tf = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST(root='./data', train=True,  download=True, transform=train_tf)
test_data  = datasets.MNIST(root='./data', train=False, download=True, transform=test_tf)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)

# ----------------------------------
# 2.  deeper CNN + Dropout
# ----------------------------------
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

# ----------------------------------
# 3. 训练 & 记录
# ----------------------------------
loss_history, step_list = [], []

def train(epochs=5):
    global_step = 0
    model.train()
    for epoch in range(epochs):
        for idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            global_step += 1
            if idx % 100 == 0:
                loss_history.append(loss.item())
                step_list.append(global_step)
                print(f"Epoch [{epoch+1}/{epochs}] Step [{idx}/{len(train_loader)}]  Loss={loss.item():.4f}")
        scheduler.step()

# ----------------------------------
# 4. 测试
# ----------------------------------
def test():
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, pred = torch.max(out, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    acc = correct / total
    print(f"Test Accuracy: {acc:.4f} ({correct}/{total})")
    return acc

# ----------------------------------
# 5. 损失 & 收敛速度 画图
# ----------------------------------
def plot_loss():
    if len(loss_history) == 0:
        print("暂无损失记录")
        return
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.set_xlabel("Iteration (every 100)")
    ax1.set_ylabel("Loss", color="tab:red")
    ax1.plot(step_list, loss_history, "r-", lw=2, label="Loss")
    ax1.tick_params(axis="y", labelcolor="red")

    # 收敛速度：|Δloss|
    ax2 = ax1.twinx()
    delta = [abs(loss_history[i] - loss_history[i - 1])
             for i in range(1, len(loss_history))]
    ax2.plot(step_list[1:], delta, "b--", lw=1, alpha=0.7, label="|ΔLoss|")
    ax2.set_ylabel("Convergence speed (|ΔLoss|)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="blue")
    ax2.set_yscale("log")

    fig.tight_layout()
    plt.title("Loss Curve & Convergence Speed")
    plt.show()

# ----------------------------------
# 6. 自定义图片预处理
# ----------------------------------
def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    # 反色 -> 二值化
    img = cv2.bitwise_not(img)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 去噪
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # 裁剪空白
    coords = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(coords)
    img = img[y:y + h, x:x + w]
    # 缩放到 20×20，再放到 28×28 中心
    scale = 20.0 / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    top = (28 - new_h) // 2
    left = (28 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = img
    # 归一化
    canvas = canvas.astype(np.float32) / 255.0
    canvas = (canvas - 0.1307) / 0.3081
    return canvas

# ----------------------------------
# 7. 自定义测试
# ----------------------------------
def test_mydata(image_path="test_6.png"):
    try:
        arr = preprocess_image(image_path)
    except FileNotFoundError as e:
        print(e)
        return
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        out = model(tensor)
        prob = torch.softmax(out, dim=1)
        conf, pred = torch.max(prob, 1)
    print(f"预测值：{pred.item()}，置信度：{conf.item():.4f}")
    plt.title(f"Predict: {pred.item()}  Confidence: {conf.item():.2f}")
    plt.imshow(arr.squeeze(), cmap="gray")
    plt.show()

# ----------------------------------
# 8. 主入口
# ----------------------------------
if __name__ == "__main__":
    os.makedirs("model2", exist_ok=True)

    train(epochs=5)
    test()
    torch.save(model.state_dict(), "model2/model.pkl")
    print("模型已保存到 ./model2/model.pkl")

    plot_loss()
    test_mydata("test_6.png")   # 换成你的图片