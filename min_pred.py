import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split


# 自定义注意力层
class AttentionLayer(nn.Module):
    def __init__(self):
        super(AttentionLayer, self).__init__()

    def forward(self, x):
        # Softmax 计算注意力权重
        attention_weights = torch.softmax(x, dim=1)
        # 加权输入
        weighted_x = x * attention_weights
        return weighted_x


# 模型定义
class AttentionModel(nn.Module):
    def __init__(self):
        super(AttentionModel, self).__init__()
        dim = 8
        self.fc1 = nn.Linear(3, dim)
        # self.attention = AttentionLayer()
        # self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = self.attention(x)
        # x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


# 数据生成
def generate_data(num_samples):
    X = np.random.rand(num_samples, 3).astype(np.float32)  # 随机生成数据
    Y = (X == X.min(axis=1, keepdims=True)).astype(np.float32)  # 标签为最小值位置为1
    return torch.tensor(X), torch.tensor(Y)


# 训练模型
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# 验证模型
def validate_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()  # 将输出转换为0或1
            correct += (predicted == labels).all(dim=1).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy


def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(seed=42)  # 42 3407


def main():
    # 超参数
    num_samples = 10000
    batch_size = 32
    num_epochs = 60
    learning_rate = 0.001

    # 检查是否有GPU可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 生成数据
    X, Y = generate_data(num_samples)
    dataset = TensorDataset(X, Y)

    # 拆分数据集为训练集和验证集
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = AttentionModel().to(device)
    criterion = nn.BCELoss()  # 二分类交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    for epoch in range(num_epochs):
        train_model(model, train_loader, criterion, optimizer, device)
        accuracy = validate_model(model, val_loader, device)
        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {accuracy:.4f}')

    # 保存模型
    torch.save(model.state_dict(), 'attention_model.pth')
    print("模型已保存到 'attention_model.pth'")


if __name__ == '__main__':
    main()
