import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split


# 模型定义
class LearnedMinToFirstModel(nn.Module):
    def __init__(self):
        super(LearnedMinToFirstModel, self).__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 自定义损失函数
def custom_loss(output, target):
    # 第一个输出值应该与标签的第一个值（最小值）接近
    min_loss = torch.nn.functional.mse_loss(output[:, 0], target[:, 0])

    # 其余两个输出值应该接近0
    zero_loss = torch.nn.functional.mse_loss(output[:, 1:], target[:, 1:])

    # 总损失是两者之和
    loss = min_loss + zero_loss
    return loss


# 数据生成
def generate_data(num_samples):
    X = np.random.rand(num_samples, 3).astype(np.float32)  # 随机生成数据
    Y = np.zeros_like(X)
    Y[:, 0] = X.min(axis=1)  # 第一个位置放最小值
    return torch.tensor(X), torch.tensor(Y)


# 训练模型
def train_model(model, train_loader, optimizer, device):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = custom_loss(outputs, labels)

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

            # 检查模型是否正确生成了输出
            condition1 = torch.isclose(outputs[:, 0], labels[:, 0], atol=1e-3)  # 检查第一个数是否与标签相近
            condition2 = torch.isclose(outputs[:, 1:], torch.zeros_like(outputs[:, 1:]), atol=1e-3)  # 检查剩下的数是否为0

            correct += (condition1 & condition2.all(dim=1)).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy


def main():
    # 超参数
    num_samples = 10000
    batch_size = 32
    num_epochs = 100
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

    # 初始化模型和优化器
    model = LearnedMinToFirstModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    for epoch in range(num_epochs):
        train_model(model, train_loader, optimizer, device)
        accuracy = validate_model(model, val_loader, device)
        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {accuracy:.4f}')

    # 保存模型
    torch.save(model.state_dict(), 'learned_min_to_first_model.pth')
    print("模型已保存到 'learned_min_to_first_model.pth'")


if __name__ == '__main__':
    main()
