import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn.init import trunc_normal_
from torch.utils.data import DataLoader, TensorDataset, random_split


class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.Q = nn.Linear(1, 2)
        self.K = nn.Linear(1, 2)
        self.scale = 1
        self.V = nn.Linear(1, 1)
        #
        self.softmax = nn.Softmax(dim=-1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        x = x.unsqueeze(-1)
        Q = self.Q(x) * self.scale
        K = self.K(x)

        QK = Q @ (K.transpose(-1, -2))
        QK_score = self.softmax(QK)
        x = (QK_score @ self.V(x)).squeeze(-1)
        return x


class Mlp3(nn.Module):
    def __init__(self, dim, hidden_dim, hidden_dim2, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim2, out_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Mlp2(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 模型定义
class SortingModel(nn.Module):
    def __init__(self):
        super(SortingModel, self).__init__()
        dim = 3
        hidden_dim = 7
        hidden_dim2 = 6  # 能行
        self.mlp2 = Mlp2(dim, hidden_dim, dim)
        self.mlp2_1 = Mlp2(dim, hidden_dim, dim)
        self.mlp2_2 = Mlp2(dim, hidden_dim, dim)
        self.mlp2_3 = Mlp2(dim, hidden_dim, dim)
        # self.mlp3 = Mlp3(dim, hidden_dim, hidden_dim, 3)
        # self.fc1 = nn.Linear(dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # # self.fc21 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc3 = nn.Linear(hidden_dim, dim)
        # 371参数（甚至可以更小dim=8，收敛也强）
        # 强过3->60 60->3 423参数
        #
        # # trunc_normal_(self.fc1.weight, std=.2)
        # nn.init.constant_(self.fc1.bias, 0)
        #
        # # trunc_normal_(self.fc2.weight, std=.2)
        # nn.init.constant_(self.fc2.bias, 0)
        #
        # # trunc_normal_(self.fc3.weight, std=.2)
        # nn.init.constant_(self.fc3.bias, 0)
        # self.att = Attention(dim)
        # self.norm2 = nn.LayerNorm(dim)
        # self.encoder = nn.Linear(3, 6)
        # self.decoder = nn.Linear(6, 3)
        #
        # self.fc1 = nn.Linear(dim, hidden_dim)
        # self.relu = nn.ReLU(inplace=True)
        # self.fc2 = nn.Linear(hidden_dim, 5)
        # # 过渡层（无relu）6可替代7（6+1），但收敛效果更差
        # self.fc1_2 = nn.Linear(5, hidden_dim)
        # self.relu2 = nn.ReLU(inplace=True)
        # self.fc2_2 = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        # torch.Tensor().un
        # x = x + self.att(x)

        # identity = x

        # x = self.norm2(x)
        # x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        # # x = torch.relu(self.fc21(x))
        # x = self.fc3(x)
        # x = self.mlp3(x)

        # 两层双层mlp
        # x = self.encoder(x)
        x = self.mlp2(x)
        # # x = torch.relu(x)# 直接保持0.13误差
        x = self.mlp2_1(x)
        x = self.mlp2_2(x)
        x = self.mlp2_3(x)

        # x = self.fc1(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        # x = self.relu(x)
        # x = self.fc1_2(x)
        # x = self.relu2(x)
        # x = self.fc2_2(x)
        # x = self.decoder(x)
        # x = identity + x

        return x


# 自定义损失函数
def custom_loss(output, target):
    # 损失函数是输出与目标排序后数组的均方误差
    # loss = torch.nn.functional.l1_loss(output, target)
    loss = torch.nn.functional.mse_loss(output, target)
    return loss


# 数据生成
def generate_data(num_samples):
    X = np.random.rand(num_samples, 3).astype(np.float32)  # 随机生成数据
    Y = np.sort(X, axis=1)  # 标签为排序后的数据
    # Y[:, :] += (Y[:, 0:1] + Y[:, 1:2] + Y[:, 2:3])/3
    # Y[:,1:] = 0
    # Y = Y[:, 0:2]
    # temp = (Y[:, 0:1] + Y[:, 1:2] + Y[:, 2:3])/3 #加这个45mlp3
    # Y = np.concatenate([Y, temp],axis=-1)
    # Y = (X == X.min(axis=1, keepdims=True)).astype(np.float32)  # 标签为最小值位置为1
    # print(Y)
    return torch.tensor(X), torch.tensor(Y)


# 训练模型
def train_model(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = custom_loss(outputs, labels)
        total_loss += loss.item()

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'train_loss:{total_loss}')


# 验证模型
def validate_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = custom_loss(outputs, labels)
            # 检查模型输出与目标是否匹配
            correct += torch.isclose(outputs, labels, atol=1e-3).all(dim=1).sum().item()
            total += labels.size(0)
            total_loss += loss.item()
    print(f'valid_loss:{total_loss}')
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


seed_everything(seed=11)  # 42 3407


def main():
    # 超参数
    num_samples = 10000
    batch_size = 32
    num_epochs = 1000
    learning_rate = 0.002

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
    model = SortingModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # 训练模型
    for epoch in range(num_epochs):
        train_model(model, train_loader, optimizer, device)
        if epoch % 1 == 0:
            accuracy = validate_model(model, val_loader, device)
            print(f'Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {accuracy:.4f}')

    # 保存模型
    torch.save(model.state_dict(), 'sorting_model.pth')
    print("模型已保存到 'sorting_model.pth'")


if __name__ == '__main__':
    main()
