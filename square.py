import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
import torch.nn.functional as F
from torch.autograd import Function
matplotlib.use('TkAgg')  # 设置为 TkAgg 后端


# 定义自定义激活函数 但不稳定
# class HRelu(nn.Module):
#     def __init__(self):
#         super(HRelu, self).__init__()
#
#     def forward(self, x):
#         # 创建常量 0 和 1，只需要创建一次
#         zero = torch.tensor(0.0, device=x.device)
#         one = torch.tensor(1.0, device=x.device)
#
#         # 原地操作避免不必要的内存分配
#         x = torch.where(x < 0, zero, torch.where(x > 1, one, x))
#
#         return x


# 创建神经网络模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1, 16)  # 增加隐藏层神经元数目
        # self.fc2 = nn.Linear(64, 64)  # 增加一个隐藏层
        # self.act = HRelu()
        self.fc3 = nn.Linear(16, 1)  # 输出层

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU 激活函数
        # x = torch.sigmoid(self.fc1(x))  # ReLU 激活函数
        # x = self.act(x)
        # x = torch.relu(self.fc2(x))  # ReLU 激活函数
        # x = torch.sum(x, dim=1, keepdim=True)
        x = self.fc3(x)  # 输出层
        return x


# 归一化数据
def normalize_data(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)


# 反归一化数据
def denormalize_data(x, x_min, x_max):
    return x * (x_max - x_min) + x_min


# 生成数据集并归一化
def generate_data(start=0, end=100, num_samples=10000):
    x = np.linspace(start, end, num_samples).reshape(-1, 1)  # 生成数据
    y = np.sin(0.2 * x)  # y = x^2

    # 归一化 x 到 [0, 1] 之间
    x_min, x_max = x.min(), x.max()
    x_normalized = normalize_data(x, x_min, x_max)

    # 归一化 y 到 [0, 1] 之间
    y_min, y_max = y.min(), y.max()
    y_normalized = normalize_data(y, y_min, y_max)

    return torch.tensor(x_normalized, dtype=torch.float32), torch.tensor(y_normalized, dtype=torch.float32), (x_min, x_max, y_min, y_max)


# 训练函数
def train(model, criterion, optimizer, train_loader, epochs=1000):
    for epoch in range(epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()  # 清空梯度
            outputs = model(inputs)  # 预测输出
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            running_loss += loss.item()

        # 每100次输出一次损失
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")


# 测试函数
def test(model, x_test, y_test, x_min, x_max, y_min, y_max):
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        predictions = model(x_test)  # 预测结果

    # 反归一化预测结果和标签
    predictions = denormalize_data(predictions, y_min, y_max)
    y_test = denormalize_data(y_test, y_min, y_max)

    mse = nn.MSELoss()(predictions, y_test)  # 计算均方误差
    print(f"Test MSE: {mse.item():.4f}")

    return predictions


# 保存模型
def save_model(model, filename="model.pth"):
    torch.save(model.state_dict(), filename)


# 加载模型
def load_model(model, filename="model.pth"):
    model.load_state_dict(torch.load(filename, weights_only=True))
    model.eval()


# 主函数：选择训练或测试
def main(mode='train', epochs=1000, batch_size=64):
    if mode == 'train':
        # 生成训练数据并归一化
        x_train, y_train, (x_min, x_max, y_min, y_max) = generate_data()

        # 使用 DataLoader 分批次训练
        dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 初始化模型、损失函数和优化器
        model = SimpleNN()
        criterion = nn.MSELoss()  # 均方误差损失
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # 训练模型
        train(model, criterion, optimizer, train_loader, epochs=epochs)

        # 保存模型
        save_model(model)
        print("Model saved to 'model.pth'")

    elif mode == 'test':
        # 生成测试数据并归一化
        x_test, y_test, (x_min, x_max, y_min, y_max) = generate_data()

        # 初始化模型
        model = SimpleNN()
        model.eval()
        # 加载训练好的模型
        load_model(model)

        # 测试模型
        predictions = test(model, x_test, y_test, x_min, x_max, y_min, y_max)
        y_test = denormalize_data(y_test, y_min, y_max)
        x_test = denormalize_data(x_test, x_min, x_max)

        # 可视化结果
        plt.figure(figsize=(10, 6))
        plt.plot(x_test.numpy(), y_test.numpy(), label='True Values', color='blue')
        plt.plot(x_test.numpy(), predictions.numpy(), label='Predictions', color='red', alpha=0.7)
        plt.xlabel('x')
        plt.ylabel('x^2')
        plt.title('True vs Predicted Values')
        plt.legend()
        plt.show()

    else:
        print("Invalid mode. Please choose 'train' or 'test'.")


# 运行主函数（训练或测试）
if __name__ == "__main__":
    # 选择 'train' 或 'test' 模式
    main(mode='train', epochs=1000, batch_size=64)  # 可修改为 'test' 进行测试
    main(mode='test', epochs=1000)
# 6 四根
# 5 概率四根
