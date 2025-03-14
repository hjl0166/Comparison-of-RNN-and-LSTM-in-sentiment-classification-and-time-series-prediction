import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from lstm.models.LSTM import LSTM

# 1. 数据预处理
# 读取数据
data = pd.read_csv('ETTh1.csv')

# 提取特征和标签
# 用.values可以得到numpy数组,可直接用做计算
# 用df[].tolist() 得到的python列表list 适合作标签
f1 = data['HUFL'].values
f2 = data['HULL'].values
f3 = data['MUFL'].values
f4 = data['MULL'].values
f5 = data['LUFL'].values
f6 = data['LULL'].values
OT = data['OT'].values

# 定义历史和未来的窗口大小
history_size = 10
future_size = 1
# 那么采样步长设为2，避免预测值重叠
step_size = 2

features = []
labels = []

for i in range(history_size,len(f1)-2,step_size):
    history_f1 = f1[i-history_size:i]
    history_f2 = f2[i-history_size:i]
    history_f3 = f3[i-history_size:i]
    history_f4 = f4[i-history_size:i]
    history_f5 = f5[i-history_size:i]
    history_f6 = f6[i-history_size:i]

    print(type(history_f1))
    # 输入特征
    feature = np.stack((history_f1, history_f2, history_f3, history_f4, history_f5, history_f6), axis=1)
    print("the shape of: ", feature.shape)
    features.append(feature)

    # 输出标签
    label = OT[i:i+future_size]  # 记录未来feature_size的标签
    labels.append(label)

# 转换为numpy
features = np.array(features)
labels = np.array(labels)

# 归一化
scaler_x = StandardScaler()
scaler_y = StandardScaler()
# reshape(a,b) a行b列的形式 (-1,b) 行自动计算
features_norm = scaler_x.fit_transform(features.reshape(-1, 6))  # 归一化只能对二维数组
labels_norm = scaler_y.fit_transform(labels.reshape(-1, 1))

features = features_norm.reshape(-1, history_size, 6)  # 将转换后的二维数组还原为三维数组
labels = labels_norm.reshape(-1, 1)

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

# 转为Tensor [样本数,时间序列,特征数]->[xx, history_size, 6]
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


# 初始化模型
input_size = 6
hidden_size = 64
num_layers = 2
output_size = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 移入GPU

model = LSTM(input_size, hidden_size, output_size).to(device)
x_train_tensor = x_train_tensor.to(device)
print(x_train_tensor.shape)
y_train_tensor = y_train_tensor.to(device)
x_test_tensor = x_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

# 损失函数
loss_f = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 保存最好的权重
best_loss = 100000.0000
best_model_weights = None

# 训练
losses = []
num_epochs = 2000
for epoch in range(num_epochs):

    model.train()
    # forward
    outputs = model(x_train_tensor)
    loss = loss_f(outputs, y_train_tensor)

    # background
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 保存最好的权重
    if loss.item() < best_loss:
        best_loss = loss.item()
        best_model_weights = model.state_dict()  # 保存当前最好的模型权重
        torch.save(best_model_weights, 'runs/exp1/best_model.pth')  # 保存权重到指定路径

    # 将每个epoch的loss添加到losses列表
    losses.append(loss.item())

    print('Epoch {}/{}, Loss: {:.8}'.format(epoch + 1, num_epochs, loss.item()))

# 存loss值到csv中
loss_df = pd.DataFrame({'epoch': range(1, num_epochs + 1), 'loss': losses})

loss_df.to_csv('runs/exp1/result.csv', index=False, encoding='utf-8-sig')  # 保存为 UTF-8 编码，防止中文乱码
# 使用保存的最佳权重进行评估
model.load_state_dict(torch.load('runs/exp1/best_model.pth'))  # 从文件中加载最佳权重

# 预测
model.eval()
with torch.no_grad():
    y_pred_tensor = model(x_test_tensor)

# 逆归一化
y_pred = scaler_y.inverse_transform(y_pred_tensor.cpu().numpy())
y_test = scaler_y.inverse_transform(y_test_tensor.cpu().numpy())
# 评估指标
r2 = r2_score(y_test, y_pred, multioutput='uniform_average')
mae_score = mae(y_test, y_pred)
print(f"R^2 score: {r2:.8f}")
print(f"MAE: {mae_score:.8f}")

# 绘制对比图, 配置中文字体支持以及调整符号的显示方式
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(10, 6))  # 创建一个新的 Matplotlib 图形，并设置图形的宽高尺寸
sample_indices = np.arange(len(y_test))  # 用于创建一个[0, len(y_test)]的等差数列的数组
plt.plot(sample_indices, y_test, color='blue', alpha=0.5, label='真实值')  # 绘制折线图，sample_indices为x轴数据，y_test为y轴数据
plt.plot(sample_indices, y_pred, color='red', alpha=0.5, label='预测值')
plt.xlabel('样本索引')
plt.ylabel('OT值')
plt.title('OT的实际值与预测值对比图')
plt.legend(['实际数据', '预测数据'])  # 显示图例
plt.grid(True)  # 显示网格线
plt.show()



# 绘制Loss曲线
plt.plot(range(1, num_epochs + 1), losses, label="Training Loss")  # 使用损失值的列表绘制
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.show()