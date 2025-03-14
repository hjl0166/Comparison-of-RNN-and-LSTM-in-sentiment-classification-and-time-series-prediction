import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchtext
from sklearn.model_selection import train_test_split
from utils import *
from RNN import RNNModel
torchtext.disable_torchtext_deprecation_warning()
# 中文情感分类数据集
import pandas as pd
import matplotlib.pyplot as plt
# 读取CSV文件
df = pd.read_csv('ChnSentiCorp_htl_all.csv')
torch.autograd.set_detect_anomaly(True)
# 确保CSV文件中列名为 'label' 和 'review'
texts = df['review'].tolist()  # 获取文本列
labels = df['label'].tolist()  # 获取标签列

# # 假设你已经加载了 DataFrame
# df = pd.read_csv('ChnSentiCorp_htl_all.csv')

# # 检查哪些位置是NaN
# nan_locations = df.isna()
# # 获取包含 NaN 值的行索引
# nan_rows = df[df.isna().any(axis=1)]
# nan_review_rows = df[df['review'].isna()]

# print(nan_review_rows)
# print(nan_rows)
# # 打印出包含NaN值的行和列
# print(nan_locations)
# # 统计每一列中NaN的数量
# nan_counts = df.isna().sum()
#
# print(nan_counts)
# # 检查导入的数据
# # print(texts[:])  # 打印前5条文本
# # print(labels[:])  # 打印前5条标签

# 划分训练集和测试集（70%训练，30%测试）
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)


# 构建词汇表
vocab = build_vocab(X_train)
pad_idx = vocab["<pad>"]
import math
# 检查X_train中的NaN值
for index, item in enumerate(X_train):
    if isinstance(item, float) and math.isnan(item):
        print(f"NaN value found at index {index}: {item}")
# 构建数据集
train_dataset = SentimentDataset(X_train,y_train,vocab)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda b: collate_batch(b, pad_idx))
test_dataset = SentimentDataset(X_test,y_test,vocab)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=lambda b: collate_batch(b, pad_idx))

# 定义超参数
vocab_size = 10000   # 词汇表大小
embedding_dim = 128  # 嵌入层维度
hidden_dim = 128     # RNN 隐藏层维度
output_dim = 1       # 输出维度（1 表示二分类任务）
epochs = 60
batch_size = 8

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNNModel(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)
print("模型结构：")
print(model)

# 训练模型
loss_f = nn.BCELoss()
# 第一轮loss值很小：1左右,后面的loss值很大29~30左右 -->降低lr=0.001-->0.0001
optimizer = optim.Adam(model.parameters(), lr=0.0001)
Best_Acc = 0.00
best_model_weights = None
losses = []
Accuracy_list = []

for epoch in range(epochs):
    correct, total = 0, 0
    model.train()
    optimizer.zero_grad()
    total_loss = 0
    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device)  # 将数据移到 GPU
        # 前向传播
        outputs = model(texts).squeeze(1)  # output->(b,1)->去除多余维度->(b)
        loss = loss_f(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # 计算Accuracy
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    Accuracy = (correct / total * 100)
    # 将每个epoch的loss添加到losses列表
    losses.append(total_loss / len(train_loader))
    Accuracy_list.append(Accuracy)
    if (Best_Acc < Accuracy):
        Best_Acc = Accuracy
        best_model_weights = model.state_dict()  # 保存当前最好的模型权重
        torch.save(best_model_weights, 'runs/exp1/best.pth')  # 保存权重到指定路径

        
    print(f'Epoch [{epoch + 1}/{epochs}],'
          f' Acc: {Accuracy:.8f}%,'
          f' Loss: {total_loss / len(train_loader):.8f}')

# 存loss值、Accuracy值到csv中
loss_df = pd.DataFrame({'epoch': range(1, epochs + 1), 'loss': losses,'acc': Accuracy_list})

loss_df.to_csv('runs/exp1/result.csv', index=False, encoding='utf-8-sig')  # 保存为 UTF-8 编码，防止中文乱码

# 使用保存的最佳权重进行评估
# model.load_state_dict(torch.load('runs/exp1/best_model.pth'))  # 从文件中加载最佳权重
# 预测
model.eval()
correct,total = 0,0
with torch.no_grad():
    for texts, labels in test_loader:
        texts, labels = texts.to(device), labels.to(device)  # 将数据移到 GPU
        outputs = model(texts).squeeze(1)
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    print(f'模型测试准确率: {correct / total * 100:.8f}%')

new_texts = ["我非常喜欢我的购买", "这个产品真的很糟糕"]
predicted_labels = predict(model, new_texts, vocab, device)
print("新文本预测结果：", predicted_labels)

# 绘制Loss曲线
plt.plot(range(1, epochs + 1), losses, label="Training Loss")  # 使用损失值的列表绘制
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.show()

# 绘制Loss曲线
plt.plot(range(1, epochs + 1), Accuracy_list, label="Training Accuracy")  # 使用损失值的列表绘制
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()
plt.show()