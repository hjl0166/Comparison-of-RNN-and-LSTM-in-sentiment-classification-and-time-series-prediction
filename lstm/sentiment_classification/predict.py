from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from utils import *
from lstm.sentiment_classification.LSTM import LSTMmodel
import pandas as pd
# 读取CSV文件
df = pd.read_csv('ChnSentiCorp_htl_all.csv')
torch.autograd.set_detect_anomaly(True)

# 确保CSV文件中列名为 'label' 和 'review'
texts = df['review'].tolist()  # 获取文本列
labels = df['label'].tolist()  # 获取标签列

# 划分训练集和测试集（70%训练，30%测试）
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)

# 构建词汇表
vocab = build_vocab(X_train)
pad_idx = vocab["<pad>"]

# 构建数据集
train_dataset = SentimentDataset(X_train,y_train,vocab)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda b: collate_batch(b, pad_idx))
test_dataset = SentimentDataset(X_test,y_test,vocab)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=lambda b: collate_batch(b, pad_idx))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMmodel(vocab_size=10000, embedding_dim=128, hidden_dim=128, output_dim=1).to(device)

# 使用保存的最佳权重进行评估
model.load_state_dict(torch.load('runs/exp1/best.pth'))  # 从文件中加载最佳权重
# 预测

correct,total = 0,0
with torch.no_grad():

    for texts, labels in test_loader:
        h = model.init_hidden(texts.size(0))  # 初始化第一个Hidden_state
        texts = texts.to(device)  # 确保输入数据在相同设备
        outputs, h = model(texts, h)
        predicted = (outputs > 0.5).float()
        labels = labels.to(device)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        # texts, labels = texts.to(device), labels.to(device)  # 将数据移到 GPU
        # outputs = model(texts).squeeze(1)
        # predicted = (outputs > 0.5).float()
        # correct += (predicted == labels).sum().item()
        # total += labels.size(0)
    print(f'模型测试准确率: {correct / total * 100:.8f}%')

new_texts = ["我非常喜欢我的购买", "这个产品真的很糟糕"]
predicted_labels = predict(model, new_texts, vocab, device)
print("新文本预测结果：", predicted_labels)