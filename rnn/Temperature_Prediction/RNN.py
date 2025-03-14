import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNModel, self).__init__()
        # 嵌入层: 将词索引转换为词向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # RNN层: 参数 batch_first=True 保证输入维度为 (batch, seq, feature)
        self.rnn = nn.RNN(embedding_dim, hidden_dim,batch_first=True)
        # 全连接层，将 RNN 的输出映射到情感分类（1维输出）
        self.fc = nn.Linear(hidden_dim, output_dim)
        # Sigmoid 激活函数，将输出转换为 0~1 之间的概率
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x:(b,seq_length)
        x = self.embedding(x)       # output->(b,seq_length,embedding_dim)
        out, _ = self.rnn(x, None)  # output->(b,seq_length,hidden_dim)
        out = out[:, -1, :]         # output->(b,hidden_dim) 取最后一个时刻的输出
        out = self.fc(out)          # output->(b,output_dim) 全连接层映射
        out = self.sigmoid(out)     # output->(b,output_dim) 输出概率
        return out
