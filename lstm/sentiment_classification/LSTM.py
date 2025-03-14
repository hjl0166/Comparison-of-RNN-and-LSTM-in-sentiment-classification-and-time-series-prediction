import torch
import torch.nn as nn

class LSTMmodel(nn.Module):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMmodel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        drop_prob = 0.5
        # 嵌入层: 将词索引转换为词向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM层: 参数 batch_first=True 保证输入维度为 (batch, seq, feature)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim,
                            dropout=drop_prob, batch_first=True)
        # 全连接层，将 LSTM 的输出映射到情感分类（1维输出）
        self.fc = nn.Linear(in_features=self.hidden_dim, out_features=self.output_dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x, hidden=None):

        # 初始化 hidden 状态（如果没有传入）
        if hidden is None:
            hidden = self.init_hidden(x.size(0))
        x = x.long()
        embeds = self.embedding(x)

        lstm_out, hidden = self.lstm(embeds, hidden)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)
        out = out[:, -1, :]
        out = out.squeeze()
        out = out.contiguous().view(-1)
        return out, hidden

    def init_hidden(self, batch_size):
        hidden = (torch.zeros(1, batch_size, self.hidden_dim).to(self.device),
                  torch.zeros(1, batch_size, self.hidden_dim).to(self.device))
        return hidden

