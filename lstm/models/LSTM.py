import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x = x.unsqueeze(1)  # 增加一个维度 (b,input)-->(b,1,input)
        # print(x.shape)
        # 初始化 h0-->LSTM的隐藏状态 (1,hidden_size)
        h0_lstm = torch.zeros(1,x.size(0),self.hidden_size).to(x.device)
        # print(h0_lstm.shape)
        # 初始化 c0-->LSTM的cell状态,负责储存长期信息
        c0_lstm = torch.zeros(1,x.size(0),self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0_lstm, c0_lstm))  # (batch_size, 1, input_size)
        out = out[:, -1]  # (batch_size, 1, hidden_size)
        out = self.fc(out)  # (batch_size, output_size)
        return out
