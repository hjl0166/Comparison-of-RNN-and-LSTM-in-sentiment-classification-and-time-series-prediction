import torch
import torch.nn as nn

class RNNmodel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNmodel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)  # 输入层特征个数，隐藏层特征个数，隐藏层个数
        self.fc = nn.Linear(hidden_size, output_size)  # 隐藏层特征个数，输出层特征个数

    def forward(self, x):
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        return self.fc(out[:, -1, :])  # 只取最后一个时间步的输出
