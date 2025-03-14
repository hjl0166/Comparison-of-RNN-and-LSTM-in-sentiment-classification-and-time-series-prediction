import pandas as pd
import matplotlib.pyplot as plt

# 手动指定 CSV 文件路径
csv_files = {
    "LSTM": "../lstm/sentiment_classification/runs/exp1/result.csv",
    "RNN": "../rnn/sentiment_classification/runs/exp1/result.csv",
    # "Model C": "path/to/result_C.csv",  # 可以添加更多模型
}

# 绘图
plt.figure(figsize=(10, 6))
for label, file_path in csv_files.items():
    # 读取 CSV 文件
    df = pd.read_csv(file_path)

    # 假设 loss 值存储在 "loss" 列，你可以根据实际情况修改列名
    plt.plot(df["epoch"], df["loss"], label=label)

# 设置图例、标题和标签
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve Comparison")
plt.legend()
plt.grid(True)

# 显示图像
plt.show()