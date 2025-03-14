from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

# 中文分词器（按字切分）
tokenizer = get_tokenizer(lambda x: list(x))

def yield_tokens(data):
    """ 迭代文本数据并分词 """
    for text in data:
        if isinstance(text, str):  # 只处理字符串类型
            yield tokenizer(text)
        else:
            print(f"Skipping non-string value: {text}")  # 输出调试信息，帮助定位问题

def build_vocab(train_texts, specials=["<pad>", "<unk>"]):
    """ 根据训练文本构建词汇表 """
    vocab = build_vocab_from_iterator(yield_tokens(train_texts), specials=specials)
    vocab.set_default_index(vocab["<unk>"])  # 设置默认索引
    return vocab

def text_pipeline(text, vocab):
    """ 将文本转换为索引序列 """
    return [vocab[token] for token in tokenizer(text)]

# 3. 创建 Dataset 类
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        """ 初始化数据集，转换文本为索引 """
        self.texts = [torch.tensor(text_pipeline(text, vocab), dtype=torch.long) for text in texts]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def collate_batch(batch, pad_idx):
    """ 批量处理函数：填充序列并转换为张量 """
    texts, labels = zip(*batch)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=pad_idx)
    labels = torch.tensor(labels, dtype=torch.float32)
    return texts_padded, labels

def predict(model,texts,vocab,device):
    """ 预测新文本的情感 """
    model = model.to(device)  # 确保模型在cuda的设备上
    model.eval()
    # 处理文本
    tensor_tokenized = [torch.tensor(text_pipeline(text, vocab), dtype=torch.long) for text in texts]
    tensor_tokenized = pad_sequence(tensor_tokenized, batch_first=True, padding_value=vocab["<pad>"])
    tensor_tokenized = tensor_tokenized.to(device)
    # 预测
    with torch.no_grad():
        outputs = model(tensor_tokenized)
        predictions = (outputs.squeeze(1) > 0.5).float()

    return predictions.cpu().numpy()
