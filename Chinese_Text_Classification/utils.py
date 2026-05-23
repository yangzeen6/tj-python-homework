import os
import time
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# ==================== 通用工具 ====================
def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def load_stopwords(filepath):
    """加载停用词表"""
    if filepath and os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def plot_confusion_matrix_custom(y_true, y_pred, class_names, title, save_path):
    """绘制并保存混淆矩阵"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=class_names, 
        cmap=plt.cm.Blues, ax=ax, xticks_rotation=45
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ==================== 传统机器学习 (SVM/NB) 数据处理 ====================
def load_ml_data(filepath, stopwords=None):
    """专门为 SVM 和朴素贝叶斯加载文本数据"""
    texts, labels = [], []
    if not os.path.exists(filepath):
        print(f"警告: 找不到数据文件 {filepath}")
        return texts, labels
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                text, label = parts[0], int(parts[1])
                # 如果传入了停用词，进行初筛
                if stopwords:
                    text = "".join([word for word in text if word not in stopwords])
                texts.append(text)
                labels.append(label)
    return texts, labels

# ==================== 深度学习 (TextCNN) 数据处理 ====================
MAX_VOCAB_SIZE = 10000
UNK, PAD = '<UNK>', '<PAD>'

def build_vocab(file_path, tokenizer, max_size, min_freq):
    """构建深度学习词表"""
    vocab_dic = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"构建词表 {file_path}"):
            lin = line.strip()
            if not lin: continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
    vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic

def build_dl_dataset(config, use_word=False):
    """为 TextCNN 构建 Dataset"""
    tokenizer = (lambda x: x.split(' ')) if use_word else (lambda x: [y for y in x])
    
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    
    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"加载DL数据 {path}"):
                lin = line.strip()
                if not lin: continue
                content, label = lin.split('\t')
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                contents.append((words_line, int(label), seq_len))
        return contents

    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return vocab, train, dev, test

class DatasetIterater(object):
    """数据迭代器"""
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = (len(batches) % self.n_batches != 0)
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            return self._to_tensor(batches)
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            return self._to_tensor(batches)

    def __iter__(self):
        return self
    
    def __len__(self):
        return self.n_batches + 1 if self.residue else self.n_batches

def build_iterator(dataset, config):
    return DatasetIterater(dataset, config.batch_size, config.device)