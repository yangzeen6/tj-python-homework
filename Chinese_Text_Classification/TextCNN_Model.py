import time
import numpy as np
import jieba
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay


class Config:
    """TextCNN 配置参数"""
    def __init__(self):
        self.dataset = 'THUCNews'
        self.train_path = f'{self.dataset}/sampled_data/train_sampled.txt'
        self.dev_path = f'{self.dataset}/sampled_data/dev_sampled.txt'
        self.test_path = f'{self.dataset}/sampled_data/test_sampled.txt'
        self.stopwords_path = f'{self.dataset}/stopwords_hit.txt'
        self.embedding_path = f'{self.dataset}/embedding_SougouNews.npz'

        self.class_list = ['finance', 'realty', 'stocks', 'education', 'science',
                           'society', 'politics', 'sports', 'game', 'entertainment']
        self.num_classes = len(self.class_list)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 模型超参
        self.dropout = 0.5
        self.num_epochs = 20
        self.batch_size = 128
        self.pad_size = 32
        self.learning_rate = 1e-3
        self.filter_sizes = (2, 3, 4)
        self.num_filters = 256                      # 卷积核数量
        self.embed = 300                            # 词向量维度
        self.n_vocab = 0                            # 词表大小，运行时赋值
        self.min_freq = 2
        self.max_vocab = 20000


config = Config()

torch.manual_seed(42)
np.random.seed(42)

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 停用词 & 数据加载（复用 baseline） ====================
def load_stopwords(filepath):
    if filepath:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return set(line.strip() for line in f if line.strip())
        except FileNotFoundError:
            pass
    return {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
            '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着',
            '没有', '看', '好', '自己', '这', '他', '她', '它', '们', '那', '些',
            '什么', '而', '且', '与', '或', '但', '如果', '因为', '所以', '然后', '可以'}


def load_data(file_path):
    texts, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            texts.append(parts[0])
            labels.append(int(parts[1]))
    return texts, labels


def cut_texts(texts, stopwords):
    return [[w for w in jieba.cut(t) if w not in stopwords and len(w) > 1]
            for t in texts]


# ==================== 构建词表 ====================
def build_vocab(tokenized_texts, min_freq=2, max_size=20000):
    word_freq = {}
    for tokens in tokenized_texts:
        for w in tokens:
            word_freq[w] = word_freq.get(w, 0) + 1

    vocab = {'<PAD>': 0, '<UNK>': 1}
    for w, f in sorted(word_freq.items(), key=lambda x: -x[1]):
        if f >= min_freq and len(vocab) < max_size:
            vocab[w] = len(vocab)
    return vocab


# ==================== Dataset ====================
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, pad_size):
        self.data = []
        self.labels = labels
        for tokens in texts:
            ids = [vocab.get(w, 1) for w in tokens]   # 1 = <UNK>
            if len(ids) > pad_size:
                ids = ids[:pad_size]
            else:
                ids += [0] * (pad_size - len(ids))    # 0 = <PAD>
            self.data.append(ids)
        self.data = torch.tensor(self.data, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ==================== TextCNN 模型 ====================
class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(
                cfg.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(
                cfg.n_vocab, cfg.embed, padding_idx=0)

        self.convs = nn.ModuleList([
            nn.Conv2d(1, cfg.num_filters, (k, cfg.embed))
            for k in cfg.filter_sizes
        ])
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(cfg.num_filters * len(cfg.filter_sizes), cfg.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x).unsqueeze(1)               # (B, 1, L, E)
        out = torch.cat([self.conv_and_pool(out, conv)
                         for conv in self.convs], dim=1)   # (B, num_filters * K)
        out = self.dropout(out)
        out = self.fc(out)
        return out


# ==================== 训练 & 评估 ====================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(config.device), y.to(config.device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds = model(x).argmax(1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(config.device), y.to(config.device)
            logits = model(x)
            total_loss += criterion(logits, y).item() * x.size(0)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    return total_loss / total, correct / total, all_preds, all_labels


def plot_confusion_matrix(y_true, y_pred, title, save_path=None):
    _, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=config.class_list,
        cmap='Blues',
        ax=ax,
        colorbar=True
    )
    ax.set_title(title, fontsize=14, pad=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存至 {save_path}")
    plt.show()


# ==================== 主流程 ====================
def main():
    print(f"Device: {config.device}")
    stopwords = load_stopwords(config.stopwords_path)

    # 加载数据
    t0 = time.time()
    X_train_raw, y_train_raw = load_data(config.train_path)
    X_dev_raw, y_dev_raw = load_data(config.dev_path)
    X_test_raw, y_test = load_data(config.test_path)
    print(f"训练集: {len(X_train_raw)}  验证集: {len(X_dev_raw)}  测试集: {len(X_test_raw)}")

    # jieba 分词
    print("\njieba 分词中...")
    X_train_cut = cut_texts(X_train_raw, stopwords)
    X_dev_cut = cut_texts(X_dev_raw, stopwords)
    X_test_cut = cut_texts(X_test_raw, stopwords)
    print(f"分词耗时: {time.time() - t0:.2f}s")

    # 统计标题分词后长度分布
    lens = [len(t) for t in X_train_cut]
    print(f"分词后平均长度: {np.mean(lens):.1f}  中位数: {np.median(lens):.1f}  "
          f"90分位: {np.percentile(lens, 90):.0f}  最大: {max(lens)}")

    # 构建词表
    vocab = build_vocab(X_train_cut, config.min_freq, config.max_vocab)
    config.n_vocab = len(vocab)
    print(f"词表大小: {config.n_vocab}")

    # 加载预训练词向量（若有词表映射则加载，否则随机初始化）
    try:
        pretrained = np.load(config.embedding_path, allow_pickle=True)
        emb_array = pretrained['embeddings'].astype('float32')
        # 检查是否有词表映射
        if 'word2idx' in pretrained:
            w2i = pretrained['word2idx'].item()
            emb_matrix = np.random.uniform(-0.25, 0.25,
                                           (config.n_vocab, config.embed)).astype('float32')
            emb_matrix[0] = 0.0
            hit = 0
            for w, i in vocab.items():
                if w in w2i:
                    emb_matrix[i] = emb_array[w2i[w]]
                    hit += 1
            print(f"预训练词嵌入命中: {hit}/{config.n_vocab} ({100*hit/config.n_vocab:.1f}%)")
            config.embedding_pretrained = torch.tensor(emb_matrix)
        else:
            print("预训练嵌入缺少 word2idx 映射，将随机初始化并从头训练")
            config.embedding_pretrained = None
    except FileNotFoundError:
        print("未找到预训练嵌入文件，将随机初始化并从头训练")
        config.embedding_pretrained = None

    # 构建 DataLoader
    train_ds = TextDataset(X_train_cut, y_train_raw, vocab, config.pad_size)
    dev_ds = TextDataset(X_dev_cut, y_dev_raw, vocab, config.pad_size)
    test_ds = TextDataset(X_test_cut, y_test, vocab, config.pad_size)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=config.batch_size)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size)

    # 初始化模型
    model = Model(config).to(config.device)
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )

    best_dev_acc = 0.0
    best_state = None

    print(f"\n开始训练 (epochs={config.num_epochs}, batch={config.batch_size}, "
          f"lr={config.learning_rate})...")
    train_start = time.time()

    for epoch in range(1, config.num_epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        dev_loss, dev_acc, _, _ = evaluate(model, dev_loader, criterion)
        scheduler.step(dev_acc)

        print(f"Epoch {epoch:2d} | train loss: {train_loss:.4f}  acc: {train_acc:.4f} | "
              f"dev loss: {dev_loss:.4f}  acc: {dev_acc:.4f} | "
              f"{time.time() - epoch_start:.1f}s")

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_time = time.time() - train_start
    print(f"\n训练完成，总耗时: {train_time:.1f}s，最佳 dev acc: {best_dev_acc:.4f}")

    # 加载最佳模型并在测试集评估
    model.load_state_dict(best_state)
    model.to(config.device)

    _, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion)

    print(f"\n{'='*50}")
    print(f"TextCNN 测试集准确率: {test_acc:.4f}")
    print(f"{'='*50}")
    print("\nTextCNN 分类效果数据：")
    print(classification_report(test_labels, test_preds,
                                target_names=config.class_list, digits=4))

    plot_confusion_matrix(test_labels, test_preds,
                          title='TextCNN Confusion Matrix',
                          save_path=f'{config.dataset}/textcnn_confusion_matrix.png')

    print(f"\n┌──────────────────────┬──────────┐")
    print(f"│ 模型                 │ 准确率   │")
    print(f"├──────────────────────┼──────────┤")
    print(f"│ TextCNN              │ {test_acc:.4f}   │")
    print(f"└──────────────────────┴──────────┘")


if __name__ == "__main__":
    main()
