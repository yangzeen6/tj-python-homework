import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE_DIR, 'THUCNews')

class Config:
    """TextCNN 配置参数"""
    def __init__(self):
        self.dataset = 'THUCNews'
        self.train_path = os.path.join(DATASET, 'sampled_data', 'train_sampled.txt')
        self.dev_path = os.path.join(DATASET, 'sampled_data', 'dev_sampled.txt')
        self.test_path = os.path.join(DATASET, 'sampled_data', 'test_sampled.txt')
        self.stopwords_path = os.path.join(DATASET, 'stopwords_hit.txt')
        self.vocab_path = os.path.join(DATASET, 'vocab.pkl')
        self.save_dir = os.path.join(DATASET, 'saved_dict')
        self.save_path = os.path.join(self.save_dir, 'textcnn_best.pth')
        self.embedding_pretrained = None  # 默认随机初始化，可通过 load_pretrained() 加载

        self.class_list = ['finance', 'home', 'stocks', 'education', 'science',
                           'society', 'politics', 'sports', 'game', 'entertainment']
        self.num_classes = len(self.class_list)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 模型超参
        self.dropout = 0.7
        self.weight_decay = 5e-4
        self.num_epochs = 20
        self.early_stop_patience = 5
        self.batch_size = 128
        self.pad_size = 32
        self.learning_rate = 1e-3
        self.embed = 300
        self.filter_sizes = (2, 3, 4)
        self.num_filters = 128
        self.n_vocab = 0  # 词表大小，运行时赋值

    def load_pretrained(self, embedding_name='embedding_SougouNews.npz'):
        """加载预训练词向量，若文件不存在则保持随机初始化"""
        emb_path = os.path.join(DATASET, embedding_name)
        if not os.path.exists(emb_path):
            print(f"预训练词向量 {emb_path} 未找到，使用随机初始化")
            return
        pretrained = np.load(emb_path, allow_pickle=True)
        emb = pretrained['embeddings'].astype('float32')
        # rescale 到 N(0,1)，与 Xavier 初始化的 Conv2d 兼容
        emb_std = emb.std()
        if emb_std > 0:
            emb = emb / emb_std
        self.embedding_pretrained = torch.tensor(emb)
        self.embed = self.embedding_pretrained.size(1)
        print(f"已加载预训练词向量: {emb_path}, 维度: {self.embed}, rescale: {1/emb_std:.1f}x")

class Model(nn.Module):
    """TextCNN 模型结构"""
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

def init_network(model, method='xavier', exclude='embedding'):
    """权重初始化"""
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass