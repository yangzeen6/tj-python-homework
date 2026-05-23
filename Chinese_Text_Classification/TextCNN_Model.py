import torch
import torch.nn as nn
import torch.nn.functional as F

class Config:
    """TextCNN 配置参数"""
    def __init__(self):
        self.dataset = 'THUCNews'
        self.train_path = f'{self.dataset}/sampled_data/train_sampled.txt'
        self.dev_path = f'{self.dataset}/sampled_data/dev_sampled.txt'
        self.test_path = f'{self.dataset}/sampled_data/test_sampled.txt'
        self.stopwords_path = f'{self.dataset}/stopwords_hit.txt'
        self.vocab_path = f'{self.dataset}/vocab.pkl'
        self.save_dir = f'{self.dataset}/saved_dict'
        self.save_path = f'{self.save_dir}/textcnn.ckpt'
        
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
        self.embed_size = 300
        self.filter_sizes = (2, 3, 4)
        self.num_filters = 256
        self.n_vocab = 0  # 词表大小，运行时赋值

class Model(nn.Module):
    """TextCNN 模型结构"""
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(config.n_vocab, config.embed_size, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed_size)) for k in config.filter_sizes])
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