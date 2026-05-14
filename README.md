# tj-python-homework: Chinese_Text_Classification

基于传统机器学习（SVM、朴素贝叶斯）的中文新闻标题分类模型

## 环境依赖

- Python 3.7（推荐）
- 所需库见 `requirements.txt`

### 安装依赖：
```bash
pip install -r requirements.txt
```

## 运行方式
### 随机抽取中文新闻与并对训练集、验证集和测试集进行分层
```bash
python new_data_sampling.py
```

### 训练朴素贝叶斯模型和SVM模型，并输出模型分类效果
```bash
python Text_Classification_Model.py
```

## 项目结构
```
.
├── THUCNews/                            # 数据集
│   ├── class.txt                        # 类别名列表
│   ├── embedding_SougouNews.npz         # 预训练词向量文件
│   ├── embedding_Tencent.npz            # 预训练词向量文件
│   ├── news_cleaned_from_thucnews.csv   # 原始CSV数据
│   ├── stopwords_hit.txt                # 停用词表
│   └── sampled_data/                    # 处理后数据
│       ├── train_sampled.txt            # 训练集
│       ├── dev_sampled.txt              # 验证集
│       └── test_sampled.txt             # 测试集
├── Text_Classification_Model.py         # 训练朴素贝叶斯模型和SVM模型，并输出模型分类效果
├── new_data_sampling.py                 # 随机抽取中文新闻与并对训练集、验证集和测试集进行分层
├── requirements.txt                     # Python依赖列表
└── README.md                            # 本文件
```