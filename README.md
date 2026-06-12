# Chinese Text Classification — 中文新闻标题分类

基于 Scikit-learn 和 PyTorch 的中文新闻标题分类项目，支持**朴素贝叶斯、SVM** 和 **TextCNN** 三种模型，提供从数据清洗到交互式演示的完整训练流程。

## 环境依赖

- Python 3.10+
- PyTorch 2.x（CUDA 推荐但非必需，代码自动 fallback 到 CPU）
- 其余见 `requirements.txt`

```bash
pip install -r requirements.txt
```

## 数据集说明

THUCNews 是清华大学公开的中文新闻分类数据集，源自新浪新闻 RSS，共 74 万条 14 类。本项目经过清洗后得到 810,041 条数据，最终选取 10 个类别用于模型训练。

原始数据需要访问此网址，选择THUCNews.zip并下载
http://thuctc.thunlp.org/

类别映射关系：

| 编号 | 英文 | 中文 | 备注 |
|------|------|------|------|
| 0 | finance | 财经 | |
| 1 | home | 家居 | 替代样本不足的 `realty`（房产） |
| 2 | stocks | 股票 | |
| 3 | education | 教育 | |
| 4 | science | 科技 | 原始数据中 `tech` 映射为 `science` |
| 5 | society | 社会 | |
| 6 | politics | 时政 | |
| 7 | sports | 体育 | |
| 8 | game | 游戏 | |
| 9 | entertainment | 娱乐 | |

## 训练流程

所有脚本在 `Chinese_Text_Classification/` 目录下运行。完整流程如下
---

### 第一步：数据清洗

将THUCNews.zip下载并解压后，放在data_clean文件夹下，运行thucnews_clean.py获取清洗得到的csv文件
如果已经有清洗好的 CSV（`THUCNews/news_cleaned_from_thucnews.csv`），可跳过此步。

```bash
cd data_clean
python thucnews_clean.py
```

这个脚本会遍历 THUCNews 原始数据文件夹（14 个中文类别子目录），对每条新闻标题去特殊字符、去重、过滤掉长度不在 6–40 字范围内的标题，最终输出 `THUCNews/news_cleaned_from_thucnews.csv`。

清洗完成后还可以运行可视化脚本，生成各类别分布、标题长度、词云等 6 张分析图：

```bash
python thucnews_visualization.py
```

图表输出到 `data_clean/output/` 目录。

---

### 第二步：数据采样

```bash
python new_data_sampling.py
```

从 `news_cleaned_from_thucnews.csv` 中筛选 10 个目标类别，每类随机抽取 **20,000 条**，按 **9:0.5:0.5** 比例分层划分为训练集（~180,000 条）、验证集（~10,000 条）和测试集（~10,000 条）。

采样过程中会过滤掉样本量不足的类别（如 `lottery`、`constellation`、`fashion`、`realty`），并将 `tech` 统一映射为 `science`。

输出文件放在 `THUCNews/sampled_data/`：
- `train_sampled.txt`
- `dev_sampled.txt`
- `test_sampled.txt`

每条数据格式为 `标题\t数字标签`。

---

### 第三步：训练模型

```bash
python train.py
```

一次运行依次完成以下训练：

**1. 加载数据与特征提取：**
- 读取采样好的训练集和测试集
- 使用 Jieba 分词并过滤停用词（`THUCNews/stopwords_hit.txt`）
- 提取 TF-IDF 特征（最大 5000 维，1-2 gram）

**2. 训练朴素贝叶斯 (NB)：**
- `MultinomialNB(alpha=0.1)`，直接使用 TF-IDF 特征（不缩放，因为 MultinomialNB 不接受负值）
- 输出分类报告和混淆矩阵 → `THUCNews/nb_confusion_matrix.png`

**3. 训练支持向量机 (SVM)：**
- 对 TF-IDF 做 `StandardScaler(with_mean=False)` 标准化（保持稀疏）
- 使用 `LinearSVC(C=1.0) + CalibratedClassifierCV(cv=3, method='sigmoid')` 进行概率校准
- 输出分类报告和混淆矩阵 → `THUCNews/svm_confusion_matrix.png`

**4. 概率集成：**
- 将 NB 和 SVM 的输出概率取平均，得到集成预测
- 输出分类报告和混淆矩阵 → `THUCNews/ensemble_confusion_matrix.png`
- 控制台打印三个模型的准确率和耗时对比

**5. 训练 TextCNN 深度学习模型：**
- 构建 char-level（字符级）词表，最大 10,000 词 → `THUCNews/vocab.pkl`
- 模型结构：Embedding(300 维) → 多尺度卷积 (2,3,4) → MaxPooling → Concat → Dropout(0.5) → 全连接
- 权重初始化：卷积层 Xavier Normal，Embedding 随机 N(0,1)
- 优化器：Adam(lr=1e-3, weight_decay=2e-4) + ReduceLROnPlateau 学习率衰减
- **双早停机制**：batch 级（3000 batch 验证 loss 不降即停）+ epoch 级（10 epoch 验证 acc 不升即停）
- 每 100 batch 评估一次验证集，保存最优模型权重 → `THUCNews/saved_dict/textcnn_best.pth`
- 训练日志写入 CSV → `THUCNews/training_log.csv`
- 测试集评估，输出分类报告和混淆矩阵 → `THUCNews/textcnn_confusion_matrix.png`

> **提示**：如果只想训练部分模型，可以编辑 `train.py` 末尾的 `__main__` 部分，注释掉 `train_machine_learning()` 或 `train_textcnn()`。

---

### 第四步：评估与分析

#### 4a. 绘制训练曲线

```bash
python plot_training.py
```

从 `training_log.csv` 读取训练日志，生成 5 张 300 DPI 高质量图：
- `curve_loss.png` — 训练/验证 Loss 双线图
- `curve_accuracy.png` — 训练/验证 Accuracy 双线图
- `curve_val_loss.png` — 验证 Loss 单独曲线
- `curve_val_accuracy.png` — 验证 Accuracy 单独曲线
- `curve_summary.png` — 四合一总览（2×2 子图）

> 需要先运行 `train.py` 生成 `training_log.csv`。

#### 4b. 混淆案例分析

```bash
python analyze_errors.py
```

对每个模型找出 Top-5 混淆类别对，展示具体的误分类标题样本，以及三个模型共同判错的"硬样本"。

#### 4c. 交互式 Web 演示

```bash
python demo.py
```

启动 Gradio Web 界面，浏览器打开 `http://127.0.0.1:7860`。输入任意中文新闻标题，即可查看 NB、SVM、TextCNN 三个模型各自的 Top-3 预测类别和置信度。

## 项目结构

```
Chinese_Text_Classification/
├── train.py                         # 训练主入口（NB + SVM + TextCNN）
├── TextCNN_Model.py                 # TextCNN 模型定义（Config / Model / 初始化）
├── utils.py                         # 工具函数（数据加载、词表构建、混淆矩阵、迭代器）
├── new_data_sampling.py             # 数据采样（筛选 10 类，每类 20,000，分层划分）
├── plot_training.py                 # 训练曲线绘制（5 张图）
├── analyze_errors.py                # 混淆案例分析（混淆对 + 样本）
├── demo.py                          # Gradio 交互式演示
├── Text_Classification_Model_Old.py # 旧版独立训练脚本（参考用）
├── requirements.txt
├── THUCNews/
│   ├── class.txt                    # 10 类别名列表
│   ├── stopwords_hit.txt            # 停用词表
│   ├── news_cleaned_from_thucnews.csv  # 清洗后数据集
│   ├── vocab.pkl                    # char-level 词表
│   ├── training_log.csv             # 训练日志
│   ├── curve_*.png                  # 训练曲线图
│   ├── *_confusion_matrix.png       # 各模型混淆矩阵
│   ├── saved_dict/
│   │   └── textcnn_best.pth         # TextCNN 最优权重
│   └── sampled_data/                # 采样后的训练/验证/测试集
│       ├── train_sampled.txt
│       ├── dev_sampled.txt
│       └── test_sampled.txt
└── data_clean/
    ├── thucnews_clean.py            # 数据清洗流水线
    ├── thucnews_visualization.py    # 数据可视化（6 张分析图）
    ├── analysis.txt                 # 探索性分析报告
    └── output/                      # 可视化图表
```
