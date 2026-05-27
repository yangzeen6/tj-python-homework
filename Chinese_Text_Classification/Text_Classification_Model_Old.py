import os
import time
import numpy as np
import jieba
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# 配置参数
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset = os.path.join(BASE_DIR, 'THUCNews')
train_path = os.path.join(dataset, 'sampled_data', 'train_sampled.txt')
test_path = os.path.join(dataset, 'sampled_data', 'test_sampled.txt')
STOPWORDS_PATH = os.path.join(dataset, 'stopwords_hit.txt')

CLASS_NAMES = ['finance', 'home', 'stocks', 'education', 'science',
               'society', 'politics', 'sports', 'game', 'entertainment']

MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)         # 词级1-2gram

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  
plt.rcParams['axes.unicode_minus'] = False

# 停用词加载
def load_stopwords(filepath):
    if filepath:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                print(f"成功加载停用词文件: {filepath}")
                return set(line.strip() for line in f if line.strip())
        except FileNotFoundError:
            print(f"警告：停用词文件 {filepath} 未找到，使用内置版停用词")
    return {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
            '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着',
            '没有', '看', '好', '自己', '这', '他', '她', '它', '们', '那', '些',
            '什么', '而', '且', '与', '或', '但', '如果', '因为', '所以', '然后', '可以'}

# 数据加载
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

# 对标题进行分词
def cut_texts(texts, stopwords):
    return [' '.join([w for w in jieba.cut(t) if w not in stopwords and len(w) > 1])
            for t in texts]

# 绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred, title, save_path=None):
    _, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=CLASS_NAMES,
        cmap='Blues',
        ax=ax,
        colorbar=True
    )
    ax.set_title(title, fontsize=14, pad=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存至{save_path}, 关闭图片以继续......")
    plt.show()

# 主流程
def main():
    stopwords = load_stopwords(STOPWORDS_PATH)

    start = time.time()
    X_train_raw, y_train = load_data(train_path)
    X_test_raw, y_test = load_data(test_path)
    print(f"训练集: {len(X_train_raw)}  测试集: {len(X_test_raw)}")

    print("\njieba 分词:")
    X_train_cut = cut_texts(X_train_raw, stopwords)
    X_test_cut = cut_texts(X_test_raw, stopwords)
    print(f"分词耗时: {time.time() - start:.2f}s")

    print("\nTF‑IDF 特征提取:")
    start = time.time()
    tfidf = TfidfVectorizer(analyzer='word', ngram_range=NGRAM_RANGE, max_features=MAX_FEATURES)
    X_train = tfidf.fit_transform(X_train_cut)
    X_test = tfidf.transform(X_test_cut)
    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"特征维度: {X_train.shape[1]}维，耗时: {time.time() - start:.2f}s")

    # 训练朴素贝叶斯模型
    print("\n朴素贝叶斯模型训练中......")
    nb_start = time.time()
    nb = MultinomialNB(alpha=0.1)
    nb.fit(X_train, y_train)
    nb_pred = nb.predict(X_test)
    nb_prob = nb.predict_proba(X_test)
    nb_acc = accuracy_score(y_test, nb_pred)
    nb_time = time.time() - nb_start
    print(f"训练耗时: {nb_time:.2f}s，准确率: {nb_acc:.4f}")

    print("\n朴素贝叶斯模型分类效果数据：")
    print(classification_report(y_test, nb_pred, target_names=CLASS_NAMES, digits=4))
    plot_confusion_matrix(y_test, nb_pred, title='Naive Bayes Confusion Matrix',
                        save_path=os.path.join(dataset, 'nb_confusion_matrix.png'))

    # 训练SVM模型
    print("\nSVM模型训练中......")
    svm_start = time.time()
    svm = SVC(kernel='linear', C=1.0, probability=True, random_state=42, max_iter=5000)
    svm.fit(X_train_scaled, y_train)
    svm_pred = svm.predict(X_test_scaled)
    svm_prob = svm.predict_proba(X_test_scaled)
    svm_acc = accuracy_score(y_test, svm_pred)
    svm_time = time.time() - svm_start
    print(f"训练耗时: {svm_time:.2f}s，准确率: {svm_acc:.4f}")

    print("\nSVM模型分类效果数据：")
    print(classification_report(y_test, svm_pred, target_names=CLASS_NAMES, digits=4))
    plot_confusion_matrix(y_test, svm_pred, title='SVM Confusion Matrix',
                        save_path=os.path.join(dataset, 'svm_confusion_matrix.png'))

    # 对朴素贝叶斯模型和SVM模型进行集成
    ensemble_prob = (nb_prob + svm_prob) / 2.0
    ensemble_pred = np.argmax(ensemble_prob, axis=1)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)

    print("\n")
    print(f"朴素贝叶斯模型  准确率: {nb_acc:.4f}  训练耗时: {nb_time:.2f}s")
    print(f"SVM 模型        准确率: {svm_acc:.4f}  训练耗时: {svm_time:.2f}s")
    print(f"集成模型        准确率: {ensemble_acc:.4f}")

    print("\n集成模型分类效果数据：")
    print(classification_report(y_test, ensemble_pred, target_names=CLASS_NAMES, digits=4))
    plot_confusion_matrix(y_test, ensemble_pred, title='Ensemble Confusion Matrix',
                        save_path=os.path.join(dataset, 'ensemble_confusion_matrix.png'))

    print("\n")
    print("│ 模型           │ 准确率   │")
    print("│ 朴素贝叶斯     │ {:.4f}   │".format(nb_acc))
    print("│ SVM            │ {:.4f}   │".format(svm_acc))
    print("│ 集成           │ {:.4f}   │".format(ensemble_acc))

if __name__ == "__main__":
    main()
