"""
分析模型预测中的典型混淆案例。
运行: python analyze_errors.py
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import defaultdict

from TextCNN_Model import Config, Model, DATASET
from utils import load_stopwords, load_ml_data, build_dl_dataset, build_iterator

# ── 配置 ──
class_names = ['finance', 'home', 'stocks', 'education', 'science',
               'society', 'politics', 'sports', 'game', 'entertainment']
TOP_K = 5  # 每个混淆对展示的样本数


def load_ml_models():
    """训练 NB 和 SVM，返回模型和原始测试文本"""
    train_path = os.path.join(DATASET, 'sampled_data', 'train_sampled.txt')
    test_path = os.path.join(DATASET, 'sampled_data', 'test_sampled.txt')
    stopwords_path = os.path.join(DATASET, 'stopwords_hit.txt')

    stopwords = load_stopwords(stopwords_path)
    train_texts, y_train = load_ml_data(train_path, stopwords)
    test_texts, y_test = load_ml_data(test_path, stopwords)

    tokenize = lambda text: ' '.join(jieba.lcut(text))
    X_train_words = [tokenize(t) for t in train_texts]
    X_test_words = [tokenize(t) for t in test_texts]

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train_words)
    X_test_tfidf = vectorizer.transform(X_test_words)

    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train_tfidf)
    X_test_scaled = scaler.transform(X_test_tfidf)

    nb = MultinomialNB(alpha=0.1)
    nb.fit(X_train_tfidf, y_train)
    nb_pred = nb.predict(X_test_tfidf)

    base_svm = LinearSVC(C=1.0, random_state=42, max_iter=5000, dual=False)
    svm = CalibratedClassifierCV(base_svm, cv=3, method='sigmoid')
    svm.fit(X_train_scaled, y_train)
    svm_pred = svm.predict(X_test_scaled)

    return nb_pred, svm_pred, y_test, test_texts


def load_textcnn_model():
    """加载训练好的 TextCNN"""
    config = Config()
    config.load_pretrained('embedding_SougouNews.npz')
    vocab, _, _, test_data = build_dl_dataset(config, use_word=False)
    config.n_vocab = len(vocab)

    model = Model(config).to(config.device)
    model.load_state_dict(torch.load(config.save_path))
    model.eval()

    test_iter = build_iterator(test_data, config)

    # 加载原始文本（与 ML 的 test_texts 对应）
    test_texts_dl = []
    with open(config.test_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if parts:
                test_texts_dl.append(parts[0])

    preds, probs = [], []
    with torch.no_grad():
        for texts, _ in test_iter:
            outputs = model(texts)
            p = F.softmax(outputs, dim=1).cpu().numpy()
            probs.append(p)
            preds.extend(outputs.argmax(1).cpu().numpy())

    return np.array(preds), test_texts_dl


def find_confusion_pairs(y_true, preds):
    """找出所有混淆对及对应的样本索引"""
    pairs = defaultdict(list)
    for i, (t, p) in enumerate(zip(y_true, preds)):
        if t != p:
            pairs[(t, p)].append(i)
    # 按混淆数量降序排列
    return sorted(pairs.items(), key=lambda x: len(x[1]), reverse=True)


def print_analysis(model_name, y_true, preds, texts):
    """打印模型分析"""
    acc = accuracy_score(y_true, preds)
    print(f"\n{'='*70}")
    print(f"【{model_name}】准确率: {acc:.4f}")
    print(f"{'='*70}")

    pairs = find_confusion_pairs(y_true, preds)
    if not pairs:
        print("  完美预测，无混淆样本！")
        return

    print(f"共 {len(pairs)} 种混淆类型，显示 Top-5:")
    for rank, ((true_id, pred_id), indices) in enumerate(pairs[:5], 1):
        true_name = class_names[true_id]
        pred_name = class_names[pred_id]
        print(f"\n  ┌─ 混淆 #{rank}: {true_name} → {pred_name} ({len(indices)} 例)")

        for j, idx in enumerate(indices[:TOP_K]):
            print(f"  │   {j+1}. 「{texts[idx][:60]}」")

        # 该混淆对中所有样本被模型判错
        sample_preds = [preds[i] for i in indices]
        # 看看 TextCNN 的概率是否很接近（边界样本）
        print(f"  └─ 共误判 {len(indices)} 例")


def compare_models(nb_pred, svm_pred, tcnn_pred, y_true, texts):
    """找出三个模型共同判错的样本"""
    all_wrong = [i for i in range(len(y_true))
                 if nb_pred[i] != y_true[i]
                 and svm_pred[i] != y_true[i]
                 and tcnn_pred[i] != y_true[i]]
    print(f"\n{'='*70}")
    print(f"【三模型共识错误】全部模型都判错的样本: {len(all_wrong)} 例")
    print(f"{'='*70}")

    for i in all_wrong[:TOP_K]:
        print(f"\n  标题: 「{texts[i]}」")
        print(f"  真实: {class_names[y_true[i]]}")
        print(f"  NB → {class_names[nb_pred[i]]}")
        print(f"  SVM → {class_names[svm_pred[i]]}")
        print(f"  TextCNN → {class_names[tcnn_pred[i]]}")


def find_boundary_cases(y_true, preds, texts, model_name):
    """找出模型预测最犹豫的样本（概率接近边界）"""
    # 对于没有概率输出的模型，找"一个模型判错但两个模型判对"的样本
    pass  # 此处仅用 TextCNN 概率


if __name__ == '__main__':
    print("加载模型...")
    nb_pred, svm_pred, y_true, ml_texts = load_ml_models()
    tcnn_pred, dl_texts = load_textcnn_model()

    # ML 和 DL 的测试集文本相同（顺序一致），取其一即可
    texts = ml_texts

    print_analysis("朴素贝叶斯 (NB)", y_true, nb_pred, texts)
    print_analysis("支持向量机 (SVM)", y_true, svm_pred, texts)
    print_analysis("TextCNN", y_true, tcnn_pred, texts)
    compare_models(nb_pred, svm_pred, tcnn_pred, y_true, texts)
