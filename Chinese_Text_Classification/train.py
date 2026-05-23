import time
import os
import jieba
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

from utils import (load_stopwords, load_ml_data, build_dl_dataset, build_iterator, 
                   plot_confusion_matrix_custom, get_time_dif)
from TextCNN_Model import Config, Model, init_network

# ========== 1. 传统机器学习训练模块 ==========
def train_machine_learning():
    print("\n" + "="*50)
    print("开始训练传统机器学习模型 (NB & SVM)...")
    print("="*50)
    dataset_dir = 'THUCNews'
    train_path = f'{dataset_dir}/sampled_data/train_sampled.txt'
    test_path = f'{dataset_dir}/sampled_data/test_sampled.txt'
    stopwords_path = f'{dataset_dir}/stopwords_hit.txt'
    class_names = ['finance', 'realty', 'stocks', 'education', 'science',
                   'society', 'politics', 'sports', 'game', 'entertainment']

    stopwords = load_stopwords(stopwords_path)
    train_texts, y_train = load_ml_data(train_path, stopwords)
    test_texts, y_test = load_ml_data(test_path, stopwords)

    print("正在进行 Jieba 分词与 TF-IDF 特征提取...")
    tokenize = lambda text: ' '.join(jieba.lcut(text))
    X_train_words = [tokenize(text) for text in train_texts]
    X_test_words = [tokenize(text) for text in test_texts]

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train_words)
    X_test_tfidf = vectorizer.transform(X_test_words)

    print("标准化数据 (StandardScaler, 保持稀疏)...")
    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train_tfidf)
    X_test_scaled = scaler.transform(X_test_tfidf)

    # --- 训练朴素贝叶斯 (使用未缩放的 TF-IDF, MultinomialNB 不接受负值) ---
    nb_start = time.time()
    nb = MultinomialNB()
    nb.fit(X_train_tfidf, y_train)
    nb_pred = nb.predict(X_test_tfidf)
    print(f"朴素贝叶斯训练完成，耗时: {time.time() - nb_start:.2f}s，准确率: {accuracy_score(y_test, nb_pred):.4f}")

    # --- 训练 SVM (LinearSVC, 高维稀疏文本的标准选择) ---
    svm_start = time.time()
    svm = LinearSVC(C=1.0, random_state=42, max_iter=5000)
    svm.fit(X_train_scaled, y_train)
    svm_pred = svm.predict(X_test_scaled)
    print(f"SVM 训练完成，耗时: {time.time() - svm_start:.2f}s，准确率: {accuracy_score(y_test, svm_pred):.4f}")
    
    plot_confusion_matrix_custom(y_test, svm_pred, class_names, 'SVM Confusion Matrix', f'{dataset_dir}/svm_confusion_matrix.png')
    print("传统模型训练与评估结束。图片已保存至 THUCNews/ 目录。\n")


# ========== 2. 深度学习 (TextCNN) 训练评估模块 ==========
def evaluate_dl(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = accuracy_score(labels_all, predict_all)
    if test:
        report = classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        return acc, loss_total / len(data_iter), report, predict_all, labels_all
    return acc, loss_total / len(data_iter)

def train_textcnn():
    print("\n" + "="*50)
    print("开始训练 TextCNN 深度学习模型...")
    print("="*50)
    config = Config()
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    start_time = time.time()
    vocab, train_data, dev_data, test_data = build_dl_dataset(config, use_word=False)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    
    config.n_vocab = len(vocab)
    model = Model(config).to(config.device)
    init_network(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    model.train()
    best_dev_acc = 0.0
    best_state = None

    for epoch in range(config.num_epochs):
        train_acc_total, train_loss_total, step = 0, 0, 0
        for texts, labels in train_iter:
            optimizer.zero_grad()
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            
            predic = torch.max(outputs.data, 1)[1].cpu()
            train_acc_total += accuracy_score(labels.data.cpu(), predic)
            train_loss_total += loss.item()
            step += 1
        
        train_acc = train_acc_total / step
        train_loss = train_loss_total / step
        dev_acc, dev_loss = evaluate_dl(config, model, dev_iter)
        scheduler.step(dev_acc)

        print(f"Epoch [{epoch+1:2d}/{config.num_epochs}] | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Dev Loss: {dev_loss:.4f} Acc: {dev_acc:.4f}")

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), config.save_path)

    print(f"\nTextCNN 训练结束，最佳验证集准确率: {best_dev_acc:.4f}")
    print("开始在测试集上进行最终评估...")
    model.load_state_dict(best_state)
    test_acc, _, test_report, test_preds, test_labels = evaluate_dl(config, model, test_iter, test=True)
    
    print(f"[{config.dataset}] TextCNN 测试集准确率: {test_acc:.4f}")
    print(test_report)
    plot_confusion_matrix_custom(test_labels, test_preds, config.class_list, 'TextCNN Confusion Matrix', f'{config.dataset}/textcnn_confusion_matrix.png')

if __name__ == '__main__':
    # 你可以把不想运行的模型注释掉
    train_machine_learning()
    train_textcnn()