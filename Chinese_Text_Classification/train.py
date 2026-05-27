import time
import os
import jieba
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from utils import (load_stopwords, load_ml_data, build_dl_dataset, build_iterator,
                   plot_confusion_matrix_custom, get_time_dif)
from TextCNN_Model import Config, Model, init_network, DATASET

# ========== 1. 传统机器学习训练模块 ==========
def train_machine_learning():
    print("\n" + "="*50)
    print("开始训练传统机器学习模型 (NB & SVM)...")
    print("="*50)
    train_path = os.path.join(DATASET, 'sampled_data', 'train_sampled.txt')
    test_path = os.path.join(DATASET, 'sampled_data', 'test_sampled.txt')
    stopwords_path = os.path.join(DATASET, 'stopwords_hit.txt')
    class_names = ['finance', 'home', 'stocks', 'education', 'science',
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
    print("\n【朴素贝叶斯】训练中...")
    nb_start = time.time()
    nb = MultinomialNB(alpha=0.1)
    nb.fit(X_train_tfidf, y_train)
    nb_pred = nb.predict(X_test_tfidf)
    nb_prob = nb.predict_proba(X_test_tfidf)
    nb_acc = accuracy_score(y_test, nb_pred)
    nb_time = time.time() - nb_start
    print(f"训练完成，耗时: {nb_time:.2f}s，准确率: {nb_acc:.4f}")
    print("\n【朴素贝叶斯分类报告】")
    print(classification_report(y_test, nb_pred, target_names=class_names, digits=4))
    plot_confusion_matrix_custom(y_test, nb_pred, class_names, 'Naive Bayes Confusion Matrix', os.path.join(DATASET, 'nb_confusion_matrix.png'))

    # --- 训练 SVM (SVC with probability for ensemble) ---
    print("\n【SVM】训练中...")
    svm_start = time.time()
    svm = SVC(kernel='linear', C=1.0, probability=True, random_state=42, max_iter=5000)
    svm.fit(X_train_scaled, y_train)
    svm_pred = svm.predict(X_test_scaled)
    svm_prob = svm.predict_proba(X_test_scaled)
    svm_acc = accuracy_score(y_test, svm_pred)
    svm_time = time.time() - svm_start
    print(f"训练完成，耗时: {svm_time:.2f}s，准确率: {svm_acc:.4f}")
    print("\n【SVM 分类报告】")
    print(classification_report(y_test, svm_pred, target_names=class_names, digits=4))
    plot_confusion_matrix_custom(y_test, svm_pred, class_names, 'SVM Confusion Matrix', os.path.join(DATASET, 'svm_confusion_matrix.png'))

    # --- 集成 (NB + SVM 平均概率融合) ---
    print("\n【集成模型】平均概率融合...")
    ensemble_prob = (nb_prob + svm_prob) / 2.0
    ensemble_pred = np.argmax(ensemble_prob, axis=1)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    print("\n【集成模型分类报告】")
    print(classification_report(y_test, ensemble_pred, target_names=class_names, digits=4))
    plot_confusion_matrix_custom(y_test, ensemble_pred, class_names, 'Ensemble Confusion Matrix', os.path.join(DATASET, 'ensemble_confusion_matrix.png'))

    # --- 对比总结 ---
    print("\n" + "=" * 50)
    print("│ 模型              │ 准确率    │ 耗时     │")
    print("│ 朴素贝叶斯        │ {:.4f}    │ {:.2f}s   │".format(nb_acc, nb_time))
    print("│ SVM               │ {:.4f}    │ {:.2f}s   │".format(svm_acc, svm_time))
    print("│ 集成 (平均概率)   │ {:.4f}    │ -        │".format(ensemble_acc))
    print("=" * 50)
    print("传统模型训练与评估结束。混淆矩阵已保存至 THUCNews/ 目录。\n")


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
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    config = Config()
    config.load_pretrained('embedding_SougouNews.npz')
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
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    model.train()
    best_dev_acc = 0.0
    best_state = None
    epochs_no_improve = 0
    best_model_path = os.path.join(config.save_dir, 'textcnn_best.pth')

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
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= config.early_stop_patience:
            print(f"\n早停触发！验证集准确率连续 {config.early_stop_patience} 个 epoch 未提升，停止训练。")
            break

    print(f"\nTextCNN 训练结束，最佳验证集准确率: {best_dev_acc:.4f}")
    print("开始在测试集上进行最终评估...")
    model.load_state_dict(best_state)
    test_acc, _, test_report, test_preds, test_labels = evaluate_dl(config, model, test_iter, test=True)
    
    print(f"[TextCNN] 测试集准确率: {test_acc:.4f}")
    print(test_report)
    plot_confusion_matrix_custom(test_labels, test_preds, config.class_list, 'TextCNN Confusion Matrix', os.path.join(DATASET, 'textcnn_confusion_matrix.png'))

if __name__ == '__main__':
    # 你可以把不想运行的模型注释掉
    # train_machine_learning()
    train_textcnn()