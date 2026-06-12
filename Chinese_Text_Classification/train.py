import time
import os
import csv
import jieba
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
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

    preprocess_start = time.time()
    tokenize = lambda text: ' '.join(jieba.lcut(text))
    X_train_words = [tokenize(text) for text in train_texts]
    X_test_words = [tokenize(text) for text in test_texts]
    t_jieba = time.time() - preprocess_start
    print(f"Jieba 分词完成，耗时: {t_jieba:.2f}s")

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train_words)
    X_test_tfidf = vectorizer.transform(X_test_words)
    t_tfidf = time.time() - preprocess_start - t_jieba
    print(f"TF-IDF 特征提取完成，耗时: {t_tfidf:.2f}s")

    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train_tfidf)
    X_test_scaled = scaler.transform(X_test_tfidf)
    t_preprocess = time.time() - preprocess_start
    print(f"数据预处理总耗时: {t_preprocess:.2f}s (Jieba+TF-IDF+Scaler)")

    # --- 训练朴素贝叶斯 (使用未缩放的 TF-IDF, MultinomialNB 不接受负值) ---
    print("\n【朴素贝叶斯】训练中...")
    nb_start = time.time()
    nb = MultinomialNB(alpha=0.1)
    nb.fit(X_train_tfidf, y_train)
    nb_pred = nb.predict(X_test_tfidf)
    nb_prob = nb.predict_proba(X_test_tfidf)
    nb_acc = accuracy_score(y_test, nb_pred)
    nb_train_time = time.time() - nb_start
    nb_time = t_preprocess + nb_train_time
    print(f"训练完成，耗时: {nb_train_time:.2f}s (不含预处理) / {nb_time:.2f}s (含预处理)，准确率: {nb_acc:.4f}")
    print("\n【朴素贝叶斯分类报告】")
    print(classification_report(y_test, nb_pred, target_names=class_names, digits=4))
    plot_confusion_matrix_custom(y_test, nb_pred, class_names, 'Naive Bayes Confusion Matrix', os.path.join(DATASET, 'nb_confusion_matrix.png'))

    # --- 训练 SVM (LinearSVC + 概率校准) ---
    print("\n【SVM】训练中...")
    svm_start = time.time()
    base_svm = LinearSVC(C=1.0, random_state=42, max_iter=5000, dual=False)
    svm = CalibratedClassifierCV(base_svm, cv=3, method='sigmoid')
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
def evaluate_dl(config, model, data_iter):
    """快速评估，返回 acc, loss"""
    model.eval()
    loss_total = 0.0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss.item()
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
    return accuracy_score(labels_all, predict_all), loss_total / len(data_iter)


def test(config, model, test_iter):
    """最终测试：加载最优模型，输出分类报告 + 文字混淆矩阵"""
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    loss_total = 0.0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in test_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss.item()
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    from sklearn import metrics
    acc = accuracy_score(labels_all, predict_all)
    report = classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
    confusion = metrics.confusion_matrix(labels_all, predict_all)
    msg = 'Test Loss: {0:>5.4f},  Test Acc: {1:>6.2%}'
    print("\n" + msg.format(loss_total / len(test_iter), acc))
    print("\n=== Precision, Recall and F1-Score ===")
    print(report)
    print("=== Confusion Matrix ===")
    print(confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    return predict_all, labels_all


def train_textcnn():
    print("\n" + "="*50)
    print("开始训练 TextCNN 深度学习模型 (batch 级实时评估)...")
    print("="*50)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    config = Config()
    # 使用纯随机初始化（不用预训练词向量）
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    eval_interval = 100           # 每 N batch 评估一次 (同原版)
    start_time = time.time()
    vocab, train_data, dev_data, test_data = build_dl_dataset(config, use_word=False)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)

    config.n_vocab = len(vocab)
    model = Model(config).to(config.device)
    init_network(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2)

    # epoch 级 CSV 日志 (供 plot_training.py 使用)
    log_path = os.path.join(DATASET, 'training_log.csv')
    log_file = open(log_path, 'w', newline='', encoding='utf-8')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['epoch', 'train_loss', 'train_acc', 'dev_loss', 'dev_acc'])

    model.train()
    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0
    flag = False
    best_dev_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(config.num_epochs):
        if flag:
            break
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        model.train()  # 每个 epoch 开头确保在训练模式

        train_acc_total, train_loss_total, step = 0.0, 0.0, 0
        for texts, labels in train_iter:
            outputs = model(texts)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            train_acc_total += accuracy_score(labels.data.cpu(),
                                              torch.max(outputs.data, 1)[1].cpu())
            train_loss_total += loss.item()
            step += 1

            # ---- 每 eval_interval batch 评估 ----
            if total_batch % eval_interval == 0:
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate_dl(config, model, dev_iter)

                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''

                time_dif = get_time_dif(start_time)
                msg = ('Iter: {:>6},  Train Loss: {:>5.4f},  Train Acc: {:>6.2%},  '
                       'Dev Loss: {:>5.4f},  Dev Acc: {:>6.2%},  Time: {} {}')
                print(msg.format(total_batch, loss.item(), train_acc,
                                 dev_loss, dev_acc, time_dif, improve))
                model.train()

            total_batch += 1

            # ---- batch 级早停 ----
            if total_batch - last_improve > config.require_improvement:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break

        # epoch 末记录 CSV
        train_acc = train_acc_total / step
        train_loss = train_loss_total / step
        dev_acc, dev_loss = evaluate_dl(config, model, dev_iter)
        scheduler.step(dev_acc)
        log_writer.writerow([epoch + 1, train_loss, train_acc, dev_loss, dev_acc])
        model.train()  # 切回训练模式，否则下个 epoch 前 100 batch dropout 不生效

        # epoch 级早停
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.early_stop_patience:
                print(f'\n早停触发！验证集准确率连续 {config.early_stop_patience} epoch 未提升。')
                break

    log_file.close()
    print(f"\n训练日志已保存至: {log_path}")

    # ---- 最终测试 ----
    print("\n开始最终测试评估...")
    test_preds, test_labels = test(config, model, test_iter)
    plot_confusion_matrix_custom(test_labels, test_preds, config.class_list,
                                 'TextCNN Confusion Matrix',
                                 os.path.join(DATASET, 'textcnn_confusion_matrix.png'))


if __name__ == '__main__':
    # 你可以把不想运行的模型注释掉
    train_machine_learning()
    train_textcnn()