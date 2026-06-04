"""
中文新闻分类 交互式演示
运行: python demo.py
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
import jieba
import pickle as pkl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import gradio as gr

from TextCNN_Model import Config, Model, DATASET
from utils import load_stopwords, load_ml_data, build_dl_dataset, build_iterator

# ── 类别 ────────────────────────
CLASS_NAMES = ['finance', 'home', 'stocks', 'education', 'science',
               'society', 'politics', 'sports', 'game', 'entertainment']
CLASS_ZH = ['财经', '家居', '股票', '教育', '科技', '社会', '时政', '体育', '游戏', '娱乐']


# ===== 加载 NB + SVM =====
def load_ml_models():
    train_path = os.path.join(DATASET, 'sampled_data', 'train_sampled.txt')
    stopwords_path = os.path.join(DATASET, 'stopwords_hit.txt')

    stopwords = load_stopwords(stopwords_path)
    train_texts, y_train = load_ml_data(train_path, stopwords)
    tokenize = lambda t: ' '.join(jieba.lcut(t))
    X_train_words = [tokenize(t) for t in train_texts]

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train_words)

    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train_tfidf)

    nb = MultinomialNB(alpha=0.1).fit(X_train_tfidf, y_train)
    svm = CalibratedClassifierCV(
        LinearSVC(C=1.0, random_state=42, max_iter=5000, dual=False),
        cv=3, method='sigmoid').fit(X_train_scaled, y_train)

    return vectorizer, scaler, nb, svm, stopwords


# ===== 加载 TextCNN =====
def load_textcnn():
    config = Config()
    config.load_pretrained('embedding_SougouNews.npz')
    _, _, _, test_data = build_dl_dataset(config, use_word=False)
    config.n_vocab = len(pkl.load(open(config.vocab_path, 'rb')))
    model = Model(config).to(config.device)
    model.load_state_dict(torch.load(config.save_path, map_location=config.device))
    model.eval()
    return model, config


# ===== 预测函数 =====
def predict(title):
    if not title.strip():
        return {c: 0.0 for c in CLASS_ZH}, {c: 0.0 for c in CLASS_ZH}, {c: 0.0 for c in CLASS_ZH}

    # ── NB & SVM ──
    tokenized = ' '.join([w for w in jieba.cut(title) if w not in sw and len(w) > 1])
    X_vec = vec.transform([tokenized])
    X_scaled = scl.transform(X_vec)

    nb_probs = nb_model.predict_proba(X_vec)[0]
    svm_probs = svm_model.predict_proba(X_scaled)[0]

    # ── TextCNN ──
    chars = list(title)
    sl = len(chars)
    if sl < 32:
        chars.extend(['<PAD>'] * (32 - sl))
    else:
        chars = chars[:32]
        sl = 32
    ids = [vocab.get(c, vocab['<UNK>']) for c in chars]
    x = torch.LongTensor([ids]).to(tcnn_config.device)
    sl_t = torch.LongTensor([sl]).to(tcnn_config.device)
    with torch.no_grad():
        logits = tcnn_model((x, sl_t))
        tcnn_probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    def to_dict(probs):
        items = sorted(zip(CLASS_ZH, probs), key=lambda x: -x[1])
        return {k: float(v) for k, v in items}

    return to_dict(nb_probs), to_dict(svm_probs), to_dict(tcnn_probs)


def predict_label(nb_r, svm_r, tcnn_r):
    """从三个模型的预测结果生成汇总标签"""
    nb_top = max(nb_r, key=nb_r.get)
    svm_top = max(svm_r, key=svm_r.get)
    tcnn_top = max(tcnn_r, key=tcnn_r.get)
    return f"NB: {nb_top}  |  SVM: {svm_top}  |  TextCNN: {tcnn_top}"


def predict_all(title):
    """一次性返回所有结果"""
    nb_r, svm_r, tcnn_r = predict(title)
    return nb_r, svm_r, tcnn_r, predict_label(nb_r, svm_r, tcnn_r)


# ===== 启动 =====
print("正在加载模型...")
vec, scl, nb_model, svm_model, sw = load_ml_models()
tcnn_model, tcnn_config = load_textcnn()
vocab = pkl.load(open(tcnn_config.vocab_path, 'rb'))
print("全部模型加载完成！")

with gr.Blocks(title="中文新闻分类系统") as demo:
    gr.Markdown("""
    # 📰 中文新闻标题分类系统
    ### 基于 NB + SVM + TextCNN 三模型集成
    输入一条中文新闻标题，查看三个模型的分类结果和置信度
    """)

    with gr.Row():
        title_input = gr.Textbox(
            label="新闻标题",
            placeholder="例如：沪深两市今日成交额突破万亿 北向资金净流入超百亿",
            lines=1
        )

    with gr.Row():
        submit_btn = gr.Button("开始分类", variant="primary")
        clear_btn = gr.Button("清空")

    with gr.Row():
        result_label = gr.Label(label="📊 综合预测（多数投票）", num_top_classes=3)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📗 朴素贝叶斯 (NB)")
            nb_output = gr.Label(label="置信度", num_top_classes=3)
        with gr.Column():
            gr.Markdown("### 📘 支持向量机 (SVM)")
            svm_output = gr.Label(label="置信度", num_top_classes=3)
        with gr.Column():
            gr.Markdown("### 📙 TextCNN (深度学习)")
            tcnn_output = gr.Label(label="置信度", num_top_classes=3)

    submit_btn.click(
        fn=predict_all,
        inputs=title_input,
        outputs=[nb_output, svm_output, tcnn_output, result_label]
    )
    clear_btn.click(
        fn=lambda: ("", None, None, None, None),
        outputs=[title_input, nb_output, svm_output, tcnn_output, result_label]
    )

    gr.Markdown("""
    ---
    ### 使用说明
    - 输入任意中文新闻标题，点击「开始分类」
    - 三个模型独立预测，输出各自的 Top-3 类别和置信度
    - TextCNN 使用预训练词向量 + batch 级实时评估训练
    """)

demo.launch(share=True)
