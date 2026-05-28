import os
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── 配置 ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE_DIR, 'THUCNews', 'training_log.csv')
SAVE_DIR = os.path.join(BASE_DIR, 'THUCNews')

# ── 样式设置 ──────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 12,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def load_log(path):
    epochs, train_loss, train_acc, dev_loss, dev_acc = [], [], [], [], []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            train_loss.append(float(row['train_loss']))
            train_acc.append(float(row['train_acc']))
            dev_loss.append(float(row['dev_loss']))
            dev_acc.append(float(row['dev_acc']))
    return epochs, train_loss, train_acc, dev_loss, dev_acc


def save_figure(fig, name):
    path = os.path.join(SAVE_DIR, name)
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    print(f'已保存: {path}')
    plt.close(fig)


def plot_loss(epochs, train_loss, dev_loss):
    """Training & Validation Loss"""
    fig, ax = plt.subplots(figsize=(10, 5))

    color_train = '#2c7bb6'
    color_dev = '#d7191c'

    ax.plot(epochs, train_loss, color=color_train, linewidth=2.2, marker='o',
            markersize=5, markerfacecolor='white', markeredgewidth=1.5, label='Training Loss')
    ax.plot(epochs, dev_loss, color=color_dev, linewidth=2.2, marker='s',
            markersize=5, markerfacecolor='white', markeredgewidth=1.5, label='Validation Loss')

    # 标注最优 dev loss
    best_idx = np.argmin(dev_loss)
    ax.annotate(f'Best: {dev_loss[best_idx]:.4f}\n(Epoch {epochs[best_idx]})',
                xy=(epochs[best_idx], dev_loss[best_idx]),
                xytext=(epochs[best_idx] + 1.2, dev_loss[best_idx] * 0.92),
                fontsize=10, color=color_dev, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=color_dev, lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color_dev, alpha=0.8))

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend(loc='upper right')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    save_figure(fig, 'curve_loss.png')


def plot_accuracy(epochs, train_acc, dev_acc):
    """Training & Validation Accuracy"""
    fig, ax = plt.subplots(figsize=(10, 5))

    color_train = '#2c7bb6'
    color_dev = '#d7191c'

    ax.plot(epochs, train_acc, color=color_train, linewidth=2.2, marker='o',
            markersize=5, markerfacecolor='white', markeredgewidth=1.5, label='Training Accuracy')
    ax.plot(epochs, dev_acc, color=color_dev, linewidth=2.2, marker='s',
            markersize=5, markerfacecolor='white', markeredgewidth=1.5, label='Validation Accuracy')

    # 标注最优 dev acc (框放在最后一个 epoch 下方)
    best_idx = np.argmax(dev_acc)
    last = len(epochs) - 1
    ax.annotate(f'Best Val Acc: {dev_acc[best_idx]:.2%}',
                xy=(epochs[best_idx], dev_acc[best_idx]),
                xytext=(epochs[last], dev_acc[last] - 0.02),
                fontsize=10, color=color_dev, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=color_dev, lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color_dev, alpha=0.8))

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training & Validation Accuracy')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.legend(loc='lower right')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    save_figure(fig, 'curve_accuracy.png')


def plot_loss_only(epochs, train_loss, dev_loss):
    """Training Loss (单独)"""
    fig, ax = plt.subplots(figsize=(10, 5))
    color = '#d7191c'
    ax.plot(epochs, dev_loss, color=color, linewidth=2.5, marker='o', markersize=6,
            markerfacecolor='white', markeredgewidth=2, zorder=5)
    ax.fill_between(epochs, dev_loss, alpha=0.08, color=color)
    best_idx = np.argmin(dev_loss)
    ax.scatter([epochs[best_idx]], [dev_loss[best_idx]], c=color, s=120, zorder=10,
               edgecolors='white', linewidth=1.5)
    y_offset = (max(dev_loss) - min(dev_loss)) * 0.15
    ax.annotate(f'Best: {dev_loss[best_idx]:.4f}',
                xy=(epochs[best_idx], dev_loss[best_idx]),
                xytext=(epochs[best_idx] + 1, dev_loss[best_idx] + y_offset),
                fontsize=11, color=color, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.8))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    save_figure(fig, 'curve_val_loss.png')


def plot_accuracy_only(epochs, train_acc, dev_acc):
    """Validation Accuracy (单独)"""
    fig, ax = plt.subplots(figsize=(10, 5))
    color = '#2c7bb6'
    ax.plot(epochs, dev_acc, color=color, linewidth=2.5, marker='s', markersize=6,
            markerfacecolor='white', markeredgewidth=2, zorder=5)
    ax.fill_between(epochs, dev_acc, alpha=0.08, color=color)
    best_idx = np.argmax(dev_acc)
    ax.scatter([epochs[best_idx]], [dev_acc[best_idx]], c=color, s=120, zorder=10,
               edgecolors='white', linewidth=1.5)
    y_offset = (max(dev_acc) - min(dev_acc)) * 0.15
    ax.annotate(f'Best: {dev_acc[best_idx]:.2%}',
                xy=(epochs[best_idx], dev_acc[best_idx]),
                xytext=(epochs[best_idx] + 1, dev_acc[best_idx] + y_offset),
                fontsize=11, color=color, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.8))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Validation Accuracy')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    save_figure(fig, 'curve_val_accuracy.png')


def plot_all_in_one(epochs, train_loss, train_acc, dev_loss, dev_acc):
    """四合一总览图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-Left: Loss
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, color='#2c7bb6', linewidth=2, marker='o', markersize=4,
            markerfacecolor='white', markeredgewidth=1.2, label='Train')
    ax.plot(epochs, dev_loss, color='#d7191c', linewidth=2, marker='s', markersize=4,
            markerfacecolor='white', markeredgewidth=1.2, label='Val')
    ax.set_title('Loss')
    ax.legend(fontsize=10)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    # Top-Right: Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, train_acc, color='#2c7bb6', linewidth=2, marker='o', markersize=4,
            markerfacecolor='white', markeredgewidth=1.2, label='Train')
    ax.plot(epochs, dev_acc, color='#d7191c', linewidth=2, marker='s', markersize=4,
            markerfacecolor='white', markeredgewidth=1.2, label='Val')
    ax.set_title('Accuracy')
    ax.legend(fontsize=10)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

    # Bottom-Left: Val Loss Only
    ax = axes[1, 0]
    ax.plot(epochs, dev_loss, color='#d7191c', linewidth=2.5)
    ax.fill_between(epochs, dev_loss, alpha=0.08, color='#d7191c')
    best = np.argmin(dev_loss)
    ax.scatter([epochs[best]], [dev_loss[best]], c='#d7191c', s=80, zorder=10)
    ax.set_title('Validation Loss (Best: {:.4f})'.format(dev_loss[best]))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    # Bottom-Right: Val Acc Only
    ax = axes[1, 1]
    ax.plot(epochs, dev_acc, color='#2c7bb6', linewidth=2.5)
    ax.fill_between(epochs, dev_acc, alpha=0.08, color='#2c7bb6')
    best = np.argmax(dev_acc)
    ax.scatter([epochs[best]], [dev_acc[best]], c='#2c7bb6', s=80, zorder=10)
    ax.set_title('Validation Accuracy (Best: {:.2%})'.format(dev_acc[best]))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

    fig.suptitle('TextCNN Training Curves', fontsize=18, fontweight='bold', y=1.01)
    for ax in axes.flat:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.subplots_adjust(top=0.92)
    save_figure(fig, 'curve_summary.png')


# ── 主流程 ──────────────────────────────────────────
if __name__ == '__main__':
    if not os.path.exists(LOG_PATH):
        print(f'错误: 找不到训练日志 {LOG_PATH}')
        print('请先运行 python train.py 生成 training_log.csv')
        exit(1)

    epochs, train_loss, train_acc, dev_loss, dev_acc = load_log(LOG_PATH)
    print(f'已加载 {len(epochs)} 个 epoch 的训练日志')

    plot_loss(epochs, train_loss, dev_loss)
    plot_accuracy(epochs, train_acc, dev_acc)
    plot_loss_only(epochs, train_loss, dev_loss)
    plot_accuracy_only(epochs, train_acc, dev_acc)
    plot_all_in_one(epochs, train_loss, train_acc, dev_loss, dev_acc)

    print('\n5 张曲线图全部生成完成！')
