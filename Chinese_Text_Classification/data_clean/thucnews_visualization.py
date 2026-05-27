import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. 读取 THUCNews 清洗后数据
# ============================================================
CSV_PATH = os.path.join(BASE_DIR, '..', 'THUCNews', 'news_cleaned_from_thucnews.csv')
df = pd.read_csv(CSV_PATH)
print(f"数据量：{len(df):,}，类别数：{df['category'].nunique()}")

# 英文→中文类别映射
CAT_ZH = {
    'sports': '体育', 'entertainment': '娱乐', 'home': '家居',
    'lottery': '彩票', 'realty': '房产', 'education': '教育',
    'fashion': '时尚', 'politics': '时政', 'constellation': '星座',
    'game': '游戏', 'society': '社会', 'tech': '科技',
    'stocks': '股票', 'finance': '财经',
}
df['category_zh'] = df['category'].map(CAT_ZH)

# ============================================================
# 2. 基本统计
# ============================================================
print("\n=== 基本统计 ===")
cat_counts = df['category_zh'].value_counts()
cat_avg_len = df.groupby('category_zh')['title_len'].mean().sort_values()
print("各类别样本数：\n", cat_counts)
print(f"\n每类平均样本数：{len(df) / df['category_zh'].nunique():.0f}")
print("\n标题长度描述统计：\n", df['title_len'].describe())

# ============================================================
# 3. 可视化
# ============================================================

# ------ 3.1 标题长度：直方图 + KDE + 统计线 ------
fig, ax = plt.subplots(figsize=(12, 5))
ax.hist(df['title_len'], bins=40, edgecolor='white', alpha=0.6, color='steelblue', density=True)
df['title_len'].plot.kde(ax=ax, color='darkorange', linewidth=2.5)
mean_val = df['title_len'].mean()
median_val = df['title_len'].median()
ax.axvline(mean_val, color='crimson', linestyle='--', linewidth=1.5, label=f'均值 {mean_val:.1f}')
ax.axvline(median_val, color='seagreen', linestyle='--', linewidth=1.5, label=f'中位数 {median_val:.0f}')
ax.axvline(mean_val - df['title_len'].std(), color='gray', linestyle=':', linewidth=1, label=f'±1 标准差')
ax.axvline(mean_val + df['title_len'].std(), color='gray', linestyle=':', linewidth=1)
ax.set_xlabel('标题长度（字符数）', fontsize=12)
ax.set_ylabel('密度', fontsize=12)
ax.set_title('标题长度分布（直方图 + KDE）', fontsize=14)
ax.legend(fontsize=9)
ax.set_xlim(0, 50)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'title_length_dist.png'), dpi=150)
plt.close()
print("[1/6] 标题长度分布图已保存")

# ------ 3.2 各类别标题长度箱线图 + 小提琴图 ------
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 箱线图
order_box = df.groupby('category_zh')['title_len'].median().sort_values().index.tolist()
sns.boxplot(data=df, x='category_zh', y='title_len', order=order_box,
            hue='category_zh', palette='Set2', ax=axes[0], width=0.6, legend=False)
axes[0].set_title('各类别标题长度 — 箱线图', fontsize=13)
axes[0].set_xlabel('')
axes[0].set_ylabel('标题长度（字符数）')
axes[0].tick_params(axis='x', rotation=45)

# 小提琴图
sns.violinplot(data=df, x='category_zh', y='title_len', order=order_box,
               hue='category_zh', palette='Set2', ax=axes[1], inner='quartile', cut=0, legend=False)
axes[1].set_title('各类别标题长度 — 小提琴图', fontsize=13)
axes[1].set_xlabel('')
axes[1].set_ylabel('标题长度（字符数）')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'title_length_by_category.png'), dpi=150)
plt.close()
print("[2/6] 各类别标题长度分析图已保存")

# ------ 3.3 类别分布：水平柱状图 ------
fig, ax = plt.subplots(figsize=(10, 8))
cat_order_count = cat_counts.index.tolist()  # 降序
colors = plt.cm.tab20(np.linspace(0, 1, len(cat_order_count)))
bars = ax.barh(range(len(cat_order_count)), cat_counts.values[::-1], color=colors[::-1])
# 反转以让最大值在顶部
cats_rev = cat_order_count[::-1]
counts_rev = cat_counts.values[::-1]
ax.clear()
ax.barh(cats_rev, counts_rev, color=colors[::-1], edgecolor='white')
ax.set_xlabel('样本数', fontsize=12)
ax.set_title('各类别样本数量分布', fontsize=14)
for i, (cat, cnt) in enumerate(zip(cats_rev, counts_rev)):
    ax.text(cnt + 1000, cat, f'{cnt:,}', va='center', fontsize=9,
            color='#333')
ax.set_xlim(0, counts_rev.max() * 1.12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'category_distribution.png'), dpi=150)
plt.close()
print("[3/6] 类别分布图已保存")

# ------ 3.4 类别数量 vs 平均标题长度 散点气泡图 ------
fig, ax = plt.subplots(figsize=(12, 7))
cat_stats = pd.DataFrame({
    'count': cat_counts,
    'avg_len': df.groupby('category_zh')['title_len'].mean(),
    'std_len': df.groupby('category_zh')['title_len'].std(),
})
cat_stats = cat_stats.reset_index()
cat_stats.columns = ['category_zh', 'count', 'avg_len', 'std_len']

scatter = ax.scatter(
    cat_stats['avg_len'], cat_stats['count'],
    s=cat_stats['count'] / 80,  # 气泡大小
    c=np.arange(len(cat_stats)), cmap='tab20', alpha=0.8, edgecolors='black', linewidth=0.5
)
for _, row in cat_stats.iterrows():
    ax.annotate(row['category_zh'], (row['avg_len'], row['count']),
                textcoords="offset points", xytext=(8, 0), fontsize=9,
                ha='left', va='center')
ax.set_xlabel('平均标题长度（字符数）', fontsize=12)
ax.set_ylabel('样本数量', fontsize=12)
ax.set_title('各类别：样本数量 × 平均标题长度（气泡图）', fontsize=14)
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'count_vs_length_bubble.png'), dpi=150)
plt.close()
print("[4/6] 数量—长度关系气泡图已保存")

# ------ 3.5 词云图 ------
font_paths = [
    'C:/Windows/Fonts/simhei.ttf',
    'C:/Windows/Fonts/msyh.ttc',
    'C:/Windows/Fonts/simsun.ttc',
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
]
font_path = None
for fp in font_paths:
    if os.path.exists(fp):
        font_path = fp
        break

if font_path is None:
    print("[5/6] [6/6] 词云图：未找到中文字体，跳过")
else:
    # 整体词云
    all_text = ' '.join(df['title'].sample(min(50000, len(df))).tolist())
    wordcloud = WordCloud(
        width=1200, height=500, background_color='white',
        font_path=font_path, max_words=200, collocations=False,
        max_font_size=120, min_font_size=10,
    ).generate(all_text)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('THUCNews 全部标题词云', fontsize=16, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'wordcloud_all.png'), dpi=150)
    plt.close()
    print("[5/6] 整体词云已保存")

    # 各类别词云（14 类 → 4×4 子图，最后两个留空）
    all_cats = cat_counts.index.tolist()
    fig, axes = plt.subplots(4, 4, figsize=(20, 18))
    axes = axes.flatten()
    for idx, cat in enumerate(all_cats):
        text = ' '.join(df[df['category_zh'] == cat]['title'].sample(
            min(3000, cat_counts[cat])).tolist())
        wc = WordCloud(
            width=350, height=220, background_color='white',
            font_path=font_path, max_words=60, collocations=False,
        ).generate(text)
        axes[idx].imshow(wc, interpolation='bilinear')
        axes[idx].axis('off')
        axes[idx].set_title(f'{cat}（{cat_counts[cat]:,}条）', fontsize=12, pad=8)
    # 隐藏多余的子图
    for idx in range(len(all_cats), 16):
        axes[idx].set_visible(False)
    plt.suptitle('各类别标题词云', fontsize=18, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'wordcloud_by_category.png'), dpi=150)
    plt.close()
    print("[6/6] 分类词云已保存")

# ============================================================
# 4. 数据洞察（控制台输出）
# ============================================================
print("\n" + "=" * 55)
print("=== 数据洞察 ===")
print("=" * 55)

imbalance = cat_counts.max() / cat_counts.min()
print(f"1. 类别不平衡度：{cat_counts.max():,}（{cat_counts.idxmax()}）/ {cat_counts.min():,}（{cat_counts.idxmin()}）= {imbalance:.1f}:1")

longest = cat_avg_len.idxmax()
shortest = cat_avg_len.idxmin()
print(f"2. 标题长度：「{longest}」平均最长({cat_avg_len[longest]:.1f}字)，「{shortest}」平均最短({cat_avg_len[shortest]:.1f}字)")

# Top-3 类别占总量比例
top3 = cat_counts.head(3).sum()
print(f"3. 前3大类别（{', '.join(cat_counts.head(3).index.tolist())}）占比：{top3 / len(df) * 100:.1f}%")

print(f"4. 整体标题长度：均值={df['title_len'].mean():.1f}，中位数={df['title_len'].median():.0f}，标准差={df['title_len'].std():.2f}")

# 标题长度最集中在哪个区间
bin_counts, bin_edges = np.histogram(df['title_len'], bins=20)
peak_range = (bin_edges[bin_counts.argmax()], bin_edges[bin_counts.argmax() + 1])
print(f"5. 标题长度高度集中在 {peak_range[0]:.0f}-{peak_range[1]:.0f} 字符区间，占总数 {bin_counts.max() / len(df) * 100:.1f}%")

# 长短标题的类别差异
print(f"6. 小类别特征：样本最少的 4 类（彩票、星座、时尚、房产），标题长度标准差分别为 "
      f"{', '.join(f'{v:.2f}' for v in cat_stats.set_index('category_zh').loc[['彩票', '星座', '时尚', '房产'], 'std_len'].values)}")

print(f"\n所有图表已保存至：{OUTPUT_DIR}/")
