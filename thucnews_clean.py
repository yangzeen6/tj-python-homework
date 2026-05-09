import os
import re
import pandas as pd

# ============================================================
# 1. 配置路径
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
THUCNEWS_DIR = os.path.join(BASE_DIR, 'THUCNews')

# 中文文件夹名 → 英文类别名 映射
CN_TO_EN = {
    '体育': 'sports',
    '娱乐': 'entertainment',
    '家居': 'home',
    '彩票': 'lottery',
    '房产': 'realty',
    '教育': 'education',
    '时尚': 'fashion',
    '时政': 'politics',
    '星座': 'constellation',
    '游戏': 'game',
    '社会': 'society',
    '科技': 'tech',
    '股票': 'stocks',
    '财经': 'finance',
}

# ============================================================
# 2. 遍历 THUCNews 子文件夹，提取标题
# ============================================================
records = []
category_counts = {}

for folder_name in os.listdir(THUCNEWS_DIR):
    folder_path = os.path.join(THUCNEWS_DIR, folder_name)
    if not os.path.isdir(folder_path):
        continue

    en_cat = CN_TO_EN.get(folder_name)
    if en_cat is None:
        print(f"警告：未知类别文件夹 '{folder_name}'，跳过")
        continue

    count = 0
    for filename in os.listdir(folder_path):
        if not filename.endswith('.txt'):
            continue
        filepath = os.path.join(folder_path, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
        except (UnicodeDecodeError, IOError):
            continue

        if first_line:
            records.append({'title': first_line, 'category': en_cat})
            count += 1
        if count % 10000 == 0 and count > 0:
            print(f"  {folder_name}: 已读取 {count} 条...")

    category_counts[en_cat] = count
    print(f"✓ {folder_name} → {en_cat}: {count} 条")

print(f"\n总读取记录数: {len(records)}")

# ============================================================
# 3. 构建 DataFrame 并清洗
# ============================================================
df = pd.DataFrame(records)

# 3.1 去重（基于标题）
before = len(df)
df.drop_duplicates(subset=['title'], keep='first', inplace=True)
print(f"去除重复标题: {before - len(df)} 条")

# 3.2 标题清洗
df['title'] = df['title'].astype(str).str.strip()

# 去除特殊字符，保留中文、英文、数字、常见标点
def clean_title(text):
    # 去除新闻标记词前缀
    text = re.sub(
        r'（组图）|（图）|\(组图\)|\(图\)|组图：|图文：|图文-|'
        r'快讯：|评论：|.*日报：|趣味测试：',
        '', text
    )
    # 保留汉字、字母、数字、空格和常用中文标点
    text = re.sub(r'[^一-龥a-zA-Z0-9\s，。、；：""（）【】《》！？…—·％％]', '', text)
    # 合并多余空白
    text = re.sub(r'\s+', '', text)
    return text

df['title'] = df['title'].apply(clean_title)

# 删除清洗后为空或过短的标题
before = len(df)
df = df[df['title'].str.len() > 0]
print(f"删除清洗后空白标题: {before - len(df)} 条")

# 3.2.5 数字替换后去重 —— 将数字替换为空，基于去数字文本找重复索引，再去重原始标题
before = len(df)
df['title_nodigit'] = df['title'].str.replace(r'\d+', '', regex=True)
dup_mask = df.duplicated(subset=['title_nodigit'], keep='first')
dup_count = dup_mask.sum()
df = df[~dup_mask]
df.drop(columns=['title_nodigit'], inplace=True)
print(f"去除仅数字差异的重复标题: {before - len(df)} 条")

# 3.3 标题长度异常过滤
df['title_len'] = df['title'].str.len()
before = len(df)
df = df[(df['title_len'] >= 6) & (df['title_len'] <= 40)]
print(f"删除标题长度异常 (<6 或 >40): {before - len(df)} 条")

# ============================================================
# 4. 保存结果
# ============================================================
df = df[['title', 'category', 'title_len']].reset_index(drop=True)

output_path = os.path.join(BASE_DIR, 'news_cleaned_from_thucnews.csv')
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n清洗后数据量: {len(df)}")
print(f"类别分布:\n{df['category'].value_counts()}")
print(f"已保存至: {output_path}")
