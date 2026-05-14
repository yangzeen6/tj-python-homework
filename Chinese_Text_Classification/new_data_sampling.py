import pandas as pd
import os

# ==================== 配置参数 ====================
CSV_PATH = "THUCNews/news_cleaned_from_thucnews.csv"
OUTPUT_DIR = "THUCNews/sampled_data"
SAMPLES_PER_CLASS = 19000       # 每类抽取数量
RANDOM_SEED = 42                # 固定随机种子

LABEL_MAP = {
    'finance': 0,
    'realty': 1,
    'stocks': 2,
    'education': 3,
    'tech': 4,
    'society': 5,
    'politics': 6,
    'sports': 7,
    'game': 8,
    'entertainment': 9,
    'home': 10
}

# 分割比例
TRAIN_RATIO = 0.9
DEV_RATIO   = 0.05
TEST_RATIO  = 0.05

# ==================== 主要处理逻辑 ====================
def main():
    # 读取数据
    df = pd.read_csv(CSV_PATH, header=None, encoding='utf-8', dtype={2: str})

    
    # 丢弃第一行
    df = df.iloc[1:].reset_index(drop=True)
    
    # 重命名列
    df.columns = ['title', 'category', 'length']
    
    # 过滤掉类别不在检测类别范围内的行
    df = df[df['category'].isin(LABEL_MAP.keys())]
    
    # 按类别抽取样本
    sampled_dfs = []
    for cat, label_id in LABEL_MAP.items():
        cat_data = df[df['category'] == cat]
        available = len(cat_data)
        if available < SAMPLES_PER_CLASS:
            print(f"警告：类别 '{cat}' 只有 {available} 条数据，不足 {SAMPLES_PER_CLASS}，将全部使用。")
            n_sample = available
        else:
            n_sample = SAMPLES_PER_CLASS
        
        # 随机抽样，固定 random_state 保证可复现
        sampled = cat_data.sample(n=n_sample, random_state=RANDOM_SEED)
        # 添加数字标签列
        sampled['label'] = label_id
        sampled_dfs.append(sampled[['title', 'label']])
    
    # 合并所有抽样数据，并按类别分层
    train_list, dev_list, test_list = [], [], []
    
    for cat_df in sampled_dfs:
        # 将该类数据随机打乱
        cat_df = cat_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        total = len(cat_df)
        train_end = int(total * TRAIN_RATIO)
        dev_end = train_end + int(total * DEV_RATIO)
        
        train_part = cat_df.iloc[:train_end]
        dev_part   = cat_df.iloc[train_end:dev_end]
        test_part  = cat_df.iloc[dev_end:]
        
        train_list.append(train_part)
        dev_list.append(dev_part)
        test_list.append(test_part)
    
    # 合并所有类
    train_df = pd.concat(train_list, ignore_index=True)
    dev_df   = pd.concat(dev_list, ignore_index=True)
    test_df  = pd.concat(test_list, ignore_index=True)
    
    # 再次整体打乱
    train_df = train_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    dev_df   = dev_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    test_df  = test_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    # 保存为所需格式：标题\t标签
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    def save_txt(df, filename):
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                f.write(f"{row['title']}\t{row['label']}\n")
        print(f"已保存: {path}，行数: {len(df)}条")
    
    save_txt(train_df, 'train_sampled.txt')
    save_txt(dev_df, 'dev_sampled.txt')
    save_txt(test_df, 'test_sampled.txt')
    
if __name__ == "__main__":
    main()