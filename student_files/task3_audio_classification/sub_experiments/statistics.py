# analyze_task3.py
import json
from collections import Counter
import numpy as np

with open('train.json') as f:
    train = json.load(f)
with open('test.json') as f:
    test = json.load(f)

print("="*60)
print("基本规模")
print("="*60)
print(f"训练集样本数: {len(train)}")
print(f"测试集样本数: {len(test)}")
print(f"训练/测试比例: {len(train)/len(test):.2f}")

# 标签分析
all_tags = []
tags_per_sample = []
for fname, tags in train.items():
    all_tags.extend(tags)
    tags_per_sample.append(len(tags))

tag_counts = Counter(all_tags)
print(f"\n{'='*60}")
print("标签集合")
print(f"{'='*60}")
print(f"标签总数（独立类别）: {len(tag_counts)}")
print(f"所有标签: {sorted(tag_counts.keys())}")

print(f"\n{'='*60}")
print("每个标签的样本数（按频率排序）")
print(f"{'='*60}")
for tag, count in tag_counts.most_common():
    pct = count / len(train) * 100
    bar = '█' * int(pct / 2)
    print(f"  {tag:15s} {count:5d} ({pct:5.2f}%)  {bar}")

print(f"\n{'='*60}")
print("每个样本的标签数分布")
print(f"{'='*60}")
tps_counter = Counter(tags_per_sample)
for n_tags in sorted(tps_counter.keys()):
    count = tps_counter[n_tags]
    pct = count / len(train) * 100
    bar = '█' * int(pct / 2)
    print(f"  {n_tags} 个标签: {count:4d} 样本 ({pct:5.2f}%)  {bar}")
print(f"\n平均每样本标签数: {np.mean(tags_per_sample):.2f}")
print(f"中位数: {np.median(tags_per_sample):.0f}")
print(f"最多: {max(tags_per_sample)}, 最少: {min(tags_per_sample)}")

# 标签共现
print(f"\n{'='*60}")
print("标签共现 TOP 15（哪些标签经常一起出现）")
print(f"{'='*60}")
cooccur = Counter()
for fname, tags in train.items():
    if len(tags) >= 2:
        sorted_tags = sorted(tags)
        for i in range(len(sorted_tags)):
            for j in range(i+1, len(sorted_tags)):
                cooccur[(sorted_tags[i], sorted_tags[j])] += 1

for (t1, t2), count in cooccur.most_common(15):
    print(f"  {t1:12s} + {t2:12s}: {count} 次")

# 长尾分析
print(f"\n{'='*60}")
print("类别不平衡程度")
print(f"{'='*60}")
counts = sorted(tag_counts.values(), reverse=True)
print(f"最多类 / 最少类 = {counts[0]} / {counts[-1]} = {counts[0]/counts[-1]:.2f}x")
print(f"前 3 类样本总数: {sum(counts[:3])} ({sum(counts[:3])/sum(counts)*100:.1f}%)")
print(f"后 3 类样本总数: {sum(counts[-3:])} ({sum(counts[-3:])/sum(counts)*100:.1f}%)")