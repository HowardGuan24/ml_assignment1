# statistics/task1_pred_sanity.py
"""
检查 predictions1.json 是否符合预期
"""
import os

with open("results/predictions1.json") as f:
    preds = eval(f.read())
with open("student_files/task1_composer_classification/test.json") as f:
    test_keys = eval(f.read())

# 1. key 完整性
missing = [k for k in test_keys if k not in preds]
extra = [k for k in preds if k not in test_keys]
print(f"预测数: {len(preds)}")
print(f"测试集 key 数: {len(test_keys)}")
print(f"缺失的预测: {len(missing)}")
print(f"多余的预测: {len(extra)}")

# 2. value 类型和范围
from collections import Counter
vals = list(preds.values())
print(f"\nvalue 类型: {set(type(v).__name__ for v in vals)}")
print(f"value 范围: {min(vals)} ~ {max(vals)}")
print(f"value 分布: {dict(sorted(Counter(vals).items()))}")

# 3. 不能全是同一个值
if len(set(vals)) == 1:
    print(f"\n⚠️ 警告：所有预测都是同一个值！")
else:
    print(f"\n✓ 预测有 {len(set(vals))} 种不同值")