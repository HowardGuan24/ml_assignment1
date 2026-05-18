# statistics/task1_compare_predictions.py
"""
对比我们的 predictions1.json 和基线版本的格式
找出差异
"""

# 加载两个文件
with open("results/predictions1.json") as f:
    ours_text = f.read()
with open("statistics/predictions1_baseline.json") as f:
    base_text = f.read()

ours = eval(ours_text)
base = eval(base_text)

print(f"=== 我们的 predictions1.json ===")
print(f"  长度: {len(ours_text)} chars")
print(f"  entry 数: {len(ours)}")
print(f"  前 3 entries:")
for k, v in list(ours.items())[:3]:
    print(f"    key={k!r} (type={type(k).__name__}), value={v!r} (type={type(v).__name__})")

print(f"\n=== 基线 predictions1_baseline.json ===")
print(f"  长度: {len(base_text)} chars")
print(f"  entry 数: {len(base)}")
print(f"  前 3 entries:")
for k, v in list(base.items())[:3]:
    print(f"    key={k!r} (type={type(k).__name__}), value={v!r} (type={type(v).__name__})")

print(f"\n=== 一致性检查 ===")
# 1. key 集合是否相同
ours_keys = set(ours.keys())
base_keys = set(base.keys())
common = ours_keys & base_keys
only_ours = ours_keys - base_keys
only_base = base_keys - ours_keys

print(f"  共同 key 数: {len(common)}")
print(f"  只在我们的: {len(only_ours)} (前 3: {list(only_ours)[:3]})")
print(f"  只在基线: {len(only_base)} (前 3: {list(only_base)[:3]})")

# 2. value 类型
ours_v_types = set(type(v).__name__ for v in ours.values())
base_v_types = set(type(v).__name__ for v in base.values())
print(f"  我们的 value 类型: {ours_v_types}")
print(f"  基线 value 类型: {base_v_types}")

# 3. key 类型
ours_k_types = set(type(k).__name__ for k in ours.keys())
base_k_types = set(type(k).__name__ for k in base.keys())
print(f"  我们的 key 类型: {ours_k_types}")
print(f"  基线 key 类型: {base_k_types}")

# 4. 文件开头格式
print(f"\n=== 文件开头对比 ===")
print(f"  我们的开头 100 字符: {ours_text[:100]!r}")
print(f"  基线的开头 100 字符: {base_text[:100]!r}")
print(f"  开头 10 字符是否一致: {ours_text[:10] == base_text[:10]}")

# 5. 看看 baseline 预测 vs 我们预测有多少不一样
diff_count = 0
for k in common:
    if ours[k] != base[k]:
        diff_count += 1
print(f"\n  预测不同的样本数: {diff_count}/{len(common)} ({diff_count/len(common)*100:.1f}%)")
print(f"  (如果 baseline 准确率是 0.25，我们的是 0.60，差异应该 > 0.35*len)")