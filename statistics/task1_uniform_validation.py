# statistics/task1_uniform_validation.py
"""
新评估协议：
- 不再用 stratified 5-fold（同分布假设错）
- 而是从每个 composer 取相同数量样本作为验证集
- 这样验证集是均匀分布，更接近真实测试集
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 加载特征缓存
cache = np.load("statistics/task1_features_G.npz", allow_pickle=True)
X, y = cache['X'], cache['y']

print("各类样本数:", {c: int((y==c).sum()) for c in range(8)})

# 取每个类样本数最少的 75% 作为训练，剩下作为验证（每类等量）
# composer 7 只有 37 个 → 训练用 27 个，验证用 10 个
# 其他类相应地取
np.random.seed(42)
n_val_per_class = 10  # 每类 10 个验证（这样验证集 80 个，完全均匀）

train_idx, val_idx = [], []
for c in range(8):
    cls_idx = np.where(y == c)[0]
    np.random.shuffle(cls_idx)
    val_idx.extend(cls_idx[:n_val_per_class])
    train_idx.extend(cls_idx[n_val_per_class:])

train_idx = np.array(train_idx)
val_idx = np.array(val_idx)

print(f"训练集: {len(train_idx)}, 验证集: {len(val_idx)} (每类 {n_val_per_class} 个)")

# 评估几个模型
models = {
    "LR no weight":            LogisticRegression(max_iter=3000, C=1.0, n_jobs=-1),
    "LR balanced":             LogisticRegression(max_iter=3000, C=1.0, class_weight='balanced', n_jobs=-1),
    "LR balanced + C=0.3":     LogisticRegression(max_iter=3000, C=0.3, class_weight='balanced', n_jobs=-1),
    "LR balanced + C=3":       LogisticRegression(max_iter=3000, C=3.0, class_weight='balanced', n_jobs=-1),
    "LR balanced + C=10":      LogisticRegression(max_iter=3000, C=10.0, class_weight='balanced', n_jobs=-1),
}

print("\n模型对照（在均匀验证集上）:")
for name, m in models.items():
    sc = StandardScaler()
    Xtr, Xva = sc.fit_transform(X[train_idx]), sc.transform(X[val_idx])
    m.fit(Xtr, y[train_idx])
    pred = m.predict(Xva)
    acc = accuracy_score(y[val_idx], pred)
    print(f"  {name:30s}: {acc:.4f}")

# 多种子稳定性
print("\n--- 用 5 个不同随机种子验证稳定性（class_weight=balanced）---")
all_accs = []
for seed in [42, 0, 1, 2, 100]:
    np.random.seed(seed)
    tr_idx, va_idx = [], []
    for c in range(8):
        cls_idx = np.where(y == c)[0]
        np.random.shuffle(cls_idx)
        va_idx.extend(cls_idx[:n_val_per_class])
        tr_idx.extend(cls_idx[n_val_per_class:])
    tr_idx, va_idx = np.array(tr_idx), np.array(va_idx)
    
    sc = StandardScaler()
    Xtr, Xva = sc.fit_transform(X[tr_idx]), sc.transform(X[va_idx])
    m = LogisticRegression(max_iter=3000, C=1.0, class_weight='balanced', n_jobs=-1)
    m.fit(Xtr, y[tr_idx])
    acc = accuracy_score(y[va_idx], m.predict(Xva))
    all_accs.append(acc)
    print(f"  seed={seed}: {acc:.4f}")
print(f"  Mean ± Std: {np.mean(all_accs):.4f} ± {np.std(all_accs):.4f}")