# task2_train.py
"""
Task 2: Temporal order prediction (True/False)
策略：
1. 对每对 (A, B) 提取非对称特征
2. 比较 A_end <-> B_start 与 B_end <-> A_start 哪个更相似
3. LR / RF / GradientBoosting 训练并比较
"""
import os
import numpy as np
import miditoolkit
from collections import Counter
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# ====== 路径 ======
DATAROOT = "student_files/task2_next_sequence_prediction"
OUTPUT_FILE = "task2_results.txt"
PRED_FILE = "predictions2.json"
# ===================

_log = []
def log(m=""):
    print(m); _log.append(str(m))


# ============================================================
# 1. 单段 MIDI 的特征
# ============================================================
def parse_notes(midi_path):
    """读 midi -> 按 start 排序的 note list"""
    midi = miditoolkit.midi.parser.MidiFile(midi_path)
    notes = []
    for inst in midi.instruments:
        if not inst.is_drum:
            notes.extend(inst.notes)
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes


def segment_histograms(notes):
    """整段的直方图特征（紧凑版，因为音符少所以维度不要太高）"""
    if len(notes) == 0:
        return np.zeros(50, dtype=np.float32)
    
    pitches = np.array([n.pitch for n in notes])
    velocities = np.array([n.velocity for n in notes])
    durations = np.array([n.end - n.start for n in notes])
    starts = np.array([n.start for n in notes])
    
    # Chroma histogram (12)
    chroma = np.zeros(12, dtype=np.float32)
    for p in pitches:
        chroma[p % 12] += 1
    chroma /= chroma.sum() + 1e-9
    
    # Interval histogram，紧凑版 (-7~+7) 即 15 维
    intervals = np.diff(pitches)
    ih = np.zeros(15, dtype=np.float32)
    for iv in np.clip(intervals, -7, 7):
        ih[iv + 7] += 1
    ih /= ih.sum() + 1e-9
    
    # Duration histogram（log-binned, 8 维）
    dur_log = np.log2(np.maximum(durations, 1))
    dh, _ = np.histogram(dur_log, bins=8, range=(0, 12))
    dh = dh.astype(np.float32) / (dh.sum() + 1e-9)
    
    # IOI histogram (8 维)
    iois = np.diff(starts)
    iois = iois[iois > 0]
    if len(iois) > 0:
        ih2_log = np.log2(np.maximum(iois, 1))
        ih2, _ = np.histogram(ih2_log, bins=8, range=(0, 14))
        ih2 = ih2.astype(np.float32) / (ih2.sum() + 1e-9)
    else:
        ih2 = np.zeros(8, dtype=np.float32)
    
    # 标量
    scalars = np.array([
        np.log1p(len(notes)),
        pitches.mean() / 127,
        pitches.std() / 24,
        velocities.mean() / 127,
        durations.mean() / 480,  # 归一化到一拍
        float(pitches.max() - pitches.min()) / 88,
        float(starts.max() - starts.min()) / 1920,  # 大约 4 拍
    ], dtype=np.float32)
    
    return np.concatenate([chroma, ih, dh, ih2, scalars])  # 12+15+8+8+7 = 50


def boundary_features(notes, side='end', n=4):
    """提取一段的边界 n 个音符的特征
    side='end': 取最后 n 个
    side='start': 取最前 n 个
    """
    if len(notes) == 0:
        return np.zeros(20, dtype=np.float32)
    
    if side == 'end':
        sub = notes[-n:]
    else:
        sub = notes[:n]
    
    pitches = np.array([nn.pitch for nn in sub])
    velocities = np.array([nn.velocity for nn in sub])
    
    # Chroma (12)
    chroma = np.zeros(12, dtype=np.float32)
    for p in pitches:
        chroma[p % 12] += 1
    chroma /= chroma.sum() + 1e-9
    
    # 标量
    scalars = np.array([
        pitches.mean() / 127,
        pitches.std() / 24 if len(pitches) > 1 else 0,
        float(pitches.max()) / 127,
        float(pitches.min()) / 127,
        velocities.mean() / 127,
        # 最后/最前的那个音符的 pitch
        float(sub[-1 if side == 'end' else 0].pitch) / 127,
        float(sub[-1 if side == 'end' else 0].pitch % 12) / 12,
        np.log1p(len(sub)),
    ], dtype=np.float32)
    
    return np.concatenate([chroma, scalars])  # 12 + 8 = 20


def pair_features(notes_A, notes_B):
    """对一对 (A, B) 提取所有特征
    设计原则：
    - 包含 A 和 B 的整体特征
    - 包含两种"假设"的边界相似度：
        - 假设 A→B：比较 A_end 和 B_start
        - 假设 B→A：比较 B_end 和 A_start
    - 包含这两个相似度的"差"（核心判别信号）
    """
    # 整段特征
    feat_A = segment_histograms(notes_A)
    feat_B = segment_histograms(notes_B)
    
    # 边界特征
    A_end = boundary_features(notes_A, 'end', n=4)
    A_start = boundary_features(notes_A, 'start', n=4)
    B_end = boundary_features(notes_B, 'end', n=4)
    B_start = boundary_features(notes_B, 'start', n=4)
    
    # 两种假设下的边界差异
    diff_AB = A_end - B_start   # 假设 A→B 时的差（应该小）
    diff_BA = B_end - A_start   # 假设 B→A 时的差（应该小）
    abs_diff_AB = np.abs(diff_AB)
    abs_diff_BA = np.abs(diff_BA)
    
    # 两种假设下的相似度差（核心标量）—— 这是判别力最强的信号
    sim_AB = -np.linalg.norm(diff_AB)   # A→B 边界相似度
    sim_BA = -np.linalg.norm(diff_BA)   # B→A 边界相似度
    sim_AB_l1 = -np.sum(abs_diff_AB)
    sim_BA_l1 = -np.sum(abs_diff_BA)
    sim_AB_cos = float(np.dot(A_end, B_start) / (np.linalg.norm(A_end) * np.linalg.norm(B_start) + 1e-9))
    sim_BA_cos = float(np.dot(B_end, A_start) / (np.linalg.norm(B_end) * np.linalg.norm(A_start) + 1e-9))
    
    # 边界处最后/最前那个音符的 pitch 差（最经典的边界信号）
    last_pitch_A = notes_A[-1].pitch if notes_A else 0
    first_pitch_A = notes_A[0].pitch if notes_A else 0
    last_pitch_B = notes_B[-1].pitch if notes_B else 0
    first_pitch_B = notes_B[0].pitch if notes_B else 0
    
    pitch_gap_AB = abs(last_pitch_A - first_pitch_B)  # A→B 边界 pitch 差
    pitch_gap_BA = abs(last_pitch_B - first_pitch_A)  # B→A 边界 pitch 差
    
    boundary_scalars = np.array([
        sim_AB, sim_BA, sim_AB - sim_BA,           # 核心信号：差值
        sim_AB_l1, sim_BA_l1, sim_AB_l1 - sim_BA_l1,
        sim_AB_cos, sim_BA_cos, sim_AB_cos - sim_BA_cos,
        pitch_gap_AB / 24, pitch_gap_BA / 24,
        (pitch_gap_AB - pitch_gap_BA) / 24,        # 核心信号：边界差
    ], dtype=np.float32)
    
    # 拼接所有
    feat = np.concatenate([
        feat_A,           # 50
        feat_B,           # 50
        A_end, A_start,   # 40
        B_end, B_start,   # 40
        diff_AB, diff_BA, # 40
        boundary_scalars, # 12
    ])  # = 232
    return feat


# ============================================================
# 2. 数据加载 + 特征提取
# ============================================================
def load_data():
    with open(os.path.join(DATAROOT, "train.json")) as f:
        train = eval(f.read())
    with open(os.path.join(DATAROOT, "test.json")) as f:
        test = eval(f.read())
    return train, test


# 缓存 parse 结果，避免重复解析同一文件
_notes_cache = {}
def get_notes(path):
    if path not in _notes_cache:
        _notes_cache[path] = parse_notes(os.path.join(DATAROOT, path))
    return _notes_cache[path]


def extract_pair_features(pairs, name="data"):
    feats = []
    for (p1, p2) in tqdm(pairs, desc=f"Extracting {name}"):
        notes_A = get_notes(p1)
        notes_B = get_notes(p2)
        feats.append(pair_features(notes_A, notes_B))
    return np.array(feats, dtype=np.float32)


# ============================================================
# 3. 训练 + 评估
# ============================================================
def cv_evaluate(X, y, model_fn, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xva = scaler.transform(X[va])
        m = model_fn()
        m.fit(Xtr, y[tr])
        pred = m.predict(Xva)
        acc = accuracy_score(y[va], pred)
        scores.append(acc)
        log(f"  Fold {fold+1}: acc = {acc:.4f}")
    log(f"  Mean ± Std: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    return np.mean(scores)


def main():
    log("=" * 70)
    log("Task 2: Temporal Order Prediction")
    log("=" * 70)
    
    train_meta, test_pairs = load_data()
    train_pairs = list(train_meta.keys())
    y = np.array([1 if train_meta[p] else 0 for p in train_pairs])
    
    log(f"训练对: {len(train_pairs)}, 测试对: {len(test_pairs)}")
    log(f"True 比例: {y.mean():.4f}")
    
    log("\n--- 特征提取 ---")
    X_train = extract_pair_features(train_pairs, "train")
    X_test = extract_pair_features(test_pairs, "test")
    log(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    log(f"涉及的不同 MIDI 文件数（缓存大小）: {len(_notes_cache)}")
    
    # CV 对比 3 个模型
    log("\n" + "=" * 70)
    log("Logistic Regression")
    log("=" * 70)
    def make_lr():
        return LogisticRegression(max_iter=3000, C=1.0, solver='lbfgs', n_jobs=-1)
    lr_acc = cv_evaluate(X_train, y, make_lr)
    
    log("\n" + "=" * 70)
    log("Random Forest")
    log("=" * 70)
    def make_rf():
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                       random_state=42, n_jobs=-1)
    rf_acc = cv_evaluate(X_train, y, make_rf)
    
    log("\n" + "=" * 70)
    log("Gradient Boosting")
    log("=" * 70)
    def make_gbm():
        return GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                           learning_rate=0.05, random_state=42)
    gbm_acc = cv_evaluate(X_train, y, make_gbm)
    
    # 选最优
    candidates = [("LR", lr_acc, make_lr), ("RF", rf_acc, make_rf), ("GBM", gbm_acc, make_gbm)]
    best_name, best_acc, make_best = max(candidates, key=lambda x: x[1])
    log(f"\n>>> 最佳模型: {best_name} (CV acc = {best_acc:.4f})")
    
    # 全数据重训 + 测试预测
    log(f"\n在全训练集重训 {best_name} 生成预测...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    final = make_best()
    final.fit(X_train_s, y)
    test_preds = final.predict(X_test_s)
    
    # 写预测：autograder 期待 bool 类型
    predictions = {pair: bool(p) for pair, p in zip(test_pairs, test_preds)}
    with open(PRED_FILE, 'w') as f:
        f.write(repr(predictions) + '\n')
    log(f"[✓] 已写入: {PRED_FILE}")
    log(f"预测数: {len(predictions)}")
    log(f"预测 True 比例: {sum(predictions.values()) / len(predictions):.4f}")
    
    with open(OUTPUT_FILE, 'w') as f:
        f.write('\n'.join(_log) + '\n')
    print(f"\n[✓] 日志已写入: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()