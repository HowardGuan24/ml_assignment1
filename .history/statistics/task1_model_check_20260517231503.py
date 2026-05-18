# statistics/task1_model_check.py
"""
Task 1 模型最终对照：
固定特征为 G 配置（v2 + pitch, 341 维）
对比：
  M1. LR + 不加 class_weight    （主推荐）
  M2. LR + class_weight=balanced (验证之前的猜测)
  M3. LR + class_weight=balanced + 调小 C（更强正则化）
  M4. RandomForest               (验证树模型是否更稳)
  M5. LR (M1) + RF (M4) 集成     (看融合是否提升)
"""
import os
import numpy as np
import miditoolkit
from collections import defaultdict
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

DATAROOT = "student_files/task1_composer_classification"
OUTPUT_FILE = "statistics/task1_model_check.txt"

_log = []
def log(m=""):
    print(m); _log.append(str(m))


# ============================================================
# 特征提取（复制自 task1_feature_v2.py）
# ============================================================
PITCH_MIN, PITCH_MAX = 21, 108
N_PITCHES = PITCH_MAX - PITCH_MIN + 1
INTERVAL_MIN, INTERVAL_MAX = -12, 12
N_INTERVALS = INTERVAL_MAX - INTERVAL_MIN + 1


def compute_polyphony(notes):
    events = []
    for n in notes:
        events.append((n.start, 1)); events.append((n.end, -1))
    events.sort()
    cur = max_p = 0
    for _, d in events:
        cur += d; max_p = max(max_p, cur)
    return max_p


def interval_hist(pitch_sequence):
    if len(pitch_sequence) < 2:
        return np.zeros(N_INTERVALS, dtype=np.float32)
    intervals = np.diff(np.array(pitch_sequence))
    ih = np.zeros(N_INTERVALS, dtype=np.float32)
    for iv in np.clip(intervals, INTERVAL_MIN, INTERVAL_MAX):
        ih[iv - INTERVAL_MIN] += 1
    ih /= ih.sum() + 1e-9
    return ih


def interval_transition_mod12(pitch_sequence):
    if len(pitch_sequence) < 3:
        return np.zeros((12, 12), dtype=np.float32)
    intervals = np.diff(np.array(pitch_sequence))
    ic = intervals % 12
    tr = np.zeros((12, 12), dtype=np.float32)
    for i in range(len(ic) - 1):
        tr[ic[i], ic[i+1]] += 1
    return tr


def ioi_hist(start_sequence):
    if len(start_sequence) < 2:
        return np.zeros(12, dtype=np.float32)
    iois = np.diff(np.array(start_sequence))
    iois = iois[iois > 0]
    if len(iois) == 0:
        return np.zeros(12, dtype=np.float32)
    il = np.log2(np.maximum(iois, 1))
    h, _ = np.histogram(il, bins=12, range=(0, 14))
    h = h.astype(np.float32) / (h.sum() + 1e-9)
    return h


def extract_G_features(midi_path):
    """直接拼接 G 配置（341 维）"""
    midi = miditoolkit.midi.parser.MidiFile(midi_path)
    notes = []
    for inst in midi.instruments:
        if not inst.is_drum:
            notes.extend(inst.notes)
    
    if len(notes) < 2:
        return np.zeros(341, dtype=np.float32)
    
    notes.sort(key=lambda n: (n.start, n.pitch))
    pitches = np.array([n.pitch for n in notes])
    velocities = np.array([n.velocity for n in notes])
    durations = np.array([n.end - n.start for n in notes])
    starts = np.array([n.start for n in notes])
    
    events = defaultdict(list)
    for n in notes:
        events[n.start].append(n)
    sorted_starts = sorted(events.keys())
    melody_pitches = [max(nn.pitch for nn in events[s]) for s in sorted_starts]
    bass_pitches = [min(nn.pitch for nn in events[s]) for s in sorted_starts]
    
    # pitch (88)
    ph = np.zeros(N_PITCHES, dtype=np.float32)
    for p in np.clip(pitches, PITCH_MIN, PITCH_MAX):
        ph[p - PITCH_MIN] += 1
    ph /= ph.sum() + 1e-9
    
    # chroma (12)
    ch = np.zeros(12, dtype=np.float32)
    for p in pitches:
        ch[p % 12] += 1
    ch /= ch.sum() + 1e-9
    
    # interval_melody (25), interval_bass (25)
    im = interval_hist(melody_pitches)
    ib = interval_hist(bass_pitches)
    
    # transition_int12 (144) = (melody trans + bass trans), row-normalized
    tr_m = interval_transition_mod12(melody_pitches)
    tr_b = interval_transition_mod12(bass_pitches)
    tr = tr_m + tr_b
    tr = tr / (tr.sum(axis=1, keepdims=True) + 1e-9)
    
    # duration (12)
    dl = np.log2(np.maximum(durations, 1))
    dh, _ = np.histogram(dl, bins=12, range=(0, 12))
    dh = dh.astype(np.float32) / (dh.sum() + 1e-9)
    
    # velocity_hist (12)
    vh, _ = np.histogram(velocities, bins=12, range=(0, 128))
    vh = vh.astype(np.float32) / (vh.sum() + 1e-9)
    
    # ioi_new (12)
    ih2 = ioi_hist(sorted_starts)
    
    # scalars (11)
    legato_ratios = []
    for i in range(len(sorted_starts) - 1):
        s_cur, s_next = sorted_starts[i], sorted_starts[i + 1]
        gap = s_next - s_cur
        if gap > 0:
            max_dur = max(nn.end - nn.start for nn in events[s_cur])
            legato_ratios.append(min(max_dur / gap, 2.0))
    legato_mean = float(np.mean(legato_ratios)) if legato_ratios else 0.0
    legato_std = float(np.std(legato_ratios)) if legato_ratios else 0.0
    
    total_ticks = max(starts.max() - starts.min(), 1)
    scalars = np.array([
        np.log1p(len(notes)), np.log1p(total_ticks / 480.0),
        len(notes) / (total_ticks / 480.0),
        float(pitches.max() - pitches.min()) / 88,
        float(pitches.std()) / 24,
        float(pitches.mean()) / 127,
        float(velocities.mean()) / 127,
        float(velocities.std()) / 64,
        compute_polyphony(notes) / 10.0,
        legato_mean, legato_std,
    ], dtype=np.float32)
    
    return np.concatenate([ph, ch, im, ib, tr.flatten(), dh, vh, ih2, scalars])


# ============================================================
# 评估
# ============================================================
def cv_eval(model_fn, X, y, n_splits=5, seed=42, return_probs=False):
    """5-fold CV，可选返回 OOF 概率（用于集成）"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    oof_probs = np.zeros((len(y), len(np.unique(y))), dtype=np.float32)
    
    for tr, va in skf.split(X, y):
        sc = StandardScaler()
        Xtr, Xva = sc.fit_transform(X[tr]), sc.transform(X[va])
        m = model_fn()
        m.fit(Xtr, y[tr])
        pred = m.predict(Xva)
        scores.append(accuracy_score(y[va], pred))
        if return_probs:
            oof_probs[va] = m.predict_proba(Xva)
    
    return np.mean(scores), oof_probs if return_probs else None


def holdout_eval(model_fn, X, y, ids):
    sort_order = np.argsort(ids)
    n_train = int(len(ids) * 0.8)
    tr_idx = sort_order[:n_train]
    va_idx = sort_order[n_train:]
    sc = StandardScaler()
    Xtr, Xva = sc.fit_transform(X[tr_idx]), sc.transform(X[va_idx])
    m = model_fn()
    m.fit(Xtr, y[tr_idx])
    return accuracy_score(y[va_idx], m.predict(Xva))


# ============================================================
# 主流程
# ============================================================
def main():
    log("=" * 75)
    log("Task 1 模型最终对照（固定 G 配置 341 维）")
    log("=" * 75)
    
    with open(os.path.join(DATAROOT, "train.json")) as f:
        train = eval(f.read())
    paths = list(train.keys())
    y = np.array([train[p] for p in paths])
    ids = np.array([int(p.split('/')[1].split('.')[0]) for p in paths])
    
    log(f"训练样本: {len(paths)}")
    
    log("\n--- 提取 G 特征 ---")
    X = np.array([extract_G_features(os.path.join(DATAROOT, p)) for p in tqdm(paths)],
                 dtype=np.float32)
    log(f"X shape: {X.shape}")
    
    # 保存特征缓存，正式训练时复用
    cache_path = "statistics/task1_features_G.npz"
    np.savez(cache_path, X=X, y=y, ids=ids, paths=np.array(paths))
    log(f"[✓] 特征缓存已保存: {cache_path}")
    
    # ---- 定义 5 个模型配置 ----
    models = {
        "M1. LR (no class_weight)":         lambda: LogisticRegression(max_iter=3000, C=1.0, n_jobs=-1),
        "M2. LR (class_weight=balanced)":   lambda: LogisticRegression(max_iter=3000, C=1.0, class_weight='balanced', n_jobs=-1),
        "M3. LR (balanced + C=0.3)":        lambda: LogisticRegression(max_iter=3000, C=0.3, class_weight='balanced', n_jobs=-1),
        "M4. RandomForest":                 lambda: RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                                                            random_state=42, n_jobs=-1),
    }
    
    log("\n" + "=" * 75)
    log("对照实验")
    log("=" * 75)
    log(f"\n{'模型':<40} | {'5-fold CV':>10} | {'ID hold-out':>12}")
    log("-" * 70)
    
    results = {}
    oof_probs = {}
    for name, model_fn in models.items():
        cv, probs = cv_eval(model_fn, X, y, return_probs=True)
        ho = holdout_eval(model_fn, X, y, ids)
        results[name] = (cv, ho)
        oof_probs[name] = probs
        log(f"{name:<40} | {cv:>10.4f} | {ho:>12.4f}")
    
    # ---- M5: LR (M1) + RF (M4) 集成 ----
    log("\n--- 集成实验 ---")
    log("M5. LR (M1) + RF (M4) 概率平均")
    # 5-fold OOF 集成
    ens_probs = (oof_probs["M1. LR (no class_weight)"] +
                 oof_probs["M4. RandomForest"]) / 2
    ens_preds = np.argmax(ens_probs, axis=1)
    ens_cv = accuracy_score(y, ens_preds)
    
    # hold-out 集成：分别在 80% 训练，预测 20% 验证
    sort_order = np.argsort(ids)
    n_train = int(len(ids) * 0.8)
    tr_idx, va_idx = sort_order[:n_train], sort_order[n_train:]
    sc = StandardScaler()
    Xtr, Xva = sc.fit_transform(X[tr_idx]), sc.transform(X[va_idx])
    
    lr = models["M1. LR (no class_weight)"]()
    lr.fit(Xtr, y[tr_idx])
    rf = models["M4. RandomForest"]()
    rf.fit(Xtr, y[tr_idx])
    p_lr = lr.predict_proba(Xva)
    p_rf = rf.predict_proba(Xva)
    ens_ho_pred = np.argmax((p_lr + p_rf) / 2, axis=1)
    ens_ho = accuracy_score(y[va_idx], ens_ho_pred)
    
    log(f"{'M5. LR + RF (avg prob)':<40} | {ens_cv:>10.4f} | {ens_ho:>12.4f}")
    
    # ---- 总结 + 推荐 ----
    log("\n" + "=" * 75)
    log("总结")
    log("=" * 75)
    
    # 找 hold-out 最高的
    all_results = list(results.items()) + [("M5. LR + RF ensemble", (ens_cv, ens_ho))]
    best = max(all_results, key=lambda x: x[1][1])
    log(f"\nhold-out 最高: {best[0]} (CV={best[1][0]:.4f}, hold-out={best[1][1]:.4f})")
    
    # class_weight 影响
    m1_cv, m1_ho = results["M1. LR (no class_weight)"]
    m2_cv, m2_ho = results["M2. LR (class_weight=balanced)"]
    log(f"\nclass_weight=balanced 的影响:")
    log(f"  CV:       {m1_cv:.4f} (无) → {m2_cv:.4f} (有), 差 {m2_cv - m1_cv:+.4f}")
    log(f"  hold-out: {m1_ho:.4f} (无) → {m2_ho:.4f} (有), 差 {m2_ho - m1_ho:+.4f}")
    if m2_ho < m1_ho - 0.005:
        log(f"  >>> balanced 在 hold-out 上有害，不要加！（验证猜测）")
    elif m2_ho > m1_ho + 0.005:
        log(f"  >>> balanced 在 hold-out 上有帮助，可以加")
    else:
        log(f"  >>> 影响不大，从简不加")
    
    # M1 confusion matrix（看哪些 composer 容易错）
    log("\n--- M1 (推荐 baseline) 5-fold OOF confusion matrix ---")
    oof = oof_probs["M1. LR (no class_weight)"]
    oof_preds = np.argmax(oof, axis=1)
    cm = confusion_matrix(y, oof_preds)
    log("       " + "  ".join(f"p{i:>2}" for i in range(8)))
    for i, row in enumerate(cm):
        log(f"  t{i:>2} | " + "  ".join(f"{v:>3d}" for v in row))
    
    # 每类的准确率
    log("\n--- 每个 composer 的预测准确率 (M1, OOF) ---")
    for cid in range(8):
        mask = (y == cid)
        if mask.sum() > 0:
            acc = (oof_preds[mask] == cid).mean()
            log(f"  composer {cid} ({mask.sum():>3} 样本): {acc:.4f}")
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        f.write('