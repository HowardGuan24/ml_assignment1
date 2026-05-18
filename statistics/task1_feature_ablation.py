# statistics/task1_feature_ablation.py
"""
Task 1 特征消融实验：
- 修复 interval (melody + bass concat)
- 对比 velocity-3 vs velocity-12
- 砍 pitch / transition 的影响
- 同时报告 5-fold CV 和 按 ID hold-out 两种评估
"""
import os
import numpy as np
import miditoolkit
from collections import defaultdict
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ====== 路径 ======
DATAROOT = "student_files/task1_composer_classification"
OUTPUT_FILE = "statistics/task1_feature_ablation.txt"
# ===================

_log = []
def log(m=""):
    print(m); _log.append(str(m))


# ============================================================
# 通用辅助
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
    """对一个 pitch 序列算 25 维 interval histogram"""
    if len(pitch_sequence) < 2:
        return np.zeros(N_INTERVALS, dtype=np.float32)
    intervals = np.diff(np.array(pitch_sequence))
    ih = np.zeros(N_INTERVALS, dtype=np.float32)
    for iv in np.clip(intervals, INTERVAL_MIN, INTERVAL_MAX):
        ih[iv - INTERVAL_MIN] += 1
    ih /= ih.sum() + 1e-9
    return ih


def ioi_hist(start_sequence):
    """对一组相邻时间点算 12 维 IOI histogram"""
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


# ============================================================
# 特征提取：返回 dict {组名: 向量}，按需 concat
# ============================================================
def extract_all_features(midi_path):
    """提取所有可能的特征，返回 dict
    新版会额外提供：
    - interval_old: 旧的（buggy）interval
    - interval_melody: melody interval（取每个时间点最高音）
    - interval_bass: bass interval（取每个时间点最低音）
    - ioi_old: 旧的 IOI（diff 所有音符 start）
    - ioi_new: 新的 IOI（diff 时间点 start）
    - velocity_hist: 12-bin histogram
    - velocity_scalars: [mean, std, range] 3 标量
    """
    midi = miditoolkit.midi.parser.MidiFile(midi_path)
    notes = []
    for inst in midi.instruments:
        if not inst.is_drum:
            notes.extend(inst.notes)
    
    groups = {}
    if len(notes) < 2:
        # 防御性：返回零向量字典
        groups['pitch']            = np.zeros(N_PITCHES, dtype=np.float32)
        groups['chroma']           = np.zeros(12, dtype=np.float32)
        groups['interval_old']     = np.zeros(N_INTERVALS, dtype=np.float32)
        groups['interval_melody']  = np.zeros(N_INTERVALS, dtype=np.float32)
        groups['interval_bass']    = np.zeros(N_INTERVALS, dtype=np.float32)
        groups['transition']       = np.zeros(144, dtype=np.float32)
        groups['duration']         = np.zeros(12, dtype=np.float32)
        groups['velocity_hist']    = np.zeros(12, dtype=np.float32)
        groups['velocity_scalars'] = np.zeros(3, dtype=np.float32)
        groups['ioi_old']          = np.zeros(12, dtype=np.float32)
        groups['ioi_new']          = np.zeros(12, dtype=np.float32)
        groups['scalars']          = np.zeros(9, dtype=np.float32)
        return groups
    
    notes.sort(key=lambda n: (n.start, n.pitch))
    pitches = np.array([n.pitch for n in notes])
    velocities = np.array([n.velocity for n in notes])
    durations = np.array([n.end - n.start for n in notes])
    starts = np.array([n.start for n in notes])
    
    # ---- Pitch histogram (88) ----
    ph = np.zeros(N_PITCHES, dtype=np.float32)
    for p in np.clip(pitches, PITCH_MIN, PITCH_MAX):
        ph[p - PITCH_MIN] += 1
    ph /= ph.sum() + 1e-9
    groups['pitch'] = ph
    
    # ---- Chroma (12) ----
    ch = np.zeros(12, dtype=np.float32)
    for p in pitches:
        ch[p % 12] += 1
    ch /= ch.sum() + 1e-9
    groups['chroma'] = ch
    
    # ---- 旧的 interval（buggy，混合和弦内+和弦间）----
    groups['interval_old'] = interval_hist(pitches.tolist())
    
    # ---- 新的 interval：melody + bass ----
    # 按 start 严格分组
    events = defaultdict(list)
    for n in notes:
        events[n.start].append(n.pitch)
    sorted_starts = sorted(events.keys())
    melody = [max(events[s]) for s in sorted_starts]
    bass = [min(events[s]) for s in sorted_starts]
    
    groups['interval_melody'] = interval_hist(melody)
    groups['interval_bass']   = interval_hist(bass)
    
    # ---- PC transition (144) ----
    pc = pitches % 12
    tr = np.zeros((12, 12), dtype=np.float32)
    for i in range(len(pc) - 1):
        tr[pc[i], pc[i+1]] += 1
    tr = tr / (tr.sum(axis=1, keepdims=True) + 1e-9)
    groups['transition'] = tr.flatten()
    
    # ---- Duration (12) ----
    dl = np.log2(np.maximum(durations, 1))
    dh, _ = np.histogram(dl, bins=12, range=(0, 12))
    dh = dh.astype(np.float32) / (dh.sum() + 1e-9)
    groups['duration'] = dh
    
    # ---- Velocity: 两种版本 ----
    vh, _ = np.histogram(velocities, bins=12, range=(0, 128))
    vh = vh.astype(np.float32) / (vh.sum() + 1e-9)
    groups['velocity_hist'] = vh
    
    groups['velocity_scalars'] = np.array([
        float(velocities.mean()) / 127.0,
        float(velocities.std()) / 64.0,
        float(velocities.max() - velocities.min()) / 127.0,
    ], dtype=np.float32)
    
    # ---- IOI: 两种版本 ----
    groups['ioi_old'] = ioi_hist(starts.tolist())
    groups['ioi_new'] = ioi_hist(sorted_starts)
    
    # ---- Scalars (9) ----
    total_ticks = max(starts.max() - starts.min(), 1)
    groups['scalars'] = np.array([
        np.log1p(len(notes)), np.log1p(total_ticks / 480.0),
        len(notes) / (total_ticks / 480.0),
        float(pitches.max() - pitches.min()) / 88,
        float(pitches.std()) / 24,
        float(pitches.mean()) / 127,
        float(velocities.mean()) / 127,
        float(velocities.std()) / 64,
        compute_polyphony(notes) / 10.0,
    ], dtype=np.float32)
    
    return groups


# ============================================================
# 评估：返回 (5-fold acc, id-holdout acc)
# ============================================================
def eval_config(X, y, ids, seed=42):
    """返回 (5-fold mean acc, 按 ID hold-out acc)"""
    # 5-fold stratified
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_scores = []
    for tr, va in skf.split(X, y):
        sc = StandardScaler()
        Xtr, Xva = sc.fit_transform(X[tr]), sc.transform(X[va])
        m = LogisticRegression(max_iter=3000, C=1.0, n_jobs=-1)
        m.fit(Xtr, y[tr])
        cv_scores.append(accuracy_score(y[va], m.predict(Xva)))
    cv_mean = np.mean(cv_scores)
    
    # 按 ID hold-out: 后 20% 验证（约模拟测试集"靠后"的特征）
    sort_order = np.argsort(ids)
    n_train = int(len(ids) * 0.8)
    tr_idx = sort_order[:n_train]
    va_idx = sort_order[n_train:]
    sc = StandardScaler()
    Xtr, Xva = sc.fit_transform(X[tr_idx]), sc.transform(X[va_idx])
    m = LogisticRegression(max_iter=3000, C=1.0, n_jobs=-1)
    m.fit(Xtr, y[tr_idx])
    holdout_acc = accuracy_score(y[va_idx], m.predict(Xva))
    
    return cv_mean, holdout_acc


# ============================================================
# 主流程
# ============================================================
def main():
    log("=" * 75)
    log("Task 1 特征消融实验")
    log("=" * 75)
    
    # 加载数据
    with open(os.path.join(DATAROOT, "train.json")) as f:
        train = eval(f.read())
    paths = list(train.keys())
    y = np.array([train[p] for p in paths])
    ids = np.array([int(p.split('/')[1].split('.')[0]) for p in paths])
    
    log(f"训练样本: {len(paths)}, composer 数: {len(set(y))}")
    log(f"ID 范围: {ids.min()} ~ {ids.max()}")
    
    # 提取所有特征组
    log("\n--- 特征提取 ---")
    all_groups = defaultdict(list)
    for p in tqdm(paths):
        groups = extract_all_features(os.path.join(DATAROOT, p))
        for name, vec in groups.items():
            all_groups[name].append(vec)
    
    for name in all_groups:
        all_groups[name] = np.array(all_groups[name], dtype=np.float32)
    
    log(f"\n各特征组维度:")
    for name, arr in all_groups.items():
        log(f"  {name:20s}: {arr.shape[1]:>4d}")
    
    # 6 个对照配置
    configs = [
        # name,                   特征组列表（按顺序 concat）
        ("A. Baseline",            ['pitch', 'chroma', 'interval_old', 'transition',
                                    'duration', 'velocity_hist', 'ioi_old', 'scalars']),
        ("B. Fix-interval",        ['pitch', 'chroma', 'interval_melody', 'interval_bass',
                                    'transition', 'duration', 'velocity_hist', 'ioi_new', 'scalars']),
        ("C. Vel-3",               ['pitch', 'chroma', 'interval_old', 'transition',
                                    'duration', 'velocity_scalars', 'ioi_old', 'scalars']),
        ("D. Fix + Vel-3",         ['pitch', 'chroma', 'interval_melody', 'interval_bass',
                                    'transition', 'duration', 'velocity_scalars', 'ioi_new', 'scalars']),
        ("E. D - pitch",           ['chroma', 'interval_melody', 'interval_bass',
                                    'transition', 'duration', 'velocity_scalars', 'ioi_new', 'scalars']),
        ("F. D - pitch - trans",   ['chroma', 'interval_melody', 'interval_bass',
                                    'duration', 'velocity_scalars', 'ioi_new', 'scalars']),
    ]
    
    log("\n" + "=" * 75)
    log("跑 6 个配置的对照实验")
    log("=" * 75)
    
    log(f"\n{'配置':<28} | {'维度':>5} | {'5-fold CV':>10} | {'ID hold-out':>12} | {'差':>7}")
    log("-" * 75)
    
    results = []
    for name, group_names in configs:
        X = np.concatenate([all_groups[n] for n in group_names], axis=1)
        cv, ho = eval_config(X, y, ids)
        diff = cv - ho
        log(f"{name:<28} | {X.shape[1]:>5} | {cv:>10.4f} | {ho:>12.4f} | {diff:>+7.4f}")
        results.append((name, X.shape[1], cv, ho))
    
    # 总结
    log("\n" + "=" * 75)
    log("解读指南")
    log("=" * 75)
    log("""
1. A→B：修复 interval/IOI 的效果（应该 CV 和 hold-out 都涨）
2. A→C：把 velocity 12-bin 换成 3 标量的效果
        - 若 CV 暴跌、hold-out 也跌 → velocity-12 是真信号，应保留
        - 若 CV 跌但 hold-out 不变或涨 → velocity-12 主要是 leakage，应砍
3. D vs F：砍掉 pitch 和 transition 的累计效果
4. CV vs hold-out 的"差"列：
   - 差 > 0.1：5-fold 严重高估，存在分布漂移
   - 差 < 0.05：两种评估一致，没有分布漂移问题
""")
    
    # 找最佳
    best_cv = max(results, key=lambda r: r[2])
    best_ho = max(results, key=lambda r: r[3])
    log(f"5-fold 最高: {best_cv[0]} ({best_cv[2]:.4f})")
    log(f"hold-out 最高: {best_ho[0]} ({best_ho[3]:.4f})")
    
    if best_cv[0] != best_ho[0]:
        log(">>> 5-fold 和 hold-out 最优配置不一致！优先看 hold-out（更接近 leaderboard）")
    else:
        log(">>> 5-fold 和 hold-out 一致，可以放心用这个配置")
    
    # 保存日志
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        f.write('\n'.join(_log) + '\n')
    log(f"\n[✓] 日志已写入: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()