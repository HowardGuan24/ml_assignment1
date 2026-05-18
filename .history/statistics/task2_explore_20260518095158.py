# statistics/task2_explore.py
"""
Task 2 数据细 explore：
1. 验证 metadata 同源假设：True 对的 metadata 是否一致？False 对呢？
2. A 和 B 的音符数 / 时长分布
3. 文件命名规律
4. 验证集构造方式（情形 1/2/3）
"""
import os
import numpy as np
import miditoolkit
from collections import Counter
from tqdm import tqdm

DATAROOT = "student_files/task2_next_sequence_prediction"
OUTPUT_FILE = "statistics/task2_explore.txt"

_log = []
def log(m=""):
    print(m); _log.append(str(m))


def get_metadata(midi_path):
    """提取一个 MIDI 的关键 metadata"""
    midi = miditoolkit.midi.parser.MidiFile(midi_path)
    notes = []
    for inst in midi.instruments:
        notes.extend(inst.notes)
    
    return {
        'first_tempo': midi.tempo_changes[0].tempo if midi.tempo_changes else 120.0,
        'time_sig': (midi.time_signature_changes[0].numerator, midi.time_signature_changes[0].denominator)
                    if midi.time_signature_changes else (4, 4),
        'key_sig': midi.key_signature_changes[0].key_name if midi.key_signature_changes else 'none',
        'n_notes': len(notes),
        'velocity_set': tuple(sorted(set(n.velocity for n in notes))) if notes else (),
        'duration_max': (max(n.end for n in notes) - min(n.start for n in notes)) if notes else 0,
        'pitch_mean': float(np.mean([n.pitch for n in notes])) if notes else 0,
        'most_common_duration': Counter([n.end - n.start for n in notes]).most_common(1)[0][0] if notes else 0,
    }


with open(os.path.join(DATAROOT, "train.json")) as f:
    train = eval(f.read())
with open(os.path.join(DATAROOT, "test.json")) as f:
    test = eval(f.read())

log("=" * 75)
log("Task 2 数据细 explore")
log("=" * 75)
log(f"训练对: {len(train)}, 测试对: {len(test)}")

# 文件命名规律
log("\n--- 文件命名规律 ---")
sample_pair = list(train.keys())[0]
log(f"训练第一对: {sample_pair}, 标签 = {train[sample_pair]}")

# 收集所有 unique 文件
all_files = set()
for pair in train.keys():
    all_files.add(pair[0])
    all_files.add(pair[1])
log(f"训练中涉及的不同 MIDI 文件数: {len(all_files)}")

# 检查命名规律：是不是 ("midis/X_1.mid", "midis/X_2.mid") 这种格式？
log("\n前 5 个 True pair 和前 5 个 False pair:")
true_pairs = [k for k, v in train.items() if v]
false_pairs = [k for k, v in train.items() if not v]
log("True pairs:")
for p in true_pairs[:5]:
    log(f"  {p}")
log("False pairs:")
for p in false_pairs[:5]:
    log(f"  {p}")

# 检查 True/False 对的命名是否有规律
log("\n--- 命名规律分析 ---")
def parse_name(path):
    """提取文件名里的数字"""
    import re
    # "midis/12_1.mid" -> (12, 1)
    m = re.match(r'midis/(\d+)_(\d+)\.mid', path)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None

# 看 True 对的命名结构
log("\nTrue 对的命名结构 (TOP 10):")
true_name_patterns = []
for p in true_pairs:
    n1, n2 = parse_name(p[0]), parse_name(p[1])
    if n1 and n2:
        # 是否同前缀（同曲子）
        same_song = (n1[0] == n2[0])
        # 后缀差
        suffix_diff = n2[1] - n1[1] if same_song else None
        true_name_patterns.append((same_song, suffix_diff))

t_cnt = Counter(true_name_patterns)
for (same, diff), cnt in t_cnt.most_common(5):
    log(f"  same_song={same}, suffix_diff={diff}: {cnt} 次")

log("\nFalse 对的命名结构 (TOP 10):")
false_name_patterns = []
for p in false_pairs:
    n1, n2 = parse_name(p[0]), parse_name(p[1])
    if n1 and n2:
        same_song = (n1[0] == n2[0])
        suffix_diff = n2[1] - n1[1] if same_song else None
        false_name_patterns.append((same_song, suffix_diff))

f_cnt = Counter(false_name_patterns)
for (same, diff), cnt in f_cnt.most_common(5):
    log(f"  same_song={same}, suffix_diff={diff}: {cnt} 次")

# 关键发现总结
n_true_same_song = sum(1 for ss, _ in true_name_patterns if ss)
n_false_same_song = sum(1 for ss, _ in false_name_patterns if ss)
log(f"\n>>> True 对中：同曲子 {n_true_same_song}/{len(true_name_patterns)}")
log(f">>> False 对中：同曲子 {n_false_same_song}/{len(false_name_patterns)}")
if n_false_same_song == len(false_name_patterns):
    log(">>> 结论：负样本 = 同源对翻转 (情形 1)，metadata 无用")
elif n_false_same_song == 0:
    log(">>> 结论：负样本 = 不同曲子 (情形 2)，metadata 大有用")
else:
    log(">>> 结论：混合情形")

# ============================================================
# Metadata 同源验证：对每个 pair, 看 A 和 B 的 metadata 是否相同
# ============================================================
log("\n" + "=" * 75)
log("Metadata 同源验证（采样 500 对）")
log("=" * 75)

# 抓 250 个 True + 250 个 False 来比较
sample_size = 250
sampled_true = true_pairs[:sample_size]
sampled_false = false_pairs[:sample_size]

# 预读 metadata 缓存
log("\n预读 metadata...")
all_sampled_files = set()
for pair in sampled_true + sampled_false:
    all_sampled_files.add(pair[0])
    all_sampled_files.add(pair[1])
meta_cache = {}
for f in tqdm(all_sampled_files):
    meta_cache[f] = get_metadata(os.path.join(DATAROOT, f))


def compare_metadata(meta_a, meta_b):
    """比较两个 metadata，返回各字段是否相同"""
    keys = ['first_tempo', 'time_sig', 'key_sig', 'velocity_set',
            'most_common_duration']
    return {k: meta_a[k] == meta_b[k] for k in keys}


def metadata_match_rate(pairs, name):
    """统计每个 metadata 字段在 pair 中的匹配率"""
    log(f"\n--- {name} 对（{len(pairs)} 个）---")
    field_match = Counter()
    field_total = Counter()
    for p in pairs:
        ma = meta_cache[p[0]]
        mb = meta_cache[p[1]]
        cmp = compare_metadata(ma, mb)
        for k, v in cmp.items():
            field_total[k] += 1
            if v:
                field_match[k] += 1
    log(f"  {'字段':<25} | {'匹配数':>8} | {'匹配率':>8}")
    log("  " + "-" * 50)
    for k in ['first_tempo', 'time_sig', 'key_sig', 'velocity_set', 'most_common_duration']:
        rate = field_match[k] / field_total[k] * 100
        log(f"  {k:<25} | {field_match[k]:>8} | {rate:>7.2f}%")


metadata_match_rate(sampled_true, "True")
metadata_match_rate(sampled_false, "False")

log("\n" + "=" * 75)
log("解读")
log("=" * 75)
log("""
- 如果 True 对的 metadata 匹配率 ≈ 100%, False 对 ≈ 100%
  → metadata 在 Task 2 上没区分度（情形 1/3）

- 如果 True 对 ≈ 100%, False 对 << 100%
  → metadata 是强力同源检测器（情形 2）, 应该重点用 diff 特征

- 如果两个都不是 100%
  → metadata 有部分信号但不是完美指纹
""")

# 看 A 和 B 的音符数
log("\n--- A 和 B 的音符数分布（采样 500 对）---")
n_notes_A = [meta_cache[p[0]]['n_notes'] for p in sampled_true[:100]]
n_notes_B = [meta_cache[p[1]]['n_notes'] for p in sampled_true[:100]]
log(f"True 对 A 的音符数: min={min(n_notes_A)}, max={max(n_notes_A)}, mean={np.mean(n_notes_A):.0f}")
log(f"True 对 B 的音符数: min={min(n_notes_B)}, max={max(n_notes_B)}, mean={np.mean(n_notes_B):.0f}")

n_notes_A = [meta_cache[p[0]]['n_notes'] for p in sampled_false[:100]]
n_notes_B = [meta_cache[p[1]]['n_notes'] for p in sampled_false[:100]]
log(f"False 对 A 的音符数: min={min(n_notes_A)}, max={max(n_notes_A)}, mean={np.mean(n_notes_A):.0f}")
log(f"False 对 B 的音符数: min={min(n_notes_B)}, max={max(n_notes_B)}, mean={np.mean(n_notes_B):.0f}")

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, 'w') as f:
    f.write('\n'.join(_log) + '\n')
print(f"\n[✓] {OUTPUT_FILE}")