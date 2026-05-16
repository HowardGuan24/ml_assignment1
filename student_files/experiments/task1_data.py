# explore_task1.py
"""
Task 1 数据探查脚本
- 使用 eval() 读取 train.json/test.json（基线代码也是这样）
- 输出写入 task1_exploration.txt
"""
import os
from collections import Counter
import numpy as np
import miditoolkit

# ====== 改成你本地路径 ======
DATAROOT = "student_files/task1_composer_classification"
OUTPUT_FILE = "student_files/experiments/task1_exploration.txt"
# ============================

_out_lines = []
def log(msg=""):
    print(msg)
    _out_lines.append(str(msg))

# 基线用 eval 读取，保持一致
with open(os.path.join(DATAROOT, "train.json")) as f:
    train = eval(f.read())
with open(os.path.join(DATAROOT, "test.json")) as f:
    test = eval(f.read())

# ---------- 基本规模 ----------
log("=" * 70)
log("基本规模")
log("=" * 70)
log(f"训练集样本数: {len(train)} (dict: path -> composer_id)")
log(f"测试集样本数: {len(test)} (list of paths)")
log(f"训练/测试比例: {len(train)/len(test):.2f}")

# ---------- 标签分布 ----------
labels = list(train.values())
label_counts = Counter(labels)
log(f"\n{'='*70}\n标签分布\n{'='*70}")
log(f"独立作曲家数: {len(label_counts)}")
log(f"标签范围: {min(label_counts)} ~ {max(label_counts)}")
total = sum(label_counts.values())
for cid in sorted(label_counts.keys()):
    cnt = label_counts[cid]
    pct = cnt / total * 100
    bar = '#' * int(pct / 2)
    log(f"  composer {cid}: {cnt:5d} ({pct:5.2f}%) {bar}")

mx, mn = max(label_counts.values()), min(label_counts.values())
log(f"\n不平衡度: 最多 / 最少 = {mx} / {mn} = {mx/mn:.2f}x")
log(f"Majority baseline (全猜最大类): {mx/total:.4f}")

# ---------- 文件 ID 切分 ----------
log(f"\n{'='*70}\n文件 ID 切分情况\n{'='*70}")
train_ids = sorted([int(k.split('/')[1].split('.')[0]) for k in train.keys()])
test_ids = sorted([int(k.split('/')[1].split('.')[0]) for k in test])
log(f"训练集 ID: {train_ids[0]} ~ {train_ids[-1]}")
log(f"测试集 ID: {test_ids[0]} ~ {test_ids[-1]}")
overlap = set(train_ids) & set(test_ids)
log(f"重叠 ID 数: {len(overlap)} (应为 0)")

# ---------- MIDI 文件结构调查（采样 200 个）----------
log(f"\n{'='*70}\nMIDI 文件结构调查（采样 200 个训练样本）\n{'='*70}")
sample_paths = list(train.keys())[:200]

n_notes_first_inst = []      # 第一个 instrument 的音符数（baseline 用这个）
n_notes_all = []              # 所有 instruments 的音符总数
n_instruments_list = []
duration_tick_list = []
duration_sec_list = []        # 用 ticks_per_beat 和 tempo 估算秒数
pitch_min_list, pitch_max_list = [], []
ticks_per_beat_set = set()
multi_instrument_count = 0
has_drum_count = 0
errors = []

for path in sample_paths:
    try:
        midi = miditoolkit.midi.parser.MidiFile(os.path.join(DATAROOT, path))
        ticks_per_beat_set.add(midi.ticks_per_beat)
        n_instruments_list.append(len(midi.instruments))
        if len(midi.instruments) > 1:
            multi_instrument_count += 1
        
        # 第一个 instrument（baseline 用的）
        first_notes = midi.instruments[0].notes
        n_notes_first_inst.append(len(first_notes))
        
        # 所有 instrument 的 notes 合并
        all_notes = []
        for inst in midi.instruments:
            all_notes.extend(inst.notes)
            if inst.is_drum:
                has_drum_count += 1
        n_notes_all.append(len(all_notes))
        
        if all_notes:
            pitches = [n.pitch for n in all_notes]
            pitch_min_list.append(min(pitches))
            pitch_max_list.append(max(pitches))
            dur_tick = max(n.end for n in all_notes) - min(n.start for n in all_notes)
            duration_tick_list.append(dur_tick)
    except Exception as e:
        errors.append((path, str(e)))

log(f"采样 {len(sample_paths)} 个文件，失败 {len(errors)} 个")
if errors[:3]:
    log(f"前 3 个错误: {errors[:3]}")

log(f"\nticks_per_beat 的不同取值: {sorted(ticks_per_beat_set)}")

log(f"\nInstrument 数量分布:")
inst_cnt = Counter(n_instruments_list)
for k in sorted(inst_cnt.keys()):
    log(f"  {k} 个 instrument: {inst_cnt[k]} 个文件")
log(f"  含多 instrument 的文件: {multi_instrument_count}/{len(sample_paths)} ({multi_instrument_count/len(sample_paths)*100:.1f}%)")
log(f"  含 drum 轨道的样本: {has_drum_count} 次（注意：这是按 instrument 计数，不是文件计数）")

log(f"\n第一 instrument 的音符数（baseline 用的）:")
log(f"  均值: {np.mean(n_notes_first_inst):.0f}, 中位数: {np.median(n_notes_first_inst):.0f}")
log(f"  最小: {min(n_notes_first_inst)}, 最大: {max(n_notes_first_inst)}")
log(f"  P10/P90: {np.percentile(n_notes_first_inst, 10):.0f} / {np.percentile(n_notes_first_inst, 90):.0f}")

log(f"\n所有 instruments 的音符总数:")
log(f"  均值: {np.mean(n_notes_all):.0f}, 中位数: {np.median(n_notes_all):.0f}")
log(f"  最小: {min(n_notes_all)}, 最大: {max(n_notes_all)}")
log(f"  注意：均值 vs 第一 inst 差异: {np.mean(n_notes_all) - np.mean(n_notes_first_inst):.0f}")
log(f"  → 如果差异大，说明应该用全部 instruments，不只第一 instrument")

log(f"\n时长（tick）:")
log(f"  均值: {np.mean(duration_tick_list):.0f}, 中位数: {np.median(duration_tick_list):.0f}")

log(f"\n音高范围:")
log(f"  全局最低音: {min(pitch_min_list)} (MIDI pitch, 0~127)")
log(f"  全局最高音: {max(pitch_max_list)}")
log(f"  平均最低: {np.mean(pitch_min_list):.1f}, 平均最高: {np.mean(pitch_max_list):.1f}")

# ---------- Instrument program 分布 ----------
log(f"\n{'='*70}\nInstrument program（音色）分布（TOP 15）\n{'='*70}")
prog_counter = Counter()
for path in sample_paths:
    try:
        midi = miditoolkit.midi.parser.MidiFile(os.path.join(DATAROOT, path))
        for inst in midi.instruments:
            prog_counter[(inst.program, inst.is_drum)] += 1
    except Exception:
        pass
for (prog, drum), cnt in prog_counter.most_common(15):
    log(f"  program={prog:3d}, is_drum={drum}: {cnt}")

# ---------- 各 composer 的样本特性差异（每个 composer 取 20 个文件看）----------
log(f"\n{'='*70}\n各作曲家的特征对比（每个作曲家采样 20 个文件）\n{'='*70}")
log(f"{'composer':>10} | {'n_samples':>10} | {'avg_n_notes':>12} | {'avg_pitch':>10} | {'avg_dur_tick':>12} | {'avg_inst':>9}")
log("-" * 80)

for cid in sorted(label_counts.keys()):
    paths_this = [p for p, l in train.items() if l == cid][:20]
    n_notes_list_c, pitch_list_c, dur_list_c, inst_list_c = [], [], [], []
    for p in paths_this:
        try:
            m = miditoolkit.midi.parser.MidiFile(os.path.join(DATAROOT, p))
            all_notes = []
            for inst in m.instruments:
                all_notes.extend(inst.notes)
            if not all_notes:
                continue
            n_notes_list_c.append(len(all_notes))
            pitch_list_c.append(np.mean([n.pitch for n in all_notes]))
            dur_list_c.append(max(n.end for n in all_notes) - min(n.start for n in all_notes))
            inst_list_c.append(len(m.instruments))
        except Exception:
            pass
    if n_notes_list_c:
        log(f"{cid:>10} | {len(paths_this):>10} | {np.mean(n_notes_list_c):>12.0f} | "
            f"{np.mean(pitch_list_c):>10.1f} | {np.mean(dur_list_c):>12.0f} | {np.mean(inst_list_c):>9.1f}")

# ---------- 一个样本文件的细节 ----------
log(f"\n{'='*70}\n样本文件 #0 详细信息\n{'='*70}")
first_path = list(train.keys())[0]
midi = miditoolkit.midi.parser.MidiFile(os.path.join(DATAROOT, first_path))
log(f"文件: {first_path}, 标签 = composer {train[first_path]}")
log(f"ticks_per_beat: {midi.ticks_per_beat}")
log(f"instruments 数: {len(midi.instruments)}")
for i, inst in enumerate(midi.instruments):
    log(f"  instrument {i}: program={inst.program}, is_drum={inst.is_drum}, "
        f"name={inst.name!r}, n_notes={len(inst.notes)}")
notes = midi.instruments[0].notes
log(f"前 5 个 notes（第一 instrument）:")
for n in notes[:5]:
    log(f"  pitch={n.pitch}, velocity={n.velocity}, start={n.start}, end={n.end}, dur={n.end-n.start}")

# 写入文件
with open(OUTPUT_FILE, 'w') as f:
    f.write('\n'.join(_out_lines) + '\n')
print(f"\n[✓] 结果已写入: {OUTPUT_FILE}")