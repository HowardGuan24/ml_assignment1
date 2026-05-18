# statistics/task1_metadata_features.py
"""
探查 MIDI 文件里所有可能的"元数据指纹"特征
看每种特征单独跑 CV 的分数
"""
import os
import numpy as np
import miditoolkit
from collections import Counter, defaultdict
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

DATAROOT = "student_files/task1_composer_classification"


def explore_one_file(midi_path):
    """挖出一个 MIDI 文件里所有可能的"指纹"信息"""
    midi = miditoolkit.midi.parser.MidiFile(midi_path)
    
    info = {}
    # 1. 文件级别 metadata
    info['ticks_per_beat'] = midi.ticks_per_beat
    info['n_tempo_changes'] = len(midi.tempo_changes)
    info['n_time_signature_changes'] = len(midi.time_signature_changes)
    info['n_key_signature_changes'] = len(midi.key_signature_changes)
    
    # 第一个 tempo
    if midi.tempo_changes:
        info['first_tempo'] = midi.tempo_changes[0].tempo
    else:
        info['first_tempo'] = -1
    
    # 第一个 time signature
    if midi.time_signature_changes:
        info['time_sig_num'] = midi.time_signature_changes[0].numerator
        info['time_sig_den'] = midi.time_signature_changes[0].denominator
    else:
        info['time_sig_num'] = -1
        info['time_sig_den'] = -1
    
    # 第一个 key signature
    if midi.key_signature_changes:
        info['key_sig'] = midi.key_signature_changes[0].key_name
    else:
        info['key_sig'] = 'none'
    
    # 2. Instrument metadata
    info['n_instruments'] = len(midi.instruments)
    info['has_drum'] = any(i.is_drum for i in midi.instruments)
    info['programs'] = tuple(sorted(i.program for i in midi.instruments))
    info['inst_names'] = tuple(i.name for i in midi.instruments)
    
    # 3. Note 级别细节
    notes = []
    for inst in midi.instruments:
        notes.extend(inst.notes)
    if notes:
        # Note start 的 mod 480（每拍内的位置）—— 如果都是整拍则是量化过的
        starts_mod = [n.start % 480 for n in notes]
        info['n_unique_start_mod'] = len(set(starts_mod))
        info['starts_quantized'] = (set(starts_mod) <= {0, 60, 120, 180, 240, 300, 360, 420})
        
        # Duration 的唯一值数（指纹）
        durs = [n.end - n.start for n in notes]
        info['n_unique_durations'] = len(set(durs))
        info['most_common_duration'] = Counter(durs).most_common(1)[0][0]
        
        # Velocity 唯一值
        vels = [n.velocity for n in notes]
        info['n_unique_velocities'] = len(set(vels))
        info['velocity_set'] = tuple(sorted(set(vels)))
    
    # 4. Lyrics / markers（有时候 MIDI 文件里包含作曲家、曲名等文字）
    info['has_lyrics'] = len(midi.lyrics) > 0
    info['has_markers'] = len(midi.markers) > 0
    if midi.lyrics:
        info['first_lyric'] = midi.lyrics[0].text
    if midi.markers:
        info['first_marker'] = midi.markers[0].text
    
    return info


# 加载训练集
with open(os.path.join(DATAROOT, "train.json")) as f:
    train = eval(f.read())
paths = list(train.keys())
y = np.array([train[p] for p in paths])

# 探索前 30 个文件，看有什么可挖的
print("=" * 80)
print("前 30 个文件的 metadata 探查")
print("=" * 80)

for i in range(30):
    info = explore_one_file(os.path.join(DATAROOT, paths[i]))
    print(f"\n--- {paths[i]} (composer {y[i]}) ---")
    for k, v in info.items():
        # 截断过长的值
        s = repr(v)
        if len(s) > 80:
            s = s[:77] + '...'
        print(f"  {k:30s}: {s}")
    if i >= 9:  # 暂时只打印 10 个，太多眼花
        break

print("\n" + "=" * 80)
print("全数据集的 metadata 分布统计")
print("=" * 80)

all_infos = []
for p in tqdm(paths):
    all_infos.append(explore_one_file(os.path.join(DATAROOT, p)))

# 对每个 metadata 字段，看是否在不同 composer 上有差异
print("\n各 metadata 字段在不同 composer 上的分布:")
print()

def composer_breakdown(field_values_per_composer, field_name):
    """看一个字段在每个 composer 上的取值分布"""
    print(f"\n[{field_name}]")
    for c in range(8):
        vals = [info[field_name] for info, label in zip(all_infos, y) if label == c]
        if not vals:
            continue
        cnt = Counter(vals)
        most_common = cnt.most_common(3)
        n_unique = len(set(vals))
        print(f"  composer {c} ({len(vals)} 样本): {n_unique} 种取值, TOP 3: {most_common}")


# 重要字段
for field in ['ticks_per_beat', 'n_tempo_changes', 'first_tempo',
              'time_sig_num', 'time_sig_den', 'key_sig',
              'n_instruments', 'programs', 'inst_names',
              'n_unique_durations', 'most_common_duration',
              'n_unique_velocities', 'velocity_set',
              'has_lyrics', 'has_markers']:
    try:
        composer_breakdown(all_infos, field)
    except Exception as e:
        print(f"\n[{field}] error: {e}")

# 看 lyric/marker 的具体内容（最可疑）
print("\n" + "=" * 80)
print("Lyric / Marker 内容（最可疑——可能直接泄露 composer）")
print("=" * 80)
lyric_count = sum(1 for info in all_infos if info.get('has_lyrics'))
marker_count = sum(1 for info in all_infos if info.get('has_markers'))
print(f"\n含 lyric 的文件: {lyric_count}/{len(all_infos)}")
print(f"含 marker 的文件: {marker_count}/{len(all_infos)}")

if lyric_count > 0:
    print("\n前 10 个 lyric 样本:")
    cnt = 0
    for info, p, label in zip(all_infos, paths, y):
        if info.get('has_lyrics'):
            print(f"  {p} (composer {label}): {info.get('first_lyric')!r}")
            cnt += 1
            if cnt >= 10:
                break

if marker_count > 0:
    print("\n前 10 个 marker 样本:")
    cnt = 0
    for info, p, label in zip(all_infos, paths, y):
        if info.get('has_markers'):
            print(f"  {p} (composer {label}): {info.get('first_marker')!r}")
            cnt += 1
            if cnt >= 10:
                break

# instrument names 也可能泄露作曲家
print("\nInstrument names 的分布:")
name_counter = Counter()
for info in all_infos:
    for n in info['inst_names']:
        if n:  # 非空名字
            name_counter[n] += 1
print(f"  非空 instrument 名字数: {sum(name_counter.values())}/{len(all_infos)}")
if name_counter:
    print(f"  TOP 10:")
    for name, cnt in name_counter.most_common(10):
        print(f"    {name!r}: {cnt}")