"""
Task 2: next-sequence prediction.

This version keeps the old validation protocol used in the previous runs:
build symmetric augmented pairs first, then run shuffled StratifiedKFold.
It adds richer segment and boundary-continuity features, compares several
sklearn models, and uses a soft-vote ensemble only when it beats the best
single model under the same CV protocol.
"""
import os
import warnings
from collections import defaultdict

import miditoolkit
import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


DATAROOT = "student_files/task2_next_sequence_prediction"
PRED_FILE = "results/predictions2.json"
LOG_FILE = "train_log/task2_results_v3.txt"

BOUNDARY_WINDOWS = (1, 2, 4, 8, 12, 16)
INTERVAL_MIN, INTERVAL_MAX = -12, 12
N_INTERVALS = INTERVAL_MAX - INTERVAL_MIN + 1
SEGMENT_FEATURE_DIM = 131
BOUNDARY_FEATURE_DIM = 134

_log = []


def log(message=""):
    print(message)
    _log.append(str(message))


def safe_mean(values, default=0.0):
    return float(np.mean(values)) if len(values) else default


def safe_std(values, default=0.0):
    return float(np.std(values)) if len(values) else default


def safe_min(values, default=0.0):
    return float(np.min(values)) if len(values) else default


def safe_max(values, default=0.0):
    return float(np.max(values)) if len(values) else default


def normalize_hist(hist):
    total = float(np.sum(hist))
    if total > 0:
        hist = hist / total
    return hist.astype(np.float32)


def chroma_hist(pitches):
    hist = np.zeros(12, dtype=np.float32)
    for pitch in pitches:
        hist[int(pitch) % 12] += 1.0
    return normalize_hist(hist)


def clipped_interval_hist(seq, lo=INTERVAL_MIN, hi=INTERVAL_MAX):
    hist = np.zeros(hi - lo + 1, dtype=np.float32)
    if len(seq) < 2:
        return hist
    for interval in np.clip(np.diff(np.asarray(seq, dtype=np.float32)), lo, hi):
        hist[int(interval) - lo] += 1.0
    return normalize_hist(hist)


def quantile_features(values, scale):
    if len(values) == 0:
        return np.zeros(5, dtype=np.float32)
    arr = np.asarray(values, dtype=np.float32)
    qs = np.quantile(arr, [0.0, 0.25, 0.5, 0.75, 1.0])
    return (qs / scale).astype(np.float32)


def parse_to_events(midi_path):
    midi = miditoolkit.midi.parser.MidiFile(midi_path)
    notes = []
    for inst in midi.instruments:
        if not inst.is_drum:
            notes.extend(inst.notes)

    grouped = defaultdict(list)
    for note in notes:
        grouped[note.start].append(note)

    starts = sorted(grouped.keys())
    melody, bass, chord_mean, chord_size = [], [], [], []
    velocity_mean, duration_mean, active_notes = [], [], []

    for start in starts:
        event_notes = grouped[start]
        pitches = np.asarray([n.pitch for n in event_notes], dtype=np.float32)
        velocities = np.asarray([n.velocity for n in event_notes], dtype=np.float32)
        durations = np.asarray([n.end - n.start for n in event_notes], dtype=np.float32)

        melody.append(float(np.max(pitches)))
        bass.append(float(np.min(pitches)))
        chord_mean.append(float(np.mean(pitches)))
        chord_size.append(float(len(event_notes)))
        velocity_mean.append(float(np.mean(velocities)))
        duration_mean.append(float(np.mean(durations)))
        active_notes.append(event_notes)

    return {
        "starts": np.asarray(starts, dtype=np.float32),
        "melody": np.asarray(melody, dtype=np.float32),
        "bass": np.asarray(bass, dtype=np.float32),
        "chord_mean": np.asarray(chord_mean, dtype=np.float32),
        "chord_size": np.asarray(chord_size, dtype=np.float32),
        "velocity_mean": np.asarray(velocity_mean, dtype=np.float32),
        "duration_mean": np.asarray(duration_mean, dtype=np.float32),
        "events": active_notes,
    }


def all_note_values(seg, attr):
    values = []
    for event_notes in seg["events"]:
        values.extend(getattr(note, attr) for note in event_notes)
    return np.asarray(values, dtype=np.float32)


def all_note_pitches(seg):
    values = []
    for event_notes in seg["events"]:
        values.extend(note.pitch for note in event_notes)
    return np.asarray(values, dtype=np.float32)


def polyphony_stats(seg):
    points = []
    for event_notes in seg["events"]:
        for note in event_notes:
            points.append((note.start, 1))
            points.append((note.end, -1))
    if not points:
        return np.zeros(3, dtype=np.float32)
    points.sort()
    cur = 0
    values = []
    for _, delta in points:
        cur += delta
        values.append(cur)
    values = np.asarray(values, dtype=np.float32)
    return np.asarray(
        [safe_mean(values) / 8.0, safe_std(values) / 8.0, safe_max(values) / 16.0],
        dtype=np.float32,
    )


def segment_stats(seg):
    starts = seg["starts"]
    melody = seg["melody"]
    bass = seg["bass"]
    chord_mean = seg["chord_mean"]
    chord_size = seg["chord_size"]
    velocity_mean = seg["velocity_mean"]
    duration_mean = seg["duration_mean"]

    if len(starts) == 0:
        return np.zeros(SEGMENT_FEATURE_DIM, dtype=np.float32)

    pitches = all_note_pitches(seg)
    velocities = all_note_values(seg, "velocity")
    durations = np.asarray(
        [note.end - note.start for event_notes in seg["events"] for note in event_notes],
        dtype=np.float32,
    )
    iois = np.diff(starts) if len(starts) >= 2 else np.asarray([], dtype=np.float32)
    length_ticks = float(starts[-1] - starts[0] + 1.0) if len(starts) else 1.0
    note_density = float(len(pitches)) / max(length_ticks / 480.0, 1.0)

    scalars = np.asarray(
        [
            np.log1p(len(starts)) / 8.0,
            np.log1p(len(pitches)) / 9.0,
            np.log1p(length_ticks) / 9.0,
            safe_mean(melody) / 127.0,
            safe_std(melody) / 24.0,
            safe_mean(bass) / 127.0,
            safe_std(bass) / 24.0,
            safe_mean(chord_mean) / 127.0,
            safe_std(chord_mean) / 24.0,
            (safe_max(pitches) - safe_min(pitches)) / 88.0,
            safe_mean(chord_size) / 8.0,
            safe_std(chord_size) / 4.0,
            safe_mean(velocity_mean) / 127.0,
            safe_std(velocity_mean) / 64.0,
            safe_mean(duration_mean) / 480.0,
            safe_std(duration_mean) / 480.0,
            safe_mean(iois) / 480.0,
            safe_std(iois) / 480.0,
            safe_min(iois) / 480.0 if len(iois) else 0.0,
            safe_max(iois) / 480.0 if len(iois) else 0.0,
            note_density / 32.0,
        ],
        dtype=np.float32,
    )

    parts = [
        chroma_hist(pitches),
        clipped_interval_hist(melody),
        clipped_interval_hist(bass),
        clipped_interval_hist(chord_mean),
        quantile_features(pitches, 127.0),
        quantile_features(velocities, 127.0),
        quantile_features(durations, 960.0),
        quantile_features(iois, 960.0),
        polyphony_stats(seg),
        scalars,
    ]
    return np.concatenate(parts).astype(np.float32)


def take_boundary(seg, side, n_events):
    n = len(seg["starts"])
    if n == 0:
        return None
    if side == "start":
        idx = np.arange(0, min(n, n_events))
    else:
        idx = np.arange(max(0, n - n_events), n)
    return idx


def boundary_features(seg, side, n_events):
    idx = take_boundary(seg, side, n_events)
    if idx is None:
        return np.zeros(BOUNDARY_FEATURE_DIM, dtype=np.float32)

    starts = seg["starts"][idx]
    melody = seg["melody"][idx]
    bass = seg["bass"][idx]
    chord_mean = seg["chord_mean"][idx]
    chord_size = seg["chord_size"][idx]
    velocity_mean = seg["velocity_mean"][idx]
    duration_mean = seg["duration_mean"][idx]
    edge = -1 if side == "end" else 0
    inner = 0 if side == "end" else -1

    note_pitches, note_velocities, note_durations = [], [], []
    for i in idx:
        for note in seg["events"][int(i)]:
            note_pitches.append(note.pitch)
            note_velocities.append(note.velocity)
            note_durations.append(note.end - note.start)
    note_pitches = np.asarray(note_pitches, dtype=np.float32)
    note_velocities = np.asarray(note_velocities, dtype=np.float32)
    note_durations = np.asarray(note_durations, dtype=np.float32)
    iois = np.diff(starts) if len(starts) >= 2 else np.asarray([], dtype=np.float32)

    scalar = np.asarray(
        [
            melody[edge] / 127.0,
            bass[edge] / 127.0,
            chord_mean[edge] / 127.0,
            chord_size[edge] / 8.0,
            velocity_mean[edge] / 127.0,
            duration_mean[edge] / 480.0,
            (melody[edge] - melody[inner]) / 24.0 if len(melody) >= 2 else 0.0,
            (bass[edge] - bass[inner]) / 24.0 if len(bass) >= 2 else 0.0,
            (chord_mean[edge] - chord_mean[inner]) / 24.0 if len(chord_mean) >= 2 else 0.0,
            safe_mean(melody) / 127.0,
            safe_std(melody) / 24.0,
            safe_mean(bass) / 127.0,
            safe_std(bass) / 24.0,
            safe_mean(chord_mean) / 127.0,
            safe_std(chord_mean) / 24.0,
            safe_mean(chord_size) / 8.0,
            safe_std(chord_size) / 4.0,
            safe_mean(velocity_mean) / 127.0,
            safe_std(velocity_mean) / 64.0,
            safe_mean(duration_mean) / 480.0,
            safe_std(duration_mean) / 480.0,
            safe_mean(iois) / 480.0,
            safe_std(iois) / 480.0,
            safe_min(iois) / 480.0 if len(iois) else 0.0,
            safe_max(iois) / 480.0 if len(iois) else 0.0,
            np.log1p(len(idx)) / 4.0,
            np.log1p(len(note_pitches)) / 6.0,
        ],
        dtype=np.float32,
    )

    parts = [
        chroma_hist(note_pitches),
        clipped_interval_hist(melody),
        clipped_interval_hist(bass),
        clipped_interval_hist(chord_mean),
        quantile_features(note_pitches, 127.0),
        quantile_features(note_velocities, 127.0),
        quantile_features(note_durations, 960.0),
        quantile_features(iois, 960.0),
        scalar,
    ]
    return np.concatenate(parts).astype(np.float32)


def transition_features(left_end, right_start):
    diff = left_end - right_start
    abs_diff = np.abs(diff)
    denom = np.abs(left_end) + np.abs(right_start) + 1e-6
    ratio = diff / denom

    l2 = float(np.linalg.norm(diff))
    l1 = float(np.sum(abs_diff))
    linf = float(np.max(abs_diff)) if len(abs_diff) else 0.0
    cosine = float(
        np.dot(left_end, right_start)
        / (np.linalg.norm(left_end) * np.linalg.norm(right_start) + 1e-9)
    )
    sign_balance = float(np.mean(np.sign(diff))) if len(diff) else 0.0

    summary = np.asarray(
        [
            -l2,
            -l1,
            -linf,
            cosine,
            safe_mean(diff),
            safe_std(diff),
            safe_mean(abs_diff),
            safe_max(abs_diff),
            safe_mean(ratio),
            safe_std(ratio),
            sign_balance,
        ],
        dtype=np.float32,
    )
    return diff, abs_diff, ratio.astype(np.float32), summary


def direct_boundary_scalars(left, right):
    left_m = left["melody"]
    right_m = right["melody"]
    left_b = left["bass"]
    right_b = right["bass"]
    left_c = left["chord_mean"]
    right_c = right["chord_mean"]
    left_d = left["duration_mean"]
    right_d = right["duration_mean"]
    left_v = left["velocity_mean"]
    right_v = right["velocity_mean"]
    left_s = left["starts"]
    right_s = right["starts"]

    if not len(left_m) or not len(right_m):
        return np.zeros(27, dtype=np.float32)

    lm_last, lm_prev = left_m[-1], left_m[-2] if len(left_m) >= 2 else left_m[-1]
    rm_first, rm_next = right_m[0], right_m[1] if len(right_m) >= 2 else right_m[0]
    lb_last, lb_prev = left_b[-1], left_b[-2] if len(left_b) >= 2 else left_b[-1]
    rb_first, rb_next = right_b[0], right_b[1] if len(right_b) >= 2 else right_b[0]
    lc_last, lc_prev = left_c[-1], left_c[-2] if len(left_c) >= 2 else left_c[-1]
    rc_first, rc_next = right_c[0], right_c[1] if len(right_c) >= 2 else right_c[0]

    left_ioi = left_s[-1] - left_s[-2] if len(left_s) >= 2 else 0.0
    right_ioi = right_s[1] - right_s[0] if len(right_s) >= 2 else 0.0
    melody_cont = (rm_first - lm_last) / 24.0
    bass_cont = (rb_first - lb_last) / 24.0
    chord_cont = (rc_first - lc_last) / 24.0

    return np.asarray(
        [
            abs(melody_cont),
            melody_cont,
            abs(bass_cont),
            bass_cont,
            abs(chord_cont),
            chord_cont,
            abs((rm_first % 12) - (lm_last % 12)) / 12.0,
            abs((rb_first % 12) - (lb_last % 12)) / 12.0,
            ((lm_last - lm_prev) - (rm_next - rm_first)) / 24.0,
            ((lb_last - lb_prev) - (rb_next - rb_first)) / 24.0,
            ((lc_last - lc_prev) - (rc_next - rc_first)) / 24.0,
            abs(left_ioi - right_ioi) / 480.0,
            (left_ioi - right_ioi) / 480.0,
            abs(left_d[-1] - right_d[0]) / 480.0,
            (left_d[-1] - right_d[0]) / 480.0,
            abs(left_v[-1] - right_v[0]) / 127.0,
            (left_v[-1] - right_v[0]) / 127.0,
            abs(left["chord_size"][-1] - right["chord_size"][0]) / 8.0,
            (left["chord_size"][-1] - right["chord_size"][0]) / 8.0,
            abs(melody_cont) - abs(bass_cont),
            abs(melody_cont) - abs(chord_cont),
            float(np.sign(melody_cont)),
            float(np.sign(bass_cont)),
            float(np.sign(chord_cont)),
            np.log1p(len(left_m)) / 8.0,
            np.log1p(len(right_m)) / 8.0,
            (np.log1p(len(left_m)) - np.log1p(len(right_m))) / 8.0,
        ],
        dtype=np.float32,
    )


def pair_features(seg_a, seg_b):
    stat_a = segment_stats(seg_a)
    stat_b = segment_stats(seg_b)
    feats = [stat_a - stat_b, np.abs(stat_a - stat_b), (stat_a + stat_b) * 0.5]

    ab_direct = direct_boundary_scalars(seg_a, seg_b)
    ba_direct = direct_boundary_scalars(seg_b, seg_a)
    feats.extend([ab_direct, ba_direct, ab_direct - ba_direct, np.abs(ab_direct - ba_direct)])

    for window in BOUNDARY_WINDOWS:
        a_end = boundary_features(seg_a, "end", window)
        a_start = boundary_features(seg_a, "start", window)
        b_end = boundary_features(seg_b, "end", window)
        b_start = boundary_features(seg_b, "start", window)

        ab_diff, ab_abs, ab_ratio, ab_summary = transition_features(a_end, b_start)
        ba_diff, ba_abs, ba_ratio, ba_summary = transition_features(b_end, a_start)
        summary_delta = ab_summary - ba_summary

        feats.extend(
            [
                ab_diff,
                ba_diff,
                ab_abs,
                ba_abs,
                ab_ratio,
                ba_ratio,
                ab_summary,
                ba_summary,
                summary_delta,
                np.abs(summary_delta),
            ]
        )

    return np.nan_to_num(np.concatenate(feats), nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32
    )


def load_data():
    with open(os.path.join(DATAROOT, "train.json")) as f:
        train = eval(f.read())
    with open(os.path.join(DATAROOT, "test.json")) as f:
        test = eval(f.read())
    return train, test


_event_cache = {}


def get_events(path):
    if path not in _event_cache:
        _event_cache[path] = parse_to_events(os.path.join(DATAROOT, path))
    return _event_cache[path]


def extract_features_for_pair(p1, p2):
    return pair_features(get_events(p1), get_events(p2))


def extract_all_pairs(pairs, name="data", augment=False):
    feats = []
    labels = []
    pair_list = []

    for p1, p2, label in tqdm(pairs, desc=f"Extracting {name}"):
        feats.append(extract_features_for_pair(p1, p2))
        labels.append(label)
        pair_list.append((p1, p2))

        if augment:
            feats.append(extract_features_for_pair(p2, p1))
            labels.append(1 - label)
            pair_list.append((p2, p1))

    return np.asarray(feats, dtype=np.float32), np.asarray(labels, dtype=np.int64), pair_list


def make_models():
    return [
        (
            "LR",
            lambda: make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=5000, C=0.75, n_jobs=-1),
            ),
        ),
        (
            "RF",
            lambda: RandomForestClassifier(
                n_estimators=800,
                max_features="sqrt",
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "ExtraTrees",
            lambda: ExtraTreesClassifier(
                n_estimators=900,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight=None,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "HistGB",
            lambda: HistGradientBoostingClassifier(
                learning_rate=0.045,
                max_iter=450,
                max_leaf_nodes=31,
                l2_regularization=0.02,
                random_state=42,
            ),
        ),
        (
            "GBM",
            lambda: GradientBoostingClassifier(
                n_estimators=450,
                learning_rate=0.045,
                max_depth=3,
                subsample=0.85,
                random_state=42,
            ),
        ),
    ]


def make_soft_vote():
    return VotingClassifier(
        estimators=[
            (
                "lr",
                make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=5000, C=0.75, n_jobs=-1),
                ),
            ),
            (
                "extra",
                ExtraTreesClassifier(
                    n_estimators=700,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
            (
                "histgb",
                HistGradientBoostingClassifier(
                    learning_rate=0.045,
                    max_iter=400,
                    max_leaf_nodes=31,
                    l2_regularization=0.02,
                    random_state=42,
                ),
            ),
            (
                "gbm",
                GradientBoostingClassifier(
                    n_estimators=400,
                    learning_rate=0.045,
                    max_depth=3,
                    subsample=0.85,
                    random_state=42,
                ),
            ),
        ],
        voting="soft",
        weights=[1, 2, 2, 2],
        n_jobs=None,
    )


def cv_evaluate(X, y, model_fn, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    pred_rates = []

    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        model = model_fn()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X[tr], y[tr])
        preds = model.predict(X[va])
        acc = accuracy_score(y[va], preds)
        scores.append(acc)
        pred_rates.append(float(np.mean(preds)))
        log(f"  Fold {fold}: acc = {acc:.4f}, pred_true = {pred_rates[-1]:.4f}")

    mean = float(np.mean(scores))
    std = float(np.std(scores))
    log(f"  Mean +/- Std: {mean:.4f} +/- {std:.4f}")
    log(f"  CV pred True ratio: {np.mean(pred_rates):.4f}")
    return mean, std


def fit_predict(model_fn, X_train, y_train, X_test):
    model = model_fn()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)
    return model.predict(X_test)


def main():
    log("=" * 72)
    log("Task 2 v3: richer boundary continuity + sklearn model search")
    log("=" * 72)

    train_meta, test_pairs = load_data()
    train_pairs = [(p1, p2, 1 if label else 0) for (p1, p2), label in train_meta.items()]

    log(f"Train pairs: {len(train_pairs)}, test pairs: {len(test_pairs)}")
    log(f"Boundary windows: {BOUNDARY_WINDOWS}")

    log("\n--- Feature extraction: train with symmetric augmentation ---")
    X_train, y_train, _ = extract_all_pairs(train_pairs, "train", augment=True)
    log(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    log(f"True ratio: {float(np.mean(y_train)):.4f}")

    log("\n--- Feature extraction: test ---")
    test_pairs_dummy = [(p1, p2, 0) for p1, p2 in test_pairs]
    X_test, _, _ = extract_all_pairs(test_pairs_dummy, "test", augment=False)
    log(f"X_test: {X_test.shape}")
    log(f"MIDI cache size: {len(_event_cache)}")

    candidates = []
    for name, model_fn in make_models():
        log("\n" + "=" * 72)
        log(name)
        log("=" * 72)
        mean, std = cv_evaluate(X_train, y_train, model_fn)
        candidates.append((name, mean, std, model_fn))

    log("\n" + "=" * 72)
    log("SoftVote")
    log("=" * 72)
    vote_mean, vote_std = cv_evaluate(X_train, y_train, make_soft_vote)
    candidates.append(("SoftVote", vote_mean, vote_std, make_soft_vote))

    best_name, best_mean, best_std, best_fn = max(candidates, key=lambda x: x[1])
    log("\n" + "=" * 72)
    log("Model ranking")
    log("=" * 72)
    for name, mean, std, _ in sorted(candidates, key=lambda x: x[1], reverse=True):
        log(f"{name:<12} mean={mean:.4f} std={std:.4f}")

    log(f"\n>>> Best model: {best_name} (CV acc = {best_mean:.4f} +/- {best_std:.4f})")
    log(f"Retraining {best_name} on all augmented training data...")

    test_preds = fit_predict(best_fn, X_train, y_train, X_test)
    predictions = {pair: bool(pred) for pair, pred in zip(test_pairs, test_preds)}

    os.makedirs(os.path.dirname(PRED_FILE), exist_ok=True)
    with open(PRED_FILE, "w") as f:
        f.write(repr(predictions) + "\n")

    pred_true_ratio = sum(predictions.values()) / len(predictions)
    log(f"\n[OK] Wrote {PRED_FILE}")
    log(f"Prediction count: {len(predictions)}")
    log(f"Prediction True ratio: {pred_true_ratio:.4f}")

    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "w") as f:
        f.write("\n".join(_log) + "\n")
    log(f"[OK] Wrote {LOG_FILE}")


if __name__ == "__main__":
    main()
