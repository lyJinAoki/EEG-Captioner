import math
import re
import random
from collections import defaultdict

PERIODS = ("early", "middle", "last")


BANDS = ("delta", "theta", "alpha", "beta", "gamma")


ROIS = ("frontal", "central", "parietal", "occipital", "temporal")


ROI_CHANNELS = {
    "frontal": ("FP1", "FP2", "F3", "F4", "F7", "F8", "FZ"),
    "central": ("C3", "C4", "CZ"),
    "parietal": ("P3", "P4", "PZ"),
    "occipital": ("O1", "O2", "OZ"),
    "temporal": ("T3", "T4", "T5", "T6"),
}


ALL_CHANNELS = frozenset(channel for channels in ROI_CHANNELS.values() for channel in channels)


TYPE_OPTIONS = {
    "dominant_band": BANDS,
    "strongest_region": ROIS,
    "peak_period": PERIODS,
    "temporal_pattern": (
        "continuously increases", "continuously decreases",
        "increases then decreases", "decreases then increases",
        "approximately stable",
    ),
    "lateral_dominance": ("left", "right", "approximately symmetric"),
}


PART_RE = re.compile(r"^# \*\*Part [123]: (Early|Middle|Last) Feature Values\*\*$")


MODEL_CHANNELS = (
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "OZ", "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ",
)


def split_filenames(samples, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Reproduce train2.py's file-level balanced split without importing it."""
    groups = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[sample[0]].append(index)
    filenames = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(filenames)
    filenames.sort(key=lambda name: len(groups[name]), reverse=True)
    ratios = {"train": train_ratio, "val": val_ratio, "test": 1 - train_ratio - val_ratio}
    targets = {name: len(samples) * ratio for name, ratio in ratios.items()}
    counts = dict.fromkeys(ratios, 0)
    split_files = {name: set() for name in ratios}
    for filename in filenames:
        split = min(ratios, key=lambda name: counts[name] / targets[name])
        split_files[split].add(filename)
        counts[split] += len(groups[filename])
    return split_files


def parse_summary_features(path):
    absolute = {}
    relative = {}
    lateralization = {}
    period = None
    table = None
    valid_channels = set()
    bad_channels = set()
    with path.open(encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            match = PART_RE.match(line)
            if line.startswith("- valid channels:"):
                valid_channels = {value.strip().upper() for value in line.split(":", 1)[1].split(",") if value.strip()}
            elif line.startswith("- bad channels:"):
                bad_channels = {value.strip().upper() for value in line.split(":", 1)[1].split(",") if value.strip()}
            elif match:
                period = match.group(1).lower()
                table = None
            elif line == "### ROI absolute band powers" and period:
                table = "absolute"
            elif line == "### ROI relative band powers" and period:
                table = "relative"
            elif line == "## **Spatial left-right asymmetry features**" and period:
                table = "lateralization"
            elif table and line.startswith("|"):
                cells = [cell.strip() for cell in line.strip("|").split("|")]
                if table in ("absolute", "relative") and cells[0].lower() in ROIS and len(cells) >= 6:
                    row = [float(value) for value in cells[1:6]]
                    if all(math.isfinite(value) for value in row):
                        target = absolute if table == "absolute" else relative
                        target[(period, cells[0].lower())] = dict(zip(BANDS, row))
                elif table == "lateralization" and len(cells) >= 4 and cells[1].lower() in BANDS:
                    try:
                        value = float(cells[3])
                    except ValueError:
                        continue
                    if math.isfinite(value):
                        lateralization[(period, cells[0], cells[1].lower())] = value
            elif table and line and not line.startswith("|"):
                table = None
    return {
        "absolute": absolute,
        "relative": relative,
        "lateralization": lateralization,
        "valid_channels": valid_channels,
        "bad_channels": bad_channels,
    }


def decisive_argmax(items, min_margin):
    ranked = sorted(items, key=lambda item: item[1], reverse=True)
    if ranked[0][1] <= 0 or (ranked[0][1] - ranked[1][1]) / ranked[0][1] < min_margin:
        return None
    return ranked[0][0]


def required_channels(question):
    if question["type"] == "strongest_region":
        return ALL_CHANNELS
    if question["type"] == "lateral_dominance":
        return frozenset(question["pair"].upper().split("-"))
    return frozenset(ROI_CHANNELS[question["roi"]])


def script_answer(question, features, min_margin):
    """Return (label, rejection_reason) for one EEG-question combination."""
    needed = required_channels(question)
    if not needed.issubset(features["valid_channels"]) or needed & features["bad_channels"]:
        return None, "bad_channel"
    case_type = question["type"]
    period, roi, band = question.get("period"), question.get("roi"), question.get("band")
    absolute, relative = features["absolute"], features["relative"]
    if case_type == "dominant_band":
        row = absolute.get((period, roi))
        if not row or not all(math.isfinite(row.get(name, math.nan)) for name in BANDS):
            return None, "nan"
        label = decisive_argmax(row.items(), min_margin)
        return (label, None) if label else (None, "close_top_two")
    if case_type == "strongest_region":
        items = []
        for name in ROIS:
            row = absolute.get((period, name))
            value = row.get(band, math.nan) if row else math.nan
            if not math.isfinite(value):
                return None, "nan"
            items.append((name, value))
        label = decisive_argmax(items, min_margin)
        return (label, None) if label else (None, "close_top_two")
    if case_type == "peak_period":
        items = []
        for name in PERIODS:
            row = absolute.get((name, roi))
            value = row.get(band, math.nan) if row else math.nan
            if not math.isfinite(value):
                return None, "nan"
            items.append((name, value))
        label = decisive_argmax(items, min_margin)
        return (label, None) if label else (None, "close_top_two")
    if case_type == "temporal_pattern":
        values = []
        for name in PERIODS:
            row = relative.get((name, roi))
            value = row.get(band, math.nan) if row else math.nan
            if not math.isfinite(value):
                return None, "nan"
            values.append(value)
        first, second = values[1] - values[0], values[2] - values[1]
        tau = question["threshold"]
        if first > tau and second > tau:
            return "continuously increases", None
        if first < -tau and second < -tau:
            return "continuously decreases", None
        if first > tau and second < -tau:
            return "increases then decreases", None
        if first < -tau and second > tau:
            return "decreases then increases", None
        if abs(first) <= tau and abs(second) <= tau:
            return "approximately stable", None
        return None, "mixed_stable_change"
    key = (period, question["pair"], band)
    value = features["lateralization"].get(key)
    if value is None or not math.isfinite(value):
        return None, "unavailable_pair"
    tau = question["threshold"]
    return (
        "right" if value > tau else "left" if value < -tau else "approximately symmetric",
        None,
    )
