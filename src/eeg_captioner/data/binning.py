import math
import json
from pathlib import Path
from .rules import PERIODS, ROIS

class Thresholds:
    def __init__(self, path: Path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.features = payload["features"]

    def quintile(self, key: str, value: float) -> int:
        spec = self.features[key]
        if spec["method"] != "quintile":
            raise ValueError(f"Expected quintile threshold for {key}")
        edges = [spec["thresholds"][f"p{value}"] for value in (20, 40, 60, 80)]
        return sum(float(value) > edge for edge in edges)

    def signed(self, key: str, value: float) -> int:
        """Return -2/-1/0/+1/+2 exactly as build_summary_bin.Binner."""

        spec = self.features[key]
        if spec["method"] != "zero_centered_magnitude":
            raise ValueError(f"Expected zero-centered threshold for {key}")
        low = float(spec["thresholds"]["abs_p33"])
        high = float(spec["thresholds"]["abs_p67"])
        value = float(value)
        if value < -high:
            return -2
        if value < -low:
            return -1
        if value <= low:
            return 0
        if value <= high:
            return 1
        return 2


def unique_bin_argmax(labels: dict[str, int]):
    maximum = max(labels.values())
    winners = [name for name, value in labels.items() if value == maximum]
    return winners[0] if len(winners) == 1 else None


def audit_case(case: dict, features: dict, thresholds: Thresholds) -> dict:
    case_type = case["type"]
    answer = case["answer_label"]
    detail = {}
    bin_answer = None
    reason = "aligned"

    if case_type == "dominant_band":
        # The binned target explicitly says both "X is the dominant band" and
        # prints the exact band-power ordering, so the answer remains visible
        # even when population-reference labels coincide.
        bin_answer = answer
        reason = "explicit_dominant_and_ordering"

    elif case_type == "strongest_region":
        period, band = case["period"], case["band"]
        values = {
            roi: features["absolute"][(period, roi)][band]
            for roi in ROIS
        }
        labels = {
            roi: thresholds.quintile(
                f"roi.bandpower.{band}.absolute", value
            )
            for roi, value in values.items()
        }
        bin_answer = unique_bin_argmax(labels)
        detail = {"bin_levels": labels}
        if bin_answer is None:
            reason = "top_options_share_bin"

    elif case_type == "peak_period":
        roi, band = case["roi"], case["band"]
        values = {
            period: features["absolute"][(period, roi)][band]
            for period in PERIODS
        }
        labels = {
            period: thresholds.quintile(
                f"roi.bandpower.{band}.absolute", value
            )
            for period, value in values.items()
        }
        bin_answer = unique_bin_argmax(labels)
        detail = {"bin_levels": labels}
        if bin_answer is None:
            reason = "top_options_share_bin"

    elif case_type == "temporal_pattern":
        roi, band = case["roi"], case["band"]
        values = [
            features["relative"][(period, roi)][band] for period in PERIODS
        ]
        key = f"change.roi.bandpower.{band}.relative"
        signs = [
            thresholds.signed(key, values[1] - values[0]),
            thresholds.signed(key, values[2] - values[1]),
        ]
        if signs[0] > 0 and signs[1] > 0:
            bin_answer = "continuously increases"
        elif signs[0] < 0 and signs[1] < 0:
            bin_answer = "continuously decreases"
        elif signs[0] > 0 and signs[1] < 0:
            bin_answer = "increases then decreases"
        elif signs[0] < 0 and signs[1] > 0:
            bin_answer = "decreases then increases"
        elif signs == [0, 0]:
            bin_answer = "approximately stable"
        else:
            reason = "bin_pattern_not_one_of_options"
        detail = {"binned_change_signs": signs}

    elif case_type == "lateral_dominance":
        key = (case["period"], case["pair"], case["band"])
        value = features["lateralization"].get(key)
        if value is None or not math.isfinite(value):
            reason = "bin_value_unavailable"
        elif abs(value) >= 1.0:
            # lateralization=(R-L)/(R+L) and asymmetry_log=ln(R)-ln(L).
            # At the limiting values, the log ratio has infinite magnitude.
            asymmetry_log = math.copysign(math.inf, value)
            signed = -2 if asymmetry_log < 0 else 2
            bin_answer = "left" if signed < 0 else "right"
            detail = {
                "lateralization_index": value,
                "derived_asymmetry_log": asymmetry_log,
                "binned_asymmetry_sign": signed,
            }
        else:
            # summary_bin renders asymmetry_log, whereas the original evaluator
            # labels lateralization_index.  Convert exactly before applying the
            # train-time text threshold: log(R/L) = 2 * atanh((R-L)/(R+L)).
            asymmetry_log = 2.0 * math.atanh(value)
            signed = thresholds.signed(
                f"centered.asymmetry.{case['band']}.log", asymmetry_log
            )
            bin_answer = "left" if signed < 0 else "right" if signed > 0 else "approximately symmetric"
            detail = {
                "lateralization_index": value,
                "derived_asymmetry_log": asymmetry_log,
                "binned_asymmetry_sign": signed,
            }
    else:
        raise ValueError(f"Unsupported case type: {case_type}")

    eligible = bin_answer == answer
    if bin_answer is not None and not eligible:
        reason = "bin_answer_disagrees_with_numeric_answer"
    return {
        "id": case["id"],
        "question_id": case["question_id"],
        "type": case_type,
        "answer_label": answer,
        "bin_answer_label": bin_answer,
        "eligible": eligible,
        "reason": reason,
        **detail,
    }
