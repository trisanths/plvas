from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np

from training.visual.diagnose_detector_curves import HIGH_RECALL_THRESHOLDS
from training.visual.prepare_webpii import DETECTOR_CLASSES


CALIBRATION_POLICY_NAME = "privacy-monotone-recall-preserving-calibration-v2"
CALIBRATION_POLICY: dict[str, Any] = {
    "schema_version": 2,
    "name": CALIBRATION_POLICY_NAME,
    "maximum_calibration_recall_drop": 0.0025,
    "minimum_threshold": 0.005,
    "maximum_threshold": 0.95,
    "maximum_candidate_threshold": "frozen-seed-baseline-per-class",
    "objective": (
        "highest per-class confidence preserving frozen-seed recall without "
        "raising a known-safe threshold"
    ),
    "selection_data": "synthetic-and-webpii-published-train-calibration",
    "published_test_used": False,
}


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _class_position(box: Any) -> dict[int, int]:
    indices = [int(value) for value in getattr(box, "ap_class_index", [])]
    expected = set(range(len(DETECTOR_CLASSES)))
    if len(indices) != len(expected) or set(indices) != expected:
        raise ValueError("calibration curves do not contain every detector class")
    return {class_index: position for position, class_index in enumerate(indices)}


def _validated_curves(box: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    px = np.asarray(getattr(box, "px", []), dtype=float)
    precision = np.asarray(getattr(box, "p_curve", []), dtype=float)
    recall = np.asarray(getattr(box, "r_curve", []), dtype=float)
    expected_shape = (len(DETECTOR_CLASSES), len(px))
    if (
        px.ndim != 1
        or len(px) < 2
        or not np.all(np.isfinite(px))
        or not np.all(np.diff(px) >= 0)
        or precision.shape != expected_shape
        or recall.shape != expected_shape
        or not np.all(np.isfinite(precision))
        or not np.all(np.isfinite(recall))
        or np.any(precision < 0)
        or np.any(precision > 1)
        or np.any(recall < 0)
        or np.any(recall > 1)
    ):
        raise ValueError("calibration P/R curves are incomplete or invalid")
    return px, precision, recall


def calibrate_threshold_profile(
    seed_box: Any,
    candidate_box: Any,
    names: dict[int, str],
    *,
    baseline_thresholds: dict[str, float] = HIGH_RECALL_THRESHOLDS,
    maximum_recall_drop: float = float(
        CALIBRATION_POLICY["maximum_calibration_recall_drop"]
    ),
) -> dict[str, Any]:
    """Calibrate one candidate without consulting guard or test outcomes."""

    normalized_names = {int(index): str(name) for index, name in names.items()}
    expected_names = {index: name for index, name in enumerate(DETECTOR_CLASSES)}
    if normalized_names != expected_names:
        raise ValueError("calibration class contract changed")
    if set(baseline_thresholds) != set(DETECTOR_CLASSES):
        raise ValueError("baseline threshold profile is incomplete")
    if (
        not math.isfinite(maximum_recall_drop)
        or not 0.0 <= maximum_recall_drop <= 0.05
    ):
        raise ValueError("maximum calibration recall drop is invalid")

    seed_positions = _class_position(seed_box)
    candidate_positions = _class_position(candidate_box)
    seed_px, _seed_precision, seed_recall = _validated_curves(seed_box)
    candidate_px, candidate_precision, candidate_recall = _validated_curves(
        candidate_box
    )
    minimum = float(CALIBRATION_POLICY["minimum_threshold"])
    maximum = float(CALIBRATION_POLICY["maximum_threshold"])

    thresholds: dict[str, float] = {}
    classes: dict[str, Any] = {}
    for class_index, name in expected_names.items():
        baseline_threshold = float(baseline_thresholds[name])
        if not 0.0 <= baseline_threshold <= 1.0:
            raise ValueError(f"baseline threshold is invalid for {name}")
        seed_position = seed_positions[class_index]
        candidate_position = candidate_positions[class_index]
        seed_operating_recall = float(
            np.interp(
                baseline_threshold,
                seed_px,
                seed_recall[seed_position],
            )
        )
        target_recall = max(0.0, seed_operating_recall - maximum_recall_drop)
        # This is a privacy-first calibrator.  Raising every class threshold can
        # also raise the global pre-profile inference floor (the minimum of the
        # profile), silently turning recoverable low-confidence boxes into
        # apparent irrecoverable misses on a shifted guard distribution.  A
        # candidate may therefore keep or lower each known-safe seed threshold,
        # but it may never raise one merely to gain calibration-set precision.
        safe_maximum = min(maximum, baseline_threshold)
        eligible = np.where(
            (candidate_px >= minimum)
            & (candidate_px <= safe_maximum)
            & (candidate_recall[candidate_position] >= target_recall)
        )[0]
        passed = bool(len(eligible))
        selected_index = int(eligible[-1]) if passed else int(
            np.abs(candidate_px - minimum).argmin()
        )
        selected_threshold = float(candidate_px[selected_index])
        selected_recall = float(candidate_recall[candidate_position, selected_index])
        selected_precision = float(
            candidate_precision[candidate_position, selected_index]
        )
        # Ultralytics' confidence grid is deterministic; fixed decimal encoding
        # prevents insignificant dtype formatting from changing artifact hashes.
        thresholds[name] = round(selected_threshold, 6)
        classes[name] = {
            "baseline_threshold": baseline_threshold,
            "maximum_safe_threshold": safe_maximum,
            "seed_operating_recall": seed_operating_recall,
            "target_recall": target_recall,
            "candidate_threshold": thresholds[name],
            "candidate_recall": selected_recall,
            "candidate_precision": selected_precision,
            "recall_margin": selected_recall - target_recall,
            "passed": (
                passed
                and selected_recall + 1e-12 >= target_recall
                and thresholds[name] <= baseline_threshold + 1e-12
            ),
        }

    passed = all(value["passed"] for value in classes.values())
    profile_sha256 = canonical_json_sha256(thresholds)
    return {
        "schema_version": 2,
        "policy": CALIBRATION_POLICY,
        "policy_sha256": canonical_json_sha256(CALIBRATION_POLICY),
        "baseline_profile": dict(baseline_thresholds),
        "baseline_profile_sha256": canonical_json_sha256(baseline_thresholds),
        "calibrated_profile": thresholds,
        "calibrated_profile_sha256": profile_sha256,
        "thresholds_never_raised": all(
            thresholds[name] <= float(baseline_thresholds[name]) + 1e-12
            for name in DETECTOR_CLASSES
        ),
        "per_class": classes,
        "passed": passed,
    }
