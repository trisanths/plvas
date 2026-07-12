from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from training.visual.compose_dataset import DETECTOR_CLASSES


HIGH_RECALL_THRESHOLDS = {
    "NAME": 0.10,
    "EMAIL": 0.10,
    "PHONE": 0.10,
    "ADDRESS": 0.10,
    "CARD_NUMBER": 0.08,
    "CVC": 0.01,
    "SECRET": 0.08,
    "SENSITIVE_FIELD": 0.08,
    "SENSITIVE_IMAGE": 0.35,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _curve_value(curve: np.ndarray, confidence_axis: np.ndarray, value: float) -> float:
    return float(np.interp(value, confidence_axis, curve))


def _smooth(values: np.ndarray, fraction: float = 0.05) -> np.ndarray:
    """Match the box filter used by Ultralytics 8.4.92 metric reporting."""

    filters = round(len(values) * fraction * 2) // 2 + 1
    padding = np.ones(filters // 2)
    padded = np.concatenate(
        (padding * values[0], values, padding * values[-1]), axis=0
    )
    return np.convolve(padded, np.ones(filters) / filters, mode="valid")


def summarize_curves(
    box: Any,
    names: dict[int, str],
    thresholds: dict[str, float] = HIGH_RECALL_THRESHOLDS,
) -> dict[str, Any]:
    """Expose threshold-independent AP and PLVA operating-point P/R.

    Ultralytics' public ``box.p`` and ``box.r`` use one confidence selected by
    maximum mean F1 across all classes. That operating point can move between
    checkpoints, so it is not a stable recall-regression measurement. This
    report also samples each class curve at PLVA's frozen high-recall threshold.
    """

    expected_names = {index: name for index, name in enumerate(DETECTOR_CLASSES)}
    normalized_names = {int(index): str(name) for index, name in names.items()}
    if normalized_names != expected_names:
        raise ValueError("detector class names differ from the PLVA contract")

    class_indices = np.asarray(box.ap_class_index, dtype=int)
    px = np.asarray(box.px, dtype=float)
    p_curve = np.asarray(box.p_curve, dtype=float)
    r_curve = np.asarray(box.r_curve, dtype=float)
    f1_curve = np.asarray(box.f1_curve, dtype=float)
    ap50 = np.asarray(box.ap50, dtype=float)
    ap5095 = np.asarray(box.ap, dtype=float)
    expected_shape = (len(class_indices), len(px))
    if (
        set(class_indices.tolist()) != set(expected_names)
        or p_curve.shape != expected_shape
        or r_curve.shape != expected_shape
        or f1_curve.shape != expected_shape
        or ap50.shape != (len(class_indices),)
        or ap5095.shape != (len(class_indices),)
    ):
        raise ValueError("Ultralytics curve metrics are incomplete or misaligned")

    # Match ultralytics.utils.metrics.ap_per_class exactly: it smooths the
    # mean F1 curve before choosing the single framework reporting threshold.
    global_index = int(_smooth(f1_curve.mean(0), 0.1).argmax())
    per_class: dict[str, Any] = {}
    for position, class_index in enumerate(class_indices.tolist()):
        name = normalized_names[class_index]
        class_best_index = int(f1_curve[position].argmax())
        threshold = float(thresholds[name])
        per_class[name] = {
            "ap50": float(ap50[position]),
            "ap50_95": float(ap5095[position]),
            "framework_global_f1": {
                "threshold": float(px[global_index]),
                "precision": float(p_curve[position, global_index]),
                "recall": float(r_curve[position, global_index]),
            },
            "class_best_f1": {
                "threshold": float(px[class_best_index]),
                "precision": float(p_curve[position, class_best_index]),
                "recall": float(r_curve[position, class_best_index]),
                "f1": float(f1_curve[position, class_best_index]),
            },
            "plva_high_recall": {
                "threshold": threshold,
                "precision": _curve_value(p_curve[position], px, threshold),
                "recall": _curve_value(r_curve[position], px, threshold),
            },
        }
    return {
        "schema_version": 1,
        "framework_global_f1_threshold": float(px[global_index]),
        "per_class": per_class,
    }


def compare_reports(seed: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    comparison: dict[str, Any] = {
        "framework_global_f1_threshold_delta": (
            candidate["framework_global_f1_threshold"]
            - seed["framework_global_f1_threshold"]
        ),
        "per_class": {},
    }
    for name in DETECTOR_CLASSES:
        seed_class = seed["per_class"][name]
        candidate_class = candidate["per_class"][name]
        comparison["per_class"][name] = {
            "ap50_delta": candidate_class["ap50"] - seed_class["ap50"],
            "ap50_95_delta": candidate_class["ap50_95"] - seed_class["ap50_95"],
            "framework_recall_delta": (
                candidate_class["framework_global_f1"]["recall"]
                - seed_class["framework_global_f1"]["recall"]
            ),
            "plva_high_recall_delta": (
                candidate_class["plva_high_recall"]["recall"]
                - seed_class["plva_high_recall"]["recall"]
            ),
        }
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare detector PR curves at PLVA's frozen operating point"
    )
    parser.add_argument("--dataset-yaml", type=Path, required=True)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    from ultralytics import YOLO

    reports: dict[str, Any] = {}
    for label, checkpoint in (("seed", args.seed), ("candidate", args.candidate)):
        if checkpoint.is_symlink() or not checkpoint.is_file():
            raise FileNotFoundError(f"{label} checkpoint is missing or unsafe")
        evaluation = YOLO(str(checkpoint)).val(
            data=str(args.dataset_yaml),
            split="val",
            imgsz=640,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            plots=False,
            verbose=False,
            save_json=False,
            project="/tmp/plva-detector-curve-diagnostics",
            name=label,
            exist_ok=True,
        )
        reports[label] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "aggregate": {
                str(key): float(value)
                for key, value in evaluation.results_dict.items()
            },
            "curves": summarize_curves(evaluation.box, evaluation.names),
        }

    document = {
        "schema_version": 1,
        "dataset_yaml": str(args.dataset_yaml),
        "checkpoints": reports,
        "candidate_minus_seed": compare_reports(
            reports["seed"]["curves"], reports["candidate"]["curves"]
        ),
    }
    serialized = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.partial")
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(args.output)
    print(serialized, end="")


if __name__ == "__main__":
    main()
