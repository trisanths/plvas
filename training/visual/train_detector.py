from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import statistics
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.visual.prepare_webpii import DETECTOR_CLASSES
from training.visual.project_classifier_checkpoint import CLASSIFIER_TENSOR_NAMES
from training.visual.safe_finetune import (
    PROTECTED_CLASSIFIER_CLASSES,
    PROTECTED_CLASSIFIER_ROWS,
)
from training.visual.runtime_safety import (
    RUNTIME_SELECTION_POLICY,
    RUNTIME_SELECTION_POLICY_NAME,
    consider_runtime_selection_candidate,
    evaluate_first_epoch_calibrated_runtime_canary,
    evaluate_runtime_release_gate,
    runtime_selection_score,
    validate_runtime_selection_candidate,
)


SECRET_VISUAL_CLASSES = ("CARD_NUMBER", "CVC", "SECRET")
OTHER_SENSITIVE_CLASSES = tuple(
    name for name in DETECTOR_CLASSES if name not in SECRET_VISUAL_CLASSES
)
SYNTHETIC_VALIDATION_SOURCE = "screen-native-synthetic-validation"
CALIBRATED_RUNTIME_SELECTION_SOURCE = (
    "calibrated-runtime-synthetic-and-webpii-train-guards"
)
PUBLISHED_TEST_SOURCE = "WebPII/test"
EXPECTED_SUPPLEMENTAL_TEST_RECORDS = 200
ULTRALYTICS_RUN_NAME = "ultralytics-run"
SAFETY_PROGRESS_NAME = "safety-selection-progress.json"
SEED_INCUMBENT_NAME = "seed-incumbent.json"
SAFETY_THRESHOLDS_NAME = "safety-best-thresholds.json"
INVOCATION_STATE_NAME = "training-invocation-state.json"
REPLAY_TRAINING_KIND = "deterministic-source-class-aware-replay-list"
PRETRAIN_BUDGET_PAUSE_STATUS = "budget-paused-before-training"
SELECTION_EPSILON = 1e-12
LEGACY_GLOBAL_F1_SELECTION_POLICY = {
    "schema_version": 1,
    "meaningful_secret_gain": 0.01,
    "maximum_individual_secret_drop": 0.01,
    "maximum_other_sensitive_mean_drop_from_incumbent": 0.03,
    "maximum_other_sensitive_mean_drop_from_high_water": 0.05,
    "maximum_individual_other_sensitive_drop": 0.05,
    "maximum_precision_mean_drop": 0.05,
    "tie_band_other_sensitive_mean_gain": 0.01,
    "precision_tiebreak_requires_no_recall_regression": True,
    "epsilon": SELECTION_EPSILON,
}
SAFETY_SELECTION_POLICY = RUNTIME_SELECTION_POLICY
# Ultralytics' natural-image defaults include 50% horizontal flips, 100%
# mosaic, +/-50% scale, +/-10% translation, and high saturation/value jitter.
# Those transforms can mirror text, tile unrelated pages, or destroy the layout
# semantics of UI screenshots. Keep only small viewport-like motion and modest
# lighting/display variation. Every detection augmentation knob used by the
# pinned 8.4.92 data pipeline is explicit here so a framework default cannot
# silently re-enter training.
SCREENSHOT_AUGMENTATION_POLICY: dict[str, float | int | str] = {
    "hsv_h": 0.0,
    "hsv_s": 0.1,
    "hsv_v": 0.1,
    "degrees": 0.0,
    "translate": 0.02,
    "scale": 0.05,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.0,
    "bgr": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "copy_paste_mode": "flip",  # Inert because copy_paste is zero.
    "close_mosaic": 0,
    "multi_scale": 0.0,
}

# Ultralytics 8.4.92 selects MuSGD for schedules over 10,000 iterations and
# triples the learning rate for model.23.cv3 classification-head parameters.
# That auto policy destabilized the warm-started detector as its default
# three-epoch warmup approached a 0.03 head LR. These settings are explicit and
# immutable across resume so framework defaults cannot silently re-enter.
STABLE_OPTIMIZER_POLICY_NAME = "safe-output-classifier-adamw-v5-fp32-trust-region"
STABLE_OPTIMIZER_POLICY: dict[str, float | int | str | bool] = {
    "optimizer": "AdamW",
    # v8 at 1e-6 improved aggregate AP but moved the box-regression branch and
    # introduced 31 WebPII low-floor misses. v9 froze geometry, but serializing
    # its 1e-7 update through Ultralytics' FP16 EMA checkpoint quantized the
    # remaining classifier changes into ~6.1e-5 steps and lost five ADDRESS
    # proposals. Keep the small trust region and persist a projected FP32 EMA.
    "lr0": 1e-7,
    "lrf": 0.1,
    "momentum": 0.9,
    "weight_decay": 0.0,
    "warmup_epochs": 1.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.0,
    "cos_lr": True,
    "nbs": 64,
    "cls_pw": 0.0,
}

SAFE_FINE_TUNE_POLICY: dict[str, Any] = {
    "schema_version": 2,
    "mode": "final-class-output-only-projected-fp32-canary",
    "freeze_layers": list(range(23)),
    "trainable_layer": 23,
    "trainable_parameters": list(CLASSIFIER_TENSOR_NAMES),
    "protected_classifier_classes": list(PROTECTED_CLASSIFIER_CLASSES),
    "protected_classifier_rows": list(PROTECTED_CLASSIFIER_ROWS),
    "protected_gradient_policy": "zero-before-optimizer-step",
    "shared_classification_features": "frozen",
    "box_regression": "frozen",
    "batchnorm_running_stats": "frozen-via-eval-every-batch",
    "batchnorm_affine_parameters": "frozen",
    "checkpoint_ema_precision": "float32",
    "checkpoint_frozen_state_projection": "exact-reference-copy",
    "immutable_seed_pre_step_state_parity": True,
    "canary_maximum_individual_recall_drop": 0.05,
}

MONITORING_POLICY = {
    "automatic_stop": False,
    "map50_absolute_drop": 0.10,
    "validation_classification_loss_multiple": 2.0,
    "consecutive_epochs": 2,
    "action": "manual-review-and-stop; never resume a failed canary",
}


class PlannedTrainingPause(RuntimeError):
    """Stop after a durable epoch without letting Ultralytics strip last.pt."""

    def __init__(self, reason: str, details: dict[str, Any]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = details


def _finite_float(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{context} is not numeric") from exc
    value = float(value)
    if not math.isfinite(value):
        raise RuntimeError(f"{context} is not finite")
    return value


def evaluate_first_epoch_canary(
    seed_incumbent: dict[str, Any], record: dict[str, Any]
) -> dict[str, Any]:
    """Require epoch one to pass independently recomputed runtime gates."""

    return evaluate_first_epoch_calibrated_runtime_canary(seed_incumbent, record)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def align_class_metrics(
    box: Any, names: dict[int, str]
) -> tuple[dict[int, float], dict[int, float], list[str]]:
    """Align Ultralytics' present-class metrics to the full detector taxonomy.

    Ultralytics omits classes with no validation labels from ``box.p`` and
    ``box.r``. ``box.ap_class_index`` is the authoritative mapping back to the
    dataset class IDs. Missing classes receive zero rather than inheriting a
    neighbouring class metric or crashing checkpoint selection.
    """
    normalized_names = {int(index): str(name) for index, name in names.items()}
    recalls = [float(value) for value in getattr(box, "r", [])]
    precisions = [float(value) for value in getattr(box, "p", [])]
    class_indices = [int(value) for value in getattr(box, "ap_class_index", [])]
    if len(recalls) != len(precisions) or len(recalls) != len(class_indices):
        raise RuntimeError(
            "Ultralytics per-class metrics cannot be aligned: p, r, and "
            "ap_class_index have different lengths"
        )
    unknown = sorted(set(class_indices) - set(normalized_names))
    if unknown:
        raise RuntimeError(
            f"Ultralytics reported metrics for unknown detector classes: {unknown}"
        )
    if len(set(class_indices)) != len(class_indices):
        raise RuntimeError(
            "Ultralytics reported duplicate detector class metric indices"
        )
    recall_by_class = {index: 0.0 for index in normalized_names}
    precision_by_class = {index: 0.0 for index in normalized_names}
    for position, class_index in enumerate(class_indices):
        recall_by_class[class_index] = recalls[position]
        precision_by_class[class_index] = precisions[position]
    reported = set(class_indices)
    missing = [
        normalized_names[index]
        for index in sorted(normalized_names)
        if index not in reported
    ]
    return recall_by_class, precision_by_class, missing


def _validated_metric_map(
    values: dict[str, float], context: str
) -> dict[str, float]:
    if not isinstance(values, dict) or set(values) != set(DETECTOR_CLASSES):
        raise ValueError(f"{context} must contain every detector class exactly once")
    normalized: dict[str, float] = {}
    for name in DETECTOR_CLASSES:
        value = values[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"{context} contains an invalid value for {name}")
        normalized[name] = float(value)
    return normalized


def summarize_safety_metrics(
    *,
    epoch: int,
    recall_by_class: dict[str, float],
    precision_by_class: dict[str, float],
) -> dict[str, Any]:
    recalls = _validated_metric_map(recall_by_class, "per-class recall")
    precisions = _validated_metric_map(precision_by_class, "per-class precision")
    return {
        "epoch": int(epoch),
        "per_class_recall": recalls,
        "per_class_precision": precisions,
        "secret_recall_minimum": min(
            recalls[name] for name in SECRET_VISUAL_CLASSES
        ),
        "other_sensitive_recall_mean": sum(
            recalls[name] for name in OTHER_SENSITIVE_CLASSES
        )
        / len(OTHER_SENSITIVE_CLASSES),
        "precision_mean": sum(precisions.values()) / len(precisions),
    }


def consider_safety_candidate(
    state: dict[str, Any] | None,
    *,
    epoch: int,
    recall_by_class: dict[str, float],
    precision_by_class: dict[str, float],
    missing_classes: list[str],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Apply the constrained PLVA checkpoint rule without recall ratcheting."""
    invalid_reason = None
    if missing_classes:
        invalid_reason = "candidate-validation-missing-classes"
    try:
        candidate = summarize_safety_metrics(
            epoch=epoch,
            recall_by_class=recall_by_class,
            precision_by_class=precision_by_class,
        )
    except (TypeError, ValueError) as exc:
        candidate = None
        invalid_reason = f"candidate-validation-invalid: {exc}"
    if invalid_reason is not None or candidate is None:
        return (
            {
                "selected": False,
                "valid": False,
                "reason": invalid_reason,
                "candidate": candidate,
                "deltas": None,
                "gates": {},
            },
            state,
        )

    if state is None:
        initialized = {
            "schema_version": 1,
            "incumbent": candidate,
            "high_water": {
                "other_sensitive_recall_mean": candidate[
                    "other_sensitive_recall_mean"
                ]
            },
        }
        return (
            {
                "selected": True,
                "valid": True,
                "reason": "initial-frozen-seed-incumbent",
                "candidate": candidate,
                "deltas": None,
                "gates": {"complete_finite_all_class_metrics": True},
            },
            initialized,
        )

    incumbent = state.get("incumbent")
    high_water = state.get("high_water")
    if not isinstance(incumbent, dict) or not isinstance(high_water, dict):
        raise RuntimeError("safety selection state is malformed")
    # Revalidate persisted state before it can influence a new decision.
    incumbent = summarize_safety_metrics(
        epoch=int(incumbent.get("epoch")),
        recall_by_class=incumbent.get("per_class_recall"),
        precision_by_class=incumbent.get("per_class_precision"),
    )
    historical_other = high_water.get("other_sensitive_recall_mean")
    if (
        isinstance(historical_other, bool)
        or not isinstance(historical_other, (int, float))
        or not math.isfinite(float(historical_other))
        or not 0.0 <= float(historical_other) <= 1.0
    ):
        raise RuntimeError("safety selection high-water mark is malformed")
    historical_other = float(historical_other)

    secret_delta = (
        candidate["secret_recall_minimum"] - incumbent["secret_recall_minimum"]
    )
    other_delta = (
        candidate["other_sensitive_recall_mean"]
        - incumbent["other_sensitive_recall_mean"]
    )
    precision_delta = candidate["precision_mean"] - incumbent["precision_mean"]
    eps = SELECTION_EPSILON
    gates = {
        "individual_secret_drop": all(
            candidate["per_class_recall"][name]
            >= incumbent["per_class_recall"][name]
            - LEGACY_GLOBAL_F1_SELECTION_POLICY["maximum_individual_secret_drop"]
            - eps
            for name in SECRET_VISUAL_CLASSES
        ),
        "other_mean_drop_from_incumbent": candidate[
            "other_sensitive_recall_mean"
        ]
        >= incumbent["other_sensitive_recall_mean"]
        - LEGACY_GLOBAL_F1_SELECTION_POLICY[
            "maximum_other_sensitive_mean_drop_from_incumbent"
        ]
        - eps,
        "other_mean_drop_from_high_water": candidate[
            "other_sensitive_recall_mean"
        ]
        >= historical_other
        - LEGACY_GLOBAL_F1_SELECTION_POLICY[
            "maximum_other_sensitive_mean_drop_from_high_water"
        ]
        - eps,
        "individual_other_sensitive_drop": all(
            candidate["per_class_recall"][name]
            >= incumbent["per_class_recall"][name]
            - LEGACY_GLOBAL_F1_SELECTION_POLICY[
                "maximum_individual_other_sensitive_drop"
            ]
            - eps
            for name in OTHER_SENSITIVE_CLASSES
        ),
        "precision_mean_drop": candidate["precision_mean"]
        >= incumbent["precision_mean"]
        - LEGACY_GLOBAL_F1_SELECTION_POLICY["maximum_precision_mean_drop"]
        - eps,
    }
    hard_gates_passed = all(gates.values())
    all_recalls_nondecreasing = all(
        candidate["per_class_recall"][name]
        >= incumbent["per_class_recall"][name] - eps
        for name in DETECTOR_CLASSES
    )
    reason = "rejected-no-meaningful-safe-improvement"
    selected = False
    if hard_gates_passed and secret_delta >= (
        LEGACY_GLOBAL_F1_SELECTION_POLICY["meaningful_secret_gain"] - eps
    ):
        selected = True
        reason = "selected-meaningful-secret-improvement"
    elif (
        hard_gates_passed
        and secret_delta >= -eps
        and secret_delta
        < LEGACY_GLOBAL_F1_SELECTION_POLICY["meaningful_secret_gain"] - eps
        and other_delta
        >= LEGACY_GLOBAL_F1_SELECTION_POLICY[
            "tie_band_other_sensitive_mean_gain"
        ]
        - eps
    ):
        selected = True
        reason = "selected-tie-band-other-sensitive-improvement"
    elif (
        hard_gates_passed
        and all_recalls_nondecreasing
        and precision_delta > eps
    ):
        selected = True
        reason = "selected-precision-tiebreak-without-recall-regression"
    elif not hard_gates_passed:
        reason = "rejected-regression-guard"

    new_state = state
    if selected:
        new_state = {
            "schema_version": 1,
            "incumbent": candidate,
            "high_water": {
                "other_sensitive_recall_mean": max(
                    historical_other, candidate["other_sensitive_recall_mean"]
                )
            },
        }
    return (
        {
            "selected": selected,
            "valid": True,
            "reason": reason,
            "candidate": candidate,
            "deltas": {
                "secret_recall_minimum": secret_delta,
                "other_sensitive_recall_mean": other_delta,
                "precision_mean": precision_delta,
                "other_sensitive_from_high_water": candidate[
                    "other_sensitive_recall_mean"
                ]
                - historical_other,
            },
            "gates": {
                **gates,
                "all_hard_gates": hard_gates_passed,
                "all_recalls_nondecreasing": all_recalls_nondecreasing,
            },
        },
        new_state,
    )


def require_valid_safety_decision(decision: dict[str, Any]) -> None:
    if decision.get("valid") is not True:
        raise RuntimeError(
            "safety selection validation is incomplete or non-finite: "
            + str(decision.get("reason"))
        )


def validation_uses_test_split(value: Any) -> bool:
    paths = value if isinstance(value, list) else [value]
    return any(
        part.lower() == "test" for item in paths for part in Path(str(item)).parts
    )


def normalize_class_names(value: Any, context: str) -> dict[int, str]:
    if isinstance(value, list):
        normalized = {index: name for index, name in enumerate(value)}
    elif isinstance(value, dict):
        normalized = {}
        for raw_index, name in value.items():
            try:
                index = int(raw_index)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"{context} has a non-integer class index") from exc
            if index in normalized:
                raise RuntimeError(f"{context} has a duplicate class index {index}")
            normalized[index] = name
    else:
        raise RuntimeError(f"{context} must declare detector class names")
    if sorted(normalized) != list(range(len(normalized))):
        raise RuntimeError(f"{context} class indices must be contiguous from zero")
    if any(not isinstance(name, str) or not name for name in normalized.values()):
        raise RuntimeError(f"{context} has an invalid detector class name")
    if len(set(normalized.values())) != len(normalized):
        raise RuntimeError(f"{context} has duplicate detector class names")
    return {index: str(normalized[index]) for index in sorted(normalized)}


def _split_paths(value: Any, context: str) -> list[Path]:
    values = value if isinstance(value, list) else [value]
    if not values or any(not isinstance(item, (str, Path)) for item in values):
        raise RuntimeError(f"{context} must contain one or more dataset paths")
    paths = [Path(str(item)) for item in values]
    if any(not str(path) for path in paths):
        raise RuntimeError(f"{context} contains an empty dataset path")
    return paths


def _resolve_split_path(dataset: dict[str, Any], path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    root = Path(str(dataset.get("path", ".")))
    return (root / path).resolve()


def _declared_manifest_candidates(value: Any, source_manifest_path: Path) -> list[Path]:
    if not isinstance(value, str) or not value:
        raise RuntimeError("composed source manifest path is missing")
    path = Path(value)
    if path.name != "manifest.json":
        raise RuntimeError("composed source manifest path must end in manifest.json")
    if path.is_absolute():
        return [path.resolve()]
    if ".." in path.parts:
        raise RuntimeError("composed source manifest path may not traverse parents")
    candidates = [path.resolve(), (source_manifest_path.parent / path).resolve()]
    return list(dict.fromkeys(candidates))


def _expected_split_paths(
    source: Any,
    source_manifest_path: Path,
    split: str,
    context: str,
) -> list[Path]:
    if not isinstance(source, dict):
        raise RuntimeError(f"composed manifest is missing its {context} source")
    return [
        manifest.parent / "images" / split
        for manifest in _declared_manifest_candidates(
            source.get("manifest"), source_manifest_path
        )
    ]


def _validate_replay_training_contract(
    replay_path: Path,
    source_manifest: dict[str, Any],
    source_manifest_path: Path,
) -> None:
    """Bind a replay list to its generated manifest and approved train roots."""

    contract = source_manifest.get("training_replay")
    if not isinstance(contract, dict):
        raise RuntimeError("replay-backed training YAML lacks a replay contract")
    if (
        contract.get("schema_version") != 1
        or contract.get("kind") != REPLAY_TRAINING_KIND
        or contract.get("physical_batch_guarantee") is not False
    ):
        raise RuntimeError("detector replay contract is unsupported or misleading")
    declared_list = contract.get("train_list")
    if (
        not isinstance(declared_list, str)
        or Path(declared_list).resolve() != replay_path.resolve()
        or replay_path.is_symlink()
        or not replay_path.is_file()
    ):
        raise RuntimeError("detector replay train list path is invalid")
    if contract.get("train_list_sha256") != sha256_file(replay_path):
        raise RuntimeError("detector replay train list SHA-256 mismatch")

    manifest_value = contract.get("manifest")
    if not isinstance(manifest_value, str):
        raise RuntimeError("detector replay manifest path is missing")
    manifest_path = Path(manifest_value)
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError("detector replay manifest is missing or unsafe")
    if contract.get("manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("detector replay manifest SHA-256 mismatch")
    replay = _read_json_object(manifest_path, "detector replay manifest")
    if (
        replay.get("schema_version") != 1
        or replay.get("kind") != REPLAY_TRAINING_KIND
        or replay.get("classes") != list(DETECTOR_CLASSES)
        or replay.get("configuration") != contract.get("configuration")
        or replay.get("configuration_sha256")
        != contract.get("configuration_sha256")
    ):
        raise RuntimeError("detector replay manifest contract changed")
    output = replay.get("output")
    consumer = replay.get("consumer_contract")
    if (
        not isinstance(output, dict)
        or output.get("train_list") != str(replay_path.resolve())
        or output.get("sha256") != sha256_file(replay_path)
        or output.get("records") != contract.get("records")
        or not isinstance(consumer, dict)
        or consumer.get("physical_batch_guarantee") is not False
    ):
        raise RuntimeError("detector replay output contract changed")

    runtime_selection = source_manifest.get("runtime_selection")
    selection_provenance = None
    if runtime_selection is not None:
        from training.visual.runtime_selection_evaluator import (
            load_validated_runtime_selection_contract,
        )

        if not isinstance(runtime_selection, dict):
            raise RuntimeError("detector runtime selection contract is malformed")
        selection_path_value = runtime_selection.get("contract")
        if not isinstance(selection_path_value, str):
            raise RuntimeError("detector runtime selection contract path is missing")
        selection_path = Path(selection_path_value)
        if (
            selection_path.is_symlink()
            or not selection_path.is_file()
            or runtime_selection.get("contract_sha256")
            != sha256_file(selection_path)
            or contract.get("runtime_selection_contract_sha256")
            != sha256_file(selection_path)
        ):
            raise RuntimeError("detector replay runtime selection binding changed")
        selection_provenance = load_validated_runtime_selection_contract(
            selection_path
        )
        eligible_records = len(selection_provenance["webpii"]["eligible"])
        if runtime_selection.get("webpii_train_eligible_records") != eligible_records:
            raise RuntimeError("detector replay WebPII eligibility count changed")

    sources = source_manifest.get("sources")
    if not isinstance(sources, dict):
        raise RuntimeError("detector replay source roots are missing")
    source_roots: dict[str, Path] = {}
    allowed_roots: list[Path] = []
    for source_name in ("webpii", "synthetic", "supplemental_ats"):
        source = sources.get(source_name)
        if not isinstance(source, dict):
            raise RuntimeError(f"detector replay lacks its {source_name} source")
        candidates = _declared_manifest_candidates(
            source.get("manifest"), source_manifest_path
        )
        existing = [
            path for path in candidates if not path.is_symlink() and path.is_file()
        ]
        if len(existing) != 1:
            raise RuntimeError(f"detector replay {source_name} manifest is ambiguous")
        source_root = existing[0].parent.resolve()
        source_roots[source_name] = source_root
        allowed_roots.append(source_root / "images/train")

    if selection_provenance is not None and (
        selection_provenance["webpii"]["source_root"]
        != source_roots["webpii"]
        or selection_provenance["synthetic"]["source_root"]
        != source_roots["synthetic"]
    ):
        raise RuntimeError(
            "runtime selection partitions are not bound to the composed sources"
        )

    try:
        rows = replay_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeError("detector replay train list is not UTF-8") from exc
    if len(rows) != contract.get("records") or not rows:
        raise RuntimeError("detector replay train list record count changed")
    resolved_rows: list[Path] = []
    for row in rows:
        if not row or "\0" in row:
            raise RuntimeError("detector replay train list contains an unsafe row")
        image = Path(row)
        if not image.is_absolute() or not any(
            image.resolve().is_relative_to(root.resolve()) for root in allowed_roots
        ):
            raise RuntimeError("detector replay image escapes approved train roots")
        if "test" in {part.lower() for part in image.parts}:
            raise RuntimeError("detector replay contains a held-out test image")
        resolved_rows.append(image.resolve())
    if selection_provenance is not None:
        from training.visual.runtime_selection_evaluator import (
            validate_replay_against_runtime_selection,
        )

        validate_replay_against_runtime_selection(
            resolved_rows, selection_provenance
        )


def validate_composed_dataset_contract(
    dataset: dict[str, Any],
    source_manifest: dict[str, Any],
    source_manifest_path: Path,
) -> dict[int, str]:
    """Bind selection and held-out evaluation to the composed data sources."""

    if source_manifest.get("published_splits_preserved") is not True:
        raise RuntimeError(
            "detector source manifest does not preserve WebPII published splits"
        )
    if source_manifest.get("test_used_for_checkpoint_selection") is not False:
        raise RuntimeError("WebPII test may not be used for checkpoint selection")
    runtime_selection = source_manifest.get("runtime_selection")
    expected_selection = (
        CALIBRATED_RUNTIME_SELECTION_SOURCE
        if isinstance(runtime_selection, dict)
        else SYNTHETIC_VALIDATION_SOURCE
    )
    if source_manifest.get("selection_data") != expected_selection:
        raise RuntimeError("checkpoint selection source contract changed")
    if isinstance(runtime_selection, dict) and (
        runtime_selection.get("kind")
        != "calibrated-runtime-selection-contract"
        or runtime_selection.get("published_test_used") is not False
    ):
        raise RuntimeError("calibrated runtime selection contract is unsafe")
    if source_manifest.get("test_source") != PUBLISHED_TEST_SOURCE:
        raise RuntimeError("published detector test source must be WebPII/test")
    if "train" not in dataset or "val" not in dataset:
        raise RuntimeError(
            "training YAML must include train and a separate tuning val split; "
            "prepare_webpii intentionally emits no val split"
        )
    if "test" not in dataset:
        raise RuntimeError(
            "training YAML must include the composed WebPII published test split"
        )
    if validation_uses_test_split(dataset["val"]):
        raise RuntimeError(
            "the tuning val path must not point at the published test split"
        )

    names = normalize_class_names(dataset.get("names"), "training YAML")
    manifest_names = normalize_class_names(
        source_manifest.get("classes"), "composed source manifest"
    )
    if names != manifest_names or tuple(names.values()) != DETECTOR_CLASSES:
        raise RuntimeError(
            "training YAML classes differ from the composed detector contract"
        )

    val_paths = _split_paths(dataset["val"], "training YAML val")
    test_paths = _split_paths(dataset["test"], "training YAML test")
    if len(val_paths) != 1 or len(test_paths) != 1:
        raise RuntimeError(
            "composed detector validation and test must each contain exactly one path"
        )
    actual_val = _resolve_split_path(dataset, val_paths[0])
    actual_test = _resolve_split_path(dataset, test_paths[0])
    sources = source_manifest.get("sources")
    if not isinstance(sources, dict):
        raise RuntimeError("composed detector manifest is missing source contracts")
    expected_val = _expected_split_paths(
        sources.get("synthetic"),
        source_manifest_path,
        "validation",
        "synthetic validation",
    )
    expected_test = _expected_split_paths(
        sources.get("webpii"),
        source_manifest_path,
        "test",
        "WebPII",
    )
    train_paths = _split_paths(dataset["train"], "training YAML train")
    resolved_train = [_resolve_split_path(dataset, path) for path in train_paths]
    if len(resolved_train) == 1 and resolved_train[0].suffix.lower() == ".txt":
        _validate_replay_training_contract(
            resolved_train[0], source_manifest, source_manifest_path
        )
    elif source_manifest.get("training_replay") is not None:
        raise RuntimeError("replay contract exists but training YAML bypasses it")
    if actual_val not in expected_val:
        raise RuntimeError(
            "training YAML val does not point at the composed synthetic validation split"
        )
    if actual_test not in expected_test:
        raise RuntimeError(
            "training YAML test does not point at the composed WebPII published test split"
        )
    if actual_val == actual_test:
        raise RuntimeError(
            "synthetic validation and WebPII test paths must be distinct"
        )
    return names


def _safe_supplemental_file(root: Path, value: Any, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{context} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"{context} contains an unsafe path")
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{context} is missing or is not a regular file")
    if root.resolve() not in path.resolve().parents:
        raise RuntimeError(f"{context} escapes the staged supplemental root")
    return path


def prepare_supplemental_test_yaml(
    source_manifest: dict[str, Any],
    source_manifest_path: Path,
    output_dir: Path,
    names: dict[int, str],
) -> Path | None:
    sources = source_manifest.get("sources")
    if not isinstance(sources, dict) or "supplemental_ats" not in sources:
        return None
    source = sources["supplemental_ats"]
    if not isinstance(source, dict):
        raise RuntimeError("composed supplemental source contract must be an object")
    if source.get("test_excluded_from_training_and_selection") is not True:
        raise RuntimeError(
            "supplemental test must be excluded from training and checkpoint selection"
        )
    candidates = _declared_manifest_candidates(
        source.get("manifest"), source_manifest_path
    )
    existing = [path for path in candidates if not path.is_symlink() and path.is_file()]
    if len(existing) != 1:
        raise RuntimeError(
            "staged supplemental manifest is missing or resolves ambiguously"
        )
    manifest_path = existing[0]
    expected_hash = source.get("manifest_sha256")
    if (
        not isinstance(expected_hash, str)
        or sha256_file(manifest_path) != expected_hash
    ):
        raise RuntimeError("staged supplemental manifest SHA-256 mismatch")
    try:
        supplemental = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "staged supplemental manifest is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(supplemental, dict):
        raise RuntimeError("staged supplemental manifest must be a JSON object")
    schema_version = supplemental.get("schema_version")
    if type(schema_version) is not int or schema_version < 2:
        raise RuntimeError(
            "staged supplemental manifest must use the verified v2 contract"
        )
    required_flags = {
        "supplemental_only": True,
        "training_use": "train-split-only",
        "test_used_for_checkpoint_selection": False,
        "values_persisted": False,
        "unmapped_source_labels_fail_closed": True,
    }
    if any(supplemental.get(key) != value for key, value in required_flags.items()):
        raise RuntimeError("staged supplemental manifest safety flags are incomplete")
    if supplemental.get("classes") != list(names.values()):
        raise RuntimeError(
            "staged supplemental classes differ from the detector contract"
        )
    content_hashes = supplemental.get("content_hashes")
    if not isinstance(content_hashes, dict) or any(
        source.get(key) != content_hashes.get(key)
        for key in ("images_aggregate_sha256", "labels_aggregate_sha256")
    ):
        raise RuntimeError(
            "composed supplemental content hashes differ from the staged manifest"
        )
    splits = supplemental.get("splits")
    test_split = splits.get("test") if isinstance(splits, dict) else None
    if not isinstance(test_split, dict):
        raise RuntimeError("staged supplemental manifest is missing its test split")
    if test_split.get("records") != EXPECTED_SUPPLEMENTAL_TEST_RECORDS:
        raise RuntimeError(
            "staged supplemental test split must contain exactly "
            f"{EXPECTED_SUPPLEMENTAL_TEST_RECORDS} records"
        )
    root = manifest_path.parent
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("staged supplemental root is missing or unsafe")
    records_path = _safe_supplemental_file(
        root,
        test_split.get("records_path"),
        "staged supplemental test records",
    )
    if records_path.relative_to(root).as_posix() != "records/test.jsonl":
        raise RuntimeError(
            "staged supplemental test records must be records/test.jsonl"
        )
    if test_split.get("records_sha256") != sha256_file(records_path):
        raise RuntimeError("staged supplemental test records SHA-256 mismatch")
    try:
        record_lines = records_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeError(
            "staged supplemental test records are not valid UTF-8"
        ) from exc
    if len(record_lines) != EXPECTED_SUPPLEMENTAL_TEST_RECORDS or any(
        not line for line in record_lines
    ):
        raise RuntimeError(
            "staged supplemental test records do not contain exactly "
            f"{EXPECTED_SUPPLEMENTAL_TEST_RECORDS} non-empty rows"
        )
    seen_images: set[str] = set()
    seen_labels: set[str] = set()
    for line_number, line in enumerate(record_lines, 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"staged supplemental test record {line_number} is invalid JSON"
            ) from exc
        if not isinstance(record, dict) or record.get("split") != "test":
            raise RuntimeError(
                f"staged supplemental test record {line_number} has the wrong split"
            )
        image_path = _safe_supplemental_file(
            root,
            record.get("image"),
            f"staged supplemental test record {line_number} image",
        )
        label_path = _safe_supplemental_file(
            root,
            record.get("label_file"),
            f"staged supplemental test record {line_number} label",
        )
        image_relative = image_path.relative_to(root).as_posix()
        label_relative = label_path.relative_to(root).as_posix()
        if (
            not image_relative.startswith("images/test/")
            or not label_relative.startswith("labels/test/")
            or image_relative in seen_images
            or label_relative in seen_labels
        ):
            raise RuntimeError(
                f"staged supplemental test record {line_number} has unsafe or reused files"
            )
        seen_images.add(image_relative)
        seen_labels.add(label_relative)
        if record.get("image_sha256") != sha256_file(image_path):
            raise RuntimeError(
                f"staged supplemental test record {line_number} image SHA-256 mismatch"
            )
        if record.get("label_sha256") != sha256_file(label_path):
            raise RuntimeError(
                f"staged supplemental test record {line_number} label SHA-256 mismatch"
            )
    for directory in (root / "images/test", root / "labels/test"):
        if directory.is_symlink() or not directory.is_dir():
            raise RuntimeError(
                "staged supplemental test image/label directories are missing or unsafe"
            )

    yaml_path = output_dir / "supplemental-test.yaml"
    yaml_lines = [
        f"path: {root.resolve()}",
        "train: images/train",
        "val: images/test",
        "test: images/test",
        f"nc: {len(names)}",
        "names:",
        *[f"  {index}: {name}" for index, name in names.items()],
        "",
    ]
    yaml_path.write_text("\n".join(yaml_lines), encoding="utf-8")
    return yaml_path


def evaluation_report(
    evaluation: Any,
    expected_names: dict[int, str],
    missing_key: str,
) -> dict[str, Any]:
    names = normalize_class_names(
        getattr(evaluation, "names", None), "Ultralytics evaluation"
    )
    if names != expected_names:
        raise RuntimeError(
            "Ultralytics evaluation classes differ from the composed detector contract"
        )
    box = getattr(evaluation, "box", None)
    aggregate = getattr(evaluation, "results_dict", None)
    if box is None or not isinstance(aggregate, dict):
        raise RuntimeError("Ultralytics evaluation did not return complete metrics")
    recalls, precisions, missing = align_class_metrics(box, names)
    per_class = {
        names[index]: {
            "precision": precisions[index],
            "recall": recalls[index],
        }
        for index in sorted(names)
    }
    missing_secret_classes = [
        name for name in SECRET_VISUAL_CLASSES if name not in per_class
    ]
    if missing_secret_classes:
        raise RuntimeError(
            "detector evaluation is missing secret visual classes: "
            f"{missing_secret_classes}"
        )
    return {
        "aggregate": aggregate,
        "per_class": per_class,
        "minimum_secret_class_recall": min(
            per_class[name]["recall"] for name in SECRET_VISUAL_CLASSES
        ),
        "missing_classes": missing,
        missing_key: missing,
    }


def _checkpoint_record(value: Any) -> dict[str, Any]:
    if value is None:
        return {"path": None, "exists": False}
    path = Path(str(value))
    if not path.is_file():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def persist_projected_fp32_ema_checkpoint(
    reference_model: Any,
    trainer: Any,
    checkpoint_path: Path,
    *,
    trainable_parameter_names: tuple[str, ...],
    protected_class_rows: tuple[int, ...],
    torch_module: Any,
) -> dict[str, Any]:
    """Replace Ultralytics' quantized EMA with an exact-boundary FP32 EMA.

    Ultralytics 8.4.92 serializes every training EMA as FP16. That is normally
    benign, but its roughly 6e-5 weight steps are hundreds of times larger than
    this detector's 1e-7 trust-region learning rate. Build the durable
    checkpoint from the live FP32 EMA, then copy every non-trainable tensor from
    the immutable invocation reference, including protected class rows inside
    otherwise trainable tensors. The resulting checkpoint is both resumable and
    safe to evaluate: only approved unprotected class-output rows can differ,
    and frozen EMA arithmetic drift cannot leak through serialization.
    """

    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise RuntimeError("Ultralytics checkpoint is missing or unsafe")
    approved = set(trainable_parameter_names)
    if approved != set(CLASSIFIER_TENSOR_NAMES):
        raise RuntimeError("FP32 checkpoint projection requires exact trainable names")
    if (
        tuple(protected_class_rows) != PROTECTED_CLASSIFIER_ROWS
        or not protected_class_rows
        or len(set(protected_class_rows)) != len(protected_class_rows)
        or any(
            type(row) is not int or row < 0 or row >= len(DETECTOR_CLASSES)
            for row in protected_class_rows
        )
    ):
        raise RuntimeError("FP32 checkpoint projection protected rows are invalid")
    reference_state_fn = getattr(reference_model, "state_dict", None)
    ema_holder = getattr(trainer, "ema", None)
    live_ema = getattr(ema_holder, "ema", None)
    if not callable(reference_state_fn) or live_ema is None:
        raise RuntimeError("FP32 checkpoint projection has no reference or live EMA")
    live_state_fn = getattr(live_ema, "state_dict", None)
    if not callable(live_state_fn):
        raise RuntimeError("FP32 checkpoint projection live EMA has no state_dict")

    projected_ema = deepcopy(live_ema).float().cpu()
    reference_state = reference_state_fn()
    projected_state = projected_ema.state_dict()
    if set(reference_state) != set(projected_state):
        raise RuntimeError("FP32 checkpoint projection state inventory changed")
    if not approved <= set(projected_state):
        raise RuntimeError("FP32 checkpoint projection trainable inventory changed")

    projected_frozen_tensors = 0
    projected_frozen_elements = 0
    projected_protected_rows = 0
    with torch_module.no_grad():
        for name in sorted(projected_state):
            expected = reference_state[name].detach().cpu()
            actual = projected_state[name]
            if tuple(expected.shape) != tuple(actual.shape):
                raise RuntimeError(
                    f"FP32 checkpoint projection shape changed for {name}"
                )
            if name in approved:
                if not actual.shape or int(actual.shape[0]) != len(DETECTOR_CLASSES):
                    raise RuntimeError(
                        f"FP32 checkpoint projection class rows changed for {name}"
                    )
                for row in protected_class_rows:
                    actual[row].copy_(expected[row])
                    projected_protected_rows += 1
                continue
            if str(expected.dtype) != str(actual.dtype):
                raise RuntimeError(
                    f"FP32 checkpoint projection dtype changed for {name}"
                )
            actual.copy_(expected)
            projected_frozen_tensors += 1
            projected_frozen_elements += int(expected.numel())

    floating_tensors = [
        tensor
        for tensor in projected_state.values()
        if bool(getattr(tensor, "is_floating_point")())
    ]
    if not floating_tensors or any(
        str(tensor.dtype) != "torch.float32" for tensor in floating_tensors
    ):
        raise RuntimeError("projected checkpoint EMA is not fully FP32")

    original_sha256 = sha256_file(checkpoint_path)
    try:
        checkpoint = torch_module.load(
            str(checkpoint_path), map_location="cpu", weights_only=False
        )
    except Exception as exc:
        raise RuntimeError("Ultralytics checkpoint could not be loaded") from exc
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("epoch") != int(trainer.epoch)
        or checkpoint.get("optimizer") is None
        or checkpoint.get("scaler") is None
    ):
        raise RuntimeError("Ultralytics checkpoint is not safely resumable")
    checkpoint["ema"] = projected_ema
    temporary = checkpoint_path.with_name(
        f".{checkpoint_path.name}.fp32-partial-{os.getpid()}"
    )
    temporary.unlink(missing_ok=True)
    try:
        torch_module.save(checkpoint, str(temporary))
        if not temporary.is_file():
            raise RuntimeError("projected FP32 checkpoint was not written")
        temporary.replace(checkpoint_path)
    finally:
        temporary.unlink(missing_ok=True)

    try:
        persisted = torch_module.load(
            str(checkpoint_path), map_location="cpu", weights_only=False
        )
    except Exception as exc:
        raise RuntimeError("projected FP32 checkpoint could not be reloaded") from exc
    persisted_ema = persisted.get("ema") if isinstance(persisted, dict) else None
    persisted_state_fn = getattr(persisted_ema, "state_dict", None)
    if not callable(persisted_state_fn):
        raise RuntimeError("projected FP32 checkpoint has no EMA state")
    persisted_state = persisted_state_fn()
    if set(persisted_state) != set(reference_state):
        raise RuntimeError("projected FP32 checkpoint state inventory changed on disk")

    changed_trainable: list[str] = []
    changed_frozen: list[str] = []
    protected_row_mismatches: list[dict[str, Any]] = []
    non_fp32_floating: list[str] = []
    for name in sorted(reference_state):
        expected = reference_state[name].detach().cpu()
        actual = persisted_state[name].detach().cpu()
        same = (
            tuple(expected.shape) == tuple(actual.shape)
            and str(expected.dtype) == str(actual.dtype)
            and bool(expected.equal(actual))
        )
        if name in approved:
            if not same:
                changed_trainable.append(name)
            for row in protected_class_rows:
                expected_row = expected[row]
                actual_row = actual[row]
                if (
                    tuple(expected_row.shape) != tuple(actual_row.shape)
                    or str(expected_row.dtype) != str(actual_row.dtype)
                    or not bool(expected_row.equal(actual_row))
                ):
                    protected_row_mismatches.append(
                        {
                            "tensor": name,
                            "row": row,
                            "class": DETECTOR_CLASSES[row],
                        }
                    )
        elif not same:
            changed_frozen.append(name)
        if bool(getattr(actual, "is_floating_point")()) and str(actual.dtype) != (
            "torch.float32"
        ):
            non_fp32_floating.append(name)
    violations: list[str] = []
    if changed_frozen:
        violations.append("projected checkpoint changed frozen tensors")
    if non_fp32_floating:
        violations.append("projected checkpoint contains non-FP32 floating tensors")
    if protected_row_mismatches:
        violations.append("projected checkpoint changed protected classifier rows")
    audit = {
        "schema_version": 1,
        "kind": "projected-full-precision-ema-checkpoint",
        "checkpoint": str(checkpoint_path),
        "original_ultralytics_fp16_sha256": original_sha256,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "approved_trainable_parameters": sorted(approved),
        "changed_trainable_tensors": changed_trainable,
        "trainable_change_observed": bool(changed_trainable),
        "changed_frozen_tensors": changed_frozen,
        "protected_classes": [
            DETECTOR_CLASSES[row] for row in protected_class_rows
        ],
        "protected_class_rows": list(protected_class_rows),
        "protected_row_mismatches": protected_row_mismatches,
        "non_fp32_floating_tensors": non_fp32_floating,
        "projected_frozen_tensors": projected_frozen_tensors,
        "projected_frozen_elements": projected_frozen_elements,
        "projected_protected_rows": projected_protected_rows,
        "floating_tensors": len(floating_tensors),
        "violations": violations,
        "passed": not violations,
    }
    if violations:
        raise RuntimeError("FP32 checkpoint projection failed: " + "; ".join(violations))
    return audit


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_json_object(path: Path, context: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{context} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{context} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{context} must contain a JSON object")
    return value


def _write_json_object_atomic(path: Path, value: dict[str, Any]) -> None:
    """Write one JSON object without exposing a partially written state file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=float) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_selected_runtime_profile(
    output_dir: Path, candidate: dict[str, Any]
) -> Path:
    normalized = validate_runtime_selection_candidate(candidate)
    path = output_dir / SAFETY_THRESHOLDS_NAME
    _write_json_object_atomic(
        path,
        {
            "schema_version": 1,
            "metric_basis": RUNTIME_SELECTION_POLICY_NAME,
            "checkpoint_sha256": normalized["checkpoint_sha256"],
            "threshold_profile": normalized["threshold_profile"],
            "threshold_profile_sha256": normalized[
                "threshold_profile_sha256"
            ],
            "calibration_contract_sha256": normalized[
                "calibration_contract_sha256"
            ],
            "selection_inventory_contract_sha256": normalized[
                "selection_inventory_contract_sha256"
            ],
            "calibration_evidence": normalized["calibration_evidence"],
        },
    )
    return path


def build_training_resume_contract(
    args: argparse.Namespace, names: dict[int, str]
) -> dict[str, Any]:
    """Fingerprint the immutable inputs that a resumed checkpoint may consume."""

    staging_path_value = getattr(args, "local_staging_manifest", None)
    staging_path = Path(staging_path_value) if staging_path_value else None
    staging_sha256 = None
    source_contract_sha256 = None
    if staging_path is not None:
        staging = _read_json_object(
            staging_path, "local detector staging manifest"
        )
        if staging.get("schema_version") != 1:
            raise RuntimeError("local detector staging manifest schema is unsupported")
        dataset_hash = sha256_file(args.dataset_yaml)
        source_hash = sha256_file(args.source_manifest)
        if staging.get("local_dataset_yaml_sha256") != dataset_hash:
            raise RuntimeError(
                "local staging manifest dataset YAML hash does not match training"
            )
        if staging.get("local_source_manifest_sha256") != source_hash:
            raise RuntimeError(
                "local staging manifest source hash does not match training"
            )
        source_contract_sha256 = staging.get("source_contract_sha256")
        if not isinstance(source_contract_sha256, str) or len(
            source_contract_sha256
        ) != 64:
            raise RuntimeError("local staging source-contract hash is invalid")
        staging_sha256 = sha256_file(staging_path)
    source_document = _read_json_object(
        args.source_manifest, "detector source manifest"
    )
    runtime_selection_path_value = getattr(
        args, "runtime_selection_contract", None
    )
    if runtime_selection_path_value is None:
        raise RuntimeError("calibrated runtime selection contract is required")
    runtime_selection_path = Path(runtime_selection_path_value)
    runtime_selection_source = source_document.get("runtime_selection")
    if (
        runtime_selection_path.is_symlink()
        or not runtime_selection_path.is_file()
        or not isinstance(runtime_selection_source, dict)
        or runtime_selection_source.get("published_test_used") is not False
        or runtime_selection_source.get("contract_sha256")
        != sha256_file(runtime_selection_path)
    ):
        raise RuntimeError("calibrated runtime selection contract is invalid")
    runtime_selection_contract = {
        "path": str(runtime_selection_path.resolve()),
        "sha256": sha256_file(runtime_selection_path),
        "policy_name": RUNTIME_SELECTION_POLICY_NAME,
        "policy_sha256": _canonical_json_sha256(RUNTIME_SELECTION_POLICY),
    }
    replay = source_document.get("training_replay")
    replay_contract = None
    if replay is not None:
        if not isinstance(replay, dict):
            raise RuntimeError("detector replay contract must be an object")
        replay_manifest = Path(str(replay.get("manifest", "")))
        replay_list = Path(str(replay.get("train_list", "")))
        if (
            replay_manifest.is_symlink()
            or replay_list.is_symlink()
            or not replay_manifest.is_file()
            or not replay_list.is_file()
        ):
            raise RuntimeError("detector replay artifacts are missing or unsafe")
        replay_contract = {
            "manifest_sha256": sha256_file(replay_manifest),
            "train_list_sha256": sha256_file(replay_list),
            "configuration_sha256": replay.get("configuration_sha256"),
            "records": replay.get("records"),
        }
    return {
        "run_id": getattr(args, "run_id", None),
        "dataset_yaml_sha256": sha256_file(args.dataset_yaml),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "staging_manifest_sha256": staging_sha256,
        "source_contract_sha256": source_contract_sha256,
        "class_contract_sha256": _canonical_json_sha256(list(names.values())),
        "augmentation_policy_sha256": _canonical_json_sha256(
            SCREENSHOT_AUGMENTATION_POLICY
        ),
        "optimizer_policy_sha256": _canonical_json_sha256(
            STABLE_OPTIMIZER_POLICY
        ),
        "safe_fine_tune_policy_sha256": _canonical_json_sha256(
            SAFE_FINE_TUNE_POLICY
        ),
        "runtime_selection_contract": runtime_selection_contract,
        "replay_contract": replay_contract,
        "attempt_id": getattr(args, "attempt_id", "default"),
        "attempt_manifest_sha256": getattr(args, "attempt_manifest_sha256", None),
        "seed_checkpoint_sha256": sha256_file(args.base_checkpoint),
        "output_dir": str(args.output_dir.resolve()),
        "local_stage_root": str(args.dataset_yaml.resolve().parent),
        "total_epochs": int(args.epochs),
    }


def validate_resume_data_contract(
    args: argparse.Namespace, contract: dict[str, Any]
) -> None:
    """Fail closed unless run identity and all staged data hashes still match."""

    run_id = contract.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise RuntimeError("visual resume requires the original run ID")
    output_dir = args.output_dir.resolve()
    expected_name = (
        "training"
        if contract.get("attempt_id") == "default"
        else f"training-{contract.get('attempt_id')}"
    )
    if (
        contract.get("output_dir") != str(output_dir)
        or output_dir.parent.name != "visual"
        or output_dir.parent.parent.name != run_id
        or output_dir.name != expected_name
    ):
        raise RuntimeError("visual resume output directory does not match the run ID")
    for path, context in (
        (args.dataset_yaml, "dataset YAML"),
        (args.source_manifest, "source manifest"),
        (getattr(args, "local_staging_manifest", None), "staging manifest"),
    ):
        if (
            path is None
            or str(Path(path).resolve().parent) != contract.get("local_stage_root")
        ):
            raise RuntimeError(f"visual resume {context} does not match the run ID")
    runtime_selection = contract.get("runtime_selection_contract")
    runtime_selection_path = Path(
        str(getattr(args, "runtime_selection_contract", ""))
    )
    if (
        not isinstance(runtime_selection, dict)
        or runtime_selection_path.is_symlink()
        or not runtime_selection_path.is_file()
        or runtime_selection.get("path") != str(runtime_selection_path.resolve())
        or runtime_selection.get("sha256") != sha256_file(runtime_selection_path)
        or runtime_selection.get("policy_name") != RUNTIME_SELECTION_POLICY_NAME
        or runtime_selection.get("policy_sha256")
        != _canonical_json_sha256(RUNTIME_SELECTION_POLICY)
    ):
        raise RuntimeError("visual resume runtime selection contract changed")

    staging = getattr(args, "data_staging", None)
    if not isinstance(staging, dict):
        raise RuntimeError("visual resume requires persisted staging metadata")
    expected = {
        "dataset_yaml": (
            args.output_dir / "staged-dataset.yaml",
            contract["dataset_yaml_sha256"],
        ),
        "source_manifest": (
            args.output_dir / "staged-source-manifest.json",
            contract["source_manifest_sha256"],
        ),
        "staging_manifest": (
            args.output_dir / "dataset-staging-manifest.json",
            contract["staging_manifest_sha256"],
        ),
        "runtime_selection_contract": (
            args.output_dir / "staged-runtime-selection-contract.json",
            runtime_selection["sha256"],
        ),
    }
    replay_contract = contract.get("replay_contract")
    if replay_contract is not None:
        if not isinstance(replay_contract, dict):
            raise RuntimeError("visual resume replay contract is invalid")
        expected.update(
            {
                "replay_manifest": (
                    args.output_dir / "staged-replay-manifest.json",
                    replay_contract.get("manifest_sha256"),
                ),
                "replay_train_list": (
                    args.output_dir / "staged-replay-train-list.txt",
                    replay_contract.get("train_list_sha256"),
                ),
            }
        )
    for name, (path, current_hash) in expected.items():
        declared_path = staging.get(f"{name}_path")
        declared_hash = staging.get(f"{name}_sha256")
        if (
            not isinstance(declared_path, str)
            or Path(declared_path).resolve() != path.resolve()
            or path.is_symlink()
            or not path.is_file()
        ):
            raise RuntimeError(f"visual resume persisted {name} path is invalid")
        actual_hash = sha256_file(path)
        if (
            not isinstance(current_hash, str)
            or declared_hash != actual_hash
            or actual_hash != current_hash
        ):
            raise RuntimeError(f"visual resume persisted {name} hash mismatch")


def _validated_progress_checkpoint(
    progress: dict[str, Any], name: str, expected_path: Path
) -> Path:
    checkpoints = progress.get("checkpoints")
    checkpoint = checkpoints.get(name) if isinstance(checkpoints, dict) else None
    if not isinstance(checkpoint, dict) or checkpoint.get("exists") is not True:
        raise RuntimeError(f"resume progress is missing the {name} checkpoint")
    declared_path = checkpoint.get("path")
    if (
        not isinstance(declared_path, str)
        or Path(declared_path).resolve() != expected_path.resolve()
        or expected_path.is_symlink()
        or not expected_path.is_file()
    ):
        raise RuntimeError(f"resume progress {name} checkpoint path is invalid")
    if checkpoint.get("bytes") != expected_path.stat().st_size:
        raise RuntimeError(f"resume progress {name} checkpoint size mismatch")
    actual_hash = sha256_file(expected_path)
    if checkpoint.get("sha256") != actual_hash:
        raise RuntimeError(f"resume progress {name} checkpoint SHA-256 mismatch")
    return expected_path


def _restore_selection_history(
    progress: dict[str, Any]
) -> tuple[
    list[dict[str, Any]], tuple[float, ...], dict[str, Any], dict[str, Any]
]:
    progress_epoch = progress.get("epoch")
    history = progress.get("history")
    if type(progress_epoch) is not int or progress_epoch < 0:
        raise RuntimeError("resume progress epoch is invalid")
    if not isinstance(history, list) or len(history) != progress_epoch + 1:
        raise RuntimeError("resume selection history is incomplete")
    seed_incumbent = progress.get("seed_incumbent")
    if not isinstance(seed_incumbent, dict):
        raise RuntimeError("resume progress seed incumbent is missing")
    seed_decision, state = consider_runtime_selection_candidate(
        None, seed_incumbent
    )
    if not seed_decision["selected"] or state is None:
        raise RuntimeError("resume progress seed incumbent is invalid")
    restored: list[dict[str, Any]] = []
    for expected_epoch, record in enumerate(history):
        if not isinstance(record, dict) or record.get("epoch") != expected_epoch:
            raise RuntimeError("resume selection history epochs are not contiguous")
        runtime_metrics = record.get("runtime_selection_metrics")
        decision, next_state = consider_runtime_selection_candidate(
            state, runtime_metrics
        )
        if (
            record.get("selected") is not decision["selected"]
            or record.get("selection_reason") != decision["reason"]
            or record.get("selection_deltas") != decision["deltas"]
            or record.get("selection_gates") != decision["gates"]
        ):
            raise RuntimeError("resume selection history has inconsistent selection")
        candidate = decision["candidate"]
        if candidate is None or runtime_metrics != candidate:
            raise RuntimeError("resume runtime selection metrics were altered")
        if next_state is not None:
            state = next_state
        restored.append(dict(record))
    if state is None:
        raise RuntimeError("resume selection state is missing")
    incumbent = state["incumbent"]
    running_best = runtime_selection_score(incumbent)
    declared_best = progress.get("best_score")
    if declared_best != list(running_best):
        raise RuntimeError("resume selection history best score is inconsistent")
    if progress.get("selection_state") != state:
        raise RuntimeError("resume selection state does not match replayed history")
    return restored, running_best, state, seed_incumbent


def load_visual_resume_state(
    args: argparse.Namespace,
    contract: dict[str, Any],
    torch_module: Any,
    *,
    allow_completed_schedule: bool = False,
) -> dict[str, Any]:
    """Validate and load exactly the durable last.pt for an interrupted run."""

    validate_resume_data_contract(args, contract)
    progress_path = args.output_dir / SAFETY_PROGRESS_NAME
    progress = _read_json_object(progress_path, "safety selection progress")
    schema_version = progress.get("schema_version")
    if schema_version != 4:
        raise RuntimeError(
            "legacy visual progress cannot resume under calibrated runtime "
            "selection; start a named frozen-seed attempt"
        )
    if progress.get("resume_contract") != contract:
        raise RuntimeError("resume progress data or run contract has changed")
    persisted_history = progress.get("history")
    first_record = (
        persisted_history[0]
        if isinstance(persisted_history, list) and persisted_history
        else None
    )
    first_canary = (
        first_record.get("canary_gate") if isinstance(first_record, dict) else None
    )
    if isinstance(first_canary, dict) and first_canary.get("passed") is not True:
        raise RuntimeError("a failed first-epoch visual canary cannot be resumed")
    persisted_seed = _read_json_object(
        args.output_dir / SEED_INCUMBENT_NAME, "frozen seed incumbent"
    )
    if persisted_seed != progress.get("seed_incumbent"):
        raise RuntimeError("frozen seed incumbent differs from resume progress")
    if (
        persisted_seed.get("checkpoint_sha256")
        != contract.get("seed_checkpoint_sha256")
        or not isinstance(persisted_seed.get("checkpoint"), str)
        or Path(persisted_seed["checkpoint"]).resolve()
        != args.base_checkpoint.resolve()
    ):
        raise RuntimeError("frozen seed incumbent does not match the immutable seed")

    weights = args.output_dir / ULTRALYTICS_RUN_NAME / "weights"
    if progress.get("status") == PRETRAIN_BUDGET_PAUSE_STATUS:
        if (
            progress.get("epoch") != -1
            or progress.get("history") != []
            or progress.get("resume_allowed") is not True
        ):
            raise RuntimeError("pre-training budget pause state is malformed")
        seed_decision, selection_state = consider_runtime_selection_candidate(
            None, persisted_seed
        )
        if not seed_decision["selected"] or selection_state is None:
            raise RuntimeError("pre-training budget pause seed state is invalid")
        incumbent = selection_state["incumbent"]
        best_key = runtime_selection_score(incumbent)
        if (
            progress.get("selection_state") != selection_state
            or progress.get("best_score") != list(best_key)
        ):
            raise RuntimeError("pre-training budget pause selection state changed")
        _validated_progress_checkpoint(
            progress, "safety_best", args.output_dir / "safety-best.pt"
        )
        _validated_progress_checkpoint(
            progress,
            "runtime_profile",
            args.output_dir / SAFETY_THRESHOLDS_NAME,
        )
        return {
            "last_path": None,
            "history": [],
            "best_key": best_key,
            "selection_state": selection_state,
            "seed_incumbent": persisted_seed,
            "progress": progress,
            "progress_sha256": sha256_file(progress_path),
            "restart_before_training": True,
            "training_complete": False,
        }

    progress_epoch = progress.get("epoch")
    if type(progress_epoch) is not int or progress_epoch < 0:
        raise RuntimeError("resume progress epoch is invalid")
    next_epoch = progress_epoch + 1
    if next_epoch > args.epochs:
        raise RuntimeError("resume progress exceeds the configured training schedule")

    # The safety-selected checkpoint is the only model artifact needed to
    # finish evaluation and write the manifest. Ultralytics final_eval() may
    # already have stripped or partially rewritten last.pt/best.pt before a
    # final validation failure, so completed-schedule recovery must not trust
    # either of those mutable framework checkpoints.
    _validated_progress_checkpoint(
        progress, "safety_best", args.output_dir / "safety-best.pt"
    )
    _validated_progress_checkpoint(
        progress,
        "runtime_profile",
        args.output_dir / SAFETY_THRESHOLDS_NAME,
    )
    history, best_key, selection_state, seed_incumbent = (
        _restore_selection_history(progress)
    )
    persisted_profile = _read_json_object(
        args.output_dir / SAFETY_THRESHOLDS_NAME,
        "selected runtime threshold profile",
    )
    incumbent = selection_state["incumbent"]
    if (
        persisted_profile.get("checkpoint_sha256")
        != incumbent["checkpoint_sha256"]
        or persisted_profile.get("threshold_profile")
        != incumbent["threshold_profile"]
        or persisted_profile.get("threshold_profile_sha256")
        != incumbent["threshold_profile_sha256"]
    ):
        raise RuntimeError("selected runtime threshold profile differs from history")
    if next_epoch == args.epochs:
        if not allow_completed_schedule:
            raise RuntimeError("the persisted visual training schedule is already complete")
        return {
            "last_path": None,
            "history": history,
            "best_key": best_key,
            "selection_state": selection_state,
            "seed_incumbent": seed_incumbent,
            "progress": progress,
            "progress_sha256": sha256_file(progress_path),
            "restart_before_training": False,
            "training_complete": True,
        }

    last_path = _validated_progress_checkpoint(
        progress, "last", weights / "last.pt"
    )
    _validated_progress_checkpoint(progress, "best", weights / "best.pt")
    try:
        checkpoint = torch_module.load(
            str(last_path), map_location="cpu", weights_only=False
        )
    except Exception as exc:
        raise RuntimeError("persisted last.pt could not be loaded for resume") from exc
    if not isinstance(checkpoint, dict):
        raise RuntimeError("persisted last.pt is not an Ultralytics checkpoint")
    if checkpoint.get("epoch") != progress.get("epoch"):
        raise RuntimeError("last.pt epoch differs from safety selection progress")
    if checkpoint.get("optimizer") is None:
        raise RuntimeError("last.pt lacks optimizer state required for resume")
    if checkpoint.get("scaler") is None or checkpoint.get("ema") is None:
        raise RuntimeError("last.pt lacks scaler or EMA state required for resume")
    train_args = checkpoint.get("train_args")
    if not isinstance(train_args, dict):
        raise RuntimeError("last.pt lacks original Ultralytics training arguments")
    if train_args.get("epochs") != args.epochs:
        raise RuntimeError("resume total epochs differ from the original schedule")
    checkpoint_data = train_args.get("data")
    if (
        not isinstance(checkpoint_data, str)
        or Path(checkpoint_data).resolve() != args.dataset_yaml.resolve()
    ):
        raise RuntimeError("last.pt dataset path differs from the resumed dataset")
    expected_save_dir = args.output_dir / ULTRALYTICS_RUN_NAME
    checkpoint_save_dir = train_args.get("save_dir")
    if (
        not isinstance(checkpoint_save_dir, str)
        or Path(checkpoint_save_dir).resolve() != expected_save_dir.resolve()
    ):
        raise RuntimeError("last.pt output directory differs from the resumed run")
    # DataLoader worker count is intentionally absent: Ultralytics explicitly
    # permits it as a resume-time resource override. Batch size and every model
    # optimization/data contract remain immutable below.
    for key, expected in (
        ("imgsz", args.image_size),
        ("batch", args.batch_size),
        ("seed", args.seed),
        ("freeze", SAFE_FINE_TUNE_POLICY["trainable_layer"]),
        *STABLE_OPTIMIZER_POLICY.items(),
        *SCREENSHOT_AUGMENTATION_POLICY.items(),
    ):
        if train_args.get(key) != expected:
            raise RuntimeError(f"last.pt training argument {key} changed on resume")
    return {
        "last_path": last_path,
        "history": history,
        "best_key": best_key,
        "selection_state": selection_state,
        "seed_incumbent": seed_incumbent,
        "progress": progress,
        "progress_sha256": sha256_file(progress_path),
        "restart_before_training": False,
        "training_complete": False,
    }


def write_safety_selection_progress(
    output_dir: Path,
    trainer: Any,
    history: list[dict[str, Any]],
    best_key: tuple[float, ...] | None,
    safety_best_path: Path,
    resume_contract: dict[str, Any],
    seed_incumbent: dict[str, Any],
    selection_state: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Atomically record every checkpoint needed to survive preemption."""

    document = {
        "schema_version": 4,
        "epoch": int(trainer.epoch),
        "selection_policy": SAFETY_SELECTION_POLICY,
        "best_score": list(best_key) if best_key is not None else None,
        "seed_incumbent": seed_incumbent,
        "selection_state": selection_state,
        "history": history,
        "resume_contract": resume_contract,
        "checkpoints": {
            "last": _checkpoint_record(getattr(trainer, "last", None)),
            "best": _checkpoint_record(getattr(trainer, "best", None)),
            "safety_best": _checkpoint_record(safety_best_path),
            "runtime_profile": _checkpoint_record(
                output_dir / SAFETY_THRESHOLDS_NAME
            ),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / SAFETY_PROGRESS_NAME
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, default=float) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path, document


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.base_license in {"", "NOASSERTION"}:
        raise RuntimeError(
            "replacement detector base checkpoint requires an explicit license"
        )
    commercial_license_approved = bool(
        getattr(args, "commercial_license_approved", False)
    )
    checkpoint_commit_hook = getattr(args, "checkpoint_commit_hook", None)
    if checkpoint_commit_hook is not None and not callable(checkpoint_commit_hook):
        raise TypeError("checkpoint_commit_hook must be callable")
    resume_requested = bool(getattr(args, "resume", False))
    if resume_requested and getattr(args, "local_staging_manifest", None) is None:
        args.local_staging_manifest = (
            args.dataset_yaml.parent / "staging_manifest.json"
        )
    if resume_requested and not isinstance(getattr(args, "data_staging", None), dict):
        persisted = {
            "dataset_yaml": args.output_dir / "staged-dataset.yaml",
            "source_manifest": args.output_dir / "staged-source-manifest.json",
            "staging_manifest": args.output_dir / "dataset-staging-manifest.json",
            "runtime_selection_contract": (
                args.output_dir / "staged-runtime-selection-contract.json"
            ),
        }
        args.data_staging = {
            f"{name}_{field}": (
                str(path) if field == "path" else sha256_file(path)
            )
            for name, path in persisted.items()
            for field in ("path", "sha256")
        }
    agpl_mode = args.base_license.startswith("AGPL-3.0")
    import torch
    import yaml
    from ultralytics import YOLO

    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    if not isinstance(source_manifest, dict):
        raise RuntimeError("detector source manifest must be a JSON object")
    if resume_requested and source_manifest.get("training_replay") is not None:
        replay_persisted = {
            "replay_manifest": args.output_dir / "staged-replay-manifest.json",
            "replay_train_list": args.output_dir / "staged-replay-train-list.txt",
        }
        for name, path in replay_persisted.items():
            if f"{name}_path" not in args.data_staging:
                args.data_staging[f"{name}_path"] = str(path)
                args.data_staging[f"{name}_sha256"] = sha256_file(path)
    dataset = yaml.safe_load(args.dataset_yaml.read_text(encoding="utf-8"))
    if not isinstance(dataset, dict):
        raise RuntimeError("training YAML must contain a dataset mapping")
    expected_names = validate_composed_dataset_contract(
        dataset, source_manifest, args.source_manifest
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    invocation_state_path = args.output_dir / INVOCATION_STATE_NAME
    if resume_requested and invocation_state_path.exists():
        invocation_state = _read_json_object(
            invocation_state_path, "visual training invocation state"
        )
        if invocation_state.get("status") == "canary-failed":
            raise RuntimeError("a failed visual canary cannot be resumed")
    resume_contract = build_training_resume_contract(args, expected_names)
    resume_state = (
        load_visual_resume_state(
            args,
            resume_contract,
            torch,
            allow_completed_schedule=True,
        )
        if resume_requested
        else None
    )
    finalize_only_recovery = bool(
        resume_state is not None and resume_state.get("training_complete") is True
    )
    restart_before_training = bool(
        resume_state is not None
        and resume_state.get("restart_before_training") is True
    )
    if finalize_only_recovery and (args.output_dir / "training_manifest.json").exists():
        raise RuntimeError("completed visual training already has a training manifest")
    framework_resume_requested = bool(
        resume_requested
        and not finalize_only_recovery
        and not restart_before_training
    )
    supplemental_test_yaml = prepare_supplemental_test_yaml(
        source_manifest,
        args.source_manifest,
        args.output_dir,
        expected_names,
    )
    best_path = args.output_dir / "safety-best.pt"
    runtime_selection_evaluator = getattr(
        args, "runtime_selection_evaluator", None
    )
    if not finalize_only_recovery and runtime_selection_evaluator is None:
        from training.visual.runtime_selection_evaluator import (
            CalibratedRuntimeSelectionEvaluator,
        )

        runtime_selection_evaluator = CalibratedRuntimeSelectionEvaluator(
            Path(args.runtime_selection_contract),
            device=str(args.device),
            batch=max(1, min(int(args.batch_size), 32)),
            workers=int(args.workers),
            output_dir=args.output_dir / "runtime-selection-evaluation",
        )
    evaluated_seed = (
        runtime_selection_evaluator.initialize_seed(args.base_checkpoint)
        if runtime_selection_evaluator is not None
        else None
    )
    if resume_state is not None:
        history: list[dict[str, Any]] = list(resume_state["history"])
        best_key: tuple[float, ...] = resume_state["best_key"]
        selection_state: dict[str, Any] = resume_state["selection_state"]
        seed_incumbent: dict[str, Any] = resume_state["seed_incumbent"]
        if evaluated_seed is not None and (
            validate_runtime_selection_candidate(seed_incumbent)
            != evaluated_seed
        ):
            raise RuntimeError(
                "recomputed frozen runtime seed differs from persisted selection state"
            )
    else:
        if evaluated_seed is None:
            raise RuntimeError("frozen runtime seed evaluation was not initialized")
        seed_decision, initialized_state = consider_runtime_selection_candidate(
            None, evaluated_seed
        )
        if not seed_decision["selected"] or initialized_state is None:
            raise RuntimeError(
                "frozen seed cannot initialize safety selection: "
                + str(seed_decision["reason"])
            )
        selection_state = initialized_state
        seed_incumbent = {
            **seed_decision["candidate"],
            "checkpoint": str(args.base_checkpoint),
            "checkpoint_sha256": sha256_file(args.base_checkpoint),
        }
        incumbent = selection_state["incumbent"]
        best_key = runtime_selection_score(incumbent)
        history = []
        shutil.copy2(args.base_checkpoint, best_path)
        if sha256_file(best_path) != seed_incumbent["checkpoint_sha256"]:
            raise RuntimeError("frozen safety incumbent copy failed verification")
        write_selected_runtime_profile(args.output_dir, seed_incumbent)
        (args.output_dir / SEED_INCUMBENT_NAME).write_text(
            json.dumps(seed_incumbent, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if checkpoint_commit_hook is not None:
            checkpoint_commit_hook()
    deadline = getattr(args, "deadline_monotonic", None)
    if deadline is not None:
        if (
            isinstance(deadline, bool)
            or not isinstance(deadline, (int, float))
            or not math.isfinite(float(deadline))
        ):
            raise RuntimeError("visual training deadline must be a finite number")
        deadline = float(deadline)
    if not finalize_only_recovery and deadline is not None:
        budget_check_time = time.monotonic()
        seconds_remaining = deadline - budget_check_time
        if seconds_remaining <= 0:
            progress_path = args.output_dir / SAFETY_PROGRESS_NAME
            if framework_resume_requested:
                if resume_state is None:
                    raise RuntimeError("pre-training resume state is missing")
                progress = resume_state["progress"]
            else:
                progress = {
                    "schema_version": 4,
                    "status": PRETRAIN_BUDGET_PAUSE_STATUS,
                    "epoch": -1,
                    "selection_policy": SAFETY_SELECTION_POLICY,
                    "best_score": list(best_key),
                    "seed_incumbent": seed_incumbent,
                    "selection_state": selection_state,
                    "history": [],
                    "resume_contract": resume_contract,
                    "resume_allowed": True,
                    "checkpoints": {
                        "last": _checkpoint_record(
                            args.output_dir
                            / ULTRALYTICS_RUN_NAME
                            / "weights/last.pt"
                        ),
                        "best": _checkpoint_record(
                            args.output_dir
                            / ULTRALYTICS_RUN_NAME
                            / "weights/best.pt"
                        ),
                        "safety_best": _checkpoint_record(best_path),
                        "runtime_profile": _checkpoint_record(
                            args.output_dir / SAFETY_THRESHOLDS_NAME
                        ),
                    },
                }
                _write_json_object_atomic(progress_path, progress)
            pause_details = {
                "epoch": progress.get("epoch"),
                "resume_allowed": True,
                "seconds_remaining": seconds_remaining,
                "deadline_monotonic": deadline,
                "checked_before_framework_training": True,
            }
            pause_manifest = {
                "schema_version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": PRETRAIN_BUDGET_PAUSE_STATUS,
                "reason": "budget-deadline-before-training",
                "details": pause_details,
                "resume_allowed": True,
                "attempt_id": getattr(args, "attempt_id", "default"),
                "total_epochs": args.epochs,
                "committed_epochs": len(history),
                "committed_epochs_this_invocation": 0,
                "last_checkpoint": progress.get("checkpoints", {}).get("last"),
                "safety_best_checkpoint": _checkpoint_record(best_path),
                "progress": {
                    "path": str(progress_path),
                    "sha256": sha256_file(progress_path),
                },
                "resume_contract": resume_contract,
            }
            _write_json_object_atomic(invocation_state_path, pause_manifest)
            if checkpoint_commit_hook is not None:
                checkpoint_commit_hook()
            return pause_manifest
    from training.visual.safe_finetune import (
        validate_frozen_state_parity,
        validate_seed_state_parity,
        wire_safe_finetune_callbacks,
    )

    model = None
    safe_finetune = None
    reference_model = None
    pre_step_parity_audit: dict[str, Any] | None = (
        history[-1].get("seed_state_parity_audit")
        if finalize_only_recovery and history
        else None
    )
    if not finalize_only_recovery:
        training_checkpoint = (
            resume_state["last_path"]
            if framework_resume_requested and resume_state is not None
            else args.base_checkpoint
        )
        reference_model = YOLO(str(training_checkpoint)).model
        model = YOLO(str(training_checkpoint))
        safe_finetune = wire_safe_finetune_callbacks(model)

    def on_pretrain_routine_end(trainer) -> None:
        nonlocal pre_step_parity_audit
        if reference_model is None:
            raise RuntimeError("training reference model is missing")
        pre_step_parity_audit = validate_seed_state_parity(
            reference_model, trainer.model
        )
        trainer.seed_state_parity_audit = pre_step_parity_audit

    if model is not None:
        model.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)
    checkpoint_commit_count = len(history)
    invocation_commit_count = 0
    epoch_durations: list[float] = []
    previous_epoch_commit = time.monotonic()

    def on_model_save(trainer) -> None:
        nonlocal best_key, selection_state
        nonlocal checkpoint_commit_count, invocation_commit_count
        nonlocal previous_epoch_commit
        if safe_finetune is None:
            raise RuntimeError("safe fine-tune controller is missing")
        if runtime_selection_evaluator is None:
            raise RuntimeError("calibrated runtime selector is missing")
        metrics = getattr(getattr(trainer, "validator", None), "metrics", None)
        box = getattr(metrics, "box", None)
        raw_names = getattr(getattr(trainer, "model", None), "names", None)
        if box is None or raw_names is None:
            if checkpoint_commit_hook is not None:
                raise RuntimeError(
                    "cannot durably commit an epoch without complete selection metrics"
                )
            return
        names = normalize_class_names(raw_names, "Ultralytics training model")
        if names != expected_names:
            raise RuntimeError(
                "Ultralytics training classes differ from the composed detector contract"
            )
        recalls, precisions, missing_validation_classes = align_class_metrics(
            box, names
        )
        recall_by_class = {names[index]: recalls[index] for index in names}
        precision_by_class = {names[index]: precisions[index] for index in names}
        raw_aggregate = getattr(trainer, "metrics", None)
        if not isinstance(raw_aggregate, dict):
            raw_aggregate = getattr(metrics, "results_dict", None)
        if not isinstance(raw_aggregate, dict):
            raise RuntimeError("training validation returned no aggregate metrics")
        aggregate_metrics: dict[str, float] = {}
        for key, value in raw_aggregate.items():
            try:
                aggregate_metrics[str(key)] = _finite_float(
                    value, f"training validation metric {key}"
                )
            except RuntimeError:
                # Reject a missing/non-finite mAP below, but do not let a
                # framework-added non-scalar diagnostic poison the report.
                continue
        map50 = _finite_float(
            aggregate_metrics.get("metrics/mAP50(B)"),
            "training validation mAP50",
        )
        checkpoint = Path(str(getattr(trainer, "last", "")))
        if checkpoint.is_symlink() or not checkpoint.is_file():
            raise RuntimeError("current epoch checkpoint is missing or unsafe")
        if reference_model is None:
            raise RuntimeError("training reference model is missing")
        current_safe_audit = getattr(
            trainer, "safe_finetune_audit", safe_finetune.last_report
        )
        trainable_parameter_names = (
            current_safe_audit.get("trainable_parameters")
            if isinstance(current_safe_audit, dict)
            else None
        )
        if not isinstance(trainable_parameter_names, list) or not all(
            isinstance(name, str) and name for name in trainable_parameter_names
        ):
            raise RuntimeError("safe fine-tune audit has no exact trainable parameters")
        checkpoint_precision_audit = persist_projected_fp32_ema_checkpoint(
            reference_model,
            trainer,
            checkpoint,
            trainable_parameter_names=tuple(trainable_parameter_names),
            protected_class_rows=PROTECTED_CLASSIFIER_ROWS,
            torch_module=torch,
        )
        checkpoint_sha256 = sha256_file(checkpoint)
        if checkpoint_precision_audit.get("checkpoint_sha256") != checkpoint_sha256:
            raise RuntimeError("projected FP32 checkpoint hash changed")
        frozen_state_parity_audit = validate_frozen_state_parity(
            reference_model,
            YOLO(str(checkpoint)).model,
            trainable_parameter_names=tuple(trainable_parameter_names),
        )
        runtime_selection_metrics = runtime_selection_evaluator.evaluate_candidate(
            checkpoint, epoch=int(trainer.epoch)
        )
        if (
            runtime_selection_metrics.get("checkpoint_sha256")
            != checkpoint_sha256
        ):
            raise RuntimeError("runtime audit checkpoint hash changed")
        decision, candidate_state = consider_runtime_selection_candidate(
            selection_state, runtime_selection_metrics
        )
        require_valid_safety_decision(decision)
        candidate = decision["candidate"]
        record = {
            "epoch": int(trainer.epoch),
            "metric_basis": RUNTIME_SELECTION_POLICY_NAME,
            "runtime_selection_metrics": candidate,
            # Ultralytics chooses one moving global max-F1 threshold. Preserve
            # those values for diagnosis only; they cannot select a checkpoint.
            "framework_global_f1_diagnostic": {
                "per_class_recall": recall_by_class,
                "per_class_precision": precision_by_class,
                "map50": map50,
                "aggregate_metrics": aggregate_metrics,
            },
            "missing_validation_classes": missing_validation_classes,
            "selected": decision["selected"],
            "selection_reason": decision["reason"],
            "selection_deltas": decision["deltas"],
            "selection_gates": decision["gates"],
            "seed_state_parity_audit": pre_step_parity_audit,
            "safe_finetune_audit": current_safe_audit,
            "checkpoint_precision_audit": checkpoint_precision_audit,
            "frozen_state_parity_audit": frozen_state_parity_audit,
        }
        if (
            pre_step_parity_audit is None
            or pre_step_parity_audit.get("passed") is not True
        ):
            raise RuntimeError("pre-step immutable checkpoint parity was not proven")
        safe_audit = record["safe_finetune_audit"]
        if not isinstance(safe_audit, dict) or safe_audit.get("passed") is not True:
            raise RuntimeError("safe fine-tune controls were not proven during training")
        if frozen_state_parity_audit.get("passed") is not True:
            raise RuntimeError("class-output-only frozen-state parity was not proven")
        if checkpoint_precision_audit.get("passed") is not True:
            raise RuntimeError("projected FP32 checkpoint parity was not proven")
        enforce_canary = bool(getattr(args, "enforce_canary", True))
        if (
            int(trainer.epoch) == 0
            and not framework_resume_requested
            and enforce_canary
        ):
            canary = evaluate_first_epoch_canary(seed_incumbent, record)
            canary["gates"]["immutable_seed_pre_step_parity"] = True
            canary["gates"]["head_only_frozen_batchnorm"] = True
            canary["gates"]["class_output_only_frozen_state_parity"] = True
            canary["gates"]["projected_fp32_checkpoint_parity"] = True
            canary["gates"]["protected_classifier_row_parity"] = not bool(
                checkpoint_precision_audit.get("protected_row_mismatches")
            )
            gradient_mask_audit = safe_audit.get(
                "protected_gradient_mask_audit"
            )
            canary["gates"]["protected_classifier_gradient_mask"] = bool(
                isinstance(gradient_mask_audit, dict)
                and gradient_mask_audit.get("passed") is True
                and gradient_mask_audit.get("protected_classes")
                == list(PROTECTED_CLASSIFIER_CLASSES)
            )
            canary["gates"]["approved_classifier_change_observed"] = bool(
                frozen_state_parity_audit.get("trainable_change_observed")
            )
            canary["passed"] = all(canary["gates"].values())
            record["canary_gate"] = canary
            if canary["passed"] is not True:
                # The canary independently recomputes every operational guard.
                # Never let a failed epoch replace the immutable safety seed.
                record["selected"] = False
                record["selection_reason"] = "rejected-first-epoch-canary"
                candidate_state = selection_state
                sentinel = {
                    "schema_version": 1,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "status": "canary-failed",
                    "epoch": int(trainer.epoch),
                    "resume_allowed": False,
                    "canary_gate": canary,
                }
                sentinel_temporary = invocation_state_path.with_name(
                    f".{invocation_state_path.name}.partial-{os.getpid()}"
                )
                sentinel_temporary.write_text(
                    json.dumps(sentinel, indent=2, sort_keys=True, default=float)
                    + "\n",
                    encoding="utf-8",
                )
                sentinel_temporary.replace(invocation_state_path)
        else:
            record["canary_gate"] = None
        history.append(record)
        if record["selected"]:
            if candidate_state is None:
                raise RuntimeError("selected checkpoint produced no safety state")
            selection_state = candidate_state
            incumbent = selection_state["incumbent"]
            best_key = runtime_selection_score(incumbent)
            shutil.copy2(checkpoint, best_path)
            if sha256_file(best_path) != checkpoint_sha256:
                raise RuntimeError("selected runtime checkpoint copy failed verification")
            write_selected_runtime_profile(args.output_dir, incumbent)
        progress_path, progress = write_safety_selection_progress(
            args.output_dir,
            trainer,
            history,
            best_key,
            best_path,
            resume_contract,
            seed_incumbent,
            selection_state,
        )
        if checkpoint_commit_hook is not None:
            missing = [
                name
                for name, checkpoint in progress["checkpoints"].items()
                if checkpoint.get("exists") is not True
            ]
            if missing:
                raise RuntimeError(
                    "cannot commit incomplete epoch checkpoints: " + ", ".join(missing)
                )
            output_root = args.output_dir.resolve()
            outside_output = [
                name
                for name, checkpoint in progress["checkpoints"].items()
                if output_root not in Path(checkpoint["path"]).resolve().parents
            ]
            if outside_output:
                raise RuntimeError(
                    "cannot commit checkpoints outside the durable output root: "
                    + ", ".join(outside_output)
                )
            if not progress_path.is_file():
                raise RuntimeError("safety selection progress was not persisted")
            checkpoint_commit_hook()
            checkpoint_commit_count += 1
            invocation_commit_count += 1

        now = time.monotonic()
        epoch_durations.append(now - previous_epoch_commit)
        previous_epoch_commit = now
        canary_gate = record.get("canary_gate")
        if isinstance(canary_gate, dict) and canary_gate.get("passed") is not True:
            raise PlannedTrainingPause(
                "canary-failed",
                {
                    "epoch": int(trainer.epoch),
                    "canary_gate": canary_gate,
                    "resume_allowed": False,
                },
            )
        deadline = getattr(args, "deadline_monotonic", None)
        if (
            isinstance(deadline, (int, float))
            and int(trainer.epoch) + 1 < args.epochs
        ):
            estimate = 1.25 * statistics.median(epoch_durations[-3:])
            finalization_reserve = 15 * 60
            remaining = float(deadline) - now
            if remaining < estimate + finalization_reserve:
                raise PlannedTrainingPause(
                    "budget-deadline",
                    {
                        "epoch": int(trainer.epoch),
                        "resume_allowed": True,
                        "seconds_remaining": remaining,
                        "estimated_next_epoch_seconds": estimate,
                        "finalization_reserve_seconds": finalization_reserve,
                    },
                )

    result = None
    if not finalize_only_recovery:
        if model is None or safe_finetune is None:
            raise RuntimeError("visual training model controls are missing")
        model.add_callback("on_model_save", on_model_save)
        try:
            result = model.train(
                data=str(args.dataset_yaml),
                epochs=args.epochs,
                imgsz=args.image_size,
                batch=args.batch_size,
                device=args.device,
                seed=args.seed,
                deterministic=True,
                project=str(args.output_dir),
                name=ULTRALYTICS_RUN_NAME,
                exist_ok=False,
                workers=args.workers,
                patience=args.patience,
                save=True,
                save_period=1,
                resume=framework_resume_requested,
                plots=False,
                verbose=True,
                **safe_finetune.training_overrides(),
                **STABLE_OPTIMIZER_POLICY,
                **SCREENSHOT_AUGMENTATION_POLICY,
            )
        except PlannedTrainingPause as pause:
            last_path = args.output_dir / ULTRALYTICS_RUN_NAME / "weights/last.pt"
            progress_path = args.output_dir / SAFETY_PROGRESS_NAME
            if last_path.is_symlink() or not last_path.is_file():
                raise RuntimeError("planned pause has no durable last.pt") from pause
            checkpoint = torch.load(
                str(last_path), map_location="cpu", weights_only=False
            )
            if (
                not isinstance(checkpoint, dict)
                or checkpoint.get("epoch") != pause.details.get("epoch")
                or checkpoint.get("optimizer") is None
                or checkpoint.get("scaler") is None
                or checkpoint.get("ema") is None
            ):
                raise RuntimeError(
                    "planned pause checkpoint is not safely resumable"
                ) from pause
            progress = _read_json_object(
                progress_path, "planned pause safety progress"
            )
            progress_last = progress.get("checkpoints", {}).get("last", {})
            if progress_last.get("sha256") != sha256_file(last_path):
                raise RuntimeError(
                    "planned pause checkpoint differs from committed progress"
                ) from pause
            status = (
                "canary-failed"
                if pause.reason == "canary-failed"
                else "budget-paused"
            )
            pause_manifest = {
                "schema_version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": status,
                "reason": pause.reason,
                "details": pause.details,
                "resume_allowed": pause.details.get("resume_allowed") is True,
                "attempt_id": getattr(args, "attempt_id", "default"),
                "total_epochs": args.epochs,
                "committed_epochs": len(history),
                "committed_epochs_this_invocation": invocation_commit_count,
                "last_checkpoint": _checkpoint_record(last_path),
                "safety_best_checkpoint": _checkpoint_record(best_path),
                "progress": {
                    "path": str(progress_path),
                    "sha256": sha256_file(progress_path),
                },
                "resume_contract": resume_contract,
            }
            _write_json_object_atomic(invocation_state_path, pause_manifest)
            if checkpoint_commit_hook is not None:
                checkpoint_commit_hook()
            return pause_manifest
    # Ultralytics finalization strips optimizer state from last.pt/best.pt
    # after the final on_model_save callback. Refresh their hashes so durable
    # progress remains truthful, while explicitly marking the finished
    # schedule non-resumable.
    completed_progress_path = args.output_dir / SAFETY_PROGRESS_NAME
    completed_progress = _read_json_object(
        completed_progress_path, "completed safety selection progress"
    )
    completed_progress["checkpoints"] = {
        "last": _checkpoint_record(
            args.output_dir / ULTRALYTICS_RUN_NAME / "weights/last.pt"
        ),
        "best": _checkpoint_record(
            args.output_dir / ULTRALYTICS_RUN_NAME / "weights/best.pt"
        ),
        "safety_best": _checkpoint_record(best_path),
        "runtime_profile": _checkpoint_record(
            args.output_dir / SAFETY_THRESHOLDS_NAME
        ),
    }
    completed_progress["training_completed"] = True
    completed_progress["resume_allowed"] = False
    completed_progress_temporary = completed_progress_path.with_name(
        f".{completed_progress_path.name}.partial-{os.getpid()}"
    )
    completed_progress_temporary.write_text(
        json.dumps(completed_progress, indent=2, sort_keys=True, default=float)
        + "\n",
        encoding="utf-8",
    )
    completed_progress_temporary.replace(completed_progress_path)
    if not best_path.exists():
        raise RuntimeError("training completed without a safety-selected checkpoint")
    selected_validation = YOLO(str(best_path)).val(
        data=str(args.dataset_yaml),
        split="val",
        imgsz=args.image_size,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        plots=False,
        verbose=False,
        project=str(args.output_dir),
        name="safety-best-validation",
        exist_ok=resume_requested,
    )
    selected_evaluation = evaluation_report(
        selected_validation,
        expected_names,
        "missing_validation_classes",
    )
    # The published WebPII and supplemental ATS tests are release gates only.
    # Freeze checkpoint selection completely before running either seed baseline.
    baseline_published_test = YOLO(str(args.base_checkpoint)).val(
        data=str(args.dataset_yaml),
        split="test",
        imgsz=args.image_size,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        plots=False,
        verbose=False,
        project=str(args.output_dir),
        name="frozen-base-webpii-test",
        exist_ok=resume_requested,
    )
    baseline_published_test_evaluation = evaluation_report(
        baseline_published_test,
        expected_names,
        "missing_test_classes",
    )
    baseline_supplemental_test_evaluation = None
    if supplemental_test_yaml is not None:
        baseline_supplemental_test = YOLO(str(args.base_checkpoint)).val(
            data=str(supplemental_test_yaml),
            split="test",
            imgsz=args.image_size,
            batch=args.batch_size,
            device=args.device,
            workers=args.workers,
            plots=False,
            verbose=False,
            project=str(args.output_dir),
            name="frozen-base-supplemental-test",
            exist_ok=resume_requested,
        )
        baseline_supplemental_test_evaluation = evaluation_report(
            baseline_supplemental_test,
            expected_names,
            "missing_test_classes",
        )
    published_test = YOLO(str(best_path)).val(
        data=str(args.dataset_yaml),
        split="test",
        imgsz=args.image_size,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        plots=False,
        verbose=False,
        project=str(args.output_dir),
        name="safety-best-webpii-test",
        exist_ok=resume_requested,
    )
    published_test_evaluation = evaluation_report(
        published_test,
        expected_names,
        "missing_test_classes",
    )
    supplemental_test_evaluation = None
    if supplemental_test_yaml is not None:
        supplemental_test = YOLO(str(best_path)).val(
            data=str(supplemental_test_yaml),
            split="test",
            imgsz=args.image_size,
            batch=args.batch_size,
            device=args.device,
            workers=args.workers,
            plots=False,
            verbose=False,
            project=str(args.output_dir),
            name="safety-best-supplemental-test",
            exist_ok=resume_requested,
        )
        supplemental_test_evaluation = evaluation_report(
            supplemental_test,
            expected_names,
            "missing_test_classes",
        )
    from training.visual.diagnose_detector_curves import HIGH_RECALL_THRESHOLDS
    from training.visual.diagnose_detector_runtime import (
        evaluate_checkpoint as evaluate_runtime_checkpoint,
    )
    from training.visual.diagnose_detector_runtime import resolve_dataset_images

    runtime_batch = max(1, min(int(args.batch_size), 32))
    published_test_images = resolve_dataset_images(args.dataset_yaml, "test")
    baseline_published_runtime, _ = evaluate_runtime_checkpoint(
        args.base_checkpoint,
        published_test_images,
        device=str(args.device),
        batch=runtime_batch,
        threshold_profiles={"high-recall": dict(HIGH_RECALL_THRESHOLDS)},
    )
    selected_profile = selection_state["incumbent"]["threshold_profile"]
    selected_published_runtime, _ = evaluate_runtime_checkpoint(
        best_path,
        published_test_images,
        device=str(args.device),
        batch=runtime_batch,
        threshold_profiles={"high-recall": selected_profile},
    )
    published_runtime_release_gate = evaluate_runtime_release_gate(
        baseline_published_runtime, selected_published_runtime
    )
    baseline_supplemental_runtime = None
    selected_supplemental_runtime = None
    supplemental_runtime_release_gate = None
    if supplemental_test_yaml is not None:
        supplemental_test_images = resolve_dataset_images(
            supplemental_test_yaml, "test"
        )
        baseline_supplemental_runtime, _ = evaluate_runtime_checkpoint(
            args.base_checkpoint,
            supplemental_test_images,
            device=str(args.device),
            batch=runtime_batch,
            threshold_profiles={"high-recall": dict(HIGH_RECALL_THRESHOLDS)},
        )
        selected_supplemental_runtime, _ = evaluate_runtime_checkpoint(
            best_path,
            supplemental_test_images,
            device=str(args.device),
            batch=runtime_batch,
            threshold_profiles={"high-recall": selected_profile},
        )
        supplemental_runtime_release_gate = evaluate_runtime_release_gate(
            baseline_supplemental_runtime, selected_supplemental_runtime
        )
    manifest = {
        "schema_version": 4,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "trained-unpublished",
        "selection_policy": SAFETY_SELECTION_POLICY,
        "seed_incumbent": seed_incumbent,
        "selection_state": selection_state,
        "base_checkpoint": {
            "path": str(args.base_checkpoint),
            "sha256": sha256_file(args.base_checkpoint),
            "license": args.base_license,
            "source": getattr(args, "base_source", None),
        },
        "data": {
            "yaml": str(args.dataset_yaml),
            "yaml_sha256": sha256_file(args.dataset_yaml),
            "source_manifest": str(args.source_manifest),
            "source_manifest_sha256": sha256_file(args.source_manifest),
            "staging": getattr(args, "data_staging", None),
            "replay_contract": resume_contract.get("replay_contract"),
            "runtime_selection_contract": resume_contract.get(
                "runtime_selection_contract"
            ),
        },
        "training_attempt": {
            "attempt_id": getattr(args, "attempt_id", "default"),
            "output_root": str(args.output_dir.resolve()),
            "attempt_manifest_sha256": getattr(
                args, "attempt_manifest_sha256", None
            ),
            "new_non_resume_seed": not resume_requested,
            "finalize_only_recovery": finalize_only_recovery,
            "restart_after_pretraining_budget_pause": restart_before_training,
        },
        "best_checkpoint": {
            "path": best_path.name,
            "sha256": sha256_file(best_path),
            "score": best_key,
            "threshold_profile": {
                "path": SAFETY_THRESHOLDS_NAME,
                "sha256": sha256_file(
                    args.output_dir / SAFETY_THRESHOLDS_NAME
                ),
                "profile_sha256": selection_state["incumbent"][
                    "threshold_profile_sha256"
                ],
            },
        },
        "history": history,
        "environment": {
            "torch": torch.__version__,
            "ultralytics": importlib.metadata.version("ultralytics"),
            "cuda_available": torch.cuda.is_available(),
            "device": args.device,
            "gpu_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "gpu_memory_bytes": (
                int(torch.cuda.get_device_properties(0).total_memory)
                if torch.cuda.is_available()
                else None
            ),
        },
        "training_config": {
            "epochs": args.epochs,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "patience": args.patience,
            "seed": args.seed,
            "resume_requested": resume_requested,
            "framework_resume_requested": framework_resume_requested,
            "finalize_only_recovery": finalize_only_recovery,
            "save": True,
            "save_period": 1,
            "optimizer_policy": {
                "name": STABLE_OPTIMIZER_POLICY_NAME,
                "ultralytics_overrides": dict(STABLE_OPTIMIZER_POLICY),
                "sha256": _canonical_json_sha256(STABLE_OPTIMIZER_POLICY),
                "auto_optimizer_disabled": True,
            },
            "safe_fine_tune_policy": {
                **SAFE_FINE_TUNE_POLICY,
                "sha256": _canonical_json_sha256(SAFE_FINE_TUNE_POLICY),
                "pre_step_parity_audit": pre_step_parity_audit,
                "callback_event_counts": (
                    safe_finetune.event_counts
                    if safe_finetune is not None
                    else None
                ),
            },
            "budget_deadline_monotonic_set": isinstance(
                getattr(args, "deadline_monotonic", None), (int, float)
            ),
            "monitoring_policy": dict(MONITORING_POLICY),
            "augmentation_policy": {
                "name": "screen-native-conservative-v1",
                "ultralytics_overrides": dict(SCREENSHOT_AUGMENTATION_POLICY),
                "disabled_ui_invalidating_transforms": [
                    "rotation",
                    "shear",
                    "perspective",
                    "horizontal-flip",
                    "vertical-flip",
                    "rgb-bgr-channel-swap",
                    "mosaic",
                    "mixup",
                    "cutmix",
                    "copy-paste",
                    "multi-scale",
                ],
                "safe_variation": {
                    "translation_fraction": 0.02,
                    "scale_gain": 0.05,
                    "saturation_gain": 0.1,
                    "value_gain": 0.1,
                    "hue_gain": 0.0,
                },
            },
        },
        "checkpoint_persistence": {
            "commit_hook_enabled": checkpoint_commit_hook is not None,
            "committed_epochs": checkpoint_commit_count,
            "committed_epochs_this_invocation": invocation_commit_count,
            "progress_path": SAFETY_PROGRESS_NAME,
            "progress_sha256": sha256_file(
                args.output_dir / SAFETY_PROGRESS_NAME
            ),
            "resume_supported": resume_contract.get("run_id") is not None,
            "resume_contract": resume_contract,
            "resumed": resume_requested,
            "resumed_from_epoch": (
                resume_state["progress"]["epoch"] if resume_state else None
            ),
            "resume_progress_sha256": (
                resume_state["progress_sha256"] if resume_state else None
            ),
            "progress_schema_version": 4,
            "seed_incumbent_path": SEED_INCUMBENT_NAME,
            "seed_incumbent_sha256": sha256_file(
                args.output_dir / SEED_INCUMBENT_NAME
            ),
        },
        "train_results": getattr(selected_validation, "results_dict", {}),
        "baseline_published_test_evaluation": {
            "dataset": PUBLISHED_TEST_SOURCE,
            "source": PUBLISHED_TEST_SOURCE,
            "split": "test",
            "timing": "post-selection-release-baseline",
            "checkpoint": str(args.base_checkpoint),
            "checkpoint_status": "frozen",
            "used_for_checkpoint_selection": False,
            **baseline_published_test_evaluation,
        },
        **(
            {
                "baseline_supplemental_test_evaluation": {
                    "dataset": "supplemental_ats/test",
                    "source": "user-provided ATS synthetic/test",
                    "split": "test",
                    "timing": "post-selection-release-baseline",
                    "checkpoint": str(args.base_checkpoint),
                    "checkpoint_status": "frozen",
                    "used_for_checkpoint_selection": False,
                    "dataset_yaml": {
                        "path": supplemental_test_yaml.name,
                        "sha256": sha256_file(supplemental_test_yaml),
                    },
                    **baseline_supplemental_test_evaluation,
                }
            }
            if baseline_supplemental_test_evaluation is not None
            and supplemental_test_yaml is not None
            else {}
        ),
        "selected_checkpoint_evaluation": {
            "dataset": "PLVA synthetic/validation",
            "source": SYNTHETIC_VALIDATION_SOURCE,
            "split": "val",
            "timing": "post-selection-framework-diagnostic",
            "used_for_checkpoint_selection": False,
            **selected_evaluation,
        },
        "runtime_checkpoint_selection_evaluation": {
            "metric_basis": RUNTIME_SELECTION_POLICY_NAME,
            "used_for_checkpoint_selection": True,
            "published_webpii_test_used": False,
            "selected": selection_state["incumbent"],
        },
        "published_test_evaluation": {
            "dataset": PUBLISHED_TEST_SOURCE,
            "source": PUBLISHED_TEST_SOURCE,
            "split": "test",
            "timing": "post-selection",
            "checkpoint": best_path.name,
            "used_for_checkpoint_selection": False,
            **published_test_evaluation,
        },
        "published_test_runtime_evaluation": {
            "metric_basis": RUNTIME_SELECTION_POLICY_NAME,
            "used_for_checkpoint_selection": False,
            "timing": "post-selection-release-gate",
            "baseline": baseline_published_runtime,
            "selected": selected_published_runtime,
            "release_gate": published_runtime_release_gate,
        },
        **(
            {
                "supplemental_test_evaluation": {
                    "dataset": "supplemental_ats/test",
                    "source": "user-provided ATS synthetic/test",
                    "split": "test",
                    "timing": "post-selection",
                    "checkpoint": best_path.name,
                    "used_for_checkpoint_selection": False,
                    "dataset_yaml": {
                        "path": supplemental_test_yaml.name,
                        "sha256": sha256_file(supplemental_test_yaml),
                    },
                    **supplemental_test_evaluation,
                }
            }
            if supplemental_test_evaluation is not None
            and supplemental_test_yaml is not None
            else {}
        ),
        **(
            {
                "supplemental_test_runtime_evaluation": {
                    "metric_basis": RUNTIME_SELECTION_POLICY_NAME,
                    "used_for_checkpoint_selection": False,
                    "timing": "post-selection-release-gate",
                    "baseline": baseline_supplemental_runtime,
                    "selected": selected_supplemental_runtime,
                    "release_gate": supplemental_runtime_release_gate,
                }
            }
            if baseline_supplemental_runtime is not None
            and selected_supplemental_runtime is not None
            else {}
        ),
        "ultralytics_default_best_results": getattr(result, "results_dict", {}),
        "distribution": {
            "mode": "agpl-development" if agpl_mode else "commercial-license",
            "commercial_license_approved": commercial_license_approved,
            "closed_source_release_allowed": commercial_license_approved,
        },
        "publication_blockers": [
            "cross-runtime ONNX parity not yet proven",
            *(
                [
                    "AGPL development output cannot ship in a closed-source product; obtain an Ultralytics Enterprise license"
                ]
                if agpl_mode and not commercial_license_approved
                else []
            ),
            "WebPII and base-checkpoint license obligations require review",
            *(
                ["published WebPII runtime privacy release gate failed"]
                if published_runtime_release_gate["passed"] is not True
                else []
            ),
            *(
                ["supplemental ATS runtime privacy release gate failed"]
                if isinstance(supplemental_runtime_release_gate, dict)
                and supplemental_runtime_release_gate.get("passed") is not True
                else []
            ),
        ],
    }
    # Ultralytics may expose NumPy or Torch scalar subclasses in results_dict.
    # Persist and return only plain JSON values so Modal clients do not need the
    # remote ML runtime merely to deserialize a completed run's metadata.
    serialized = json.dumps(manifest, indent=2, sort_keys=True, default=float)
    manifest_document = json.loads(serialized)
    (args.output_dir / "training_manifest.json").write_text(
        serialized + "\n",
        encoding="utf-8",
    )
    completion_state = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "resume_allowed": False,
        "finalize_only_recovery": finalize_only_recovery,
        "attempt_id": getattr(args, "attempt_id", "default"),
        "total_epochs": args.epochs,
        "committed_epochs": len(history),
        "training_manifest_sha256": sha256_file(
            args.output_dir / "training_manifest.json"
        ),
        "safety_best_checkpoint": _checkpoint_record(best_path),
    }
    temporary_state = invocation_state_path.with_name(
        f".{invocation_state_path.name}.partial-{os.getpid()}"
    )
    temporary_state.write_text(
        json.dumps(completion_state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_state.replace(invocation_state_path)
    return manifest_document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the optional PLVA visual detector"
    )
    parser.add_argument("--dataset-yaml", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--base-license", required=True)
    parser.add_argument("--base-source")
    parser.add_argument("--commercial-license-approved", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--local-staging-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1311)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(train(parse_args()), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
