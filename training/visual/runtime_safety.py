from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from training.visual.prepare_webpii import DETECTOR_CLASSES


SECRET_VISUAL_CLASSES = ("CARD_NUMBER", "CVC", "SECRET")
OTHER_SENSITIVE_CLASSES = tuple(
    name for name in DETECTOR_CLASSES if name not in SECRET_VISUAL_CLASSES
)
RUNTIME_SELECTION_POLICY_NAME = "runtime-top1-fixed-profile-v1"
RUNTIME_SELECTION_POLICY: dict[str, float | int | str] = {
    "schema_version": 1,
    "name": RUNTIME_SELECTION_POLICY_NAME,
    "maximum_individual_privacy_recall_drop": 0.05,
    "maximum_individual_compatible_recall_drop": 0.05,
    "maximum_individual_secret_recall_drop": 0.01,
    "maximum_overall_privacy_recall_drop": 0.01,
    "maximum_other_sensitive_mean_drop": 0.03,
    "maximum_fixed_profile_precision_mean_drop": 0.05,
    "maximum_map50_drop": 0.03,
    "maximum_individual_ap50_drop": 0.03,
    "maximum_individual_ap50_95_drop": 0.03,
    "maximum_irrecoverable_miss_increase": 0,
    "maximum_hard_negative_mask_rate_increase": 0.01,
    "maximum_predictions_per_image_multiple": 1.10,
    "meaningful_secret_privacy_gain": 0.005,
    "epsilon": 1e-12,
}


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _finite_unit(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} is not numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{context} must be finite and between zero and one")
    return normalized


def _finite_nonnegative(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} is not numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{context} must be finite and nonnegative")
    return normalized


def _nonnegative_int(value: Any, context: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{context} must be a nonnegative integer")
    return value


def _class_unit_map(value: Any, context: str) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != set(DETECTOR_CLASSES):
        raise ValueError(f"{context} must contain every detector class exactly once")
    return {
        name: _finite_unit(value[name], f"{context} {name}")
        for name in DETECTOR_CLASSES
    }


def _class_count_map(value: Any, context: str) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != set(DETECTOR_CLASSES):
        raise ValueError(f"{context} must contain every detector class exactly once")
    return {
        name: _nonnegative_int(value[name], f"{context} {name}")
        for name in DETECTOR_CLASSES
    }


def validate_runtime_candidate(value: Any) -> dict[str, Any]:
    """Normalize one compact checkpoint summary used for durable selection."""

    if not isinstance(value, dict):
        raise ValueError("runtime safety candidate must be an object")
    if value.get("metric_basis") != RUNTIME_SELECTION_POLICY_NAME:
        raise ValueError("runtime safety candidate metric basis changed")
    epoch = value.get("epoch")
    if type(epoch) is not int or epoch < -1:
        raise ValueError("runtime safety candidate epoch is invalid")
    runtime_contract_sha256 = value.get("runtime_contract_sha256")
    if (
        not isinstance(runtime_contract_sha256, str)
        or len(runtime_contract_sha256) != 64
    ):
        raise ValueError("runtime safety contract SHA-256 is invalid")
    checkpoint_sha256 = value.get("checkpoint_sha256")
    image_inventory_sha256 = value.get("image_inventory_sha256")
    runtime_geometry_sha256 = value.get("runtime_geometry_sha256")
    threshold_profile_sha256 = value.get("threshold_profile_sha256")
    for digest, context in (
        (checkpoint_sha256, "checkpoint"),
        (image_inventory_sha256, "image inventory"),
        (runtime_geometry_sha256, "runtime geometry"),
        (threshold_profile_sha256, "threshold profile"),
    ):
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"runtime safety {context} SHA-256 is invalid")

    privacy = _class_unit_map(
        value.get("per_class_privacy_coverage_recall"),
        "per-class privacy coverage recall",
    )
    compatible = _class_unit_map(
        value.get("per_class_compatible_coverage_recall"),
        "per-class compatible coverage recall",
    )
    precision = _class_unit_map(
        value.get("per_class_fixed_profile_precision"),
        "per-class fixed-profile precision",
    )
    fixed_recall = _class_unit_map(
        value.get("per_class_fixed_profile_recall"),
        "per-class fixed-profile recall",
    )
    ap50 = _class_unit_map(value.get("per_class_ap50"), "per-class AP50")
    ap50_95 = _class_unit_map(
        value.get("per_class_ap50_95"), "per-class AP50-95"
    )
    irrecoverable_by_class = _class_count_map(
        value.get("per_class_irrecoverable_misses"),
        "per-class irrecoverable misses",
    )
    support_by_class = _class_count_map(
        value.get("per_class_support"), "per-class support"
    )
    if any(count <= 0 for count in support_by_class.values()):
        raise ValueError("runtime safety selection requires positive support per class")
    hard_negative_images = _nonnegative_int(
        value.get("hard_negative_images"), "hard-negative images"
    )
    hard_negative_images_with_masks = _nonnegative_int(
        value.get("hard_negative_images_with_masks"),
        "hard-negative images with masks",
    )
    if hard_negative_images_with_masks > hard_negative_images:
        raise ValueError("masked hard-negative images exceed hard-negative images")
    truth = _nonnegative_int(value.get("truth"), "runtime truth count")
    images = _nonnegative_int(value.get("images"), "runtime image count")
    if images <= 0:
        raise ValueError("runtime image inventory is empty")
    predictions = _nonnegative_int(value.get("predictions"), "runtime prediction count")
    irrecoverable = _nonnegative_int(
        value.get("irrecoverable_misses"), "irrecoverable misses"
    )
    if irrecoverable != sum(irrecoverable_by_class.values()):
        raise ValueError("per-class irrecoverable misses do not sum to the total")
    if truth != sum(support_by_class.values()):
        raise ValueError("per-class support does not sum to runtime truth")
    preprofile_saturated_images = _nonnegative_int(
        value.get("preprofile_saturated_images"),
        "preprofile saturated images",
    )

    normalized = {
        "schema_version": 1,
        "metric_basis": RUNTIME_SELECTION_POLICY_NAME,
        "epoch": epoch,
        "runtime_contract_sha256": runtime_contract_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "image_inventory_sha256": image_inventory_sha256,
        "runtime_geometry_sha256": runtime_geometry_sha256,
        "threshold_profile_sha256": threshold_profile_sha256,
        "map50": _finite_unit(value.get("map50"), "mAP50"),
        "per_class_ap50": ap50,
        "per_class_ap50_95": ap50_95,
        "per_class_fixed_profile_precision": precision,
        "per_class_fixed_profile_recall": fixed_recall,
        "fixed_profile_precision_mean": sum(precision.values()) / len(precision),
        "per_class_privacy_coverage_recall": privacy,
        "per_class_compatible_coverage_recall": compatible,
        "privacy_coverage_recall": _finite_unit(
            value.get("privacy_coverage_recall"), "privacy coverage recall"
        ),
        "compatible_coverage_recall": _finite_unit(
            value.get("compatible_coverage_recall"),
            "compatible coverage recall",
        ),
        "secret_privacy_recall_minimum": min(
            privacy[name] for name in SECRET_VISUAL_CLASSES
        ),
        "secret_compatible_recall_minimum": min(
            compatible[name] for name in SECRET_VISUAL_CLASSES
        ),
        "other_sensitive_privacy_recall_mean": sum(
            privacy[name] for name in OTHER_SENSITIVE_CLASSES
        )
        / len(OTHER_SENSITIVE_CLASSES),
        "other_sensitive_compatible_recall_mean": sum(
            compatible[name] for name in OTHER_SENSITIVE_CLASSES
        )
        / len(OTHER_SENSITIVE_CLASSES),
        "irrecoverable_misses": irrecoverable,
        "per_class_irrecoverable_misses": irrecoverable_by_class,
        "per_class_support": support_by_class,
        "hard_negative_images": hard_negative_images,
        "hard_negative_images_with_masks": hard_negative_images_with_masks,
        "hard_negative_mask_rate": (
            hard_negative_images_with_masks / hard_negative_images
            if hard_negative_images
            else 0.0
        ),
        "truth": truth,
        "images": images,
        "predictions": predictions,
        "predictions_per_image": predictions / images,
        "preprofile_saturated_images": preprofile_saturated_images,
    }
    framework_diagnostic = value.get("framework_global_f1_diagnostic")
    if framework_diagnostic is not None:
        if not isinstance(framework_diagnostic, dict):
            raise ValueError("framework global-F1 diagnostic must be an object")
        normalized["framework_global_f1_diagnostic"] = framework_diagnostic
    return normalized


def summarize_runtime_candidate(
    *,
    epoch: int,
    map50: float,
    curve_report: dict[str, Any],
    runtime_report: dict[str, Any],
    runtime_contract: dict[str, Any],
) -> dict[str, Any]:
    """Combine threshold-independent AP with PLVA's actual masking policy."""

    curves = curve_report.get("per_class")
    if (
        runtime_report.get("runtime_contract") != runtime_contract
        or runtime_report.get("runtime_contract_sha256")
        != canonical_json_sha256(runtime_contract)
    ):
        raise ValueError("runtime checkpoint contract changed during evaluation")
    profiles = runtime_report.get("profiles")
    high_recall = profiles.get("high-recall") if isinstance(profiles, dict) else None
    if not isinstance(curves, dict) or not isinstance(high_recall, dict):
        raise ValueError("runtime checkpoint evaluation is incomplete")
    runtime_classes = high_recall.get("per_class")
    if not isinstance(runtime_classes, dict):
        raise ValueError("runtime checkpoint per-class evaluation is incomplete")

    per_class_ap50: dict[str, float] = {}
    per_class_ap50_95: dict[str, float] = {}
    per_class_precision: dict[str, float] = {}
    per_class_fixed_recall: dict[str, float] = {}
    per_class_privacy: dict[str, float] = {}
    per_class_compatible: dict[str, float] = {}
    per_class_irrecoverable: dict[str, int] = {}
    per_class_support: dict[str, int] = {}
    framework_precision: dict[str, float] = {}
    framework_recall: dict[str, float] = {}
    for name in DETECTOR_CLASSES:
        curve = curves.get(name)
        runtime = runtime_classes.get(name)
        if not isinstance(curve, dict) or not isinstance(runtime, dict):
            raise ValueError(f"runtime checkpoint evaluation is missing {name}")
        fixed = curve.get("plva_high_recall")
        framework = curve.get("framework_global_f1")
        if not isinstance(fixed, dict) or not isinstance(framework, dict):
            raise ValueError(f"runtime curve evaluation is incomplete for {name}")
        per_class_ap50[name] = curve.get("ap50")
        per_class_ap50_95[name] = curve.get("ap50_95")
        per_class_precision[name] = fixed.get("precision")
        per_class_fixed_recall[name] = fixed.get("recall")
        per_class_privacy[name] = runtime.get("coverage_recall")
        per_class_compatible[name] = runtime.get(
            "class_compatible_coverage_recall"
        )
        per_class_irrecoverable[name] = runtime.get("irrecoverable_misses")
        per_class_support[name] = runtime.get("support")
        framework_precision[name] = framework.get("precision")
        framework_recall[name] = framework.get("recall")

    runtime_contract_without_profile = dict(runtime_contract)
    runtime_contract_without_profile.pop("threshold_profiles", None)
    threshold_profiles = runtime_contract.get("threshold_profiles")
    high_recall_thresholds = (
        threshold_profiles.get("high-recall")
        if isinstance(threshold_profiles, dict)
        else None
    )
    image_inventory = runtime_report.get("image_inventory")
    if not isinstance(image_inventory, dict) or not isinstance(
        high_recall_thresholds, dict
    ):
        raise ValueError("runtime inventory or high-recall profile is missing")
    return validate_runtime_candidate(
        {
            "schema_version": 1,
            "metric_basis": RUNTIME_SELECTION_POLICY_NAME,
            "epoch": epoch,
            "runtime_contract_sha256": canonical_json_sha256(runtime_contract),
            "checkpoint_sha256": runtime_report.get("checkpoint_sha256"),
            "image_inventory_sha256": image_inventory.get(
                "ordered_absolute_paths_sha256"
            ),
            "runtime_geometry_sha256": canonical_json_sha256(
                runtime_contract_without_profile
            ),
            "threshold_profile_sha256": canonical_json_sha256(
                high_recall_thresholds
            ),
            "map50": map50,
            "per_class_ap50": per_class_ap50,
            "per_class_ap50_95": per_class_ap50_95,
            "per_class_fixed_profile_precision": per_class_precision,
            "per_class_fixed_profile_recall": per_class_fixed_recall,
            "per_class_privacy_coverage_recall": per_class_privacy,
            "per_class_compatible_coverage_recall": per_class_compatible,
            "privacy_coverage_recall": high_recall.get("coverage_recall"),
            "compatible_coverage_recall": high_recall.get(
                "class_compatible_coverage_recall"
            ),
            "irrecoverable_misses": high_recall.get("irrecoverable_misses"),
            "per_class_irrecoverable_misses": per_class_irrecoverable,
            "per_class_support": per_class_support,
            "hard_negative_images": high_recall.get("hard_negative_images"),
            "hard_negative_images_with_masks": high_recall.get(
                "hard_negative_images_with_masks"
            ),
            "truth": high_recall.get("truth"),
            "images": high_recall.get("images"),
            "predictions": high_recall.get("predictions"),
            "preprofile_saturated_images": high_recall.get(
                "preprofile_saturated_images"
            ),
            # Diagnostic only: never consumed by a gate or selection score.
            "framework_global_f1_diagnostic": {
                "threshold": curve_report.get("framework_global_f1_threshold"),
                "per_class_precision": framework_precision,
                "per_class_recall": framework_recall,
            },
        }
    )


def _gates_against(
    candidate: dict[str, Any], baseline: dict[str, Any], label: str
) -> tuple[dict[str, bool], dict[str, Any]]:
    policy = RUNTIME_SELECTION_POLICY
    eps = float(policy["epsilon"])
    privacy_drop = {
        name: baseline["per_class_privacy_coverage_recall"][name]
        - candidate["per_class_privacy_coverage_recall"][name]
        for name in DETECTOR_CLASSES
    }
    compatible_drop = {
        name: baseline["per_class_compatible_coverage_recall"][name]
        - candidate["per_class_compatible_coverage_recall"][name]
        for name in DETECTOR_CLASSES
    }
    ap50_drop = {
        name: baseline["per_class_ap50"][name]
        - candidate["per_class_ap50"][name]
        for name in DETECTOR_CLASSES
    }
    ap50_95_drop = {
        name: baseline["per_class_ap50_95"][name]
        - candidate["per_class_ap50_95"][name]
        for name in DETECTOR_CLASSES
    }
    per_class_irrecoverable_increase = {
        name: candidate["per_class_irrecoverable_misses"][name]
        - baseline["per_class_irrecoverable_misses"][name]
        for name in DETECTOR_CLASSES
    }
    overall_privacy_drop = (
        baseline["privacy_coverage_recall"]
        - candidate["privacy_coverage_recall"]
    )
    other_privacy_drop = (
        baseline["other_sensitive_privacy_recall_mean"]
        - candidate["other_sensitive_privacy_recall_mean"]
    )
    other_compatible_drop = (
        baseline["other_sensitive_compatible_recall_mean"]
        - candidate["other_sensitive_compatible_recall_mean"]
    )
    precision_drop = (
        baseline["fixed_profile_precision_mean"]
        - candidate["fixed_profile_precision_mean"]
    )
    map50_drop = baseline["map50"] - candidate["map50"]
    irrecoverable_increase = (
        candidate["irrecoverable_misses"] - baseline["irrecoverable_misses"]
    )
    hard_negative_rate_increase = (
        candidate["hard_negative_mask_rate"] - baseline["hard_negative_mask_rate"]
    )
    predictions_multiple = (
        candidate["predictions_per_image"] / baseline["predictions_per_image"]
        if baseline["predictions_per_image"] > 0
        else (1.0 if candidate["predictions_per_image"] == 0 else math.inf)
    )
    gates = {
        f"{label}_runtime_geometry_unchanged": (
            candidate["runtime_geometry_sha256"]
            == baseline["runtime_geometry_sha256"]
        ),
        f"{label}_image_inventory_unchanged": (
            candidate["image_inventory_sha256"]
            == baseline["image_inventory_sha256"]
            and candidate["images"] == baseline["images"]
            and candidate["truth"] == baseline["truth"]
            and candidate["per_class_support"] == baseline["per_class_support"]
        ),
        f"{label}_preprofile_not_saturated": (
            candidate["preprofile_saturated_images"] == 0
        ),
        f"{label}_individual_privacy_recall_drop": all(
            drop
            <= float(policy["maximum_individual_privacy_recall_drop"]) + eps
            for drop in privacy_drop.values()
        ),
        f"{label}_individual_compatible_recall_drop": all(
            drop
            <= float(policy["maximum_individual_compatible_recall_drop"]) + eps
            for drop in compatible_drop.values()
        ),
        f"{label}_individual_secret_privacy_recall_drop": all(
            privacy_drop[name]
            <= float(policy["maximum_individual_secret_recall_drop"]) + eps
            for name in SECRET_VISUAL_CLASSES
        ),
        f"{label}_individual_secret_compatible_recall_drop": all(
            compatible_drop[name]
            <= float(policy["maximum_individual_secret_recall_drop"]) + eps
            for name in SECRET_VISUAL_CLASSES
        ),
        f"{label}_overall_privacy_recall_drop": (
            overall_privacy_drop
            <= float(policy["maximum_overall_privacy_recall_drop"]) + eps
        ),
        f"{label}_other_sensitive_privacy_mean_drop": (
            other_privacy_drop
            <= float(policy["maximum_other_sensitive_mean_drop"]) + eps
        ),
        f"{label}_other_sensitive_compatible_mean_drop": (
            other_compatible_drop
            <= float(policy["maximum_other_sensitive_mean_drop"]) + eps
        ),
        f"{label}_fixed_profile_precision_mean_drop": (
            precision_drop
            <= float(policy["maximum_fixed_profile_precision_mean_drop"]) + eps
        ),
        f"{label}_map50_drop": (
            map50_drop <= float(policy["maximum_map50_drop"]) + eps
        ),
        f"{label}_individual_ap50_drop": all(
            drop <= float(policy["maximum_individual_ap50_drop"]) + eps
            for drop in ap50_drop.values()
        ),
        f"{label}_individual_ap50_95_drop": all(
            drop <= float(policy["maximum_individual_ap50_95_drop"]) + eps
            for drop in ap50_95_drop.values()
        ),
        f"{label}_irrecoverable_miss_increase": (
            irrecoverable_increase
            <= int(policy["maximum_irrecoverable_miss_increase"])
            and all(
                increase <= int(policy["maximum_irrecoverable_miss_increase"])
                for increase in per_class_irrecoverable_increase.values()
            )
        ),
        f"{label}_hard_negative_mask_rate_increase": (
            hard_negative_rate_increase
            <= float(policy["maximum_hard_negative_mask_rate_increase"]) + eps
        ),
        f"{label}_predictions_per_image_multiple": (
            predictions_multiple
            <= float(policy["maximum_predictions_per_image_multiple"]) + eps
        ),
    }
    deltas = {
        "per_class_privacy_recall_drop": privacy_drop,
        "per_class_compatible_recall_drop": compatible_drop,
        "per_class_ap50_drop": ap50_drop,
        "per_class_ap50_95_drop": ap50_95_drop,
        "per_class_irrecoverable_miss_increase": per_class_irrecoverable_increase,
        "overall_privacy_recall_drop": overall_privacy_drop,
        "other_sensitive_privacy_mean_drop": other_privacy_drop,
        "other_sensitive_compatible_mean_drop": other_compatible_drop,
        "fixed_profile_precision_mean_drop": precision_drop,
        "map50_drop": map50_drop,
        "irrecoverable_miss_increase": irrecoverable_increase,
        "hard_negative_mask_rate_increase": hard_negative_rate_increase,
        "predictions_per_image_multiple": predictions_multiple,
    }
    return gates, deltas


def _selection_score(candidate: dict[str, Any]) -> tuple[float, ...]:
    return (
        candidate["secret_privacy_recall_minimum"],
        candidate["privacy_coverage_recall"],
        candidate["secret_compatible_recall_minimum"],
        candidate["compatible_coverage_recall"],
        candidate["map50"],
        candidate["fixed_profile_precision_mean"],
    )


def consider_runtime_safety_candidate(
    state: dict[str, Any] | None,
    candidate_value: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Select a checkpoint against both the frozen reference and incumbent."""

    try:
        candidate = validate_runtime_candidate(candidate_value)
    except (TypeError, ValueError) as exc:
        return (
            {
                "selected": False,
                "valid": False,
                "reason": f"candidate-runtime-metrics-invalid: {exc}",
                "candidate": None,
                "deltas": None,
                "gates": {},
            },
            state,
        )

    if state is None:
        initialized = {
            "schema_version": 2,
            "metric_basis": RUNTIME_SELECTION_POLICY_NAME,
            "reference": candidate,
            "incumbent": candidate,
        }
        return (
            {
                "selected": True,
                "valid": True,
                "reason": "initial-frozen-runtime-reference",
                "candidate": candidate,
                "deltas": None,
                "gates": {
                    "complete_finite_runtime_metrics": True,
                    "all_hard_gates": True,
                },
            },
            initialized,
        )

    if (
        not isinstance(state, dict)
        or state.get("schema_version") != 2
        or state.get("metric_basis") != RUNTIME_SELECTION_POLICY_NAME
    ):
        raise RuntimeError("runtime safety selection state is malformed")
    reference = validate_runtime_candidate(state.get("reference"))
    incumbent = validate_runtime_candidate(state.get("incumbent"))
    reference_gates, reference_deltas = _gates_against(
        candidate, reference, "reference"
    )
    incumbent_gates, incumbent_deltas = _gates_against(
        candidate, incumbent, "incumbent"
    )
    gates = {**reference_gates, **incumbent_gates}
    hard_gates_passed = all(gates.values())

    eps = float(RUNTIME_SELECTION_POLICY["epsilon"])
    secret_delta = (
        candidate["secret_privacy_recall_minimum"]
        - incumbent["secret_privacy_recall_minimum"]
    )
    privacy_delta = (
        candidate["privacy_coverage_recall"]
        - incumbent["privacy_coverage_recall"]
    )
    compatible_delta = (
        candidate["compatible_coverage_recall"]
        - incumbent["compatible_coverage_recall"]
    )
    runtime_nondecreasing = all(
        candidate["per_class_privacy_coverage_recall"][name]
        >= incumbent["per_class_privacy_coverage_recall"][name] - eps
        and candidate["per_class_compatible_coverage_recall"][name]
        >= incumbent["per_class_compatible_coverage_recall"][name] - eps
        for name in DETECTOR_CLASSES
    )
    selected = False
    reason = "rejected-no-safe-runtime-improvement"
    if hard_gates_passed and secret_delta >= (
        float(RUNTIME_SELECTION_POLICY["meaningful_secret_privacy_gain"]) - eps
    ):
        selected = True
        reason = "selected-meaningful-secret-privacy-gain"
    elif hard_gates_passed and secret_delta >= -eps and privacy_delta > eps:
        selected = True
        reason = "selected-net-privacy-coverage-gain"
    elif (
        hard_gates_passed
        and secret_delta >= -eps
        and privacy_delta >= -eps
        and compatible_delta > eps
    ):
        selected = True
        reason = "selected-compatible-coverage-gain"
    elif (
        hard_gates_passed
        and runtime_nondecreasing
        and _selection_score(candidate) > _selection_score(incumbent)
    ):
        selected = True
        reason = "selected-runtime-safe-quality-tiebreak"
    elif not hard_gates_passed:
        reason = "rejected-runtime-regression-guard"

    next_state = state
    if selected:
        next_state = {
            "schema_version": 2,
            "metric_basis": RUNTIME_SELECTION_POLICY_NAME,
            "reference": reference,
            "incumbent": candidate,
        }
    return (
        {
            "selected": selected,
            "valid": True,
            "reason": reason,
            "candidate": candidate,
            "deltas": {
                "from_reference": reference_deltas,
                "from_incumbent": incumbent_deltas,
                "secret_privacy_recall_minimum": secret_delta,
                "privacy_coverage_recall": privacy_delta,
                "compatible_coverage_recall": compatible_delta,
            },
            "gates": {
                **gates,
                "all_hard_gates": hard_gates_passed,
                "all_runtime_recalls_nondecreasing": runtime_nondecreasing,
            },
        },
        next_state,
    )


def selection_score(candidate_value: dict[str, Any]) -> tuple[float, ...]:
    return _selection_score(validate_runtime_candidate(candidate_value))


def evaluate_first_epoch_runtime_canary(
    seed_value: dict[str, Any], record: dict[str, Any]
) -> dict[str, Any]:
    """Independently recompute the epoch-one hard gates against the seed."""

    seed = validate_runtime_candidate(seed_value)
    candidate = validate_runtime_candidate(record.get("runtime_safety_metrics"))
    gates, deltas = _gates_against(candidate, seed, "seed")
    declared_hard_gate = (
        record.get("selection_gates", {}).get("all_hard_gates") is True
    )
    gates["checkpoint_selection_runtime_guards"] = declared_hard_gate
    return {
        "schema_version": 2,
        "metric_basis": RUNTIME_SELECTION_POLICY_NAME,
        "passed": all(gates.values()),
        "gates": gates,
        "deltas_from_seed": deltas,
    }


REQUIRED_SELECTION_SLICES = ("synthetic-guard", "webpii-guard")


def validate_runtime_selection_candidate(value: Any) -> dict[str, Any]:
    """Validate a calibrated checkpoint across both non-test guard domains."""

    if not isinstance(value, dict):
        raise ValueError("runtime selection candidate must be an object")
    if value.get("metric_basis") != RUNTIME_SELECTION_POLICY_NAME:
        raise ValueError("runtime selection candidate metric basis changed")
    epoch = value.get("epoch")
    if type(epoch) is not int or epoch < -1:
        raise ValueError("runtime selection candidate epoch is invalid")
    checkpoint_sha256 = value.get("checkpoint_sha256")
    calibration_contract_sha256 = value.get("calibration_contract_sha256")
    selection_inventory_contract_sha256 = value.get(
        "selection_inventory_contract_sha256"
    )
    threshold_profile_sha256 = value.get("threshold_profile_sha256")
    for digest, context in (
        (checkpoint_sha256, "checkpoint"),
        (calibration_contract_sha256, "calibration contract"),
        (selection_inventory_contract_sha256, "selection inventory contract"),
        (threshold_profile_sha256, "threshold profile"),
    ):
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"runtime selection {context} SHA-256 is invalid")
    threshold_profile = _class_unit_map(
        value.get("threshold_profile"), "runtime threshold profile"
    )
    if canonical_json_sha256(threshold_profile) != threshold_profile_sha256:
        raise ValueError("runtime threshold profile hash mismatch")
    if value.get("calibration_passed") is not True:
        raise ValueError("runtime threshold calibration did not pass")
    calibration_evidence = value.get("calibration_evidence")
    if (
        not isinstance(calibration_evidence, dict)
        or calibration_evidence.get("passed") is not True
        or calibration_evidence.get("profile_sha256")
        != threshold_profile_sha256
        or not isinstance(calibration_evidence.get("kind"), str)
        or not calibration_evidence["kind"]
    ):
        raise ValueError("runtime threshold calibration evidence is incomplete")
    raw_slices = value.get("slices")
    if not isinstance(raw_slices, dict) or set(raw_slices) != set(
        REQUIRED_SELECTION_SLICES
    ):
        raise ValueError("runtime selection guard slices are incomplete")
    slices = {
        name: validate_runtime_candidate(raw_slices[name])
        for name in REQUIRED_SELECTION_SLICES
    }
    for name, summary in slices.items():
        if (
            summary["epoch"] != epoch
            or summary["checkpoint_sha256"] != checkpoint_sha256
            or summary["threshold_profile_sha256"] != threshold_profile_sha256
        ):
            raise ValueError(f"runtime selection {name} binding changed")

    return {
        "schema_version": 1,
        "metric_basis": RUNTIME_SELECTION_POLICY_NAME,
        "epoch": epoch,
        "checkpoint_sha256": checkpoint_sha256,
        "calibration_contract_sha256": calibration_contract_sha256,
        "selection_inventory_contract_sha256": (
            selection_inventory_contract_sha256
        ),
        "threshold_profile_sha256": threshold_profile_sha256,
        "threshold_profile": threshold_profile,
        "calibration_passed": True,
        "calibration_evidence": calibration_evidence,
        "slices": slices,
        "secret_privacy_recall_minimum": min(
            summary["secret_privacy_recall_minimum"]
            for summary in slices.values()
        ),
        "privacy_coverage_recall_mean": sum(
            summary["privacy_coverage_recall"] for summary in slices.values()
        )
        / len(slices),
        "secret_compatible_recall_minimum": min(
            summary["secret_compatible_recall_minimum"]
            for summary in slices.values()
        ),
        "compatible_coverage_recall_mean": sum(
            summary["compatible_coverage_recall"] for summary in slices.values()
        )
        / len(slices),
        "map50_mean": sum(summary["map50"] for summary in slices.values())
        / len(slices),
        "predictions_per_image_mean": sum(
            summary["predictions_per_image"] for summary in slices.values()
        )
        / len(slices),
    }


def _combined_gates_against(
    candidate: dict[str, Any], baseline: dict[str, Any], label: str
) -> tuple[dict[str, bool], dict[str, Any]]:
    gates = {
        f"{label}_calibration_contract_unchanged": (
            candidate["calibration_contract_sha256"]
            == baseline["calibration_contract_sha256"]
        ),
        f"{label}_selection_inventory_contract_unchanged": (
            candidate["selection_inventory_contract_sha256"]
            == baseline["selection_inventory_contract_sha256"]
        ),
    }
    deltas: dict[str, Any] = {"slices": {}}
    for slice_name in REQUIRED_SELECTION_SLICES:
        slice_label = f"{label}_{slice_name.replace('-', '_')}"
        slice_gates, slice_deltas = _gates_against(
            candidate["slices"][slice_name],
            baseline["slices"][slice_name],
            slice_label,
        )
        gates.update(slice_gates)
        deltas["slices"][slice_name] = slice_deltas
    deltas.update(
        {
            "secret_privacy_recall_minimum": (
                candidate["secret_privacy_recall_minimum"]
                - baseline["secret_privacy_recall_minimum"]
            ),
            "privacy_coverage_recall_mean": (
                candidate["privacy_coverage_recall_mean"]
                - baseline["privacy_coverage_recall_mean"]
            ),
            "secret_compatible_recall_minimum": (
                candidate["secret_compatible_recall_minimum"]
                - baseline["secret_compatible_recall_minimum"]
            ),
            "compatible_coverage_recall_mean": (
                candidate["compatible_coverage_recall_mean"]
                - baseline["compatible_coverage_recall_mean"]
            ),
            "map50_mean": candidate["map50_mean"] - baseline["map50_mean"],
        }
    )
    return gates, deltas


def runtime_selection_score(candidate_value: dict[str, Any]) -> tuple[float, ...]:
    candidate = validate_runtime_selection_candidate(candidate_value)
    return (
        candidate["secret_privacy_recall_minimum"],
        candidate["privacy_coverage_recall_mean"],
        candidate["secret_compatible_recall_minimum"],
        candidate["compatible_coverage_recall_mean"],
        candidate["map50_mean"],
        -candidate["predictions_per_image_mean"],
    )


def consider_runtime_selection_candidate(
    state: dict[str, Any] | None,
    candidate_value: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Fail-closed operational selection with a non-ratcheting reference."""

    try:
        candidate = validate_runtime_selection_candidate(candidate_value)
    except (TypeError, ValueError) as exc:
        return (
            {
                "selected": False,
                "valid": False,
                "reason": f"candidate-runtime-selection-invalid: {exc}",
                "candidate": None,
                "deltas": None,
                "gates": {},
            },
            state,
        )
    if state is None:
        initialized = {
            "schema_version": 3,
            "metric_basis": RUNTIME_SELECTION_POLICY_NAME,
            "reference": candidate,
            "incumbent": candidate,
        }
        return (
            {
                "selected": True,
                "valid": True,
                "reason": "initial-calibrated-runtime-reference",
                "candidate": candidate,
                "deltas": None,
                "gates": {
                    "complete_finite_calibrated_runtime_metrics": True,
                    "all_hard_gates": True,
                },
            },
            initialized,
        )
    if (
        not isinstance(state, dict)
        or state.get("schema_version") != 3
        or state.get("metric_basis") != RUNTIME_SELECTION_POLICY_NAME
    ):
        raise RuntimeError("calibrated runtime selection state is malformed")
    reference = validate_runtime_selection_candidate(state.get("reference"))
    incumbent = validate_runtime_selection_candidate(state.get("incumbent"))
    reference_gates, reference_deltas = _combined_gates_against(
        candidate, reference, "reference"
    )
    incumbent_gates, incumbent_deltas = _combined_gates_against(
        candidate, incumbent, "incumbent"
    )
    gates = {**reference_gates, **incumbent_gates}
    hard_gates_passed = all(gates.values())
    eps = float(RUNTIME_SELECTION_POLICY["epsilon"])
    secret_delta = (
        candidate["secret_privacy_recall_minimum"]
        - incumbent["secret_privacy_recall_minimum"]
    )
    privacy_delta = (
        candidate["privacy_coverage_recall_mean"]
        - incumbent["privacy_coverage_recall_mean"]
    )
    compatible_delta = (
        candidate["compatible_coverage_recall_mean"]
        - incumbent["compatible_coverage_recall_mean"]
    )
    runtime_nondecreasing = all(
        candidate["slices"][slice_name][metric][class_name]
        >= incumbent["slices"][slice_name][metric][class_name] - eps
        for slice_name in REQUIRED_SELECTION_SLICES
        for metric in (
            "per_class_privacy_coverage_recall",
            "per_class_compatible_coverage_recall",
        )
        for class_name in DETECTOR_CLASSES
    )

    selected = False
    reason = "rejected-no-calibrated-runtime-improvement"
    if hard_gates_passed and secret_delta >= (
        float(RUNTIME_SELECTION_POLICY["meaningful_secret_privacy_gain"]) - eps
    ):
        selected = True
        reason = "selected-meaningful-calibrated-secret-privacy-gain"
    elif hard_gates_passed and secret_delta >= -eps and privacy_delta > eps:
        selected = True
        reason = "selected-calibrated-net-privacy-gain"
    elif (
        hard_gates_passed
        and secret_delta >= -eps
        and privacy_delta >= -eps
        and compatible_delta > eps
    ):
        selected = True
        reason = "selected-calibrated-compatible-coverage-gain"
    elif (
        hard_gates_passed
        and runtime_nondecreasing
        and runtime_selection_score(candidate)
        > runtime_selection_score(incumbent)
    ):
        selected = True
        reason = "selected-calibrated-runtime-quality-tiebreak"
    elif not hard_gates_passed:
        reason = "rejected-calibrated-runtime-regression-guard"

    next_state = state
    if selected:
        next_state = {
            "schema_version": 3,
            "metric_basis": RUNTIME_SELECTION_POLICY_NAME,
            "reference": reference,
            "incumbent": candidate,
        }
    return (
        {
            "selected": selected,
            "valid": True,
            "reason": reason,
            "candidate": candidate,
            "deltas": {
                "from_reference": reference_deltas,
                "from_incumbent": incumbent_deltas,
            },
            "gates": {
                **gates,
                "all_hard_gates": hard_gates_passed,
                "all_runtime_recalls_nondecreasing": runtime_nondecreasing,
            },
        },
        next_state,
    )


def evaluate_first_epoch_calibrated_runtime_canary(
    seed_value: dict[str, Any], record: dict[str, Any]
) -> dict[str, Any]:
    seed = validate_runtime_selection_candidate(seed_value)
    candidate = validate_runtime_selection_candidate(
        record.get("runtime_selection_metrics")
    )
    gates, deltas = _combined_gates_against(candidate, seed, "seed")
    gates["checkpoint_selection_runtime_guards"] = (
        record.get("selection_gates", {}).get("all_hard_gates") is True
    )
    return {
        "schema_version": 3,
        "metric_basis": RUNTIME_SELECTION_POLICY_NAME,
        "passed": all(gates.values()),
        "gates": gates,
        "deltas_from_seed": deltas,
    }


def evaluate_runtime_release_gate(
    seed_report: dict[str, Any],
    candidate_report: dict[str, Any],
    *,
    profile: str = "high-recall",
) -> dict[str, Any]:
    """Apply privacy-only release guards on a test split after selection."""

    seed_profiles = seed_report.get("profiles")
    candidate_profiles = candidate_report.get("profiles")
    seed = seed_profiles.get(profile) if isinstance(seed_profiles, dict) else None
    candidate = (
        candidate_profiles.get(profile)
        if isinstance(candidate_profiles, dict)
        else None
    )
    if not isinstance(seed, dict) or not isinstance(candidate, dict):
        raise ValueError("runtime release reports are missing the requested profile")
    seed_classes = seed.get("per_class")
    candidate_classes = candidate.get("per_class")
    if (
        not isinstance(seed_classes, dict)
        or not isinstance(candidate_classes, dict)
        or set(seed_classes) != set(DETECTOR_CLASSES)
        or set(candidate_classes) != set(DETECTOR_CLASSES)
    ):
        raise ValueError("runtime release class metrics are incomplete")
    privacy_drop = {
        name: _finite_unit(
            seed_classes[name].get("coverage_recall"),
            f"release seed {name} privacy recall",
        )
        - _finite_unit(
            candidate_classes[name].get("coverage_recall"),
            f"release candidate {name} privacy recall",
        )
        for name in DETECTOR_CLASSES
    }
    compatible_drop = {
        name: _finite_unit(
            seed_classes[name].get("class_compatible_coverage_recall"),
            f"release seed {name} compatible recall",
        )
        - _finite_unit(
            candidate_classes[name].get("class_compatible_coverage_recall"),
            f"release candidate {name} compatible recall",
        )
        for name in DETECTOR_CLASSES
    }
    seed_inventory = seed_report.get("image_inventory")
    candidate_inventory = candidate_report.get("image_inventory")
    inventory_gate = (
        isinstance(seed_inventory, dict)
        and seed_inventory == candidate_inventory
        and seed.get("truth") == candidate.get("truth")
        and all(
            seed_classes[name].get("support")
            == candidate_classes[name].get("support")
            for name in DETECTOR_CLASSES
        )
    )
    overall_drop = _finite_unit(
        seed.get("coverage_recall"), "release seed privacy recall"
    ) - _finite_unit(
        candidate.get("coverage_recall"), "release candidate privacy recall"
    )
    hard_negative_increase = _finite_unit(
        candidate.get("hard_negative_mask_rate", 0.0),
        "release candidate hard-negative rate",
    ) - _finite_unit(
        seed.get("hard_negative_mask_rate", 0.0),
        "release seed hard-negative rate",
    )
    irrecoverable_increase = _nonnegative_int(
        candidate.get("irrecoverable_misses"),
        "release candidate irrecoverable misses",
    ) - _nonnegative_int(
        seed.get("irrecoverable_misses"),
        "release seed irrecoverable misses",
    )
    eps = float(RUNTIME_SELECTION_POLICY["epsilon"])
    gates = {
        "image_inventory_unchanged": inventory_gate,
        "overall_privacy_recall_drop": overall_drop
        <= float(RUNTIME_SELECTION_POLICY["maximum_overall_privacy_recall_drop"])
        + eps,
        "secret_privacy_recall_drop": all(
            privacy_drop[name]
            <= float(
                RUNTIME_SELECTION_POLICY["maximum_individual_secret_recall_drop"]
            )
            + eps
            for name in SECRET_VISUAL_CLASSES
        ),
        "other_privacy_recall_drop": all(
            privacy_drop[name]
            <= float(
                RUNTIME_SELECTION_POLICY[
                    "maximum_individual_privacy_recall_drop"
                ]
            )
            + eps
            for name in OTHER_SENSITIVE_CLASSES
        ),
        "compatible_recall_drop": all(
            compatible_drop[name]
            <= float(
                RUNTIME_SELECTION_POLICY[
                    "maximum_individual_compatible_recall_drop"
                ]
            )
            + eps
            for name in DETECTOR_CLASSES
        ),
        "hard_negative_mask_rate_increase": hard_negative_increase
        <= float(
            RUNTIME_SELECTION_POLICY["maximum_hard_negative_mask_rate_increase"]
        )
        + eps,
        "irrecoverable_miss_increase": irrecoverable_increase
        <= int(RUNTIME_SELECTION_POLICY["maximum_irrecoverable_miss_increase"]),
        "preprofile_not_saturated": candidate.get(
            "preprofile_saturated_images"
        )
        == 0,
    }
    return {
        "schema_version": 1,
        "metric_basis": RUNTIME_SELECTION_POLICY_NAME,
        "profile": profile,
        "passed": all(gates.values()),
        "gates": gates,
        "deltas": {
            "overall_privacy_recall_drop": overall_drop,
            "per_class_privacy_recall_drop": privacy_drop,
            "per_class_compatible_recall_drop": compatible_drop,
            "hard_negative_mask_rate_increase": hard_negative_increase,
            "irrecoverable_miss_increase": irrecoverable_increase,
        },
    }
