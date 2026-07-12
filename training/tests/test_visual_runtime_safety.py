from __future__ import annotations

import copy
import unittest

from training.visual.prepare_webpii import DETECTOR_CLASSES
from training.visual.runtime_safety import (
    REQUIRED_SELECTION_SLICES,
    RUNTIME_SELECTION_POLICY_NAME,
    canonical_json_sha256,
    consider_runtime_selection_candidate,
    evaluate_first_epoch_calibrated_runtime_canary,
    evaluate_runtime_release_gate,
    validate_runtime_selection_candidate,
)


HASHES = {
    "checkpoint": "1" * 64,
    "inventory": "2" * 64,
    "geometry": "3" * 64,
    "profile": "4" * 64,
    "runtime": "5" * 64,
    "calibration": "6" * 64,
    "selection": "7" * 64,
}


def class_map(value: float) -> dict[str, float]:
    return {name: value for name in DETECTOR_CLASSES}


def count_map(value: int) -> dict[str, int]:
    return {name: value for name in DETECTOR_CLASSES}


def slice_metrics(
    *,
    epoch: int = -1,
    privacy: float = 0.99,
    compatible: float = 0.97,
    map50: float = 0.95,
    profile_sha256: str = HASHES["profile"],
    inventory_sha256: str = HASHES["inventory"],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "metric_basis": RUNTIME_SELECTION_POLICY_NAME,
        "epoch": epoch,
        "runtime_contract_sha256": HASHES["runtime"],
        "checkpoint_sha256": HASHES["checkpoint"],
        "image_inventory_sha256": inventory_sha256,
        "runtime_geometry_sha256": HASHES["geometry"],
        "threshold_profile_sha256": profile_sha256,
        "map50": map50,
        "per_class_ap50": class_map(map50),
        "per_class_ap50_95": class_map(map50 - 0.1),
        "per_class_fixed_profile_precision": class_map(0.9),
        "per_class_fixed_profile_recall": class_map(0.95),
        "per_class_privacy_coverage_recall": class_map(privacy),
        "per_class_compatible_coverage_recall": class_map(compatible),
        "privacy_coverage_recall": privacy,
        "compatible_coverage_recall": compatible,
        "irrecoverable_misses": 0,
        "per_class_irrecoverable_misses": count_map(0),
        "per_class_support": count_map(100),
        "hard_negative_images": 100,
        "hard_negative_images_with_masks": 5,
        "truth": 900,
        "images": 1000,
        "predictions": 1000,
        "preprofile_saturated_images": 0,
        "framework_global_f1_diagnostic": {
            "threshold": 0.25,
            "per_class_recall": class_map(0.5),
        },
    }


def selection_candidate(
    *,
    epoch: int = -1,
    privacy: float = 0.99,
    compatible: float = 0.97,
    map50: float = 0.95,
    profile_value: float = 0.1,
) -> dict[str, object]:
    threshold_profile = class_map(profile_value)
    profile_sha256 = canonical_json_sha256(threshold_profile)
    return {
        "schema_version": 1,
        "metric_basis": RUNTIME_SELECTION_POLICY_NAME,
        "epoch": epoch,
        "checkpoint_sha256": HASHES["checkpoint"],
        "calibration_contract_sha256": HASHES["calibration"],
        "selection_inventory_contract_sha256": HASHES["selection"],
        "threshold_profile_sha256": profile_sha256,
        "threshold_profile": threshold_profile,
        "calibration_passed": True,
        "calibration_evidence": {
            "kind": "fixture-calibration",
            "passed": True,
            "profile_sha256": profile_sha256,
        },
        "slices": {
            name: slice_metrics(
                epoch=epoch,
                privacy=privacy,
                compatible=compatible,
                map50=map50,
                profile_sha256=profile_sha256,
                inventory_sha256=str(index + 8) * 64,
            )
            for index, name in enumerate(REQUIRED_SELECTION_SLICES)
        },
    }


class RuntimeSafetySelectionTests(unittest.TestCase):
    def test_release_gate_uses_operational_coverage_and_exact_inventory(self) -> None:
        def report(recall: float) -> dict[str, object]:
            return {
                "image_inventory": {"images": 100, "sha256": "a" * 64},
                "profiles": {
                    "high-recall": {
                        "truth": 900,
                        "coverage_recall": recall,
                        "hard_negative_mask_rate": 0.05,
                        "irrecoverable_misses": 0,
                        "preprofile_saturated_images": 0,
                        "per_class": {
                            name: {
                                "support": 100,
                                "coverage_recall": recall,
                                "class_compatible_coverage_recall": recall,
                            }
                            for name in DETECTOR_CLASSES
                        },
                    }
                },
            }

        self.assertTrue(evaluate_runtime_release_gate(report(0.99), report(0.99))["passed"])
        failed = evaluate_runtime_release_gate(report(0.99), report(0.90))
        self.assertFalse(failed["passed"])
        self.assertFalse(failed["gates"]["overall_privacy_recall_drop"])

    def test_v7_synthetic_runtime_fixture_passes_misleading_global_f1_drop(self) -> None:
        seed = selection_candidate()
        seed_slice = seed["slices"]["synthetic-guard"]
        candidate = copy.deepcopy(seed)
        candidate["epoch"] = 0
        candidate["checkpoint_sha256"] = "a" * 64
        candidate["threshold_profile"] = class_map(0.05)
        candidate["threshold_profile_sha256"] = canonical_json_sha256(
            candidate["threshold_profile"]
        )
        candidate["calibration_evidence"]["profile_sha256"] = candidate[
            "threshold_profile_sha256"
        ]
        for summary in candidate["slices"].values():
            summary["epoch"] = 0
            summary["checkpoint_sha256"] = "a" * 64
            summary["threshold_profile_sha256"] = candidate[
                "threshold_profile_sha256"
            ]
        candidate_slice = candidate["slices"]["synthetic-guard"]
        seed_slice["privacy_coverage_recall"] = 3566 / 3600
        seed_slice["compatible_coverage_recall"] = 3516 / 3600
        seed_slice["predictions"] = 5265
        candidate_slice["privacy_coverage_recall"] = 3572 / 3600
        candidate_slice["compatible_coverage_recall"] = 3526 / 3600
        candidate_slice["predictions"] = 5068
        for summary, address, secret in (
            (seed_slice, 0.99, 0.975),
            (candidate_slice, 0.98, 0.995),
        ):
            summary["per_class_privacy_coverage_recall"]["ADDRESS"] = address
            summary["per_class_compatible_coverage_recall"]["ADDRESS"] = address
            summary["per_class_privacy_coverage_recall"]["SECRET"] = secret
            summary["per_class_compatible_coverage_recall"]["SECRET"] = secret
        candidate_slice["framework_global_f1_diagnostic"] = {
            "threshold": 0.23,
            "per_class_recall": {**class_map(0.8), "ADDRESS": 0.735},
        }

        seed_decision, state = consider_runtime_selection_candidate(None, seed)
        self.assertTrue(seed_decision["selected"])
        decision, _ = consider_runtime_selection_candidate(state, candidate)
        self.assertTrue(decision["valid"])
        self.assertTrue(decision["gates"]["all_hard_gates"])
        self.assertTrue(decision["selected"])

    def test_candidate_must_pass_reference_and_current_incumbent(self) -> None:
        reference = validate_runtime_selection_candidate(
            selection_candidate(privacy=0.99)
        )
        incumbent = validate_runtime_selection_candidate(
            selection_candidate(privacy=0.95)
        )
        state = {
            "schema_version": 3,
            "metric_basis": RUNTIME_SELECTION_POLICY_NAME,
            "reference": reference,
            "incumbent": incumbent,
        }
        candidate = selection_candidate(epoch=0, privacy=0.94)
        for summary in candidate["slices"].values():
            summary["epoch"] = 0
        decision, _ = consider_runtime_selection_candidate(state, candidate)
        self.assertFalse(decision["gates"]["all_hard_gates"])
        self.assertFalse(decision["selected"])
        self.assertFalse(
            decision["gates"]["reference_synthetic_guard_overall_privacy_recall_drop"]
        )

        state["reference"] = validate_runtime_selection_candidate(
            selection_candidate(privacy=0.90)
        )
        state["incumbent"] = validate_runtime_selection_candidate(
            selection_candidate(privacy=0.99)
        )
        decision, _ = consider_runtime_selection_candidate(state, candidate)
        self.assertFalse(
            decision["gates"]["incumbent_synthetic_guard_overall_privacy_recall_drop"]
        )

    def test_ap95_inventory_saturation_and_calibration_fail_closed(self) -> None:
        seed = selection_candidate()
        _, state = consider_runtime_selection_candidate(None, seed)
        candidate = selection_candidate(epoch=0)
        for summary in candidate["slices"].values():
            summary["epoch"] = 0
        synthetic = candidate["slices"]["synthetic-guard"]
        synthetic["per_class_ap50_95"]["ADDRESS"] -= 0.04
        synthetic["preprofile_saturated_images"] = 1
        synthetic["image_inventory_sha256"] = "f" * 64
        decision, _ = consider_runtime_selection_candidate(state, candidate)
        self.assertFalse(decision["gates"]["all_hard_gates"])
        self.assertFalse(
            decision["gates"]["reference_synthetic_guard_individual_ap50_95_drop"]
        )
        self.assertFalse(
            decision["gates"]["reference_synthetic_guard_preprofile_not_saturated"]
        )
        self.assertFalse(
            decision["gates"]["reference_synthetic_guard_image_inventory_unchanged"]
        )

        candidate["calibration_passed"] = False
        decision, next_state = consider_runtime_selection_candidate(state, candidate)
        self.assertFalse(decision["valid"])
        self.assertIs(next_state, state)

    def test_canary_recomputes_hard_gates_instead_of_trusting_record(self) -> None:
        seed = selection_candidate()
        candidate = selection_candidate(epoch=0, privacy=0.90)
        for summary in candidate["slices"].values():
            summary["epoch"] = 0
        record = {
            "runtime_selection_metrics": candidate,
            "selection_gates": {"all_hard_gates": True},
        }
        canary = evaluate_first_epoch_calibrated_runtime_canary(seed, record)
        self.assertFalse(canary["passed"])


if __name__ == "__main__":
    unittest.main()
