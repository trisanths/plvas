from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from training.visual.compose_dataset import DETECTOR_CLASSES
from training.visual.calibrate_detector_runtime import calibrate_threshold_profile
from training.visual.diagnose_detector_curves import (
    HIGH_RECALL_THRESHOLDS,
    compare_reports,
    summarize_curves,
)
from training.visual.diagnose_detector_runtime import (
    compare_status,
    intersection_over_truth,
    label_path_for_image,
    pad_and_clip_boxes,
    runtime_nms,
    select_profile,
)


class DetectorCurveDiagnosticTests(unittest.TestCase):
    def fixture(self, *, address_recall: float = 0.7) -> SimpleNamespace:
        classes = len(DETECTOR_CLASSES)
        px = np.array([0.0, 0.1, 0.2])
        precision = np.tile(np.array([1.0, 0.9, 0.8]), (classes, 1))
        recall = np.tile(np.array([0.8, 0.7, 0.6]), (classes, 1))
        recall[DETECTOR_CLASSES.index("ADDRESS"), 1] = address_recall
        f1 = 2 * precision * recall / (precision + recall)
        return SimpleNamespace(
            ap_class_index=np.arange(classes),
            px=px,
            p_curve=precision,
            r_curve=recall,
            f1_curve=f1,
            ap50=np.linspace(0.90, 0.98, classes),
            ap=np.linspace(0.70, 0.78, classes),
        )

    def test_samples_each_class_at_runtime_high_recall_threshold(self) -> None:
        names = {index: name for index, name in enumerate(DETECTOR_CLASSES)}
        report = summarize_curves(self.fixture(), names)
        address = report["per_class"]["ADDRESS"]
        self.assertEqual(
            address["plva_high_recall"]["threshold"],
            HIGH_RECALL_THRESHOLDS["ADDRESS"],
        )
        self.assertAlmostEqual(address["plva_high_recall"]["precision"], 0.9)
        self.assertAlmostEqual(address["plva_high_recall"]["recall"], 0.7)
        self.assertIn("ap50", address)
        self.assertIn("framework_global_f1_threshold", report)

    def test_comparison_separates_fixed_threshold_and_framework_deltas(self) -> None:
        names = {index: name for index, name in enumerate(DETECTOR_CLASSES)}
        seed = summarize_curves(self.fixture(address_recall=0.70), names)
        candidate = summarize_curves(self.fixture(address_recall=0.72), names)
        comparison = compare_reports(seed, candidate)
        self.assertAlmostEqual(
            comparison["per_class"]["ADDRESS"]["plva_high_recall_delta"],
            0.02,
        )

    def test_rejects_changed_class_contract(self) -> None:
        names = {index: name for index, name in enumerate(DETECTOR_CLASSES)}
        names[0] = "WRONG"
        with self.assertRaisesRegex(ValueError, "PLVA contract"):
            summarize_curves(self.fixture(), names)

    def test_runtime_geometry_and_class_aware_nms_match_browser_contract(self) -> None:
        boxes = np.array(
            [
                [10.0, 10.0, 20.0, 20.0],
                [10.5, 10.5, 20.5, 20.5],
                [10.5, 10.5, 20.5, 20.5],
            ]
        )
        padded = pad_and_clip_boxes(boxes, width=100, height=100)
        self.assertTrue(np.array_equal(padded[0], [8.0, 8.0, 22.0, 22.0]))
        scores = np.array([0.9, 0.8, 0.7])
        classes = np.array([3, 3, 6])
        kept = runtime_nms(padded, scores, classes)
        self.assertEqual(kept.tolist(), [0, 2])
        self.assertAlmostEqual(
            intersection_over_truth(padded[[0]], np.array([10, 10, 20, 20]))[0],
            1.0,
        )

    def test_runtime_profile_uses_best_class_threshold_before_nms(self) -> None:
        boxes = np.array([[0.0, 0.0, 20.0, 20.0], [30.0, 30.0, 40.0, 40.0]])
        scores = np.array([0.09, 0.09])
        classes = np.array([3, 6])
        selected_boxes, selected_scores, selected_classes = select_profile(
            boxes,
            scores,
            classes,
            HIGH_RECALL_THRESHOLDS,
        )
        self.assertEqual(len(selected_boxes), 1)
        self.assertEqual(selected_scores.tolist(), [0.09])
        self.assertEqual(selected_classes.tolist(), [6])

    def test_runtime_comparison_attributes_threshold_and_top1_losses(self) -> None:
        seed = {
            "one#0": {
                "profiles": {
                    "high-recall": {"covered": True, "class_compatible_covered": True},
                    "balanced": {"covered": True, "class_compatible_covered": True},
                },
                "low_confidence_covering": [],
            },
            "two#0": {
                "profiles": {
                    "high-recall": {"covered": True, "class_compatible_covered": True},
                    "balanced": {"covered": True, "class_compatible_covered": True},
                },
                "low_confidence_covering": [],
            },
        }
        candidate = {
            "one#0": {
                "profiles": {
                    "high-recall": {"covered": False, "class_compatible_covered": False},
                    "balanced": {"covered": False, "class_compatible_covered": False},
                },
                "low_confidence_covering": [{"class": "ADDRESS", "score": 0.09}],
            },
            "two#0": {
                "profiles": {
                    "high-recall": {"covered": True, "class_compatible_covered": False},
                    "balanced": {"covered": True, "class_compatible_covered": False},
                },
                "low_confidence_covering": [],
            },
        }
        report = compare_status(seed, candidate)["profiles"]["high-recall"]
        self.assertEqual(report["lost_privacy_coverage"], 1)
        self.assertEqual(report["lost_class_compatible_coverage"], 2)
        self.assertEqual(report["loss_attribution"]["below_profile_threshold"], 1)
        self.assertEqual(
            report["loss_attribution"]["wrong_top1_but_still_redacted"], 1
        )

    def test_runtime_label_path_requires_one_images_component(self) -> None:
        self.assertEqual(
            label_path_for_image(Path("/root/images/validation/example.png")),
            Path("/root/labels/validation/example.txt"),
        )
        with self.assertRaisesRegex(ValueError, "unique images"):
            label_path_for_image(Path("/root/example.png"))

    def test_calibration_lowers_only_enough_to_preserve_seed_recall(self) -> None:
        classes = len(DETECTOR_CLASSES)
        px = np.array([0.0, 0.005, 0.01, 0.05, 0.1, 0.2, 0.35])
        precision = np.tile(
            np.array([0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95]),
            (classes, 1),
        )
        seed_recall = np.tile(
            np.array([1.0, 0.99, 0.98, 0.95, 0.90, 0.50, 0.40]),
            (classes, 1),
        )
        candidate_recall = seed_recall.copy()
        address = DETECTOR_CLASSES.index("ADDRESS")
        candidate_recall[address] = np.array(
            [1.0, 0.99, 0.98, 0.94, 0.88, 0.40, 0.30]
        )
        seed = SimpleNamespace(
            ap_class_index=np.arange(classes),
            px=px,
            p_curve=precision,
            r_curve=seed_recall,
        )
        candidate = SimpleNamespace(
            ap_class_index=np.arange(classes),
            px=px,
            p_curve=precision,
            r_curve=candidate_recall,
        )
        names = {index: name for index, name in enumerate(DETECTOR_CLASSES)}
        calibrated = calibrate_threshold_profile(seed, candidate, names)
        self.assertTrue(calibrated["passed"])
        self.assertEqual(calibrated["calibrated_profile"]["ADDRESS"], 0.05)
        self.assertEqual(calibrated["calibrated_profile"]["NAME"], 0.1)
        self.assertTrue(calibrated["per_class"]["ADDRESS"]["passed"])
        self.assertTrue(calibrated["thresholds_never_raised"])

    def test_calibration_never_raises_known_safe_seed_thresholds(self) -> None:
        classes = len(DETECTOR_CLASSES)
        px = np.array([0.0, 0.005, 0.01, 0.05, 0.08, 0.1, 0.2, 0.35])
        precision = np.tile(np.linspace(0.4, 0.95, len(px)), (classes, 1))
        seed_recall = np.tile(
            np.array([1.0, 0.995, 0.99, 0.98, 0.97, 0.96, 0.90, 0.80]),
            (classes, 1),
        )
        # A stronger candidate would satisfy the recall target at every grid
        # point.  Precision maximization alone must still not raise any of the
        # established PLVA high-recall thresholds.
        candidate_recall = np.ones((classes, len(px)))
        seed = SimpleNamespace(
            ap_class_index=np.arange(classes),
            px=px,
            p_curve=precision,
            r_curve=seed_recall,
        )
        candidate = SimpleNamespace(
            ap_class_index=np.arange(classes),
            px=px,
            p_curve=precision,
            r_curve=candidate_recall,
        )
        names = {index: name for index, name in enumerate(DETECTOR_CLASSES)}

        calibrated = calibrate_threshold_profile(seed, candidate, names)

        self.assertTrue(calibrated["passed"])
        self.assertTrue(calibrated["thresholds_never_raised"])
        self.assertEqual(
            calibrated["calibrated_profile"],
            HIGH_RECALL_THRESHOLDS,
        )
        for name in DETECTOR_CLASSES:
            self.assertLessEqual(
                calibrated["per_class"][name]["candidate_threshold"],
                calibrated["per_class"][name]["baseline_threshold"],
            )


if __name__ == "__main__":
    unittest.main()
