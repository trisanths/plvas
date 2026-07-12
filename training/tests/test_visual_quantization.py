from __future__ import annotations

import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training.schema import sha256_file
from training.visual.prepare_webpii import DETECTOR_CLASSES
from training.visual.runtime_safety import (
    RUNTIME_SELECTION_POLICY_NAME,
    canonical_json_sha256,
)
from training.visual.runtime_threshold_artifact import (
    RUNTIME_THRESHOLD_ARTIFACT_NAME,
    load_exported_runtime_threshold_artifact,
    load_runtime_threshold_artifact,
)
from training.visual.quantize_detector_onnx import (
    _evenly_spaced,
    gate_report,
    quantize_and_validate,
)
from training.visual.export_detector_onnx import (
    stage_runtime_threshold_artifact,
    write_artifact_metadata,
)


def write_runtime_thresholds(
    path: Path, checkpoint_sha256: str = "c" * 64
) -> dict:
    profile = {
        name: round(0.01 * (index + 1), 6)
        for index, name in enumerate(DETECTOR_CLASSES)
    }
    profile_sha256 = canonical_json_sha256(profile)
    value = {
        "schema_version": 1,
        "metric_basis": RUNTIME_SELECTION_POLICY_NAME,
        "checkpoint_sha256": checkpoint_sha256,
        "threshold_profile": profile,
        "threshold_profile_sha256": profile_sha256,
        "calibration_contract_sha256": "d" * 64,
        "selection_inventory_contract_sha256": "e" * 64,
        "calibration_evidence": {
            "kind": "combined-training-calibration",
            "passed": True,
            "profile_sha256": profile_sha256,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return value


def write_detector_manifest(
    path: Path, detector: Path, runtime_thresholds: Path
) -> dict:
    normalized = load_runtime_threshold_artifact(runtime_thresholds)
    binding = {
        "path": runtime_thresholds.name,
        "sha256": normalized["sha256"],
        "metric_basis": normalized["metric_basis"],
        "checkpoint_sha256": normalized["checkpoint_sha256"],
        "threshold_profile_sha256": normalized["threshold_profile_sha256"],
        "calibration_contract_sha256": normalized[
            "calibration_contract_sha256"
        ],
        "selection_inventory_contract_sha256": normalized[
            "selection_inventory_contract_sha256"
        ],
    }
    value = {
        "schema_version": 2,
        "artifact": {"path": detector.name, "sha256": sha256_file(detector)},
        "operational_thresholds": binding,
    }
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return value


def passing_report() -> dict:
    per_class = {
        name: {
            "support": 20,
            "matched": 16,
            "detections": 20,
            "recall": 0.8,
            "precision": 0.8,
        }
        for name in DETECTOR_CLASSES
    }
    model_metrics = {
        "support": 20 * len(DETECTOR_CLASSES),
        "matched": 16 * len(DETECTOR_CLASSES),
        "detections": 20 * len(DETECTOR_CLASSES),
        "recall": 0.8,
        "precision": 0.8,
        "per_class": per_class,
    }
    candidate_hash = "a" * 64
    return {
        "source": {"bytes": 1_000, "sha256": "b" * 64},
        "candidate": {"bytes": 400, "sha256": candidate_hash},
        "graph_contract": {"passed": True},
        "data": {
            "calibration_images": 256,
            "evaluation_images": 400,
            "calibration_evaluation_hash_overlap": 0,
            "class_support": {name: 20 for name in DETECTOR_CLASSES},
        },
        "comparison": {
            "raw_output": {
                "non_finite_candidate_values": 0,
                "class_argmax_agreement": 0.995,
                "mean_class_score_absolute_error": 0.001,
                "maximum_class_score_absolute_error": 0.1,
            },
            "profiles": {
                profile: {
                    "fp32": copy.deepcopy(model_metrics),
                    "candidate": copy.deepcopy(model_metrics),
                    "fp32_proposal_coverage": 0.995,
                }
                for profile in ("high-recall",)
            },
        },
        "operational_thresholds": {
            "profile_name": "high-recall",
            "threshold_profile_sha256": "f" * 64,
        },
        "cross_runtime": {
            "model_sha256": candidate_hash,
            "backends": {
                "node": {
                    "passed": True,
                    "package": "onnxruntime-node",
                    "package_version": "1.27.0",
                    "vectors_checked": 104,
                },
                "wasm": {
                    "passed": True,
                    "package": "onnxruntime-web",
                    "package_version": "1.27.0",
                    "vectors_checked": 104,
                },
            },
        },
    }


class DetectorQuantizationGateTests(unittest.TestCase):
    def test_fp32_export_declares_quantization_gated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.pt"
            destination = root / "detector.onnx"
            source.write_bytes(b"source")
            destination.write_bytes(b"fp32")
            runtime_source = root / "training" / RUNTIME_THRESHOLD_ARTIFACT_NAME
            write_runtime_thresholds(
                runtime_source, checkpoint_sha256=sha256_file(source)
            )
            args = argparse.Namespace(
                source=source,
                source_license="AGPL-3.0-only",
                training_manifest=None,
                runtime_thresholds=runtime_source,
                output_dir=root,
                opset=17,
            )
            operational_thresholds = stage_runtime_threshold_artifact(
                args, expected_checkpoint_sha256=sha256_file(source)
            )
            write_artifact_metadata(
                args,
                {
                    "created_at": "2026-07-11T00:00:00+00:00",
                    "source_sha256": sha256_file(source),
                    "output_sha256": sha256_file(destination),
                    "operational_thresholds": operational_thresholds,
                },
                destination,
            )
            manifest = json.loads(
                (root / "detector_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["operational_thresholds"], operational_thresholds
            )
            self.assertTrue(
                (root / RUNTIME_THRESHOLD_ARTIFACT_NAME).is_file()
            )
            self.assertEqual(
                manifest["quantization"]["status"], "gated-not-produced"
            )
            self.assertTrue(manifest["quantization"]["fp32_remains_authoritative"])
            self.assertIn(
                "detector INT8 quantization not yet approved",
                manifest["publication_blockers"],
            )

    def test_gate_requires_size_quality_and_browser_parity(self) -> None:
        report = passing_report()
        result = gate_report(report)
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["size_reduction_fraction"], 0.6)

        report["candidate"]["bytes"] = 900
        del report["cross_runtime"]["backends"]["wasm"]
        result = gate_report(report)
        self.assertFalse(result["passed"])
        self.assertTrue(any("model bytes" in value for value in result["failures"]))
        self.assertTrue(any("wasm parity" in value for value in result["failures"]))

    def test_gate_allows_no_secret_recall_regression(self) -> None:
        report = passing_report()
        report["comparison"]["profiles"]["high-recall"]["candidate"][
            "per_class"
        ]["SECRET"] = {
            "support": 20,
            "matched": 15,
            "detections": 20,
            "recall": 0.75,
            "precision": 0.75,
        }
        result = gate_report(report)
        self.assertFalse(result["passed"])
        self.assertIn(
            "high-recall SECRET recall regresses beyond its allowed bound",
            result["failures"],
        )

    def test_sampling_is_deterministic_and_spans_the_input(self) -> None:
        paths = [Path(f"{index:03d}.png") for index in range(10)]
        self.assertEqual(
            _evenly_spaced(paths, 4),
            [Path("000.png"), Path("002.png"), Path("005.png"), Path("007.png")],
        )
        self.assertEqual(_evenly_spaced(paths, 0), paths)

    def test_exported_runtime_profile_is_hash_and_bundle_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            detector = root / "detector.onnx"
            detector.write_bytes(b"detector")
            runtime_thresholds = root / RUNTIME_THRESHOLD_ARTIFACT_NAME
            write_runtime_thresholds(runtime_thresholds)
            manifest = root / "detector_manifest.json"
            write_detector_manifest(manifest, detector, runtime_thresholds)

            normalized = load_exported_runtime_threshold_artifact(
                detector, runtime_thresholds, manifest
            )
            self.assertEqual(
                normalized["threshold_profile"]["NAME"], 0.01
            )

            detector.write_bytes(b"another-detector")
            with self.assertRaisesRegex(RuntimeError, "another ONNX"):
                load_exported_runtime_threshold_artifact(
                    detector, runtime_thresholds, manifest
                )

    def test_failed_candidate_is_not_promoted_or_left_behind(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "detector.onnx"
            source.write_bytes(b"s" * 1_000)
            runtime_thresholds = root / RUNTIME_THRESHOLD_ARTIFACT_NAME
            runtime_value = write_runtime_thresholds(runtime_thresholds)
            detector_manifest = root / "detector_manifest.json"
            write_detector_manifest(
                detector_manifest, source, runtime_thresholds
            )
            calibration_root = root / "calibration"
            evaluation_root = root / "evaluation"
            labels_root = root / "labels"
            output_dir = root / "output"
            for directory in (calibration_root, evaluation_root, labels_root):
                directory.mkdir()
            output_dir.mkdir()
            previous = output_dir / "detector.int8.onnx"
            previous.write_bytes(b"previous-approved-model")
            calibration_paths = [Path(f"cal-{index}.png") for index in range(256)]
            evaluation_paths = [Path(f"eval-{index}.png") for index in range(400)]
            support = {index: 20 for index in range(len(DETECTOR_CLASSES))}

            def fake_quantize(_source, destination, _images) -> None:
                destination.write_bytes(b"q" * 400)

            rejected_comparison = passing_report()["comparison"]
            rejected_comparison["raw_output"]["class_argmax_agreement"] = 0.50

            args = argparse.Namespace(
                source=source,
                runtime_thresholds=runtime_thresholds,
                detector_manifest=detector_manifest,
                calibration_images=calibration_root,
                evaluation_images=evaluation_root,
                evaluation_labels=labels_root,
                output_dir=output_dir,
                runtime_parity_dir=root,
                max_calibration_images=512,
                max_evaluation_images=0,
                force=True,
                require_output=False,
            )
            with (
                patch(
                    "training.visual.quantize_detector_onnx._images",
                    side_effect=[calibration_paths, evaluation_paths],
                ),
                patch(
                    "training.visual.quantize_detector_onnx._class_support",
                    return_value=support,
                ),
                patch(
                    "training.visual.quantize_detector_onnx._content_hashes",
                    side_effect=[{"calibration"}, {"evaluation"}],
                ),
                patch(
                    "training.visual.quantize_detector_onnx.quantize_qdq",
                    side_effect=fake_quantize,
                ),
                patch(
                    "training.visual.quantize_detector_onnx.graph_contract",
                    return_value={"passed": True},
                ),
                patch(
                    "training.visual.quantize_detector_onnx.compare_models",
                    return_value=rejected_comparison,
                ),
                patch(
                    "training.visual.quantize_detector_onnx.run_cross_runtime_parity",
                    side_effect=lambda candidate, *_: {
                        **passing_report()["cross_runtime"],
                        "model_sha256": sha256_file(candidate),
                    },
                ),
            ):
                result = quantize_and_validate(args)

            self.assertEqual(result["status"], "rejected")
            self.assertIsNone(result["output"])
            self.assertEqual(previous.read_bytes(), b"previous-approved-model")
            self.assertEqual(
                result["preserved_previous_output"]["sha256"], sha256_file(previous)
            )
            self.assertFalse(any(output_dir.glob("*candidate*.onnx")))
            self.assertTrue((output_dir / "detector_quantization_report.json").is_file())
            self.assertEqual(
                result["operational_thresholds"]["threshold_profile"],
                runtime_value["threshold_profile"],
            )

    def test_approved_candidate_copies_frozen_threshold_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "detector.onnx"
            source.write_bytes(b"s" * 1_000)
            runtime_thresholds = root / RUNTIME_THRESHOLD_ARTIFACT_NAME
            write_runtime_thresholds(runtime_thresholds)
            detector_manifest = root / "detector_manifest.json"
            write_detector_manifest(
                detector_manifest, source, runtime_thresholds
            )
            calibration_root = root / "calibration"
            evaluation_root = root / "evaluation"
            labels_root = root / "labels"
            output_dir = root / "output"
            for directory in (
                calibration_root,
                evaluation_root,
                labels_root,
                output_dir,
            ):
                directory.mkdir()
            calibration_paths = [Path(f"cal-{index}.png") for index in range(256)]
            evaluation_paths = [Path(f"eval-{index}.png") for index in range(400)]
            support = {index: 20 for index in range(len(DETECTOR_CLASSES))}

            args = argparse.Namespace(
                source=source,
                runtime_thresholds=runtime_thresholds,
                detector_manifest=detector_manifest,
                calibration_images=calibration_root,
                evaluation_images=evaluation_root,
                evaluation_labels=labels_root,
                output_dir=output_dir,
                runtime_parity_dir=root,
                max_calibration_images=512,
                max_evaluation_images=0,
                force=False,
                require_output=True,
            )

            def fake_quantize(_source, destination, _images) -> None:
                destination.write_bytes(b"q" * 400)

            with (
                patch(
                    "training.visual.quantize_detector_onnx._images",
                    side_effect=[calibration_paths, evaluation_paths],
                ),
                patch(
                    "training.visual.quantize_detector_onnx._class_support",
                    return_value=support,
                ),
                patch(
                    "training.visual.quantize_detector_onnx._content_hashes",
                    side_effect=[{"calibration"}, {"evaluation"}],
                ),
                patch(
                    "training.visual.quantize_detector_onnx.quantize_qdq",
                    side_effect=fake_quantize,
                ),
                patch(
                    "training.visual.quantize_detector_onnx.graph_contract",
                    return_value={"passed": True},
                ),
                patch(
                    "training.visual.quantize_detector_onnx.compare_models",
                    return_value=passing_report()["comparison"],
                ) as compare_models,
                patch(
                    "training.visual.quantize_detector_onnx.run_cross_runtime_parity",
                    side_effect=lambda candidate, *_: {
                        **passing_report()["cross_runtime"],
                        "model_sha256": sha256_file(candidate),
                    },
                ),
            ):
                result = quantize_and_validate(args)

            copied_thresholds = output_dir / RUNTIME_THRESHOLD_ARTIFACT_NAME
            used_profiles = compare_models.call_args.args[-1]
            self.assertEqual(
                used_profiles,
                {
                    "high-recall": load_runtime_threshold_artifact(
                        runtime_thresholds
                    )["threshold_profile"]
                },
            )
            self.assertEqual(result["status"], "approved")
            self.assertTrue((output_dir / "detector.int8.onnx").is_file())
            self.assertEqual(
                copied_thresholds.read_bytes(), runtime_thresholds.read_bytes()
            )
            self.assertEqual(
                result["output"]["runtime_thresholds"]["sha256"],
                sha256_file(runtime_thresholds),
            )


if __name__ == "__main__":
    unittest.main()
