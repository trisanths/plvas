from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.schema import sha256_file
from training.visual.runtime_threshold_artifact import (
    RUNTIME_THRESHOLD_ARTIFACT_NAME,
    load_runtime_threshold_artifact,
)


def _selected_checkpoint_sha256(args: argparse.Namespace, source_sha256: str) -> str:
    training_manifest_value = getattr(args, "training_manifest", None)
    if training_manifest_value is None:
        if args.source.suffix.lower() == ".pt":
            return source_sha256
        raise RuntimeError(
            "ONNX export requires a training manifest to bind runtime thresholds"
        )
    training_manifest_path = Path(training_manifest_value)
    if not training_manifest_path.is_file():
        raise RuntimeError("training manifest is missing")
    try:
        training = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("training manifest is not valid UTF-8 JSON") from exc
    checkpoint_sha256 = training.get("best_checkpoint", {}).get("sha256")
    if not isinstance(checkpoint_sha256, str):
        raise RuntimeError("training manifest has no selected checkpoint SHA-256")
    if args.source.suffix.lower() == ".pt" and checkpoint_sha256 != source_sha256:
        raise RuntimeError("training manifest selected another checkpoint")
    return checkpoint_sha256


def stage_runtime_threshold_artifact(
    args: argparse.Namespace,
    *,
    expected_checkpoint_sha256: str,
) -> dict[str, Any]:
    source_value = getattr(args, "runtime_thresholds", None)
    if source_value is None:
        raise RuntimeError("export requires the selected runtime threshold artifact")
    source = Path(source_value)
    normalized = load_runtime_threshold_artifact(
        source,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    destination = args.output_dir / RUNTIME_THRESHOLD_ARTIFACT_NAME
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    staged = load_runtime_threshold_artifact(
        destination,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    if staged["sha256"] != normalized["sha256"]:
        raise RuntimeError("exported runtime threshold artifact changed while copying")
    return {
        "path": destination.name,
        "sha256": staged["sha256"],
        "metric_basis": staged["metric_basis"],
        "checkpoint_sha256": staged["checkpoint_sha256"],
        "threshold_profile_sha256": staged["threshold_profile_sha256"],
        "calibration_contract_sha256": staged[
            "calibration_contract_sha256"
        ],
        "selection_inventory_contract_sha256": staged[
            "selection_inventory_contract_sha256"
        ],
    }


def normalize_onnx(path: Path) -> None:
    """Remove volatile metadata and serialize protobuf fields deterministically."""
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    metadata = {
        item.key: item.value for item in model.metadata_props if item.key != "date"
    }
    metadata["description"] = "PLVA fine-grained visual detector"
    del model.metadata_props[:]
    for key, value in sorted(metadata.items()):
        item = model.metadata_props.add()
        item.key = key
        item.value = value
    onnx.checker.check_model(model)
    path.write_bytes(model.SerializeToString(deterministic=True))


def write_artifact_metadata(
    args: argparse.Namespace,
    report: dict[str, Any],
    destination: Path,
) -> tuple[str, str]:
    operational_thresholds = report.get("operational_thresholds")
    if not isinstance(operational_thresholds, dict):
        raise RuntimeError("export report has no operational threshold binding")
    training_manifest_path = getattr(args, "training_manifest", None)
    training = (
        json.loads(training_manifest_path.read_text(encoding="utf-8"))
        if training_manifest_path and training_manifest_path.exists()
        else {}
    )
    epochs = int(
        training.get("training_config", {}).get(
            "epochs", len(training.get("history", []))
        )
    )
    smoke_only = epochs <= 2
    blockers = list(training.get("publication_blockers", []))
    selected_evaluation = training.get("selected_checkpoint_evaluation", {})
    runtime_selection = training.get(
        "runtime_checkpoint_selection_evaluation", {}
    )
    selected_runtime = runtime_selection.get("selected", {})
    selected_secret_recall = selected_runtime.get(
        "secret_privacy_recall_minimum"
    )
    if smoke_only:
        blockers.append("two-epoch development smoke is not a quality candidate")
    if selected_secret_recall is not None and float(selected_secret_recall) < 1.0:
        blockers.append("selected checkpoint does not meet the zero-secret-miss gate")
    if "cross-runtime ONNX parity not yet proven" not in blockers:
        blockers.append("cross-runtime ONNX parity not yet proven")
    if "detector INT8 quantization not yet approved" not in blockers:
        blockers.append("detector INT8 quantization not yet approved")
    manifest = {
        "schema_version": 2,
        "created_at": report["created_at"],
        "status": "development-smoke" if smoke_only else "trained-unpublished",
        "release_eligible": False,
        "artifact": {
            "path": destination.name,
            "bytes": destination.stat().st_size,
            "sha256": report["output_sha256"],
            "format": "onnx",
            "opset": args.opset,
            "input": {
                "name": "images",
                "layout": "NCHW",
                "dtype": "float32",
                "dynamic_axes": ["batch", "height", "width"],
            },
        },
        "source": {
            "path": str(args.source),
            "sha256": report["source_sha256"],
            "license": args.source_license,
        },
        "operational_thresholds": operational_thresholds,
        "training_manifest": (
            {
                "path": str(training_manifest_path),
                "sha256": sha256_file(training_manifest_path),
            }
            if training_manifest_path and training_manifest_path.exists()
            else None
        ),
        "training": {
            "epochs": epochs,
            "metrics": training.get("train_results", {}),
            "selection": training.get("best_checkpoint", {}).get("score"),
            "selected_checkpoint_evaluation": selected_evaluation,
            "runtime_checkpoint_selection_evaluation": runtime_selection,
            "distribution": training.get("distribution", {}),
        },
        "quantization": {
            "status": "gated-not-produced",
            "runtime_artifact": None,
            "fp32_remains_authoritative": True,
            "candidate_strategy": {
                "format": "QDQ",
                "activation_type": "QUInt8",
                "weight_type": "QInt8",
                "operators": ["Conv"],
                "runtime_target": "onnxruntime-web@1.27.0/wasm",
            },
            "approval_command": (
                "python -m training.visual.quantize_detector_onnx "
                "--source detector.onnx "
                f"--runtime-thresholds {operational_thresholds['path']} "
                "--detector-manifest detector_manifest.json"
            ),
            "required_evidence": [
                "at least 50% artifact byte reduction",
                "frozen all-class holdout safety parity",
                "full-output numerical comparison",
                "pinned ONNX Runtime 1.27.0 Node and WASM parity",
            ],
        },
        "publication_blockers": blockers,
    }
    manifest_name = "detector_manifest.json"
    (args.output_dir / manifest_name).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metrics = manifest["training"]["metrics"]
    secret_recall = selected_runtime.get("secret_privacy_recall_minimum")
    privacy_recall = selected_runtime.get("privacy_coverage_recall_mean")
    compatible_recall = selected_runtime.get(
        "compatible_coverage_recall_mean"
    )
    card = "\n".join(
        [
            "# PLVA fine-grained visual detector",
            "",
            f"Status: {manifest['status']}; not approved for release.",
            "",
            "This detector proposes class-aware sensitive regions for PLVA's local ",
            "visual path. OCR plus semantic detection remains an independent safety path.",
            "",
            "## Artifact",
            "",
            f"- ONNX: `{destination.name}` ({destination.stat().st_size:,} bytes)",
            f"- SHA-256: `{report['output_sha256']}`",
            f"- Opset: {args.opset}",
            f"- Training epochs represented: {epochs}",
            f"- Runtime thresholds: `{operational_thresholds['path']}`",
            f"- Runtime profile SHA-256: `{operational_thresholds['threshold_profile_sha256']}`",
            "- Quantization: gated; FP32 remains the only exported artifact",
            "",
            "## Operational selection metrics",
            "",
            f"- Mean privacy coverage: {float(privacy_recall or 0.0):.6f}",
            f"- Mean class-compatible coverage: {float(compatible_recall or 0.0):.6f}",
            f"- Minimum secret privacy coverage: {float(secret_recall or 0.0):.6f}",
            "- Metric basis: calibrated per-class thresholds with PLVA top-1, padding, and NMS",
            "",
            "## Framework diagnostics (not checkpoint-selection metrics)",
            "",
            f"- mAP50: {float(metrics.get('metrics/mAP50(B)', 0.0)):.6f}",
            f"- mAP50-95: {float(metrics.get('metrics/mAP50-95(B)', 0.0)):.6f}",
            f"- Precision: {float(metrics.get('metrics/precision(B)', 0.0)):.6f}",
            f"- Recall: {float(metrics.get('metrics/recall(B)', 0.0)):.6f}",
            "",
            "These measurements validate the training/export path only; they are not ",
            "release-quality evidence.",
            "",
            "## License and release status",
            "",
            f"The source checkpoint is recorded as `{args.source_license}`. AGPL ",
            "development output cannot ship in a closed-source product without an ",
            "Ultralytics Enterprise license. Cross-runtime parity, frozen holdout ",
            "quality gates, dataset-license review, and WebGPU evidence also remain ",
            "required.",
            "",
            "Static Conv INT8 is not emitted automatically. A separate QDQ candidate ",
            "must pass size, frozen holdout recall, full-output numerical, and pinned ",
            "Node/WASM parity gates before it can be promoted.",
            "",
        ]
    )
    card_name = "MODEL_CARD.md"
    (args.output_dir / card_name).write_text(card, encoding="utf-8")
    return manifest_name, card_name


def export(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.source.suffix.lower()
    report: dict[str, Any] = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source),
        "source_sha256": sha256_file(args.source),
        "source_license": args.source_license,
    }
    if suffix == ".xml":
        weights = args.source.with_suffix(".bin")
        if not weights.exists():
            raise FileNotFoundError(weights)
        report |= {
            "status": "blocked",
            "output": None,
            "reason": (
                "The public WebRedact release is OpenVINO IR only. Supported OpenVINO "
                "conversion imports ONNX/PyTorch into IR; it does not reconstruct a "
                "source-equivalent ONNX graph from IR."
            ),
            "weights_sha256": sha256_file(weights),
            "required_resolution": (
                "Obtain a licensed ONNX/source checkpoint from upstream or train the "
                "PLVA replacement detector and compare it against this reference."
            ),
        }
    elif suffix == ".onnx":
        import onnx

        onnx.checker.check_model(onnx.load(str(args.source), load_external_data=False))
        destination = args.output_dir / "detector.onnx"
        shutil.copy2(args.source, destination)
        report |= {
            "status": "copied-verified-onnx",
            "output": destination.name,
            "output_sha256": sha256_file(destination),
        }
    elif suffix == ".pt":
        if args.source_license in {"", "NOASSERTION"}:
            raise RuntimeError("PyTorch source checkpoint requires an explicit license")
        # BatchNorm fusion otherwise varies by a few float32 ULPs with CPU
        # reduction/thread scheduling, changing the artifact hash for the same
        # checkpoint. Export on one deterministic Torch thread.
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "install a reviewed, pinned Ultralytics build to export a .pt checkpoint"
            ) from exc
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        model = YOLO(str(args.source))
        exported = Path(
            model.export(
                format="onnx",
                imgsz=args.image_size,
                opset=args.opset,
                dynamic=True,
                simplify=False,
            )
        )
        destination = args.output_dir / "detector.onnx"
        shutil.copy2(exported, destination)
        normalize_onnx(destination)
        report |= {
            "status": "exported",
            "output": destination.name,
            "output_sha256": sha256_file(destination),
            "deterministic_export": {
                "torch_threads": 1,
                "volatile_metadata_removed": ["date"],
                "protobuf_serialization": "deterministic",
            },
        }
    else:
        raise ValueError(f"unsupported detector source: {args.source}")

    if report["output"] is not None:
        report["operational_thresholds"] = stage_runtime_threshold_artifact(
            args,
            expected_checkpoint_sha256=_selected_checkpoint_sha256(
                args, report["source_sha256"]
            ),
        )
        manifest_name, card_name = write_artifact_metadata(args, report, destination)
        report["artifact_manifest"] = manifest_name
        report["model_card"] = card_name
    report_path = args.output_dir / "conversion_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.require_output and report["output"] is None:
        raise RuntimeError(report["reason"])
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a reproducible visual detector ONNX"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-license", default="NOASSERTION")
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument(
        "--runtime-thresholds",
        type=Path,
        help="checkpoint-bound safety-best-thresholds.json from training",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--require-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(export(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
