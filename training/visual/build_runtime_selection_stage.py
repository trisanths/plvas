from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from training.schema import sha256_file
from training.visual.build_synthetic_selection_split import (
    build_synthetic_selection_split,
)
from training.visual.build_webpii_guard_split import build_webpii_guard_split
from training.visual.calibrate_detector_runtime import CALIBRATION_POLICY
from training.visual.diagnose_detector_curves import HIGH_RECALL_THRESHOLDS
from training.visual.prepare_webpii import DETECTOR_CLASSES


CONTRACT_NAME = "runtime-selection-contract.json"
COMBINED_CALIBRATION_IMAGES_NAME = "combined-calibration-images.txt"
COMBINED_CALIBRATION_DATASET_NAME = "dataset-combined-calibration.yaml"


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _image_lines(path: Path) -> list[str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"selection image list is missing or unsafe: {path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError("selection image list is not UTF-8") from exc
    if not lines or any(not line for line in lines) or len(lines) != len(set(lines)):
        raise ValueError("selection image list is empty or duplicated")
    return lines


def _artifact(path: Path, records: int | None = None) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"selection artifact is missing or unsafe: {path}")
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if records is not None:
        result["records"] = records
    return result


def _dataset_yaml(image_list: Path) -> str:
    return "\n".join(
        [
            "path: /",
            f"train: {image_list.resolve()}",
            f"val: {image_list.resolve()}",
            f"test: {image_list.resolve()}",
            f"nc: {len(DETECTOR_CLASSES)}",
            "names:",
            *[
                f"  {index}: {class_name}"
                for index, class_name in enumerate(DETECTOR_CLASSES)
            ],
            "",
        ]
    )


def materialize_runtime_selection_stage(
    *,
    webpii_root: Path,
    synthetic_root: Path,
    output_dir: Path,
    seed: int = 1311,
) -> dict[str, Any]:
    """Build calibration/guard inputs without reading WebPII published test."""

    output_dir = output_dir.resolve()
    webpii = build_webpii_guard_split(
        webpii_root,
        output_dir / "webpii",
        seed=seed,
        heldout_fraction=0.10,
        force=True,
    )
    synthetic = build_synthetic_selection_split(
        synthetic_root,
        output_dir / "synthetic",
        seed=seed,
    )

    webpii_calibration = Path(webpii["calibration_images"])
    webpii_guard = Path(webpii["guard_images"])
    webpii_eligible = Path(webpii["eligible_images"])
    synthetic_calibration = Path(synthetic["calibration"]["image_list"])
    synthetic_guard = Path(synthetic["guard"]["image_list"])
    combined_lines = sorted(
        _image_lines(webpii_calibration) + _image_lines(synthetic_calibration)
    )
    if len(combined_lines) != len(set(combined_lines)):
        raise RuntimeError("combined calibration inventories overlap")
    combined_list = output_dir / COMBINED_CALIBRATION_IMAGES_NAME
    combined_yaml = output_dir / COMBINED_CALIBRATION_DATASET_NAME
    _atomic_write(combined_list, "".join(f"{line}\n" for line in combined_lines))
    _atomic_write(combined_yaml, _dataset_yaml(combined_list))

    immutable_inputs = {
        "schema_version": 2,
        "seed": seed,
        "calibration_policy": CALIBRATION_POLICY,
        "frozen_seed_threshold_profile": HIGH_RECALL_THRESHOLDS,
        "webpii_split_manifest_sha256": webpii["manifest_sha256"],
        "synthetic_split_manifest_sha256": synthetic["manifest_sha256"],
        "combined_calibration_images_sha256": sha256_file(combined_list),
        "synthetic_guard_images_sha256": sha256_file(synthetic_guard),
        "webpii_guard_images_sha256": sha256_file(webpii_guard),
        "webpii_eligible_images_sha256": sha256_file(webpii_eligible),
        "published_webpii_test_used": False,
    }
    selection_inventory = {
        "synthetic-guard": _artifact(
            synthetic_guard, synthetic["guard"]["records"]
        ),
        "webpii-guard": _artifact(
            webpii_guard, webpii["statistics"]["guard"]["records"]
        ),
    }
    partitions = {
        "webpii": {
            "kind": "webpii-published-train-partitions",
            "source_root": str(webpii_root.resolve()),
            "allowed_image_root": str(
                (webpii_root.resolve() / "images/train").resolve()
            ),
            "forbidden_test_root": str(
                (webpii_root.resolve() / "images/test").resolve()
            ),
            "split_manifest": _artifact(Path(webpii["manifest_path"])),
            "calibration": _artifact(
                webpii_calibration,
                webpii["statistics"]["calibration"]["records"],
            ),
            "guard": _artifact(
                webpii_guard,
                webpii["statistics"]["guard"]["records"],
            ),
            "eligible": _artifact(
                webpii_eligible,
                webpii["statistics"]["eligible"]["records"],
            ),
        },
        "synthetic": {
            "kind": "synthetic-validation-partitions",
            "source_root": str(synthetic_root.resolve()),
            "allowed_image_root": str(
                (synthetic_root.resolve() / "images/validation").resolve()
            ),
            "split_manifest": _artifact(Path(synthetic["manifest_path"])),
            "calibration": _artifact(
                synthetic_calibration, synthetic["calibration"]["records"]
            ),
            "guard": _artifact(
                synthetic_guard, synthetic["guard"]["records"]
            ),
        },
    }
    contract = {
        "schema_version": 2,
        "kind": "calibrated-runtime-selection-contract",
        "classes": list(DETECTOR_CLASSES),
        "immutable_inputs": immutable_inputs,
        "contract_sha256": _canonical_sha256(immutable_inputs),
        "calibration": {
            "dataset_yaml": str(combined_yaml),
            "dataset_yaml_sha256": sha256_file(combined_yaml),
            "image_list": str(combined_list),
            "image_list_sha256": sha256_file(combined_list),
            "records": len(combined_lines),
            "sources": ["synthetic-validation-calibration", "WebPII/train-calibration"],
        },
        "guards": {
            "synthetic-guard": {
                "dataset_yaml": synthetic["guard"]["dataset_yaml"],
                "dataset_yaml_sha256": synthetic["guard"]["dataset_yaml_sha256"],
                "image_list": synthetic["guard"]["image_list"],
                "image_list_sha256": synthetic["guard"]["image_list_sha256"],
                "records": synthetic["guard"]["records"],
            },
            "webpii-guard": {
                "dataset_yaml": webpii["guard_dataset_yaml"],
                "dataset_yaml_sha256": webpii["guard_dataset_yaml_sha256"],
                "image_list": webpii["guard_images"],
                "image_list_sha256": webpii["guard_images_sha256"],
                "records": webpii["statistics"]["guard"]["records"],
            },
        },
        "selection_inventory": selection_inventory,
        "partitions": partitions,
        "training_eligibility": {
            "webpii_images": str(webpii_eligible),
            "webpii_images_sha256": sha256_file(webpii_eligible),
            "records": webpii["statistics"]["eligible"]["records"],
        },
        "source_manifests": {
            "webpii": {
                "path": webpii["manifest_path"],
                "sha256": webpii["manifest_sha256"],
            },
            "synthetic": {
                "path": synthetic["manifest_path"],
                "sha256": synthetic["manifest_sha256"],
            },
        },
        "published_test_used": False,
        "values_persisted": False,
    }
    contract_path = output_dir / CONTRACT_NAME
    _atomic_write(contract_path, json.dumps(contract, indent=2, sort_keys=True) + "\n")
    return {
        **contract,
        "contract_path": str(contract_path),
        "contract_file_sha256": sha256_file(contract_path),
        "webpii_eligible_images": str(webpii_eligible),
        "webpii_eligible_images_sha256": sha256_file(webpii_eligible),
        "webpii": webpii,
        "synthetic": synthetic,
    }
