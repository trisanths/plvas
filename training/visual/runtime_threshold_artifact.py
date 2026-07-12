from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from training.schema import sha256_file
from training.visual.prepare_webpii import DETECTOR_CLASSES
from training.visual.runtime_safety import (
    RUNTIME_SELECTION_POLICY_NAME,
    canonical_json_sha256,
)


RUNTIME_THRESHOLD_ARTIFACT_NAME = "safety-best-thresholds.json"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _digest(value: Any, context: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise RuntimeError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _read_json_object(path: Path, context: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{context} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{context} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{context} must contain a JSON object")
    return value


def load_runtime_threshold_artifact(
    path: Path,
    *,
    expected_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Load and verify the checkpoint-bound calibrated runtime profile."""

    value = _read_json_object(path, "runtime threshold artifact")
    if value.get("schema_version") != 1:
        raise RuntimeError("runtime threshold artifact schema changed")
    if value.get("metric_basis") != RUNTIME_SELECTION_POLICY_NAME:
        raise RuntimeError("runtime threshold artifact metric basis changed")

    checkpoint_sha256 = _digest(
        value.get("checkpoint_sha256"), "runtime threshold checkpoint"
    )
    if (
        expected_checkpoint_sha256 is not None
        and checkpoint_sha256
        != _digest(expected_checkpoint_sha256, "expected checkpoint")
    ):
        raise RuntimeError("runtime threshold artifact belongs to another checkpoint")

    profile_value = value.get("threshold_profile")
    if not isinstance(profile_value, dict) or set(profile_value) != set(
        DETECTOR_CLASSES
    ):
        raise RuntimeError(
            "runtime threshold profile must contain every detector class exactly once"
        )
    profile: dict[str, float] = {}
    for name in DETECTOR_CLASSES:
        raw = profile_value[name]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise RuntimeError(f"runtime threshold for {name} is not numeric")
        threshold = float(raw)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise RuntimeError(
                f"runtime threshold for {name} must be finite and between zero and one"
            )
        profile[name] = threshold

    profile_sha256 = _digest(
        value.get("threshold_profile_sha256"), "runtime threshold profile"
    )
    if canonical_json_sha256(profile) != profile_sha256:
        raise RuntimeError("runtime threshold profile SHA-256 mismatch")

    calibration_contract_sha256 = _digest(
        value.get("calibration_contract_sha256"),
        "runtime threshold calibration contract",
    )
    selection_inventory_contract_sha256 = _digest(
        value.get("selection_inventory_contract_sha256"),
        "runtime threshold selection inventory contract",
    )
    evidence = value.get("calibration_evidence")
    if (
        not isinstance(evidence, dict)
        or evidence.get("passed") is not True
        or evidence.get("profile_sha256") != profile_sha256
        or not isinstance(evidence.get("kind"), str)
        or not evidence["kind"]
    ):
        raise RuntimeError("runtime threshold calibration evidence is incomplete")

    return {
        "schema_version": 1,
        "metric_basis": RUNTIME_SELECTION_POLICY_NAME,
        "path": str(path),
        "sha256": sha256_file(path),
        "checkpoint_sha256": checkpoint_sha256,
        "threshold_profile": profile,
        "threshold_profile_sha256": profile_sha256,
        "calibration_contract_sha256": calibration_contract_sha256,
        "selection_inventory_contract_sha256": (
            selection_inventory_contract_sha256
        ),
        "calibration_evidence": evidence,
    }


def load_exported_runtime_threshold_artifact(
    source: Path,
    runtime_thresholds: Path,
    detector_manifest: Path,
) -> dict[str, Any]:
    """Verify that one exported ONNX, manifest, and runtime profile are a bundle."""

    source_sha256 = sha256_file(source)
    manifest = _read_json_object(detector_manifest, "detector artifact manifest")
    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict) or artifact.get("sha256") != source_sha256:
        raise RuntimeError("detector manifest belongs to another ONNX artifact")
    binding = manifest.get("operational_thresholds")
    if not isinstance(binding, dict):
        raise RuntimeError("detector manifest has no operational threshold binding")

    normalized = load_runtime_threshold_artifact(
        runtime_thresholds,
        expected_checkpoint_sha256=binding.get("checkpoint_sha256"),
    )
    expected_values = {
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
    for key, expected in expected_values.items():
        if binding.get(key) != expected:
            raise RuntimeError(
                f"detector manifest operational threshold {key} binding changed"
            )
    normalized["detector_manifest"] = str(detector_manifest)
    normalized["detector_manifest_sha256"] = sha256_file(detector_manifest)
    normalized["detector_sha256"] = source_sha256
    return normalized
