"""Project a detector candidate onto the narrowest safe classifier boundary.

This utility is intentionally separate from training.  It starts from an
immutable seed detector, copies only the three final classification
convolutions from a candidate, and optionally restores selected class rows from
the seed.  Every other tensor is taken from the seed byte-for-byte at the tensor
level.

The pure :func:`project_classifier_state_dict` function is useful for tests and
offline analysis.  The CLI loads and saves Ultralytics checkpoints, reloads the
new file, repeats the exact parity audit, and emits a provenance JSON document.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.visual.prepare_webpii import DETECTOR_CLASSES


SCHEMA_VERSION = 1
PROJECTION_KIND = "rejected-detector-classifier-boundary-projection"
CLASSIFIER_SCALES = (0, 1, 2)
CLASSIFIER_TENSOR_NAMES = tuple(
    f"model.23.cv3.{scale}.2.{attribute}"
    for scale in CLASSIFIER_SCALES
    for attribute in ("weight", "bias")
)
_CLASSIFIER_OUTPUT_PATTERN = re.compile(r"^model\.23\.cv3\.(\d+)\.2\.(weight|bias)$")

StateDict = Mapping[str, Any]
YoloFactory = Callable[[str], Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_class_names(value: Any, *, context: str) -> tuple[str, ...]:
    """Normalize an Ultralytics name list/mapping and enforce the PLVA order."""

    if isinstance(value, Mapping):
        normalized: dict[int, str] = {}
        for raw_index, raw_name in value.items():
            if isinstance(raw_index, bool):
                raise ValueError(f"{context} has a non-integer class index")
            try:
                index = int(raw_index)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{context} has a non-integer class index") from error
            if index in normalized:
                raise ValueError(f"{context} has duplicate class index {index}")
            normalized[index] = raw_name
        if sorted(normalized) != list(range(len(normalized))):
            raise ValueError(f"{context} class indices must be contiguous from zero")
        names = tuple(normalized[index] for index in range(len(normalized)))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        names = tuple(value)
    else:
        raise ValueError(f"{context} must be a class-name list or mapping")

    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f"{context} contains an invalid class name")
    if len(set(names)) != len(names):
        raise ValueError(f"{context} contains duplicate class names")
    if names != DETECTOR_CLASSES:
        raise ValueError(
            f"{context} must exactly match the PLVA detector class order: "
            f"{list(DETECTOR_CLASSES)}"
        )
    return names


def normalize_protected_class_names(
    value: Sequence[str], *, class_names: tuple[str, ...]
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError("protected classes must be a sequence of exact class names")
    protected = tuple(value)
    if any(not isinstance(name, str) or not name for name in protected):
        raise ValueError("protected classes contain an invalid class name")
    if len(set(protected)) != len(protected):
        raise ValueError("protected classes contain duplicates")
    unknown = sorted(set(protected) - set(class_names))
    if unknown:
        raise ValueError(f"unknown protected classes: {unknown}")
    protected_set = set(protected)
    return tuple(name for name in class_names if name in protected_set)


def _shape(tensor: Any, *, name: str) -> tuple[int, ...]:
    raw_shape = getattr(tensor, "shape", None)
    if raw_shape is None:
        raise TypeError(f"state tensor {name!r} has no shape")
    try:
        return tuple(int(dimension) for dimension in raw_shape)
    except (TypeError, ValueError) as error:
        raise TypeError(f"state tensor {name!r} has an invalid shape") from error


def _dtype(tensor: Any, *, name: str) -> str:
    raw_dtype = getattr(tensor, "dtype", None)
    if raw_dtype is None:
        raise TypeError(f"state tensor {name!r} has no dtype")
    return str(raw_dtype)


def _cpu_tensor(tensor: Any, *, name: str) -> Any:
    detach = getattr(tensor, "detach", None)
    if not callable(detach):
        raise TypeError(f"state tensor {name!r} cannot be detached")
    detached = detach()
    cpu = getattr(detached, "cpu", None)
    if not callable(cpu):
        raise TypeError(f"state tensor {name!r} cannot be moved to CPU")
    return cpu()


def _clone_tensor(tensor: Any, *, name: str) -> Any:
    cpu_tensor = _cpu_tensor(tensor, name=name)
    clone = getattr(cpu_tensor, "clone", None)
    if not callable(clone):
        raise TypeError(f"state tensor {name!r} cannot be cloned")
    return clone()


def snapshot_state_dict(state: StateDict) -> dict[str, Any]:
    """Return detached CPU clones so subsequent model loads cannot mutate input."""

    _validate_state_mapping(state, context="state dict")
    return {name: _clone_tensor(tensor, name=name) for name, tensor in state.items()}


def _tensor_equal(left: Any, right: Any, *, name: str) -> bool:
    if _shape(left, name=name) != _shape(right, name=name):
        return False
    if _dtype(left, name=name) != _dtype(right, name=name):
        return False
    left_cpu = _cpu_tensor(left, name=name)
    right_cpu = _cpu_tensor(right, name=name)
    equal = getattr(left_cpu, "equal", None)
    if not callable(equal):
        raise TypeError(f"state tensor {name!r} does not support exact equality")
    return bool(equal(right_cpu))


def _row(tensor: Any, row_index: int, *, name: str) -> Any:
    try:
        return tensor[row_index]
    except (IndexError, KeyError, TypeError) as error:
        raise TypeError(f"state tensor {name!r} does not expose class rows") from error


def _row_equal(left: Any, right: Any, row_index: int, *, name: str) -> bool:
    return _tensor_equal(
        _row(left, row_index, name=name),
        _row(right, row_index, name=name),
        name=f"{name}[{row_index}]",
    )


def _restore_row(target: Any, source: Any, row_index: int, *, name: str) -> None:
    target_row = _row(target, row_index, name=name)
    source_row = _row(source, row_index, name=name)
    copy = getattr(target_row, "copy_", None)
    if not callable(copy):
        raise TypeError(f"state tensor {name!r} does not support row restoration")
    copy(source_row)


def _validate_state_mapping(state: StateDict, *, context: str) -> None:
    if not isinstance(state, Mapping) or not state:
        raise TypeError(f"{context} must be a non-empty state-dict mapping")
    if any(not isinstance(name, str) or not name for name in state):
        raise ValueError(f"{context} contains an invalid tensor name")


def _validate_state_contract(
    seed_state: StateDict,
    candidate_state: StateDict,
    *,
    class_names: tuple[str, ...],
) -> None:
    _validate_state_mapping(seed_state, context="seed state dict")
    _validate_state_mapping(candidate_state, context="candidate state dict")
    seed_keys = set(seed_state)
    candidate_keys = set(candidate_state)
    if seed_keys != candidate_keys:
        missing = sorted(seed_keys - candidate_keys)
        unexpected = sorted(candidate_keys - seed_keys)
        raise ValueError(
            "candidate state-dict keys differ from seed; "
            f"missing={missing}, unexpected={unexpected}"
        )

    discovered_outputs = {
        name for name in seed_keys if _CLASSIFIER_OUTPUT_PATTERN.fullmatch(name)
    }
    expected_outputs = set(CLASSIFIER_TENSOR_NAMES)
    if discovered_outputs != expected_outputs:
        missing = sorted(expected_outputs - discovered_outputs)
        unexpected = sorted(discovered_outputs - expected_outputs)
        raise ValueError(
            "final classifier tensor names differ from the exact projection "
            f"boundary; missing={missing}, unexpected={unexpected}"
        )

    for name in sorted(seed_keys):
        seed_shape = _shape(seed_state[name], name=name)
        candidate_shape = _shape(candidate_state[name], name=name)
        if seed_shape != candidate_shape:
            raise ValueError(
                f"candidate tensor {name!r} shape {candidate_shape} differs "
                f"from seed shape {seed_shape}"
            )
        seed_dtype = _dtype(seed_state[name], name=name)
        candidate_dtype = _dtype(candidate_state[name], name=name)
        if seed_dtype != candidate_dtype:
            raise ValueError(
                f"candidate tensor {name!r} dtype {candidate_dtype} differs "
                f"from seed dtype {seed_dtype}"
            )

    class_count = len(class_names)
    for scale in CLASSIFIER_SCALES:
        weight_name = f"model.23.cv3.{scale}.2.weight"
        bias_name = f"model.23.cv3.{scale}.2.bias"
        weight_shape = _shape(seed_state[weight_name], name=weight_name)
        bias_shape = _shape(seed_state[bias_name], name=bias_name)
        if len(weight_shape) != 4 or weight_shape[0] != class_count:
            raise ValueError(
                f"{weight_name!r} must have shape [classes, channels, 1, 1]"
            )
        if weight_shape[2:] != (1, 1):
            raise ValueError(f"{weight_name!r} is not a final 1x1 classifier conv")
        if bias_shape != (class_count,):
            raise ValueError(f"{bias_name!r} must have one row per detector class")


def audit_classifier_projection(
    seed_state: StateDict,
    candidate_state: StateDict,
    projected_state: StateDict,
    *,
    class_names: Sequence[str] | Mapping[int, str] = DETECTOR_CLASSES,
    protected_class_names: Sequence[str] = (),
    strict: bool = True,
) -> dict[str, Any]:
    """Audit exact seed/candidate parity for a projected state dict."""

    normalized_names = normalize_class_names(class_names, context="class names")
    protected = normalize_protected_class_names(
        protected_class_names, class_names=normalized_names
    )
    _validate_state_contract(seed_state, candidate_state, class_names=normalized_names)
    _validate_state_mapping(projected_state, context="projected state dict")

    seed_keys = set(seed_state)
    projected_keys = set(projected_state)
    missing = sorted(seed_keys - projected_keys)
    unexpected = sorted(projected_keys - seed_keys)
    frozen_mismatches: list[str] = []
    projected_shape_mismatches: list[str] = []
    projected_dtype_mismatches: list[str] = []
    candidate_changed_frozen: list[str] = []
    protected_row_mismatches: list[dict[str, Any]] = []
    candidate_row_mismatches: list[dict[str, Any]] = []
    changed_allowed_rows: list[dict[str, Any]] = []
    protected_set = set(protected)

    for name in sorted(seed_keys & projected_keys):
        seed_shape = _shape(seed_state[name], name=name)
        projected_shape = _shape(projected_state[name], name=name)
        if seed_shape != projected_shape:
            projected_shape_mismatches.append(name)
            continue
        if _dtype(seed_state[name], name=name) != _dtype(
            projected_state[name], name=name
        ):
            projected_dtype_mismatches.append(name)
            continue
        if name not in CLASSIFIER_TENSOR_NAMES:
            if not _tensor_equal(seed_state[name], projected_state[name], name=name):
                frozen_mismatches.append(name)
            if not _tensor_equal(seed_state[name], candidate_state[name], name=name):
                candidate_changed_frozen.append(name)
            continue

        for row_index, class_name in enumerate(normalized_names):
            row_record = {
                "tensor": name,
                "row_index": row_index,
                "class_name": class_name,
            }
            if class_name in protected_set:
                if not _row_equal(
                    seed_state[name], projected_state[name], row_index, name=name
                ):
                    protected_row_mismatches.append(row_record)
                continue
            if not _row_equal(
                candidate_state[name], projected_state[name], row_index, name=name
            ):
                candidate_row_mismatches.append(row_record)
            if not _row_equal(
                seed_state[name], projected_state[name], row_index, name=name
            ):
                changed_allowed_rows.append(row_record)

    violations: list[str] = []
    if missing:
        violations.append("projected state dict is missing seed tensors")
    if unexpected:
        violations.append("projected state dict has unexpected tensors")
    if projected_shape_mismatches:
        violations.append("projected tensor shapes differ from the seed")
    if projected_dtype_mismatches:
        violations.append("projected tensor dtypes differ from the seed")
    if frozen_mismatches:
        violations.append("frozen tensors do not exactly match the seed")
    if protected_row_mismatches:
        violations.append("protected classifier rows do not exactly match the seed")
    if candidate_row_mismatches:
        violations.append("unprotected classifier rows do not match the candidate")
    if not changed_allowed_rows:
        violations.append("projection contains no allowed classifier row change")

    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "classifier-boundary-projection-audit",
        "classes": list(normalized_names),
        "protected_classes": list(protected),
        "allowed_tensor_names": list(CLASSIFIER_TENSOR_NAMES),
        "reference_tensors": len(seed_state),
        "candidate_changed_frozen_tensors": candidate_changed_frozen,
        "changed_allowed_rows": changed_allowed_rows,
        "allowed_change_observed": bool(changed_allowed_rows),
        "missing_tensors": missing,
        "unexpected_tensors": unexpected,
        "shape_mismatches": projected_shape_mismatches,
        "dtype_mismatches": projected_dtype_mismatches,
        "frozen_tensor_mismatches": frozen_mismatches,
        "protected_row_mismatches": protected_row_mismatches,
        "candidate_row_mismatches": candidate_row_mismatches,
        "violations": violations,
        "passed": not violations,
    }
    if strict and violations:
        raise RuntimeError(
            "classifier projection audit failed: " + "; ".join(violations)
        )
    return report


def project_classifier_state_dict(
    seed_state: StateDict,
    candidate_state: StateDict,
    *,
    class_names: Sequence[str] | Mapping[int, str] = DETECTOR_CLASSES,
    protected_class_names: Sequence[str] = (),
) -> dict[str, Any]:
    """Return a new state dict containing only allowed candidate changes.

    Inputs are never mutated.  All returned tensors are detached CPU clones.
    The function fails if the resulting projection is a no-op.
    """

    normalized_names = normalize_class_names(class_names, context="class names")
    protected = normalize_protected_class_names(
        protected_class_names, class_names=normalized_names
    )
    _validate_state_contract(seed_state, candidate_state, class_names=normalized_names)

    projected = {
        name: _clone_tensor(
            candidate_state[name] if name in CLASSIFIER_TENSOR_NAMES else tensor,
            name=name,
        )
        for name, tensor in seed_state.items()
    }
    class_to_row = {name: index for index, name in enumerate(normalized_names)}
    for tensor_name in CLASSIFIER_TENSOR_NAMES:
        for class_name in protected:
            _restore_row(
                projected[tensor_name],
                seed_state[tensor_name],
                class_to_row[class_name],
                name=tensor_name,
            )

    audit_classifier_projection(
        seed_state,
        candidate_state,
        projected,
        class_names=normalized_names,
        protected_class_names=protected,
    )
    return projected


def _safe_input_checkpoint(path: Path, *, context: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{context} checkpoint is missing or unsafe: {path}")
    if path.suffix.lower() != ".pt":
        raise ValueError(f"{context} checkpoint must use a .pt suffix")
    return path.resolve()


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _assert_expected_hash(actual: str, expected: str | None, *, context: str) -> None:
    if expected is None:
        return
    normalized = expected.lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError(f"expected {context} SHA-256 must be 64 lowercase hex digits")
    if actual != normalized:
        raise RuntimeError(
            f"{context} checkpoint SHA-256 mismatch: expected {normalized}, got {actual}"
        )


def _load_model_state(
    yolo: Any, *, context: str
) -> tuple[tuple[str, ...], dict[str, Any]]:
    model = getattr(yolo, "model", None)
    if model is None or not callable(getattr(model, "state_dict", None)):
        raise TypeError(f"{context} Ultralytics object has no model state dict")
    names = normalize_class_names(
        getattr(model, "names", getattr(yolo, "names", None)),
        context=f"{context} model names",
    )
    return names, snapshot_state_dict(model.state_dict())


def project_detector_checkpoints(
    *,
    seed_checkpoint: Path,
    candidate_checkpoint: Path,
    output_checkpoint: Path,
    provenance_path: Path,
    protected_class_names: Sequence[str] = (),
    expected_seed_sha256: str | None = None,
    expected_candidate_sha256: str | None = None,
    yolo_factory: YoloFactory | None = None,
) -> dict[str, Any]:
    """Project, atomically save, reload, audit, and document a checkpoint."""

    seed_path = _safe_input_checkpoint(Path(seed_checkpoint), context="seed")
    candidate_path = _safe_input_checkpoint(
        Path(candidate_checkpoint), context="candidate"
    )
    requested_output = Path(output_checkpoint).absolute()
    requested_provenance = Path(provenance_path).absolute()
    if requested_output.is_symlink():
        raise FileExistsError(
            f"refusing to replace output checkpoint symlink: {requested_output}"
        )
    if requested_provenance.is_symlink():
        raise FileExistsError(
            f"refusing to replace provenance symlink: {requested_provenance}"
        )
    output_path = requested_output.resolve()
    provenance_output = requested_provenance.resolve()
    if seed_path == candidate_path or seed_path.samefile(candidate_path):
        raise ValueError("seed and candidate checkpoints must be different files")
    if output_path in {seed_path, candidate_path}:
        raise ValueError("output checkpoint must not overwrite an input checkpoint")
    if output_path.suffix.lower() != ".pt":
        raise ValueError("output checkpoint must use a .pt suffix")
    if provenance_output.suffix.lower() != ".json":
        raise ValueError("provenance path must use a .json suffix")
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite output checkpoint: {output_path}")
    if provenance_output.exists() or provenance_output.is_symlink():
        raise FileExistsError(f"refusing to overwrite provenance: {provenance_output}")

    seed_record = _artifact_record(seed_path)
    candidate_record = _artifact_record(candidate_path)
    _assert_expected_hash(seed_record["sha256"], expected_seed_sha256, context="seed")
    _assert_expected_hash(
        candidate_record["sha256"],
        expected_candidate_sha256,
        context="candidate",
    )

    if yolo_factory is None:
        try:
            from ultralytics import YOLO
        except ImportError as error:  # pragma: no cover - exercised in Modal image
            raise RuntimeError("checkpoint projection requires Ultralytics") from error
        yolo_factory = YOLO

    seed_yolo = yolo_factory(str(seed_path))
    candidate_yolo = yolo_factory(str(candidate_path))
    seed_names, seed_state = _load_model_state(seed_yolo, context="seed")
    candidate_names, candidate_state = _load_model_state(
        candidate_yolo, context="candidate"
    )
    if _artifact_record(seed_path) != seed_record:
        raise RuntimeError("seed checkpoint bytes changed while loading")
    if _artifact_record(candidate_path) != candidate_record:
        raise RuntimeError("candidate checkpoint bytes changed while loading")
    if candidate_names != seed_names:
        raise RuntimeError("candidate detector class names differ from the seed")
    protected = normalize_protected_class_names(
        protected_class_names, class_names=seed_names
    )
    projected_state = project_classifier_state_dict(
        seed_state,
        candidate_state,
        class_names=seed_names,
        protected_class_names=protected,
    )
    in_memory_audit = audit_classifier_projection(
        seed_state,
        candidate_state,
        projected_state,
        class_names=seed_names,
        protected_class_names=protected,
    )

    load_state_dict = getattr(seed_yolo.model, "load_state_dict", None)
    if not callable(load_state_dict):
        raise TypeError("seed Ultralytics model cannot load a projected state dict")
    load_result = load_state_dict(projected_state, strict=True)
    missing_keys = list(getattr(load_result, "missing_keys", ()))
    unexpected_keys = list(getattr(load_result, "unexpected_keys", ()))
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "strict projected state load was incomplete; "
            f"missing={missing_keys}, unexpected={unexpected_keys}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_checkpoint = output_path.with_name(
        f".{output_path.stem}.partial-{os.getpid()}.pt"
    )
    try:
        save = getattr(seed_yolo, "save", None)
        if not callable(save):
            raise TypeError("seed Ultralytics object cannot save a checkpoint")
        save(str(temporary_checkpoint))
        if temporary_checkpoint.is_symlink() or not temporary_checkpoint.is_file():
            raise RuntimeError("Ultralytics did not create a safe projected checkpoint")

        reloaded_yolo = yolo_factory(str(temporary_checkpoint))
        reloaded_names, reloaded_state = _load_model_state(
            reloaded_yolo, context="reloaded projection"
        )
        if reloaded_names != seed_names:
            raise RuntimeError("reloaded projected detector class names changed")
        reload_audit = audit_classifier_projection(
            seed_state,
            candidate_state,
            reloaded_state,
            class_names=reloaded_names,
            protected_class_names=protected,
        )
        temporary_checkpoint.replace(output_path)
    finally:
        temporary_checkpoint.unlink(missing_ok=True)

    output_record = _artifact_record(output_path)
    policy = {
        "classes": list(seed_names),
        "allowed_tensor_names": list(CLASSIFIER_TENSOR_NAMES),
        "protected_classes": list(protected),
        "protected_class_rows": {name: seed_names.index(name) for name in protected},
        "all_other_tensors_source": "immutable-seed",
        "unprotected_allowed_rows_source": "candidate",
        "require_allowed_change": True,
        "reload_before_atomic_publish": True,
    }
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "kind": PROJECTION_KIND,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": policy,
        "policy_sha256": canonical_json_sha256(policy),
        "inputs": {
            "seed_checkpoint": seed_record,
            "rejected_candidate_checkpoint": candidate_record,
        },
        "output": {"projected_checkpoint": output_record},
        "in_memory_audit": in_memory_audit,
        "reloaded_checkpoint_audit": reload_audit,
    }
    provenance["payload_sha256"] = canonical_json_sha256(provenance)
    _atomic_write_json(provenance_output, provenance)
    return {
        **provenance,
        "provenance_artifact": _artifact_record(provenance_output),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Project a rejected Ultralytics detector onto final classifier convs"
        )
    )
    parser.add_argument("--seed-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument(
        "--protected-class",
        action="append",
        default=[],
        help="exact PLVA class name whose classifier row must stay at the seed",
    )
    parser.add_argument("--expected-seed-sha256")
    parser.add_argument("--expected-candidate-sha256")
    return parser


def main(
    argv: Sequence[str] | None = None, *, yolo_factory: YoloFactory | None = None
) -> int:
    args = build_parser().parse_args(argv)
    result = project_detector_checkpoints(
        seed_checkpoint=args.seed_checkpoint,
        candidate_checkpoint=args.candidate_checkpoint,
        output_checkpoint=args.output_checkpoint,
        provenance_path=args.provenance,
        protected_class_names=args.protected_class,
        expected_seed_sha256=args.expected_seed_sha256,
        expected_candidate_sha256=args.expected_candidate_sha256,
        yolo_factory=yolo_factory,
    )
    print(
        json.dumps(
            {
                "projected_checkpoint": result["output"]["projected_checkpoint"],
                "provenance": result["provenance_artifact"],
                "protected_classes": result["policy"]["protected_classes"],
                "audit_passed": result["reloaded_checkpoint_audit"]["passed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
