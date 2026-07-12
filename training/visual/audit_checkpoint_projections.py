from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from training.schema import sha256_file
from training.visual.build_replay_train_list import (
    DEFAULT_SOURCE_QUOTA,
    materialize_replay_training_stage,
)
from training.visual.calibrate_detector_runtime import canonical_json_sha256
from training.visual.runtime_safety import (
    consider_runtime_selection_candidate,
    evaluate_first_epoch_calibrated_runtime_canary,
)
from training.visual.runtime_selection_evaluator import (
    CalibratedRuntimeSelectionEvaluator,
)
from training.visual.stage_training_dataset import (
    COPY_WORKERS,
    stage_composed_dataset,
)


SCHEMA_VERSION = 1
AUDIT_KIND = "offline-calibrated-runtime-checkpoint-projection-audit"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


@dataclass(frozen=True)
class CandidateSpec:
    label: str
    checkpoint: Path
    expected_sha256: str | None = None


def _normalized_expected_sha256(value: str | None, *, context: str) -> str | None:
    if value is None:
        return None
    normalized = value.lower()
    if _SHA256.fullmatch(normalized) is None:
        raise ValueError(f"expected {context} SHA-256 must be 64 hex digits")
    return normalized


def _parse_assignment(value: str, *, context: str) -> tuple[str, str]:
    label, separator, assigned = value.partition("=")
    if not separator or _LABEL.fullmatch(label) is None or not assigned:
        raise ValueError(
            f"{context} must use a safe LABEL=VALUE assignment; got {value!r}"
        )
    if label == "seed":
        raise ValueError(f"{context} label 'seed' is reserved")
    return label, assigned


def parse_candidate_specs(
    candidate_values: Sequence[str],
    expected_hash_values: Sequence[str] = (),
) -> tuple[CandidateSpec, ...]:
    """Parse repeatable CLI assignments and reject ambiguous candidates."""

    if not candidate_values:
        raise ValueError("at least one candidate checkpoint is required")
    candidates: dict[str, Path] = {}
    for value in candidate_values:
        label, raw_path = _parse_assignment(value, context="candidate")
        if label in candidates:
            raise ValueError(f"duplicate candidate label: {label}")
        candidates[label] = Path(raw_path)

    hashes: dict[str, str] = {}
    for value in expected_hash_values:
        label, raw_hash = _parse_assignment(value, context="expected candidate SHA-256")
        if label in hashes:
            raise ValueError(f"duplicate expected candidate SHA-256 label: {label}")
        if label not in candidates:
            raise ValueError(
                f"expected candidate SHA-256 has no matching candidate: {label}"
            )
        hashes[label] = _normalized_expected_sha256(
            raw_hash, context=f"candidate {label}"
        )

    return tuple(
        CandidateSpec(
            label=label,
            checkpoint=checkpoint,
            expected_sha256=hashes.get(label),
        )
        for label, checkpoint in candidates.items()
    )


def _safe_regular_file(path: Path, *, context: str, suffix: str | None = None) -> Path:
    requested = Path(path).absolute()
    if requested.is_symlink() or not requested.is_file():
        raise RuntimeError(f"{context} is missing or unsafe: {requested}")
    resolved = requested.resolve()
    if suffix is not None and resolved.suffix.lower() != suffix:
        raise ValueError(f"{context} must use a {suffix} suffix")
    return resolved


def _safe_source_root(path: Path, *, context: str) -> Path:
    requested = Path(path).absolute()
    if requested.is_symlink() or not requested.is_dir():
        raise RuntimeError(f"{context} is missing or unsafe: {requested}")
    return requested.resolve()


def _safe_directory_destination(path: Path, *, context: str) -> Path:
    requested = Path(path).absolute()
    if requested.is_symlink() or (requested.exists() and not requested.is_dir()):
        raise RuntimeError(f"{context} is unsafe: {requested}")
    return requested.resolve()


def _path_trees_overlap(one: Path, other: Path) -> bool:
    return one == other or one.is_relative_to(other) or other.is_relative_to(one)


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _assert_artifact_unchanged(
    path: Path, expected: Mapping[str, Any], *, context: str
) -> None:
    if path.is_symlink() or not path.is_file() or _artifact(path) != dict(expected):
        raise RuntimeError(f"{context} bytes changed during projection audit")


def _assert_expected_hash(
    artifact: Mapping[str, Any], expected: str | None, *, context: str
) -> None:
    normalized = _normalized_expected_sha256(expected, context=context)
    if normalized is not None and artifact.get("sha256") != normalized:
        raise RuntimeError(
            f"{context} SHA-256 mismatch: expected {normalized}, "
            f"got {artifact.get('sha256')}"
        )


def _safe_output_path(path: Path) -> Path:
    requested = Path(path).absolute()
    if requested.suffix.lower() != ".json":
        raise ValueError("projection audit output must use a .json suffix")
    if requested.is_symlink() or (requested.exists() and not requested.is_file()):
        raise RuntimeError(f"projection audit output is unsafe: {requested}")
    parent = requested.parent
    nearest = parent
    while not nearest.exists():
        if nearest == nearest.parent:
            break
        nearest = nearest.parent
    if nearest.is_symlink() or not nearest.is_dir():
        raise RuntimeError(f"projection audit output parent is unsafe: {nearest}")
    return requested


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir() or path.is_symlink():
        raise RuntimeError(f"projection audit output became unsafe: {path}")
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.is_symlink():
            raise RuntimeError(f"refusing to replace output symlink: {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_checkpoint_set(
    seed_checkpoint: Path,
    candidates: Sequence[CandidateSpec],
    *,
    expected_seed_sha256: str | None,
) -> tuple[
    Path, dict[str, Any], tuple[tuple[CandidateSpec, Path, dict[str, Any]], ...]
]:
    seed = _safe_regular_file(
        seed_checkpoint, context="immutable seed checkpoint", suffix=".pt"
    )
    seed_artifact = _artifact(seed)
    _assert_expected_hash(
        seed_artifact, expected_seed_sha256, context="seed checkpoint"
    )

    labels: set[str] = set()
    identities: set[tuple[int, int]] = {(seed.stat().st_dev, seed.stat().st_ino)}
    validated: list[tuple[CandidateSpec, Path, dict[str, Any]]] = []
    for candidate in candidates:
        if _LABEL.fullmatch(candidate.label) is None or candidate.label == "seed":
            raise ValueError(f"unsafe or reserved candidate label: {candidate.label!r}")
        if candidate.label in labels:
            raise ValueError(f"duplicate candidate label: {candidate.label}")
        labels.add(candidate.label)
        checkpoint = _safe_regular_file(
            candidate.checkpoint,
            context=f"candidate {candidate.label} checkpoint",
            suffix=".pt",
        )
        identity = (checkpoint.stat().st_dev, checkpoint.stat().st_ino)
        if identity in identities:
            raise ValueError(
                f"candidate {candidate.label} duplicates the seed or another candidate"
            )
        identities.add(identity)
        artifact = _artifact(checkpoint)
        _assert_expected_hash(
            artifact,
            candidate.expected_sha256,
            context=f"candidate {candidate.label} checkpoint",
        )
        validated.append((candidate, checkpoint, artifact))
    if not validated:
        raise ValueError("at least one candidate checkpoint is required")
    return seed, seed_artifact, tuple(validated)


def audit_checkpoint_projections(
    *,
    dataset_yaml: Path,
    composed_manifest: Path,
    source_roots: Mapping[str, Path],
    stage_root: Path,
    seed_checkpoint: Path,
    candidates: Sequence[CandidateSpec],
    output: Path,
    expected_seed_sha256: str | None = None,
    evaluation_output_dir: Path | None = None,
    device: str = "0",
    batch: int = 16,
    workers: int = 4,
    replay_seed: int = 1311,
    replay_batch_size: int = 64,
    source_quota: Mapping[str, int] = DEFAULT_SOURCE_QUOTA,
    copy_workers: int = COPY_WORKERS,
    force_stage: bool = False,
) -> dict[str, Any]:
    """Stage once, then audit checkpoint projections against one frozen seed."""

    if set(source_roots) != {"webpii", "synthetic", "supplemental_ats"}:
        raise ValueError(
            "source_roots must contain exactly webpii, synthetic, supplemental_ats"
        )
    if not isinstance(device, str) or not device:
        raise ValueError("evaluation device must be a non-empty string")
    if type(batch) is not int or batch <= 0 or type(workers) is not int or workers < 0:
        raise ValueError("evaluation batch/workers are invalid")
    if type(replay_seed) is not int:
        raise ValueError("replay seed must be an integer")
    if type(replay_batch_size) is not int or replay_batch_size <= 0:
        raise ValueError("replay batch size must be a positive integer")
    if type(copy_workers) is not int or copy_workers <= 0:
        raise ValueError("copy workers must be a positive integer")
    if set(source_quota) != {"webpii", "synthetic", "supplemental_ats"}:
        raise ValueError("source quota has unexpected or missing sources")
    normalized_quota = dict(source_quota)
    if any(type(value) is not int or value < 0 for value in normalized_quota.values()):
        raise ValueError("source quotas must be non-negative integers")
    if sum(normalized_quota.values()) != replay_batch_size:
        raise ValueError("source quotas must sum to replay batch size")

    dataset = _safe_regular_file(dataset_yaml, context="composed dataset YAML")
    manifest = _safe_regular_file(
        composed_manifest, context="composed dataset manifest", suffix=".json"
    )
    dataset_artifact = _artifact(dataset)
    manifest_artifact = _artifact(manifest)
    roots = {
        name: _safe_source_root(root, context=f"{name} source root")
        for name, root in source_roots.items()
    }
    destination = _safe_directory_destination(stage_root, context="stage root")
    for name, root in roots.items():
        if _path_trees_overlap(destination, root):
            raise ValueError(f"stage root overlaps the {name} source tree")

    seed, seed_artifact, validated_candidates = _validate_checkpoint_set(
        seed_checkpoint,
        candidates,
        expected_seed_sha256=expected_seed_sha256,
    )
    for _, checkpoint, _ in validated_candidates:
        if checkpoint.is_relative_to(destination):
            raise ValueError("candidate checkpoints must be outside the mutable stage")
    if seed.is_relative_to(destination):
        raise ValueError("immutable seed checkpoint must be outside the mutable stage")

    output_path = _safe_output_path(output)
    for name, root in roots.items():
        if output_path.is_relative_to(root):
            raise ValueError(f"projection audit output must be outside {name}")
    evaluation_dir = _safe_directory_destination(
        evaluation_output_dir or destination / "projection-runtime-evaluations",
        context="runtime evaluation output directory",
    )
    for name, root in roots.items():
        if _path_trees_overlap(evaluation_dir, root):
            raise ValueError(f"runtime evaluation output overlaps {name}")
    protected_stage_trees = tuple(
        destination / name
        for name in (
            "webpii",
            "synthetic",
            "supplemental_ats",
            "replay",
            "selection",
        )
    )
    for protected in protected_stage_trees:
        if _path_trees_overlap(evaluation_dir, protected):
            raise ValueError("runtime evaluation output overlaps staged audit inputs")
        if output_path.is_relative_to(protected):
            raise ValueError("projection audit output overlaps staged audit inputs")
    if seed.is_relative_to(evaluation_dir) or any(
        checkpoint.is_relative_to(evaluation_dir)
        for _, checkpoint, _ in validated_candidates
    ):
        raise ValueError("runtime evaluation output contains an input checkpoint")

    composed_stage = stage_composed_dataset(
        dataset,
        manifest,
        roots,
        destination,
        force=force_stage,
        copy_workers=copy_workers,
    )
    if (
        composed_stage.get("metadata", {}).get("test_used_for_checkpoint_selection")
        is not False
    ):
        raise RuntimeError("staged dataset does not explicitly exclude published test")
    replay_stage = materialize_replay_training_stage(
        composed_stage,
        seed=replay_seed,
        batch_size=replay_batch_size,
        source_quota=normalized_quota,
        runtime_selection=True,
    )
    selection = replay_stage.get("runtime_selection")
    if (
        not isinstance(selection, dict)
        or selection.get("published_test_used") is not False
    ):
        raise RuntimeError("runtime selection stage does not exclude published test")
    contract_value = replay_stage.get("runtime_selection_contract")
    if not isinstance(contract_value, str) or not contract_value:
        raise RuntimeError("runtime selection stage did not produce a contract")
    contract = _safe_regular_file(
        Path(contract_value), context="runtime selection contract", suffix=".json"
    )
    contract_sha256 = sha256_file(contract)
    if replay_stage.get("runtime_selection_contract_sha256") != contract_sha256:
        raise RuntimeError("runtime selection contract SHA-256 mismatch")
    contract_artifact = _artifact(contract)

    _assert_artifact_unchanged(dataset, dataset_artifact, context="dataset YAML")
    _assert_artifact_unchanged(
        manifest, manifest_artifact, context="composed dataset manifest"
    )
    _assert_artifact_unchanged(seed, seed_artifact, context="seed checkpoint")
    _assert_artifact_unchanged(
        contract, contract_artifact, context="runtime selection contract"
    )
    for candidate, checkpoint, artifact in validated_candidates:
        _assert_artifact_unchanged(
            checkpoint, artifact, context=f"candidate {candidate.label} checkpoint"
        )

    evaluator = CalibratedRuntimeSelectionEvaluator(
        contract,
        device=device,
        batch=batch,
        workers=workers,
        output_dir=evaluation_dir,
    )
    seed_metrics = evaluator.initialize_seed(seed)
    if seed_metrics.get("checkpoint_sha256") != seed_artifact["sha256"]:
        raise RuntimeError("seed evaluator reported a different checkpoint SHA-256")
    seed_decision, selection_state = consider_runtime_selection_candidate(
        None, seed_metrics
    )
    if (
        seed_decision.get("valid") is not True
        or seed_decision.get("selected") is not True
        or selection_state is None
    ):
        raise RuntimeError("immutable seed could not initialize runtime selection")
    seed_selection_state_sha256 = canonical_json_sha256(selection_state)

    candidate_records: list[dict[str, Any]] = []
    for input_index, (candidate, checkpoint, artifact) in enumerate(
        validated_candidates
    ):
        _assert_artifact_unchanged(
            checkpoint, artifact, context=f"candidate {candidate.label} checkpoint"
        )
        metrics = evaluator.evaluate_candidate(checkpoint, epoch=0)
        if metrics.get("checkpoint_sha256") != artifact["sha256"]:
            raise RuntimeError(
                f"candidate {candidate.label} evaluator reported a different SHA-256"
            )
        # Projections are unordered alternatives, not a training chronology.
        # Give every one an isolated copy of the exact seed-initialized state so
        # its selection gates and first-epoch canary cannot depend on CLI order.
        candidate_input_state = deepcopy(selection_state)
        if canonical_json_sha256(candidate_input_state) != seed_selection_state_sha256:
            raise RuntimeError("fresh candidate selection state differs from the seed")
        decision, next_state = consider_runtime_selection_candidate(
            candidate_input_state, metrics
        )
        if decision.get("valid") is not True or next_state is None:
            raise RuntimeError(
                f"candidate {candidate.label} produced invalid runtime metrics"
            )
        canary = evaluate_first_epoch_calibrated_runtime_canary(
            seed_metrics,
            {
                "runtime_selection_metrics": metrics,
                "selection_gates": decision.get("gates", {}),
            },
        )
        if canonical_json_sha256(selection_state) != seed_selection_state_sha256:
            raise RuntimeError("candidate evaluation mutated the immutable seed state")
        _assert_artifact_unchanged(
            checkpoint, artifact, context=f"candidate {candidate.label} checkpoint"
        )
        candidate_records.append(
            {
                "label": candidate.label,
                "input_index": input_index,
                "evaluation_epoch": 0,
                "checkpoint": artifact,
                "comparison_baseline": "immutable-seed",
                "selection_input_state_sha256": seed_selection_state_sha256,
                "runtime_selection_metrics": metrics,
                "selection_decision": decision,
                "resulting_selection_state": next_state,
                "calibrated_runtime_canary": canary,
            }
        )

    _assert_artifact_unchanged(seed, seed_artifact, context="seed checkpoint")
    _assert_artifact_unchanged(dataset, dataset_artifact, context="dataset YAML")
    _assert_artifact_unchanged(
        manifest, manifest_artifact, context="composed dataset manifest"
    )
    _assert_artifact_unchanged(
        contract, contract_artifact, context="runtime selection contract"
    )
    for candidate, checkpoint, artifact in validated_candidates:
        _assert_artifact_unchanged(
            checkpoint, artifact, context=f"candidate {candidate.label} checkpoint"
        )

    summary_candidates = [
        {
            "label": record["label"],
            "sha256": record["checkpoint"]["sha256"],
            "selected": record["selection_decision"]["selected"],
            "reason": record["selection_decision"]["reason"],
            "canary_passed": record["calibrated_runtime_canary"]["passed"],
        }
        for record in candidate_records
    ]
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": AUDIT_KIND,
        "published_test_used": False,
        "configuration": {
            "device": device,
            "evaluation_batch": batch,
            "evaluation_workers": workers,
            "replay_seed": replay_seed,
            "replay_batch_size": replay_batch_size,
            "source_quota": normalized_quota,
            "copy_workers": copy_workers,
            "force_stage": force_stage,
        },
        "inputs": {
            "dataset_yaml": dataset_artifact,
            "composed_manifest": manifest_artifact,
            "source_roots": {name: str(root) for name, root in sorted(roots.items())},
            "seed_checkpoint": {
                **seed_artifact,
                "expected_sha256": _normalized_expected_sha256(
                    expected_seed_sha256, context="seed checkpoint"
                ),
                "stable_during_audit": True,
            },
            "candidate_checkpoints": [
                {
                    "label": candidate.label,
                    **artifact,
                    "expected_sha256": _normalized_expected_sha256(
                        candidate.expected_sha256,
                        context=f"candidate {candidate.label} checkpoint",
                    ),
                    "stable_during_audit": True,
                }
                for candidate, _, artifact in validated_candidates
            ],
        },
        "staging": {
            "composed": composed_stage,
            "replay": replay_stage,
            "runtime_selection_contract": contract_artifact,
        },
        "runtime_selection": {
            "evaluator_instances": 1,
            "seed_initializations": 1,
            "candidate_comparison_mode": (
                "order-independent-fresh-immutable-seed-state"
            ),
            "invariants": {
                "single_evaluator_instance": True,
                "single_seed_evaluation": True,
                "candidate_order_independent": True,
                "each_candidate_considered_from_immutable_seed_state": True,
                "each_canary_compared_to_immutable_seed": True,
            },
            "seed": {
                "label": "seed",
                "checkpoint": seed_artifact,
                "runtime_selection_metrics": seed_metrics,
                "selection_decision": seed_decision,
                "selection_state": selection_state,
                "selection_state_sha256": seed_selection_state_sha256,
            },
            "candidates": candidate_records,
        },
        "summary": {
            "candidate_count": len(candidate_records),
            "all_canaries_passed": all(
                record["canary_passed"] for record in summary_candidates
            ),
            "candidate_order_independent": True,
            "selected_candidate_labels": [
                record["label"] for record in summary_candidates if record["selected"]
            ],
            "candidates": summary_candidates,
        },
    }
    document["payload_sha256"] = canonical_json_sha256(document)
    _atomic_write_json(output_path, document)
    return document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit projected detector checkpoints on frozen calibrated-runtime "
            "selection slices"
        )
    )
    parser.add_argument("--dataset-yaml", type=Path, required=True)
    parser.add_argument("--composed-manifest", type=Path, required=True)
    parser.add_argument("--webpii-root", type=Path, required=True)
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument(
        "--supplemental-ats-root", "--ats-root", type=Path, required=True
    )
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--seed-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="candidate checkpoint; repeat for order-independent alternatives",
    )
    parser.add_argument("--expected-seed-sha256")
    parser.add_argument(
        "--expected-candidate-sha256",
        action="append",
        default=[],
        metavar="LABEL=SHA256",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-output-dir", type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--replay-seed", type=int, default=1311)
    parser.add_argument("--replay-batch-size", type=int, default=64)
    parser.add_argument("--webpii-per-batch", type=int, default=40)
    parser.add_argument("--synthetic-per-batch", type=int, default=22)
    parser.add_argument("--ats-per-batch", type=int, default=2)
    parser.add_argument("--copy-workers", type=int, default=COPY_WORKERS)
    parser.add_argument("--force-stage", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    candidates = parse_candidate_specs(args.candidate, args.expected_candidate_sha256)
    result = audit_checkpoint_projections(
        dataset_yaml=args.dataset_yaml,
        composed_manifest=args.composed_manifest,
        source_roots={
            "webpii": args.webpii_root,
            "synthetic": args.synthetic_root,
            "supplemental_ats": args.supplemental_ats_root,
        },
        stage_root=args.stage_root,
        seed_checkpoint=args.seed_checkpoint,
        candidates=candidates,
        output=args.output,
        expected_seed_sha256=args.expected_seed_sha256,
        evaluation_output_dir=args.evaluation_output_dir,
        device=args.device,
        batch=args.batch,
        workers=args.workers,
        replay_seed=args.replay_seed,
        replay_batch_size=args.replay_batch_size,
        source_quota={
            "webpii": args.webpii_per_batch,
            "synthetic": args.synthetic_per_batch,
            "supplemental_ats": args.ats_per_batch,
        },
        copy_workers=args.copy_workers,
        force_stage=args.force_stage,
    )
    print(
        json.dumps(
            {
                "output": _artifact(
                    _safe_regular_file(args.output, context="audit output")
                ),
                **result["summary"],
                "published_test_used": result["published_test_used"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
