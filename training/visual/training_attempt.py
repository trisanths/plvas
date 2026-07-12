from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping

from training.schema import sha256_file


ATTEMPT_SCHEMA_VERSION = 1
DEFAULT_ATTEMPT_ID = "default"
ATTEMPT_MANIFEST_NAME = "training-attempt.json"
SEED_CHECKPOINT_NAME = "frozen-seed.pt"
SEED_PROVENANCE_NAME = "frozen-seed-provenance.json"
SAFETY_PROGRESS_NAME = "safety-selection-progress.json"
PERSISTED_STAGE_FILES = {
    "dataset_yaml": "staged-dataset.yaml",
    "source_manifest": "staged-source-manifest.json",
    "staging_manifest": "dataset-staging-manifest.json",
    "replay_manifest": "staged-replay-manifest.json",
    "replay_train_list": "staged-replay-train-list.txt",
    "runtime_selection_contract": "staged-runtime-selection-contract.json",
}
PRE_EPOCH_DISPOSABLE_FILES = {
    "safety-best.pt",
    "safety-best-thresholds.json",
    "seed-incumbent.json",
    "supplemental-test.yaml",
}
PRE_EPOCH_DISPOSABLE_DIRS = {
    "runtime-selection-evaluation",
    "frozen-base-webpii-test",
    "frozen-base-supplemental-test",
    "ultralytics-run",
}


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_attempt_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,31}", value):
        raise ValueError("visual attempt must be 1-32 safe filename characters")
    if value in {".", ".."}:
        raise ValueError("visual attempt cannot be a path component")
    return value


def attempt_output_dir(run_root: Path, attempt_id: str) -> Path:
    attempt_id = validate_attempt_id(attempt_id)
    name = "training" if attempt_id == DEFAULT_ATTEMPT_ID else f"training-{attempt_id}"
    return run_root / "visual" / name


def attempt_artifact_dir(run_root: Path, attempt_id: str) -> Path:
    attempt_id = validate_attempt_id(attempt_id)
    name = "artifacts" if attempt_id == DEFAULT_ATTEMPT_ID else f"artifacts-{attempt_id}"
    return run_root / "visual" / name


def resolve_frozen_seed(
    value: str,
    *,
    volume_root: Path = Path("/vol"),
    snapshot_root: Path | None = None,
) -> Path:
    """Resolve a regular checkpoint confined to the immutable snapshot tree."""
    volume_root = volume_root.resolve()
    allowed_root = (snapshot_root or volume_root / "snapshots").resolve()
    raw = Path(value)
    candidate = raw if raw.is_absolute() else volume_root / raw
    resolved = candidate.resolve()
    if resolved == allowed_root or allowed_root not in resolved.parents:
        raise RuntimeError("visual seed must be below the Volume snapshots root")
    if candidate.is_symlink() or resolved.is_symlink() or not resolved.is_file():
        raise RuntimeError("visual seed checkpoint is missing or unsafe")
    return resolved


def _read_optional_json(path: Path | None, context: str) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{context} is unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{context} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{context} must contain a JSON object")
    return value


def _staged_metadata_sources(
    stage: Mapping[str, Any] | None,
) -> dict[str, Path]:
    if stage is None:
        return {}
    required = {"dataset_yaml", "source_manifest", "staging_manifest"}
    missing = sorted(name for name in required if not stage.get(name))
    if missing:
        raise RuntimeError(
            "seeded attempt staging metadata is incomplete: " + ", ".join(missing)
        )
    replay_present = {
        name for name in ("replay_manifest", "replay_train_list") if stage.get(name)
    }
    if replay_present and replay_present != {"replay_manifest", "replay_train_list"}:
        raise RuntimeError("seeded attempt replay staging metadata is incomplete")
    names = required | replay_present
    if stage.get("runtime_selection_contract"):
        names.add("runtime_selection_contract")
    sources: dict[str, Path] = {}
    for name in sorted(names):
        source = Path(str(stage[name]))
        if source.is_symlink() or not source.is_file():
            raise RuntimeError(f"seeded attempt {name} staging metadata is unsafe")
        sources[name] = source
    return sources


def _load_staged_metadata_contract(
    attempt_root: Path, manifest: Mapping[str, Any]
) -> dict[str, str] | None:
    stage = manifest.get("pre_epoch_stage")
    if stage is None:
        return None
    if not isinstance(stage, dict) or stage.get("schema_version") != 1:
        raise RuntimeError("visual training attempt staging contract is invalid")
    files = stage.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("visual training attempt staging files are invalid")
    unknown = sorted(set(files) - set(PERSISTED_STAGE_FILES))
    if unknown:
        raise RuntimeError(
            "visual training attempt staging contract has unknown files: "
            + ", ".join(unknown)
        )
    required = {"dataset_yaml", "source_manifest", "staging_manifest"}
    if not required.issubset(files):
        raise RuntimeError("visual training attempt staging contract is incomplete")
    replay_present = set(files) & {"replay_manifest", "replay_train_list"}
    if replay_present and replay_present != {"replay_manifest", "replay_train_list"}:
        raise RuntimeError("visual training attempt replay staging contract is incomplete")
    result: dict[str, str] = {}
    for name, record in sorted(files.items()):
        if not isinstance(record, dict):
            raise RuntimeError("visual training attempt staging record is invalid")
        expected_filename = PERSISTED_STAGE_FILES[name]
        if record.get("filename") != expected_filename:
            raise RuntimeError("visual training attempt staging filename changed")
        path = attempt_root / expected_filename
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                f"visual training attempt staged {name} is missing or unsafe"
            )
        size = record.get("bytes")
        digest = record.get("sha256")
        if type(size) is not int or size < 0 or path.stat().st_size != size:
            raise RuntimeError(
                f"visual training attempt staged {name} byte count changed"
            )
        if not isinstance(digest, str) or sha256_file(path) != digest:
            raise RuntimeError(
                f"visual training attempt staged {name} SHA-256 changed"
            )
        result[f"{name}_path"] = str(path)
        result[f"{name}_sha256"] = digest
    return result


def initialize_seeded_attempt(
    *,
    attempt_root: Path,
    run_id: str,
    attempt_id: str,
    source_checkpoint: Path,
    expected_sha256: str,
    seed_provenance: Mapping[str, Any],
    optimizer_policy: Mapping[str, Any],
    license_path: Path,
    inherited_provenance_path: Path | None = None,
    staged_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically create a new immutable-seed attempt without overwriting state."""
    attempt_id = validate_attempt_id(attempt_id)
    if attempt_id == DEFAULT_ATTEMPT_ID:
        raise RuntimeError("a frozen-seed retry requires a non-default attempt ID")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("visual seed SHA-256 must be 64 lowercase hex characters")
    if attempt_root.exists():
        raise FileExistsError(
            f"visual attempt already exists; resume it explicitly: {attempt_root}"
        )
    if source_checkpoint.is_symlink() or not source_checkpoint.is_file():
        raise RuntimeError("visual seed checkpoint is missing or unsafe")
    source_sha256 = sha256_file(source_checkpoint)
    if source_sha256 != expected_sha256:
        raise RuntimeError("visual seed checkpoint SHA-256 does not match the CLI pin")
    if license_path.is_symlink() or not license_path.is_file():
        raise RuntimeError("detector seed license is missing or unsafe")
    license_text = license_path.read_text(encoding="utf-8").strip()
    if not license_text or license_text == "NOASSERTION":
        raise RuntimeError("detector seed requires an explicit license")
    inherited = _read_optional_json(
        inherited_provenance_path, "inherited detector provenance"
    )
    optimizer_document = dict(optimizer_policy)
    staged_sources = _staged_metadata_sources(staged_metadata)
    provenance_document = {
        "schema_version": 1,
        "kind": "frozen-volume-snapshot-seed",
        "source": {
            "path": str(source_checkpoint.resolve()),
            "bytes": source_checkpoint.stat().st_size,
            "sha256": source_sha256,
            "expected_sha256": expected_sha256,
        },
        "declared_provenance": dict(seed_provenance),
        "license": {
            "value": license_text,
            "path": str(license_path.resolve()),
            "sha256": sha256_file(license_path),
        },
        "inherited_detector_provenance": inherited,
        "inherited_detector_provenance_sha256": (
            sha256_file(inherited_provenance_path)
            if inherited is not None and inherited_provenance_path is not None
            else None
        ),
    }
    provenance_text = (
        json.dumps(provenance_document, indent=2, sort_keys=True) + "\n"
    )
    provenance_sha256 = hashlib.sha256(provenance_text.encode("utf-8")).hexdigest()
    manifest = {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "attempt_root": str(attempt_root.resolve()),
        "mode": "new-non-resume-from-frozen-seed",
        "resume_allowed": True,
        "seed": {
            "checkpoint": SEED_CHECKPOINT_NAME,
            "provenance": SEED_PROVENANCE_NAME,
            "source_path": str(source_checkpoint.resolve()),
            "bytes": source_checkpoint.stat().st_size,
            "sha256": source_sha256,
            "provenance_bytes": len(provenance_text.encode("utf-8")),
            "provenance_sha256": provenance_sha256,
        },
        "optimizer_policy": optimizer_document,
        "optimizer_policy_sha256": _canonical_hash(optimizer_document),
    }
    if staged_sources:
        manifest["pre_epoch_stage"] = {
            "schema_version": 1,
            "files": {
                name: {
                    "filename": PERSISTED_STAGE_FILES[name],
                    "bytes": source.stat().st_size,
                    "sha256": sha256_file(source),
                }
                for name, source in sorted(staged_sources.items())
            },
        }

    attempt_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = attempt_root.with_name(
        f".{attempt_root.name}.initializing-{uuid.uuid4().hex}"
    )
    if temporary.exists():
        raise RuntimeError(f"stale attempt initializer exists: {temporary}")
    temporary.mkdir()
    try:
        staged_seed = temporary / SEED_CHECKPOINT_NAME
        shutil.copy2(source_checkpoint, staged_seed)
        if (
            staged_seed.stat().st_size != source_checkpoint.stat().st_size
            or sha256_file(staged_seed) != source_sha256
        ):
            raise RuntimeError("staged visual seed failed byte/hash verification")
        (temporary / SEED_PROVENANCE_NAME).write_text(
            provenance_text, encoding="utf-8"
        )
        for name, source in sorted(staged_sources.items()):
            destination = temporary / PERSISTED_STAGE_FILES[name]
            shutil.copy2(source, destination)
            expected = manifest["pre_epoch_stage"]["files"][name]
            if (
                destination.stat().st_size != expected["bytes"]
                or sha256_file(destination) != expected["sha256"]
            ):
                raise RuntimeError(
                    f"staged visual {name} metadata failed byte/hash verification"
                )
        (temporary / ATTEMPT_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(attempt_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_seeded_attempt(
        attempt_root=attempt_root,
        run_id=run_id,
        attempt_id=attempt_id,
        optimizer_policy=optimizer_policy,
        license_path=license_path,
    )


def load_seeded_attempt(
    *,
    attempt_root: Path,
    run_id: str,
    attempt_id: str,
    optimizer_policy: Mapping[str, Any],
    license_path: Path,
) -> dict[str, Any]:
    attempt_id = validate_attempt_id(attempt_id)
    if attempt_root.is_symlink() or not attempt_root.is_dir():
        raise RuntimeError("visual training attempt root is missing or unsafe")
    manifest = _read_optional_json(
        attempt_root / ATTEMPT_MANIFEST_NAME, "visual training attempt manifest"
    )
    if manifest is None:
        raise RuntimeError("visual training attempt manifest is missing")
    expected_policy = dict(optimizer_policy)
    checks = {
        "schema_version": manifest.get("schema_version") == ATTEMPT_SCHEMA_VERSION,
        "run_id": manifest.get("run_id") == run_id,
        "attempt_id": manifest.get("attempt_id") == attempt_id,
        "attempt_root": manifest.get("attempt_root") == str(attempt_root.resolve()),
        "mode": manifest.get("mode") == "new-non-resume-from-frozen-seed",
        "resume_allowed": manifest.get("resume_allowed") is True,
        "optimizer_policy": manifest.get("optimizer_policy") == expected_policy,
        "optimizer_policy_sha256": manifest.get("optimizer_policy_sha256")
        == _canonical_hash(expected_policy),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            "visual training attempt contract mismatch: " + ", ".join(failed)
        )
    seed = manifest.get("seed")
    if not isinstance(seed, dict) or seed.get("checkpoint") != SEED_CHECKPOINT_NAME:
        raise RuntimeError("visual training attempt seed contract is invalid")
    checkpoint = attempt_root / SEED_CHECKPOINT_NAME
    provenance = attempt_root / SEED_PROVENANCE_NAME
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise RuntimeError("visual training attempt seed is missing or unsafe")
    if provenance.is_symlink() or not provenance.is_file():
        raise RuntimeError("visual training seed provenance is missing or unsafe")
    if checkpoint.stat().st_size != seed.get("bytes"):
        raise RuntimeError("visual training attempt seed byte count changed")
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != seed.get("sha256"):
        raise RuntimeError("visual training attempt seed SHA-256 changed")
    if "provenance_sha256" in seed:
        if (
            provenance.stat().st_size != seed.get("provenance_bytes")
            or sha256_file(provenance) != seed.get("provenance_sha256")
        ):
            raise RuntimeError("visual training attempt provenance bytes changed")
    provenance_document = _read_optional_json(
        provenance, "visual training seed provenance"
    )
    if provenance_document is None:
        raise RuntimeError("visual training seed provenance is missing")
    provenance_source = provenance_document.get("source")
    if (
        provenance_document.get("schema_version") != 1
        or provenance_document.get("kind") != "frozen-volume-snapshot-seed"
        or not isinstance(provenance_source, dict)
        or provenance_source.get("path") != seed.get("source_path")
        or provenance_source.get("bytes") != seed.get("bytes")
        or provenance_source.get("sha256") != seed.get("sha256")
        or provenance_source.get("expected_sha256") != seed.get("sha256")
    ):
        raise RuntimeError("visual training seed provenance contract changed")
    license_entry = provenance_document.get("license")
    if not isinstance(license_entry, dict):
        raise RuntimeError("visual training seed license provenance is missing")
    if license_path.is_symlink() or not license_path.is_file():
        raise RuntimeError("detector seed license is missing or unsafe")
    if sha256_file(license_path) != license_entry.get("sha256"):
        raise RuntimeError("detector seed license changed after attempt creation")
    staged_metadata = _load_staged_metadata_contract(attempt_root, manifest)
    return {
        "manifest": manifest,
        "manifest_path": attempt_root / ATTEMPT_MANIFEST_NAME,
        "manifest_sha256": sha256_file(attempt_root / ATTEMPT_MANIFEST_NAME),
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "provenance": provenance_document,
        "provenance_path": provenance,
        "provenance_sha256": sha256_file(provenance),
        "license": license_entry["value"],
        "staged_metadata": staged_metadata,
    }


def validate_seeded_attempt_invocation(
    attempt: Mapping[str, Any],
    *,
    source_checkpoint: Path,
    expected_sha256: str,
    seed_provenance: Mapping[str, Any],
) -> None:
    """Prove a repeated fresh command names the exact immutable attempt."""

    if source_checkpoint.is_symlink() or not source_checkpoint.is_file():
        raise RuntimeError("visual seed checkpoint is missing or unsafe")
    manifest = attempt.get("manifest")
    provenance = attempt.get("provenance")
    if not isinstance(manifest, Mapping) or not isinstance(provenance, Mapping):
        raise RuntimeError("visual training attempt contract is incomplete")
    seed = manifest.get("seed")
    if not isinstance(seed, Mapping):
        raise RuntimeError("visual training attempt seed contract is invalid")
    checks = {
        "CLI SHA-256": expected_sha256 == attempt.get("checkpoint_sha256"),
        "source SHA-256": sha256_file(source_checkpoint)
        == attempt.get("checkpoint_sha256"),
        "source path": seed.get("source_path") == str(source_checkpoint.resolve()),
        "declared provenance": provenance.get("declared_provenance")
        == dict(seed_provenance),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            "existing visual attempt differs from this fresh invocation: "
            + ", ".join(failed)
        )


def _validate_disposable_pre_epoch_tree(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(
            f"visual pre-epoch disposable output is unsafe: {path.name}"
        )
    for candidate in path.rglob("*"):
        relative = candidate.relative_to(path)
        if candidate.is_symlink():
            raise RuntimeError(
                "visual pre-epoch disposable output contains a symlink: "
                + str(relative)
            )
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise RuntimeError(
                "visual pre-epoch disposable output contains a special file: "
                + str(relative)
            )
        lowered = candidate.name.lower()
        if (
            candidate.suffix.lower() == ".pt"
            or "optimizer" in lowered
            or "scaler" in lowered
            or lowered in {"last.pt", "best.pt"}
        ):
            raise RuntimeError(
                "visual pre-epoch recovery found checkpoint or optimizer state: "
                + str(relative)
            )
        if lowered == "results.csv":
            try:
                rows = [
                    line
                    for line in candidate.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            except (OSError, UnicodeDecodeError) as exc:
                raise RuntimeError(
                    "visual pre-epoch results history is unreadable"
                ) from exc
            if len(rows) > 1:
                raise RuntimeError(
                    "visual pre-epoch recovery found recorded epoch history"
                )
        elif "history" in lowered and candidate.stat().st_size:
            raise RuntimeError(
                "visual pre-epoch recovery found recorded training history: "
                + str(relative)
            )


def _verify_loaded_attempt_unchanged(
    attempt_root: Path, attempt: Mapping[str, Any]
) -> None:
    expected = {
        ATTEMPT_MANIFEST_NAME: attempt.get("manifest_sha256"),
        SEED_CHECKPOINT_NAME: attempt.get("checkpoint_sha256"),
        SEED_PROVENANCE_NAME: attempt.get("provenance_sha256"),
    }
    for name, digest in expected.items():
        path = attempt_root / name
        if (
            not isinstance(digest, str)
            or path.is_symlink()
            or not path.is_file()
            or sha256_file(path) != digest
        ):
            raise RuntimeError(
                f"visual pre-epoch immutable attempt file changed: {name}"
            )
    manifest = attempt.get("manifest")
    if not isinstance(manifest, Mapping):
        raise RuntimeError("visual pre-epoch attempt manifest is invalid")
    staged = _load_staged_metadata_contract(attempt_root, manifest)
    if staged != attempt.get("staged_metadata"):
        raise RuntimeError("visual pre-epoch staged metadata changed")


def seeded_attempt_start_mode(
    attempt_root: Path,
    attempt: Mapping[str, Any],
    *,
    discard_pre_epoch_outputs: bool = False,
) -> str:
    """Classify an immutable named attempt as a resume or pristine restart.

    A no-progress recovery never consumes mutable model output.  It is accepted
    only while the directory contains the verified immutable attempt files and
    the staging files atomically included in that attempt's manifest.
    """

    progress_path = attempt_root / SAFETY_PROGRESS_NAME
    if progress_path.is_symlink():
        raise RuntimeError("visual safety selection progress is unsafe")
    if progress_path.exists():
        progress = _read_optional_json(
            progress_path, "visual safety selection progress"
        )
        if progress is None or progress.get("schema_version") != 4:
            raise RuntimeError(
                "visual resume requires schema-4 safety selection progress"
            )
        return "resume"

    allowed = {
        ATTEMPT_MANIFEST_NAME,
        SEED_CHECKPOINT_NAME,
        SEED_PROVENANCE_NAME,
    }
    manifest = attempt.get("manifest")
    seed = manifest.get("seed") if isinstance(manifest, Mapping) else None
    if (
        not isinstance(seed, Mapping)
        or seed.get("provenance_sha256") != attempt.get("provenance_sha256")
    ):
        raise RuntimeError(
            "visual pre-epoch recovery requires hash-bound seed provenance"
        )
    staged_metadata = attempt.get("staged_metadata")
    if staged_metadata is not None:
        if not isinstance(staged_metadata, Mapping):
            raise RuntimeError("visual training attempt staging state is invalid")
        allowed.update(
            Path(str(path)).name
            for key, path in staged_metadata.items()
            if key.endswith("_path")
        )
    unsafe: list[str] = []
    mutable: list[str] = []
    disposable: list[Path] = []
    for child in attempt_root.iterdir():
        if child.is_symlink():
            unsafe.append(child.name)
        elif child.name in allowed:
            continue
        elif child.name in PRE_EPOCH_DISPOSABLE_FILES:
            if not child.is_file():
                unsafe.append(child.name)
            else:
                disposable.append(child)
        elif child.name.startswith(".safety-best-thresholds.json.partial-"):
            if not child.is_file():
                unsafe.append(child.name)
            else:
                disposable.append(child)
        elif child.name in PRE_EPOCH_DISPOSABLE_DIRS:
            _validate_disposable_pre_epoch_tree(child)
            disposable.append(child)
        else:
            mutable.append(child.name)
    if unsafe:
        raise RuntimeError(
            "visual pre-epoch recovery found unsafe artifacts: "
            + ", ".join(sorted(unsafe))
        )
    if mutable:
        raise RuntimeError(
            "visual pre-epoch recovery found ambiguous mutable training artifacts: "
            + ", ".join(sorted(mutable))
        )
    if staged_metadata is None:
        raise RuntimeError(
            "visual pre-epoch recovery requires atomically persisted staging metadata"
        )
    if disposable and not discard_pre_epoch_outputs:
        raise RuntimeError(
            "visual pre-epoch recovery found disposable training outputs: "
            + ", ".join(sorted(path.name for path in disposable))
        )
    for path in sorted(disposable, key=lambda value: value.name):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    _verify_loaded_attempt_unchanged(attempt_root, attempt)
    return "fresh-from-seed"
