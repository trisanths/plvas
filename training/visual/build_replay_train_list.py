from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from training.schema import sha256_file
from training.visual.prepare_webpii import DETECTOR_CLASSES


SCHEMA_VERSION = 1
SOURCE_ORDER = ("webpii", "synthetic", "supplemental_ats")
DEFAULT_SOURCE_QUOTA = {
    # Match the eligible unique-source inventory (36,407/20,000/1,800) while
    # retaining full coverage and deterministic rare-class replay. The prior
    # 32/28/4 policy overexposed synthetic and ATS examples; its first canary
    # improved synthetic metrics while regressing the WebPII guard.
    "webpii": 40,
    "synthetic": 22,
    "supplemental_ats": 2,
}
DEFAULT_RARE_CLASSES = ("CVC", "SECRET", "SENSITIVE_IMAGE")
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
REPLAY_DATASET_NAME = "dataset-replay.yaml"
REPLAY_SOURCE_MANIFEST_NAME = "source-manifest-replay.json"
REPLAY_STAGING_MANIFEST_NAME = "staging-manifest-replay.json"


@dataclass(frozen=True)
class ImageLabelPair:
    source: str
    image: Path
    label: Path
    relative_image: str
    relative_label: str
    classes: tuple[int, ...]
    class_instances: tuple[int, ...]
    label_sha256: str
    image_bytes: int


class DeterministicReplaySequence:
    """Coverage-first deterministic weighted sequence.

    Every candidate is emitted once in a weighted permutation before weighted
    replacement begins. This preserves source coverage when a source is
    oversampled while still applying inverse-sqrt replay pressure to repeats.
    """

    def __init__(
        self,
        candidates: Sequence[ImageLabelPair],
        weights: Mapping[Path, float],
        *,
        seed: int,
        context: str,
    ) -> None:
        if not candidates:
            raise ValueError(f"{context} candidate pool is empty")
        self._candidates = tuple(candidates)
        self._weights = tuple(float(weights[item.image]) for item in candidates)
        if any(not math.isfinite(weight) or weight <= 0 for weight in self._weights):
            raise ValueError(f"{context} contains a non-positive sampling weight")
        self._seed = int(seed)
        self._context = context
        self._coverage = sorted(
            self._candidates,
            key=lambda item: _weighted_rank(
                seed=self._seed,
                context=f"{self._context}:coverage",
                identity=item.relative_image,
                weight=weights[item.image],
            ),
        )
        self._coverage_index = 0
        self._replacement_index = 0
        cumulative: list[float] = []
        running = 0.0
        for weight in self._weights:
            running += weight
            cumulative.append(running)
        self._cumulative = tuple(cumulative)
        self._weight_sum = running

    def next(self) -> ImageLabelPair:
        if self._coverage_index < len(self._coverage):
            value = self._coverage[self._coverage_index]
            self._coverage_index += 1
            return value
        unit = _unit_interval(
            self._seed,
            f"{self._context}:replacement:{self._replacement_index}",
        )
        self._replacement_index += 1
        index = bisect.bisect_left(self._cumulative, unit * self._weight_sum)
        return self._candidates[min(index, len(self._candidates) - 1)]


def build_replay_train_list(
    source_roots: Mapping[str, Path],
    output_dir: Path,
    *,
    seed: int = 1311,
    logical_batches: int | None = None,
    batch_size: int = 64,
    source_quota: Mapping[str, int] = DEFAULT_SOURCE_QUOTA,
    rare_classes: Sequence[str] = DEFAULT_RARE_CLASSES,
    eligible_images_by_source: Mapping[str, set[Path]] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build a deterministic, source/class-aware Ultralytics train list.

    The resulting text file is a valid Ultralytics image-list input. Each
    logical 64-row block has the requested source quota and contains a
    synthetic replay example for every required rare class. Ultralytics 8.4.92
    sorts image-list entries and then globally shuffles them when ``rect=False``;
    therefore these are exact epoch-level quotas and construction-time block
    guarantees, not physical DataLoader batch guarantees. A custom BatchSampler
    is required if the same quota must survive consumer shuffling.
    """

    roots = _validate_source_roots(source_roots)
    quota = _validate_quota(source_quota, batch_size)
    rare_indices = _validate_rare_classes(rare_classes)
    output_dir = Path(output_dir)
    list_path = output_dir / "train-replay.txt"
    manifest_path = output_dir / "replay-manifest.json"
    if not force:
        for path in (list_path, manifest_path):
            if path.exists():
                raise FileExistsError(f"replay output already exists: {path}")

    full_inventories = {
        source: _inventory_source(source, roots[source]) for source in SOURCE_ORDER
    }
    eligible_images_by_source = eligible_images_by_source or {}
    unknown_eligibility_sources = set(eligible_images_by_source) - set(SOURCE_ORDER)
    if unknown_eligibility_sources:
        raise ValueError(
            "eligible image filters contain unknown sources: "
            + ", ".join(sorted(unknown_eligibility_sources))
        )
    inventories: dict[str, list[ImageLabelPair]] = {}
    excluded_inventories: dict[str, list[ImageLabelPair]] = {}
    for source in SOURCE_ORDER:
        eligible = eligible_images_by_source.get(source)
        if eligible is None:
            inventories[source] = full_inventories[source]
            excluded_inventories[source] = []
            continue
        normalized = {Path(path).resolve() for path in eligible}
        available = {item.image for item in full_inventories[source]}
        if not normalized or not normalized <= available:
            raise ValueError(f"{source} eligible image filter is empty or contains unknown files")
        inventories[source] = [
            item for item in full_inventories[source] if item.image in normalized
        ]
        excluded_inventories[source] = [
            item for item in full_inventories[source] if item.image not in normalized
        ]
        if not inventories[source]:
            raise ValueError(f"{source} eligible image filter removes the full source")
    total_images = sum(len(items) for items in inventories.values())
    if logical_batches is None:
        zero_coverage = [
            source for source in SOURCE_ORDER if quota[source] == 0
        ]
        if zero_coverage:
            raise ValueError(
                "automatic replay coverage requires a positive quota for: "
                + ", ".join(zero_coverage)
            )
        synthetic_shared_quota = quota["synthetic"] - len(rare_indices)
        if synthetic_shared_quota <= 0:
            raise ValueError(
                "automatic replay coverage requires synthetic quota beyond "
                "the reserved rare-class slots"
            )
        effective_quota = {
            **quota,
            "synthetic": synthetic_shared_quota,
        }
        logical_batches = max(
            math.ceil(len(inventories[source]) / effective_quota[source])
            for source in SOURCE_ORDER
        )
    if type(logical_batches) is not int or logical_batches <= 0:
        raise ValueError("logical_batches must be a positive integer")

    original_counts: Counter[int] = Counter()
    for items in inventories.values():
        for item in items:
            original_counts.update(item.class_instances)
    missing_classes = [
        name
        for index, name in enumerate(DETECTOR_CLASSES)
        if original_counts[index] == 0
    ]
    if missing_classes:
        raise ValueError(
            "replay sources have no labels for detector classes: "
            + ", ".join(missing_classes)
        )

    raw_class_weights = {
        index: 1.0 / math.sqrt(original_counts[index])
        for index in range(len(DETECTOR_CLASSES))
    }
    mean_weight = sum(raw_class_weights.values()) / len(raw_class_weights)
    class_weights = {
        index: raw_class_weights[index] / mean_weight
        for index in raw_class_weights
    }
    pair_weights: dict[Path, float] = {}
    for items in inventories.values():
        for item in items:
            pair_weights[item.image] = (
                max(class_weights[index] for index in item.classes)
                if item.classes
                else 1.0
            )

    sequences = {
        source: DeterministicReplaySequence(
            inventories[source],
            pair_weights,
            seed=seed,
            context=source,
        )
        for source in SOURCE_ORDER
    }
    synthetic_rare_sequences: dict[int, DeterministicReplaySequence] = {}
    for class_index in rare_indices:
        candidates = [
            item for item in inventories["synthetic"] if class_index in item.classes
        ]
        if not candidates:
            raise ValueError(
                f"synthetic source has no guaranteed {DETECTOR_CLASSES[class_index]} examples"
            )
        synthetic_rare_sequences[class_index] = DeterministicReplaySequence(
            candidates,
            pair_weights,
            seed=seed,
            context=f"synthetic-guarantee-{DETECTOR_CLASSES[class_index]}",
        )
    if quota["synthetic"] < len(rare_indices):
        raise ValueError(
            "synthetic quota must have at least one slot for every guaranteed rare class"
        )

    selected: list[ImageLabelPair] = []
    block_contracts: list[dict[str, Any]] = []
    for block_index in range(logical_batches):
        block: list[ImageLabelPair] = []
        block.extend(sequences["webpii"].next() for _ in range(quota["webpii"]))
        block.extend(
            synthetic_rare_sequences[class_index].next()
            for class_index in rare_indices
        )
        block.extend(
            sequences["synthetic"].next()
            for _ in range(quota["synthetic"] - len(rare_indices))
        )
        block.extend(
            sequences["supplemental_ats"].next()
            for _ in range(quota["supplemental_ats"])
        )
        block.sort(
            key=lambda item: _stable_digest(
                seed,
                f"block:{block_index}:{item.source}:{item.relative_image}",
            )
        )
        _validate_logical_block(block, quota, rare_indices, block_index)
        selected.extend(block)
        block_contracts.append(
            {
                "index": block_index,
                "records": len(block),
                "source_counts": dict(sorted(Counter(item.source for item in block).items())),
                "guaranteed_rare_classes": [
                    DETECTOR_CLASSES[index] for index in rare_indices
                ],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    encoded_list = "".join(f"{item.image.resolve()}\n" for item in selected)
    list_path.write_text(encoded_list, encoding="utf-8")
    list_sha256 = sha256_file(list_path)

    selected_by_source = Counter(item.source for item in selected)
    selected_classes: Counter[int] = Counter()
    selected_unique: dict[str, set[str]] = {
        source: set() for source in SOURCE_ORDER
    }
    for item in selected:
        selected_classes.update(item.class_instances)
        selected_unique[item.source].add(item.relative_image)
    incomplete_sources = [
        source
        for source in SOURCE_ORDER
        if len(selected_unique[source]) != len(inventories[source])
    ]
    if incomplete_sources:
        raise ValueError(
            "logical replay plan does not cover every training image for: "
            + ", ".join(incomplete_sources)
        )

    configuration = {
        "seed": seed,
        "logical_batches": logical_batches,
        "batch_size": batch_size,
        "source_quota": dict(sorted(quota.items())),
        "rare_classes": [DETECTOR_CLASSES[index] for index in rare_indices],
        "image_weight": "max normalized inverse-sqrt original class frequency; negatives=1.0",
        "oversampling": "coverage-first then deterministic weighted replacement",
        "coverage": "every source image appears at least once per epoch",
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "deterministic-source-class-aware-replay-list",
        "classes": list(DETECTOR_CLASSES),
        "configuration": configuration,
        "configuration_sha256": _canonical_json_sha256(configuration),
        "consumer_contract": {
            "format": "ultralytics-image-list",
            "image_label_pairing": "images/<split>/<stem> -> labels/<split>/<stem>.txt",
            "quota_scope": "exact in constructed logical blocks and epoch multiset",
            "physical_batch_guarantee": False,
            "physical_batch_note": (
                "Ultralytics rect=False globally shuffles this multiset; use a custom "
                "BatchSampler to preserve block quotas in physical batches."
            ),
        },
        "output": {
            "train_list": str(list_path.resolve()),
            "records": len(selected),
            "sha256": list_sha256,
        },
        "original": {
            "images": total_images,
            "full_source_images_before_selection_holdout": sum(
                len(items) for items in full_inventories.values()
            ),
            "selection_holdout_images": sum(
                len(items) for items in excluded_inventories.values()
            ),
            "class_counts": {
                DETECTOR_CLASSES[index]: original_counts[index]
                for index in range(len(DETECTOR_CLASSES))
            },
            "normalized_inverse_sqrt_class_weights": {
                DETECTOR_CLASSES[index]: class_weights[index]
                for index in range(len(DETECTOR_CLASSES))
            },
        },
        "selected": {
            "source_counts": dict(sorted(selected_by_source.items())),
            "unique_images_by_source": {
                source: len(selected_unique[source]) for source in SOURCE_ORDER
            },
            "duplicate_rows_by_source": {
                source: selected_by_source[source] - len(selected_unique[source])
                for source in SOURCE_ORDER
            },
            "coverage_ratio_by_source": {
                source: len(selected_unique[source]) / len(inventories[source])
                for source in SOURCE_ORDER
            },
            "class_exposures": {
                DETECTOR_CLASSES[index]: selected_classes[index]
                for index in range(len(DETECTOR_CLASSES))
            },
        },
        "sources": {
            source: {
                **_source_provenance(roots[source], inventories[source]),
                "full_source_images": len(full_inventories[source]),
                "eligible_training_images": len(inventories[source]),
                "selection_holdout_images": len(excluded_inventories[source]),
                "selection_holdout_pair_inventory_sha256": _pair_inventory_sha256(
                    excluded_inventories[source]
                ),
            }
            for source in SOURCE_ORDER
        },
        "logical_block_contract": {
            "blocks": logical_batches,
            "records_per_block": batch_size,
            "source_quota": dict(sorted(quota.items())),
            "guaranteed_rare_classes": [
                DETECTOR_CLASSES[index] for index in rare_indices
            ],
            "all_blocks_validated": True,
            "block_contracts_sha256": _canonical_json_sha256(block_contracts),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path.resolve())
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest


def materialize_replay_training_stage(
    stage: Mapping[str, Any],
    *,
    seed: int = 1311,
    batch_size: int = 64,
    source_quota: Mapping[str, int] = DEFAULT_SOURCE_QUOTA,
    rare_classes: Sequence[str] = DEFAULT_RARE_CLASSES,
    runtime_selection: bool = False,
) -> dict[str, Any]:
    """Derive an immutable replay-backed dataset contract from a verified stage.

    ``stage_composed_dataset`` first verifies and copies the three source trees.
    This function then creates a deterministic replay list and new YAML/source/
    staging manifests without mutating those verified inputs. Rebuilding it in a
    resume container produces byte-identical files at the same per-run path.
    """

    root_value = stage.get("root")
    if not isinstance(root_value, str) or not root_value:
        raise ValueError("verified detector stage is missing its root")
    root = Path(root_value)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("verified detector stage root is missing or unsafe")
    root = root.resolve()
    original_dataset = _read_json_or_yaml_mapping(
        Path(str(stage.get("dataset_yaml", ""))), "verified staged dataset YAML"
    )
    original_source_path = Path(str(stage.get("source_manifest", "")))
    original_stage_path = Path(str(stage.get("staging_manifest", "")))
    original_source = _read_json_mapping(
        original_source_path, "verified staged source manifest"
    )
    original_stage = _read_json_mapping(
        original_stage_path, "verified staged staging manifest"
    )
    if original_stage.get("schema_version") != 1:
        raise ValueError("verified detector staging manifest schema is unsupported")

    roots = {source: root / source for source in SOURCE_ORDER}
    runtime_selection_contract = None
    eligible_images_by_source = None
    if runtime_selection:
        from training.visual.build_runtime_selection_stage import (
            materialize_runtime_selection_stage,
        )

        runtime_selection_contract = materialize_runtime_selection_stage(
            webpii_root=roots["webpii"],
            synthetic_root=roots["synthetic"],
            output_dir=root / "selection",
            seed=seed,
        )
        eligible_path = Path(
            runtime_selection_contract["webpii_eligible_images"]
        )
        eligible_images_by_source = {
            "webpii": {
                Path(line).resolve()
                for line in eligible_path.read_text(encoding="utf-8").splitlines()
                if line
            }
        }
    replay = build_replay_train_list(
        roots,
        root / "replay",
        seed=seed,
        batch_size=batch_size,
        source_quota=source_quota,
        rare_classes=rare_classes,
        eligible_images_by_source=eligible_images_by_source,
        force=True,
    )
    replay_list = Path(replay["output"]["train_list"])
    replay_manifest = Path(replay["manifest_path"])

    dataset_path = root / REPLAY_DATASET_NAME
    dataset_lines = [
        "path: /",
        f"train: {replay_list}",
        f"val: {root / 'synthetic/images/validation'}",
        f"test: {root / 'webpii/images/test'}",
        f"nc: {len(DETECTOR_CLASSES)}",
        "names:",
        *[f"  {index}: {name}" for index, name in enumerate(DETECTOR_CLASSES)],
        "",
    ]
    _atomic_write_text(dataset_path, "\n".join(dataset_lines))

    replay_contract = {
        "schema_version": SCHEMA_VERSION,
        "kind": "deterministic-source-class-aware-replay-list",
        "manifest": str(replay_manifest.resolve()),
        "manifest_sha256": sha256_file(replay_manifest),
        "train_list": str(replay_list.resolve()),
        "train_list_sha256": sha256_file(replay_list),
        "records": replay["output"]["records"],
        "configuration": deepcopy(replay["configuration"]),
        "configuration_sha256": replay["configuration_sha256"],
        "physical_batch_guarantee": False,
        "source_stage_contract_sha256": original_stage.get(
            "source_contract_sha256"
        ),
        "runtime_selection_contract_sha256": (
            runtime_selection_contract["contract_file_sha256"]
            if runtime_selection_contract is not None
            else None
        ),
    }
    source_document = deepcopy(original_source)
    source_document["dataset_yaml"] = str(dataset_path.resolve())
    source_document["dataset_yaml_sha256"] = sha256_file(dataset_path)
    source_document["training_replay"] = replay_contract
    if runtime_selection_contract is not None:
        source_document["selection_data"] = (
            "calibrated-runtime-synthetic-and-webpii-train-guards"
        )
        source_document["runtime_selection"] = {
            "schema_version": 1,
            "kind": "calibrated-runtime-selection-contract",
            "contract": runtime_selection_contract["contract_path"],
            "contract_sha256": runtime_selection_contract[
                "contract_file_sha256"
            ],
            "published_test_used": False,
            "webpii_train_eligible_records": runtime_selection_contract[
                "training_eligibility"
            ]["records"],
        }
    source_path = root / REPLAY_SOURCE_MANIFEST_NAME
    _atomic_write_text(
        source_path,
        json.dumps(source_document, indent=2, sort_keys=True) + "\n",
    )

    stage_document = deepcopy(original_stage)
    stage_document["local_dataset_yaml"] = str(dataset_path.resolve())
    stage_document["local_dataset_yaml_sha256"] = sha256_file(dataset_path)
    stage_document["local_source_manifest"] = str(source_path.resolve())
    stage_document["local_source_manifest_sha256"] = sha256_file(source_path)
    stage_document["training_sources"] = ["replay/train-replay.txt"]
    stage_document["training_replay"] = replay_contract
    if runtime_selection_contract is not None:
        stage_document["runtime_selection"] = source_document[
            "runtime_selection"
        ]
    stage_path = root / REPLAY_STAGING_MANIFEST_NAME
    _atomic_write_text(
        stage_path,
        json.dumps(stage_document, indent=2, sort_keys=True) + "\n",
    )

    # Preserve unrelated stage metadata for callers, but make every consumed
    # path point at the replay-derived, hash-bound contract.
    result = dict(stage)
    result.update(
        {
            "mode": "verified-local-stage-with-deterministic-replay",
            "dataset_yaml": str(dataset_path.resolve()),
            "source_manifest": str(source_path.resolve()),
            "staging_manifest": str(stage_path.resolve()),
            "replay_manifest": str(replay_manifest.resolve()),
            "replay_manifest_sha256": sha256_file(replay_manifest),
            "replay_train_list": str(replay_list.resolve()),
            "replay_train_list_sha256": sha256_file(replay_list),
            "replay": replay,
            "runtime_selection_contract": (
                runtime_selection_contract["contract_path"]
                if runtime_selection_contract is not None
                else None
            ),
            "runtime_selection_contract_sha256": (
                runtime_selection_contract["contract_file_sha256"]
                if runtime_selection_contract is not None
                else None
            ),
            "runtime_selection": runtime_selection_contract,
        }
    )
    # Catch accidental loss of the selection/test split while deriving YAML.
    if "val" not in original_dataset or "test" not in original_dataset:
        raise ValueError("verified staged dataset lacks validation or test splits")
    return result


def _validate_source_roots(source_roots: Mapping[str, Path]) -> dict[str, Path]:
    if set(source_roots) != set(SOURCE_ORDER):
        raise ValueError(
            "source_roots must contain exactly: " + ", ".join(SOURCE_ORDER)
        )
    roots: dict[str, Path] = {}
    for source in SOURCE_ORDER:
        root = Path(source_roots[source])
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f"{source} root is missing or unsafe: {root}")
        roots[source] = root.resolve()
    return roots


def _validate_quota(source_quota: Mapping[str, int], batch_size: int) -> dict[str, int]:
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if set(source_quota) != set(SOURCE_ORDER):
        raise ValueError("source_quota has an unexpected source")
    quota = {source: source_quota[source] for source in SOURCE_ORDER}
    if any(type(value) is not int or value < 0 for value in quota.values()):
        raise ValueError("source quotas must be non-negative integers")
    if sum(quota.values()) != batch_size:
        raise ValueError("source quotas must sum to batch_size")
    return quota


def _validate_rare_classes(rare_classes: Sequence[str]) -> tuple[int, ...]:
    names = tuple(rare_classes)
    if not names or len(names) != len(set(names)):
        raise ValueError("rare_classes must be a non-empty unique sequence")
    unknown = [name for name in names if name not in DETECTOR_CLASSES]
    if unknown:
        raise ValueError("unknown rare classes: " + ", ".join(unknown))
    return tuple(DETECTOR_CLASSES.index(name) for name in names)


def _inventory_source(source: str, root: Path) -> list[ImageLabelPair]:
    image_root = root / "images/train"
    label_root = root / "labels/train"
    for directory, description in ((image_root, "image"), (label_root, "label")):
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"{source} {description} directory is missing or unsafe")
        symlinks = [path for path in directory.rglob("*") if path.is_symlink()]
        if symlinks:
            raise ValueError(f"{source} {description} tree contains a symbolic link")
    source_manifest = _read_source_manifest(source, root)
    images = sorted(
        path
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"{source} has no training images")
    pairs: list[ImageLabelPair] = []
    seen_labels: set[Path] = set()
    for image in images:
        if image.is_symlink() or "\n" in str(image) or "\0" in str(image):
            raise ValueError(f"{source} contains an unsafe image path: {image}")
        relative_image_path = image.relative_to(image_root)
        label = label_root / relative_image_path.with_suffix(".txt")
        if label.is_symlink() or not label.is_file():
            raise ValueError(f"{source} image lacks a paired label: {relative_image_path}")
        if label in seen_labels:
            raise ValueError(f"{source} label is paired with multiple images: {label}")
        seen_labels.add(label)
        class_instances = _parse_yolo_label(label, source)
        pairs.append(
            ImageLabelPair(
                source=source,
                image=image.resolve(),
                label=label.resolve(),
                relative_image=relative_image_path.as_posix(),
                relative_label=label.relative_to(label_root).as_posix(),
                classes=tuple(sorted(set(class_instances))),
                class_instances=class_instances,
                label_sha256=sha256_file(label),
                image_bytes=image.stat().st_size,
            )
        )
    extra_labels = {
        path.resolve()
        for path in label_root.rglob("*.txt")
        if path.is_file() and not path.is_symlink()
    } - {path.resolve() for path in seen_labels}
    if extra_labels:
        raise ValueError(f"{source} contains {len(extra_labels)} unpaired label files")
    split = source_manifest.get("splits", {}).get("train")
    if isinstance(split, dict):
        if split.get("records") != len(pairs):
            raise ValueError(f"{source} manifest train record count differs from files")
        expected_counts = split.get("class_counts")
        measured_counts = Counter(
            DETECTOR_CLASSES[index]
            for item in pairs
            for index in item.class_instances
        )
        if isinstance(expected_counts, dict) and {
            str(name): int(count) for name, count in expected_counts.items()
        } != dict(sorted(measured_counts.items())):
            raise ValueError(f"{source} manifest train class counts differ from labels")
    return pairs


def _read_source_manifest(source: str, root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{source} manifest is missing or unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{source} manifest is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{source} manifest must be an object")
    if value.get("classes") != list(DETECTOR_CLASSES):
        raise ValueError(f"{source} manifest class order differs from detector contract")
    return value


def _parse_yolo_label(path: Path, source: str) -> tuple[int, ...]:
    classes: list[int] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"{source} label is not valid UTF-8: {path}") from exc
    for line_number, line in enumerate(lines, 1):
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{source} label {path}:{line_number} is not YOLO format")
        try:
            class_index = int(parts[0])
            coordinates = [float(value) for value in parts[1:]]
        except ValueError as exc:
            raise ValueError(
                f"{source} label {path}:{line_number} is not numeric"
            ) from exc
        if not 0 <= class_index < len(DETECTOR_CLASSES):
            raise ValueError(f"{source} label {path}:{line_number} has an unknown class")
        center_x, center_y, width, height = coordinates
        if (
            any(not math.isfinite(value) for value in coordinates)
            or not 0 <= center_x <= 1
            or not 0 <= center_y <= 1
            or not 0 < width <= 1
            or not 0 < height <= 1
        ):
            raise ValueError(
                f"{source} label {path}:{line_number} has invalid normalized coordinates"
            )
        classes.append(class_index)
    # Preserve repeated instances for the class-frequency contract. The caller
    # separately derives the unique per-image class set used for replay weight.
    return tuple(sorted(classes))


def _read_json_mapping(path: Path, context: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{context} is missing or unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{context} must contain an object")
    return value


def _read_json_or_yaml_mapping(path: Path, context: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{context} is missing or unsafe")
    try:
        import yaml

        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"{context} is not valid UTF-8 YAML") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{context} must contain a mapping")
    return value


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _validate_logical_block(
    block: Sequence[ImageLabelPair],
    quota: Mapping[str, int],
    rare_indices: Iterable[int],
    block_index: int,
) -> None:
    actual = Counter(item.source for item in block)
    if actual != Counter(quota):
        raise RuntimeError(f"logical replay block {block_index} violates source quota")
    synthetic_classes = {
        class_index
        for item in block
        if item.source == "synthetic"
        for class_index in item.classes
    }
    missing = [
        DETECTOR_CLASSES[index]
        for index in rare_indices
        if index not in synthetic_classes
    ]
    if missing:
        raise RuntimeError(
            f"logical replay block {block_index} lacks guaranteed classes: {missing}"
        )


def _source_provenance(
    root: Path, pairs: Sequence[ImageLabelPair]
) -> dict[str, Any]:
    digest = hashlib.sha256()
    for item in pairs:
        digest.update(item.relative_image.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item.image_bytes).encode("ascii"))
        digest.update(b"\0")
        digest.update(item.relative_label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.label_sha256.encode("ascii"))
        digest.update(b"\n")
    manifest_path = root / "manifest.json"
    manifest = None
    if manifest_path.is_file() and not manifest_path.is_symlink():
        manifest = {
            "path": str(manifest_path.resolve()),
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
        }
    return {
        "root": str(root),
        "images": len(pairs),
        "pair_inventory_contract": (
            "relative-image\\0image-bytes\\0relative-label\\0label-sha256\\n"
        ),
        "pair_inventory_sha256": digest.hexdigest(),
        "manifest": manifest,
    }


def _pair_inventory_sha256(pairs: Sequence[ImageLabelPair]) -> str:
    digest = hashlib.sha256()
    for item in sorted(pairs, key=lambda value: value.relative_image):
        digest.update(item.relative_image.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item.image_bytes).encode("ascii"))
        digest.update(b"\0")
        digest.update(item.relative_label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.label_sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _weighted_rank(*, seed: int, context: str, identity: str, weight: float) -> float:
    unit = _unit_interval(seed, f"{context}:{identity}")
    return -math.log(unit) / weight


def _unit_interval(seed: int, context: str) -> float:
    digest = _stable_digest(seed, context)
    integer = int.from_bytes(digest[:8], "big")
    return (integer + 1) / ((1 << 64) + 1)


def _stable_digest(seed: int, context: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{context}".encode("utf-8")).digest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic source/class-aware Ultralytics replay list"
    )
    parser.add_argument("--webpii-root", type=Path, required=True)
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--supplemental-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1311)
    parser.add_argument("--logical-batches", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--webpii-per-batch", type=int, default=40)
    parser.add_argument("--synthetic-per-batch", type=int, default=22)
    parser.add_argument("--supplemental-per-batch", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_replay_train_list(
        {
            "webpii": args.webpii_root,
            "synthetic": args.synthetic_root,
            "supplemental_ats": args.supplemental_root,
        },
        args.output_dir,
        seed=args.seed,
        logical_batches=args.logical_batches,
        batch_size=args.batch_size,
        source_quota={
            "webpii": args.webpii_per_batch,
            "synthetic": args.synthetic_per_batch,
            "supplemental_ats": args.supplemental_per_batch,
        },
        force=args.force,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
