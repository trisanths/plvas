from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from training.schema import sha256_file
from training.visual.prepare_webpii import (
    DETECTOR_CLASSES,
    WEBPII_ID,
    WEBPII_REVISION,
)


SCHEMA_VERSION = 1
DEFAULT_SEED = 1311
DEFAULT_HELDOUT_FRACTION = 0.10
CALIBRATION_IMAGES_NAME = "calibration-images.txt"
GUARD_IMAGES_NAME = "guard-images.txt"
ELIGIBLE_IMAGES_NAME = "eligible-images.txt"
CALIBRATION_DATASET_NAME = "dataset-webpii-calibration.yaml"
GUARD_DATASET_NAME = "dataset-webpii-guard.yaml"
ELIGIBLE_DATASET_NAME = "dataset-webpii-eligible.yaml"
MANIFEST_NAME = "webpii-guard-manifest.json"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

GroupKey = tuple[str, str, str]
StratumKey = tuple[str, str]


@dataclass(frozen=True)
class WebPIIRecord:
    group: GroupKey
    stratum: StratumKey
    variant: str
    image: Path
    label: Path
    relative_image: str
    relative_label: str
    classes: tuple[int, ...]
    image_bytes: int
    label_sha256: str


def build_webpii_guard_split(
    webpii_root: Path,
    output_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    heldout_fraction: float = DEFAULT_HELDOUT_FRACTION,
    force: bool = False,
) -> dict[str, Any]:
    """Create calibration, guard, and eligible sets from WebPII published train.

    The source tree is never mutated. Selection happens at
    ``(company, page_type, source_id)`` group granularity, so every variant of
    one source identity is assigned together. About ten percent of groups are
    held out, then split into disjoint calibration and checkpoint-guard halves.
    The returned paths are directly consumable by replay-stage materialization
    and Ultralytics validation.
    """

    root = _validate_root(webpii_root)
    output_dir = Path(output_dir).resolve()
    seed = _validate_seed(seed)
    heldout_fraction = _validate_fraction(heldout_fraction)
    manifest_path = root / "manifest.json"
    source_manifest = _read_json_object(manifest_path, "WebPII manifest")
    train_contract, records_path = _validate_source_manifest(
        root, source_manifest, manifest_path
    )
    records = _read_records(root, records_path, train_contract)
    _validate_complete_pair_inventory(root, records)

    grouped: dict[GroupKey, list[WebPIIRecord]] = defaultdict(list)
    for record in records:
        grouped[record.group].append(record)
    heldout_groups, heldout_allocation = _select_heldout_groups(
        grouped, seed=seed, heldout_fraction=heldout_fraction
    )
    all_groups = set(grouped)
    calibration_groups, guard_groups, calibration_allocation = _split_heldout_groups(
        grouped,
        heldout_groups,
        heldout_allocation,
        seed=seed,
    )
    eligible_groups = all_groups - heldout_groups
    _validate_group_partition(
        all_groups,
        {
            "calibration": calibration_groups,
            "guard": guard_groups,
            "eligible": eligible_groups,
        },
    )

    calibration_records = sorted(
        (record for record in records if record.group in calibration_groups),
        key=lambda item: item.relative_image.encode("utf-8"),
    )
    guard_records = sorted(
        (record for record in records if record.group in guard_groups),
        key=lambda item: item.relative_image.encode("utf-8"),
    )
    eligible_records = sorted(
        (record for record in records if record.group in eligible_groups),
        key=lambda item: item.relative_image.encode("utf-8"),
    )
    if not calibration_records or not guard_records or not eligible_records:
        raise ValueError("WebPII guard construction requires non-empty split outputs")
    if len(calibration_records) + len(guard_records) + len(eligible_records) != len(
        records
    ):
        raise RuntimeError("WebPII guard record assignment is not exhaustive")

    if output_dir == root or root in output_dir.parents:
        raise ValueError("guard output may not be inside the WebPII source tree")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.is_symlink():
        raise ValueError("guard output directory may not be a symbolic link")
    if output_dir.exists() and (not output_dir.is_dir() or not force):
        if output_dir.is_dir():
            raise FileExistsError(f"guard output already exists: {output_dir}")
        raise ValueError(f"guard output path is not a directory: {output_dir}")

    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.partial-", dir=str(output_dir.parent)
        )
    )
    try:
        calibration_path = temporary / CALIBRATION_IMAGES_NAME
        guard_path = temporary / GUARD_IMAGES_NAME
        eligible_path = temporary / ELIGIBLE_IMAGES_NAME
        calibration_dataset_path = temporary / CALIBRATION_DATASET_NAME
        guard_dataset_path = temporary / GUARD_DATASET_NAME
        eligible_dataset_path = temporary / ELIGIBLE_DATASET_NAME
        persisted_manifest_path = temporary / MANIFEST_NAME

        calibration_path.write_text(
            _encode_image_list(calibration_records), encoding="utf-8"
        )
        guard_path.write_text(_encode_image_list(guard_records), encoding="utf-8")
        eligible_path.write_text(_encode_image_list(eligible_records), encoding="utf-8")
        calibration_dataset_path.write_text(
            _dataset_yaml(output_dir, validation=CALIBRATION_IMAGES_NAME),
            encoding="utf-8",
        )
        guard_dataset_path.write_text(
            _dataset_yaml(output_dir, validation=GUARD_IMAGES_NAME),
            encoding="utf-8",
        )
        eligible_dataset_path.write_text(
            _dataset_yaml(output_dir, validation=None),
            encoding="utf-8",
        )

        total_stats = _statistics(records)
        calibration_stats = _statistics(calibration_records)
        guard_stats = _statistics(guard_records)
        eligible_stats = _statistics(eligible_records)
        _validate_partition_statistics(
            total_stats,
            calibration_stats,
            guard_stats,
            eligible_stats,
        )
        configuration = {
            "seed": seed,
            "heldout_fraction": heldout_fraction,
            "selection_unit": ["company", "page_type", "source_id"],
            "stratification": ["company", "page_type"],
            "allocation": (
                "global round-half-up 10-percent heldout target; Hamilton "
                "largest-remainder quotas per company+page_type; deterministic "
                "class-support repair; heldout groups divided into seeded, "
                "stratified calibration and guard halves"
            ),
        }
        published_test = source_manifest["splits"]["test"]
        persisted: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "kind": "webpii-train-derived-identity-disjoint-guard-split",
            "classes": list(DETECTOR_CLASSES),
            "configuration": configuration,
            "configuration_sha256": _canonical_sha256(configuration),
            "source": {
                "dataset": source_manifest["dataset"],
                "revision": source_manifest["revision"],
                "root": str(root),
                "manifest": {
                    "path": str(manifest_path),
                    "bytes": manifest_path.stat().st_size,
                    "sha256": sha256_file(manifest_path),
                },
                "train_records": {
                    "path": str(records_path),
                    "records": len(records),
                    "bytes": records_path.stat().st_size,
                    "sha256": sha256_file(records_path),
                },
                "pair_inventory_sha256": _pair_inventory_sha256(records),
                "pair_inventory_contract": (
                    "relative-image\\0image-bytes\\0relative-label\\0label-sha256\\n"
                ),
            },
            "published_test": {
                "excluded": True,
                "used_for_selection": False,
                "emitted_in_dataset_yaml": False,
                "declared_records": published_test.get("records"),
            },
            "selection": {
                "groups": {
                    "total": len(all_groups),
                    "heldout": len(heldout_groups),
                    "calibration": len(calibration_groups),
                    "guard": len(guard_groups),
                    "eligible": len(eligible_groups),
                    "pairwise_overlap": {
                        "calibration_guard": 0,
                        "calibration_eligible": 0,
                        "guard_eligible": 0,
                    },
                },
                "records": {
                    "total": len(records),
                    "heldout": len(calibration_records) + len(guard_records),
                    "calibration": len(calibration_records),
                    "guard": len(guard_records),
                    "eligible": len(eligible_records),
                    "pairwise_overlap": {
                        "calibration_guard": 0,
                        "calibration_eligible": 0,
                        "guard_eligible": 0,
                    },
                },
                "actual_heldout_group_fraction": len(heldout_groups) / len(all_groups),
                "actual_heldout_record_fraction": (
                    len(calibration_records) + len(guard_records)
                )
                / len(records),
                "class_coverage": _class_coverage_contract(
                    grouped,
                    calibration_groups,
                    guard_groups,
                ),
                "strata": _allocation_statistics(
                    heldout_allocation,
                    calibration_allocation,
                    grouped,
                    calibration_groups,
                    guard_groups,
                ),
            },
            "statistics": {
                "all": total_stats,
                "calibration": calibration_stats,
                "guard": guard_stats,
                "eligible": eligible_stats,
            },
            "invariants": {
                "identity_disjoint": True,
                "group_overlap": 0,
                "record_overlap": 0,
                "groups_exhaustive": True,
                "records_exhaustive": True,
                "all_variants_assigned_with_identity": True,
                "image_label_regular_file_pairs_verified": True,
                "published_test_excluded": True,
                "raw_pii_values_persisted": False,
            },
            "outputs": {
                "calibration_images": _output_contract(
                    output_dir / CALIBRATION_IMAGES_NAME,
                    calibration_path,
                    len(calibration_records),
                ),
                "guard_images": _output_contract(
                    output_dir / GUARD_IMAGES_NAME,
                    guard_path,
                    len(guard_records),
                ),
                "eligible_images": _output_contract(
                    output_dir / ELIGIBLE_IMAGES_NAME,
                    eligible_path,
                    len(eligible_records),
                ),
                "calibration_dataset_yaml": _output_contract(
                    output_dir / CALIBRATION_DATASET_NAME,
                    calibration_dataset_path,
                    None,
                ),
                "guard_dataset_yaml": _output_contract(
                    output_dir / GUARD_DATASET_NAME,
                    guard_dataset_path,
                    None,
                ),
                "eligible_dataset_yaml": _output_contract(
                    output_dir / ELIGIBLE_DATASET_NAME,
                    eligible_dataset_path,
                    None,
                ),
            },
            "values_persisted": False,
        }
        persisted_manifest_path.write_text(
            json.dumps(persisted, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _publish_directory(temporary, output_dir, force=force)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    final_manifest_path = output_dir / MANIFEST_NAME
    result = dict(persisted)
    result.update(
        {
            "root": str(output_dir),
            "manifest_path": str(final_manifest_path),
            "manifest_sha256": sha256_file(final_manifest_path),
            "calibration_images": str(output_dir / CALIBRATION_IMAGES_NAME),
            "calibration_images_sha256": sha256_file(
                output_dir / CALIBRATION_IMAGES_NAME
            ),
            "guard_images": str(output_dir / GUARD_IMAGES_NAME),
            "guard_images_sha256": sha256_file(output_dir / GUARD_IMAGES_NAME),
            "eligible_images": str(output_dir / ELIGIBLE_IMAGES_NAME),
            "eligible_images_sha256": sha256_file(output_dir / ELIGIBLE_IMAGES_NAME),
            "calibration_dataset_yaml": str(output_dir / CALIBRATION_DATASET_NAME),
            "calibration_dataset_yaml_sha256": sha256_file(
                output_dir / CALIBRATION_DATASET_NAME
            ),
            "guard_dataset_yaml": str(output_dir / GUARD_DATASET_NAME),
            "guard_dataset_yaml_sha256": sha256_file(output_dir / GUARD_DATASET_NAME),
            "eligible_dataset_yaml": str(output_dir / ELIGIBLE_DATASET_NAME),
            "eligible_dataset_yaml_sha256": sha256_file(
                output_dir / ELIGIBLE_DATASET_NAME
            ),
        }
    )
    return result


def _validate_root(value: Path) -> Path:
    root = Path(value)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"WebPII root is missing or unsafe: {root}")
    return root.resolve()


def _validate_seed(value: int) -> int:
    if type(value) is not int:
        raise ValueError("WebPII guard seed must be an integer")
    return value


def _validate_fraction(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("WebPII guard fraction must be numeric")
    fraction = float(value)
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("WebPII guard fraction must be strictly between zero and one")
    return fraction


def _read_json_object(path: Path, context: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{context} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{context} must contain a JSON object")
    return value


def _validate_source_manifest(
    root: Path,
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> tuple[dict[str, Any], Path]:
    if manifest.get("schema_version") != 1:
        raise ValueError("WebPII manifest schema is unsupported")
    if (
        manifest.get("dataset") != WEBPII_ID
        or manifest.get("revision") != WEBPII_REVISION
    ):
        raise ValueError(
            "WebPII manifest is not pinned to the approved source revision"
        )
    if manifest.get("classes") != list(DETECTOR_CLASSES):
        raise ValueError("WebPII manifest class order differs from detector contract")
    if manifest.get("published_splits_preserved") is not True:
        raise ValueError("WebPII manifest does not preserve published splits")
    if manifest.get("test_used_for_checkpoint_selection") is not False:
        raise ValueError("WebPII published test must remain excluded from selection")
    if manifest.get("values_persisted") is not False:
        raise ValueError(
            "WebPII source manifest must attest that values are not persisted"
        )
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "test"}:
        raise ValueError("WebPII manifest must declare exactly train and test splits")
    train = splits.get("train")
    test = splits.get("test")
    if not isinstance(train, dict) or not isinstance(test, dict):
        raise ValueError("WebPII split contracts must be objects")
    if type(train.get("records")) is not int or train["records"] <= 1:
        raise ValueError("WebPII train split must declare at least two records")
    if type(test.get("records")) is not int or test["records"] < 0:
        raise ValueError("WebPII test split record count is invalid")
    records_value = train.get("records_path")
    if not isinstance(records_value, str) or not records_value:
        raise ValueError("WebPII train records path is missing")
    records_path = _resolve_source_relative(root, records_value, "train records")
    expected = root / "records/train.jsonl"
    if records_path != expected.resolve():
        raise ValueError("WebPII train records must be records/train.jsonl")
    expected_hash = train.get("records_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("WebPII train records SHA-256 is missing")
    if records_path.is_symlink() or not records_path.is_file():
        raise ValueError("WebPII train records are missing or unsafe")
    if sha256_file(records_path) != expected_hash:
        raise ValueError("WebPII train records SHA-256 mismatch")
    if manifest_path != root / "manifest.json":
        raise RuntimeError("WebPII manifest path changed during validation")
    return dict(train), records_path


def _resolve_source_relative(root: Path, value: str, context: str) -> Path:
    if "\0" in value or "\n" in value or "\r" in value:
        raise ValueError(f"WebPII {context} path contains control characters")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"WebPII {context} path must remain source-relative")
    resolved = (root / relative).resolve()
    if root not in resolved.parents:
        raise ValueError(f"WebPII {context} path escapes the source root")
    return resolved


def _read_records(
    root: Path,
    records_path: Path,
    train_contract: Mapping[str, Any],
) -> list[WebPIIRecord]:
    try:
        lines = records_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError("WebPII train records are not valid UTF-8") from exc
    if any(not line.strip() for line in lines):
        raise ValueError("WebPII train records contain a blank JSONL row")
    if len(lines) != train_contract["records"]:
        raise ValueError("WebPII train records count differs from manifest")

    records: list[WebPIIRecord] = []
    seen_images: set[str] = set()
    seen_labels: set[str] = set()
    seen_variants: set[tuple[GroupKey, str]] = set()
    for index, encoded in enumerate(lines, 1):
        try:
            row = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise ValueError(f"WebPII train record {index} is not valid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"WebPII train record {index} must be an object")
        if row.get("published_split") != "train":
            raise ValueError(f"WebPII train record {index} is not published train data")
        company = _required_string(row, "company", index)
        page_type = _required_string(row, "page_type", index)
        source_id = _required_string(row, "source_id", index)
        variant = _required_string(row, "variant", index)
        group = (company, page_type, source_id)
        group_variant = (group, variant)
        if group_variant in seen_variants:
            raise ValueError(f"WebPII train record {index} duplicates a group variant")
        seen_variants.add(group_variant)

        image_value = _required_string(row, "image", index)
        label_value = _required_string(row, "label_file", index)
        image = _resolve_source_relative(root, image_value, f"record {index} image")
        label = _resolve_source_relative(root, label_value, f"record {index} label")
        image_relative = Path(image_value)
        label_relative = Path(label_value)
        if image_relative.parts[:2] != ("images", "train"):
            raise ValueError(
                f"WebPII train record {index} image is outside images/train"
            )
        if label_relative.parts[:2] != ("labels", "train"):
            raise ValueError(
                f"WebPII train record {index} label is outside labels/train"
            )
        if image_relative.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(
                f"WebPII train record {index} has an unsupported image type"
            )
        if label_relative.suffix.lower() != ".txt":
            raise ValueError(f"WebPII train record {index} label is not a text file")
        if image_relative.relative_to("images/train").with_suffix(
            ""
        ) != label_relative.relative_to("labels/train").with_suffix(""):
            raise ValueError(f"WebPII train record {index} image/label stems differ")
        _require_regular_file(root, image, f"record {index} image")
        _require_regular_file(root, label, f"record {index} label")
        if image.stat().st_size <= 0:
            raise ValueError(f"WebPII train record {index} image is empty")
        if image_value in seen_images or label_value in seen_labels:
            raise ValueError(f"WebPII train record {index} repeats an image or label")
        seen_images.add(image_value)
        seen_labels.add(label_value)

        classes = _parse_yolo_label(label, index)
        annotations = row.get("annotations")
        if not isinstance(annotations, list):
            raise ValueError(f"WebPII train record {index} annotations must be a list")
        annotation_classes: list[int] = []
        for annotation in annotations:
            if not isinstance(annotation, dict):
                raise ValueError(
                    f"WebPII train record {index} annotation must be an object"
                )
            class_name = annotation.get("class")
            if class_name not in DETECTOR_CLASSES:
                raise ValueError(
                    f"WebPII train record {index} annotation class is invalid"
                )
            annotation_classes.append(DETECTOR_CLASSES.index(class_name))
        if Counter(annotation_classes) != Counter(classes):
            raise ValueError(
                f"WebPII train record {index} annotations differ from its label"
            )
        records.append(
            WebPIIRecord(
                group=group,
                stratum=(company, page_type),
                variant=variant,
                image=image,
                label=label,
                relative_image=image_relative.as_posix(),
                relative_label=label_relative.as_posix(),
                classes=classes,
                image_bytes=image.stat().st_size,
                label_sha256=sha256_file(label),
            )
        )

    _validate_declared_statistics(train_contract, records)
    return records


def _required_string(row: Mapping[str, Any], key: str, index: int) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError(f"WebPII train record {index} has an invalid {key}")
    return value


def _require_regular_file(root: Path, path: Path, context: str) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"WebPII {context} escapes the source root") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"WebPII {context} contains a symbolic link")
    if not path.is_file():
        raise ValueError(f"WebPII {context} is missing or not a regular file")


def _parse_yolo_label(path: Path, record_index: int) -> tuple[int, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"WebPII train record {record_index} label is not valid UTF-8"
        ) from exc
    classes: list[int] = []
    for line_number, line in enumerate(lines, 1):
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(
                f"WebPII label {path}:{line_number} is not YOLO detection format"
            )
        try:
            class_index = int(parts[0])
            coordinates = tuple(float(value) for value in parts[1:])
        except ValueError as exc:
            raise ValueError(
                f"WebPII label {path}:{line_number} contains a non-numeric value"
            ) from exc
        if not 0 <= class_index < len(DETECTOR_CLASSES):
            raise ValueError(f"WebPII label {path}:{line_number} has an unknown class")
        center_x, center_y, width, height = coordinates
        if (
            any(not math.isfinite(value) for value in coordinates)
            or not 0.0 <= center_x <= 1.0
            or not 0.0 <= center_y <= 1.0
            or not 0.0 < width <= 1.0
            or not 0.0 < height <= 1.0
        ):
            raise ValueError(
                f"WebPII label {path}:{line_number} has invalid normalized coordinates"
            )
        classes.append(class_index)
    return tuple(classes)


def _validate_complete_pair_inventory(
    root: Path, records: Sequence[WebPIIRecord]
) -> None:
    image_root = root / "images/train"
    label_root = root / "labels/train"
    for directory, context in ((image_root, "image"), (label_root, "label")):
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"WebPII train {context} directory is missing or unsafe")
    for directory, context in ((image_root, "image"), (label_root, "label")):
        if any(path.is_symlink() for path in directory.rglob("*")):
            raise ValueError(f"WebPII train {context} tree contains a symbolic link")
    images = {
        path.resolve()
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    labels = {path.resolve() for path in label_root.rglob("*.txt") if path.is_file()}
    recorded_images = {record.image for record in records}
    recorded_labels = {record.label for record in records}
    if images != recorded_images:
        raise ValueError("WebPII train records are not exhaustive over image files")
    if labels != recorded_labels:
        raise ValueError("WebPII train records are not exhaustive over label files")


def _validate_declared_statistics(
    train_contract: Mapping[str, Any], records: Sequence[WebPIIRecord]
) -> None:
    measured_class_counts = Counter(
        DETECTOR_CLASSES[class_index]
        for record in records
        for class_index in record.classes
    )
    measured_companies = Counter(record.group[0] for record in records)
    measured_page_types = Counter(record.group[1] for record in records)
    expected: tuple[tuple[str, Mapping[str, int]], ...] = (
        ("class_counts", measured_class_counts),
        ("companies", measured_companies),
        ("page_types", measured_page_types),
    )
    for field, measured in expected:
        declared = train_contract.get(field)
        if not isinstance(declared, dict):
            raise ValueError(f"WebPII train manifest is missing {field}")
        normalized = {str(key): int(value) for key, value in declared.items()}
        if normalized != dict(sorted(measured.items())):
            raise ValueError(f"WebPII train manifest {field} differs from records")
    annotations = sum(len(record.classes) for record in records)
    negative_images = sum(not record.classes for record in records)
    if train_contract.get("annotations") != annotations:
        raise ValueError("WebPII train manifest annotation count differs from labels")
    if train_contract.get("negative_images", 0) != negative_images:
        raise ValueError(
            "WebPII train manifest negative-image count differs from labels"
        )


def _select_heldout_groups(
    grouped: Mapping[GroupKey, Sequence[WebPIIRecord]],
    *,
    seed: int,
    heldout_fraction: float,
) -> tuple[set[GroupKey], dict[StratumKey, int]]:
    if len(grouped) < 3:
        raise ValueError(
            "WebPII guard construction requires at least three identity groups"
        )
    strata = _stratify_groups(grouped)
    total_groups = len(grouped)
    target = int(math.floor(total_groups * heldout_fraction + 0.5))
    target = min(total_groups - 1, max(2, target))
    allocation = _hamilton_allocation(
        {stratum: len(groups) for stratum, groups in strata.items()},
        target=target,
        seed=seed,
        domain="heldout-stratum",
    )
    heldout: set[GroupKey] = set()
    for stratum in sorted(strata, key=lambda value: _canonical_json(value)):
        ranked_groups = sorted(
            strata[stratum],
            key=lambda group: _stable_digest(seed, "heldout-group", group),
        )
        heldout.update(ranked_groups[: allocation[stratum]])
    heldout = _repair_selected_class_support(
        grouped,
        heldout,
        allocation,
        seed=seed,
    )
    if len(heldout) != target:
        raise RuntimeError(
            "WebPII stratified heldout selection produced the wrong size"
        )
    return heldout, allocation


def _split_heldout_groups(
    grouped: Mapping[GroupKey, Sequence[WebPIIRecord]],
    heldout: set[GroupKey],
    heldout_allocation: Mapping[StratumKey, int],
    *,
    seed: int,
) -> tuple[set[GroupKey], set[GroupKey], dict[StratumKey, int]]:
    heldout_strata: dict[StratumKey, list[GroupKey]] = defaultdict(list)
    for group in heldout:
        heldout_strata[(group[0], group[1])].append(group)
    target = len(heldout) // 2
    target = min(len(heldout) - 1, max(1, target))
    calibration_allocation = _hamilton_allocation(
        {stratum: len(groups) for stratum, groups in heldout_strata.items()},
        target=target,
        seed=seed,
        domain="calibration-stratum",
    )
    calibration: set[GroupKey] = set()
    for stratum in sorted(heldout_strata, key=_canonical_json):
        ranked = sorted(
            heldout_strata[stratum],
            key=lambda group: _stable_digest(seed, "calibration-group", group),
        )
        calibration.update(ranked[: calibration_allocation[stratum]])
    calibration = _repair_dual_class_support(
        grouped,
        heldout,
        calibration,
        calibration_allocation,
        seed=seed,
    )
    guard = heldout - calibration
    if not calibration or not guard:
        raise RuntimeError(
            "WebPII heldout split produced an empty calibration or guard"
        )
    if calibration & guard or calibration | guard != heldout:
        raise RuntimeError("WebPII calibration and guard groups are not a partition")
    actual_allocation = Counter((group[0], group[1]) for group in calibration)
    for stratum in heldout_allocation:
        actual_allocation.setdefault(stratum, 0)
    return calibration, guard, dict(actual_allocation)


def _stratify_groups(
    grouped: Mapping[GroupKey, Sequence[WebPIIRecord]],
) -> dict[StratumKey, list[GroupKey]]:
    strata: dict[StratumKey, list[GroupKey]] = defaultdict(list)
    for group, records in grouped.items():
        if not records or any(record.group != group for record in records):
            raise RuntimeError("WebPII identity grouping is inconsistent")
        stratum = (group[0], group[1])
        if any(record.stratum != stratum for record in records):
            raise RuntimeError("WebPII group crosses company/page_type strata")
        strata[stratum].append(group)
    return strata


def _hamilton_allocation(
    counts: Mapping[StratumKey, int],
    *,
    target: int,
    seed: int,
    domain: str,
) -> dict[StratumKey, int]:
    total = sum(counts.values())
    if not counts or target < 0 or target > total:
        raise ValueError("invalid WebPII stratified allocation target")
    if total == 0:
        raise ValueError("cannot allocate an empty WebPII group set")
    exact = {stratum: target * count / total for stratum, count in counts.items()}
    allocation = {stratum: int(math.floor(value)) for stratum, value in exact.items()}
    remaining = target - sum(allocation.values())
    ranked = sorted(
        counts,
        key=lambda stratum: (
            -(exact[stratum] - allocation[stratum]),
            _stable_digest(seed, domain, stratum),
        ),
    )
    for stratum in ranked:
        if remaining == 0:
            break
        if allocation[stratum] < counts[stratum]:
            allocation[stratum] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError("WebPII Hamilton allocation could not reach its target")
    return allocation


def _group_class_support(
    grouped: Mapping[GroupKey, Sequence[WebPIIRecord]],
) -> dict[GroupKey, frozenset[int]]:
    return {
        group: frozenset(
            class_index for record in records for class_index in record.classes
        )
        for group, records in grouped.items()
    }


def _support_counts(
    groups: Iterable[GroupKey],
    classes: Mapping[GroupKey, frozenset[int]],
) -> Counter[int]:
    return Counter(class_index for group in groups for class_index in classes[group])


def _deficit_score(counts: Mapping[int, int], targets: Mapping[int, int]) -> int:
    return sum(
        max(0, target - counts.get(index, 0)) for index, target in targets.items()
    )


def _repair_selected_class_support(
    grouped: Mapping[GroupKey, Sequence[WebPIIRecord]],
    selected: set[GroupKey],
    allocation: Mapping[StratumKey, int],
    *,
    seed: int,
) -> set[GroupKey]:
    classes = _group_class_support(grouped)
    source_counts = _support_counts(grouped, classes)
    targets = {
        index: min(2, source_counts[index]) for index in range(len(DETECTOR_CLASSES))
    }
    chosen = set(selected)
    for _ in range(len(DETECTOR_CLASSES) * 3):
        counts = _support_counts(chosen, classes)
        before = _deficit_score(counts, targets)
        if before == 0:
            break
        missing = sorted(
            (index for index in targets if counts[index] < targets[index]),
            key=lambda index: (source_counts[index], index),
        )
        best: tuple[int, bytes, GroupKey, GroupKey] | None = None
        for class_index in missing:
            candidates = sorted(
                (
                    group
                    for group in grouped
                    if group not in chosen and class_index in classes[group]
                ),
                key=lambda group: _stable_digest(
                    seed, "heldout-repair-candidate", group
                ),
            )
            for candidate in candidates:
                stratum = (candidate[0], candidate[1])
                if allocation.get(stratum, 0) == 0:
                    continue
                victims = sorted(
                    (group for group in chosen if (group[0], group[1]) == stratum),
                    key=lambda group: _stable_digest(
                        seed, "heldout-repair-victim", group
                    ),
                )
                for victim in victims:
                    revised = counts.copy()
                    revised.subtract(classes[victim])
                    revised.update(classes[candidate])
                    improvement = before - _deficit_score(revised, targets)
                    if improvement <= 0:
                        continue
                    tie = _stable_digest(
                        seed, "heldout-repair-swap", (candidate, victim)
                    )
                    choice = (-improvement, tie, candidate, victim)
                    if best is None or choice < best:
                        best = choice
        if best is None:
            break
        _, _, candidate, victim = best
        chosen.remove(victim)
        chosen.add(candidate)
    return chosen


def _repair_dual_class_support(
    grouped: Mapping[GroupKey, Sequence[WebPIIRecord]],
    heldout: set[GroupKey],
    calibration: set[GroupKey],
    ideal_allocation: Mapping[StratumKey, int],
    *,
    seed: int,
) -> set[GroupKey]:
    classes = _group_class_support(grouped)
    heldout_counts = _support_counts(heldout, classes)
    targets = {
        index: 1 if heldout_counts[index] >= 2 else 0
        for index in range(len(DETECTOR_CLASSES))
    }
    chosen = set(calibration)
    for _ in range(len(DETECTOR_CLASSES) * 3):
        guard = heldout - chosen
        calibration_counts = _support_counts(chosen, classes)
        guard_counts = _support_counts(guard, classes)
        before = _deficit_score(calibration_counts, targets) + _deficit_score(
            guard_counts, targets
        )
        if before == 0:
            break
        actual = Counter((group[0], group[1]) for group in chosen)
        best: tuple[int, int, bytes, GroupKey, GroupKey] | None = None
        for candidate in sorted(
            guard,
            key=lambda group: _stable_digest(
                seed, "calibration-repair-candidate", group
            ),
        ):
            candidate_stratum = (candidate[0], candidate[1])
            for victim in sorted(
                chosen,
                key=lambda group: _stable_digest(
                    seed, "calibration-repair-victim", group
                ),
            ):
                victim_stratum = (victim[0], victim[1])
                if candidate_stratum != victim_stratum:
                    candidate_after = actual[candidate_stratum] + 1
                    victim_after = actual[victim_stratum] - 1
                    if (
                        abs(candidate_after - ideal_allocation[candidate_stratum]) > 1
                        or abs(victim_after - ideal_allocation[victim_stratum]) > 1
                    ):
                        continue
                revised_calibration = calibration_counts.copy()
                revised_calibration.subtract(classes[victim])
                revised_calibration.update(classes[candidate])
                revised_guard = guard_counts.copy()
                revised_guard.subtract(classes[candidate])
                revised_guard.update(classes[victim])
                after = _deficit_score(revised_calibration, targets) + _deficit_score(
                    revised_guard, targets
                )
                improvement = before - after
                if improvement <= 0:
                    continue
                stratum_penalty = int(candidate_stratum != victim_stratum)
                tie = _stable_digest(
                    seed, "calibration-repair-swap", (candidate, victim)
                )
                choice = (-improvement, stratum_penalty, tie, candidate, victim)
                if best is None or choice < best:
                    best = choice
        if best is None:
            break
        _, _, _, candidate, victim = best
        chosen.remove(victim)
        chosen.add(candidate)
    return chosen


def _validate_group_partition(
    all_groups: set[GroupKey],
    splits: Mapping[str, set[GroupKey]],
) -> None:
    names = tuple(splits)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if splits[left] & splits[right]:
                raise RuntimeError(f"WebPII {left} and {right} identity groups overlap")
    union = set().union(*splits.values())
    if union != all_groups:
        raise RuntimeError("WebPII split identity groups are not exhaustive")


def _stable_digest(seed: int, domain: str, value: Any) -> bytes:
    encoded = f"{seed}\0{domain}\0{_canonical_json(value)}".encode("utf-8")
    return hashlib.sha256(encoded).digest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _statistics(records: Sequence[WebPIIRecord]) -> dict[str, Any]:
    groups = {record.group for record in records}
    class_instances = Counter(
        class_index for record in records for class_index in record.classes
    )
    class_support = Counter(
        class_index for record in records for class_index in set(record.classes)
    )
    companies = _dimension_statistics(records, 0)
    page_types = _dimension_statistics(records, 1)
    stratum_records: dict[StratumKey, list[WebPIIRecord]] = defaultdict(list)
    for record in records:
        stratum_records[record.stratum].append(record)
    return {
        "records": len(records),
        "groups": len(groups),
        "annotations": sum(class_instances.values()),
        "labeled_images": sum(bool(record.classes) for record in records),
        "negative_images": sum(not record.classes for record in records),
        "class_instances": {
            name: class_instances[index] for index, name in enumerate(DETECTOR_CLASSES)
        },
        "class_image_support": {
            name: class_support[index] for index, name in enumerate(DETECTOR_CLASSES)
        },
        "companies": companies,
        "page_types": page_types,
        "company_page_strata": [
            {
                "company": stratum[0],
                "page_type": stratum[1],
                "records": len(stratum_records[stratum]),
                "groups": len({record.group for record in stratum_records[stratum]}),
            }
            for stratum in sorted(stratum_records, key=_canonical_json)
        ],
    }


def _dimension_statistics(
    records: Sequence[WebPIIRecord], group_index: int
) -> dict[str, dict[str, int]]:
    by_value: dict[str, list[WebPIIRecord]] = defaultdict(list)
    for record in records:
        by_value[record.group[group_index]].append(record)
    return {
        value: {
            "records": len(items),
            "groups": len({item.group for item in items}),
        }
        for value, items in sorted(by_value.items())
    }


def _validate_partition_statistics(
    total: Mapping[str, Any],
    calibration: Mapping[str, Any],
    guard: Mapping[str, Any],
    eligible: Mapping[str, Any],
) -> None:
    for field in (
        "records",
        "groups",
        "annotations",
        "labeled_images",
        "negative_images",
    ):
        if total[field] != calibration[field] + guard[field] + eligible[field]:
            raise RuntimeError(
                f"WebPII guard statistics are not exhaustive for {field}"
            )
    for field in ("class_instances", "class_image_support"):
        for name in DETECTOR_CLASSES:
            if total[field][name] != (
                calibration[field][name] + guard[field][name] + eligible[field][name]
            ):
                raise RuntimeError(
                    f"WebPII guard statistics are not exhaustive for {field}/{name}"
                )


def _allocation_statistics(
    heldout_allocation: Mapping[StratumKey, int],
    calibration_allocation: Mapping[StratumKey, int],
    grouped: Mapping[GroupKey, Sequence[WebPIIRecord]],
    calibration_groups: set[GroupKey],
    guard_groups: set[GroupKey],
) -> list[dict[str, Any]]:
    strata: dict[StratumKey, list[GroupKey]] = defaultdict(list)
    for group in grouped:
        strata[(group[0], group[1])].append(group)
    return [
        {
            "company": stratum[0],
            "page_type": stratum[1],
            "groups": len(strata[stratum]),
            "heldout_groups": heldout_allocation.get(stratum, 0),
            "calibration_groups": calibration_allocation.get(stratum, 0),
            "guard_groups": sum(group in guard_groups for group in strata[stratum]),
            "eligible_groups": len(strata[stratum])
            - heldout_allocation.get(stratum, 0),
            "records": sum(len(grouped[group]) for group in strata[stratum]),
            "calibration_records": sum(
                len(grouped[group])
                for group in strata[stratum]
                if group in calibration_groups
            ),
            "guard_records": sum(
                len(grouped[group])
                for group in strata[stratum]
                if group in guard_groups
            ),
        }
        for stratum in sorted(strata, key=_canonical_json)
    ]


def _class_coverage_contract(
    grouped: Mapping[GroupKey, Sequence[WebPIIRecord]],
    calibration_groups: set[GroupKey],
    guard_groups: set[GroupKey],
) -> dict[str, Any]:
    classes = _group_class_support(grouped)
    source_counts = _support_counts(grouped, classes)
    heldout_counts = _support_counts(calibration_groups | guard_groups, classes)
    calibration_counts = _support_counts(calibration_groups, classes)
    guard_counts = _support_counts(guard_groups, classes)
    calibration_missing = [
        name
        for index, name in enumerate(DETECTOR_CLASSES)
        if calibration_counts[index] == 0
    ]
    guard_missing = [
        name for index, name in enumerate(DETECTOR_CLASSES) if guard_counts[index] == 0
    ]
    dual_possible_from_heldout = [
        name
        for index, name in enumerate(DETECTOR_CLASSES)
        if heldout_counts[index] >= 2
    ]
    return {
        "source_group_support": {
            name: source_counts[index] for index, name in enumerate(DETECTOR_CLASSES)
        },
        "heldout_group_support": {
            name: heldout_counts[index] for index, name in enumerate(DETECTOR_CLASSES)
        },
        "calibration_group_support": {
            name: calibration_counts[index]
            for index, name in enumerate(DETECTOR_CLASSES)
        },
        "guard_group_support": {
            name: guard_counts[index] for index, name in enumerate(DETECTOR_CLASSES)
        },
        "calibration_missing_classes": calibration_missing,
        "guard_missing_classes": guard_missing,
        "classes_with_dual_support_possible_from_heldout": dual_possible_from_heldout,
        "dual_all_class_support_achieved": not calibration_missing
        and not guard_missing,
    }


def _encode_image_list(records: Iterable[WebPIIRecord]) -> str:
    paths = [str(record.image) for record in records]
    if any("\n" in path or "\r" in path or "\0" in path for path in paths):
        raise ValueError("WebPII image-list path contains a control character")
    return "".join(f"{path}\n" for path in paths)


def _dataset_yaml(output_dir: Path, *, validation: str | None) -> str:
    lines = [
        f"path: {json.dumps(str(output_dir), ensure_ascii=False)}",
        f"train: {json.dumps(ELIGIBLE_IMAGES_NAME)}",
    ]
    if validation is not None:
        lines.append(f"val: {json.dumps(validation)}")
    lines.extend(
        [
            f"nc: {len(DETECTOR_CLASSES)}",
            "names:",
            *[
                f"  {index}: {json.dumps(name)}"
                for index, name in enumerate(DETECTOR_CLASSES)
            ],
            "",
        ]
    )
    return "\n".join(lines)


def _output_contract(
    final_path: Path, temporary_path: Path, records: int | None
) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "path": str(final_path),
        "bytes": temporary_path.stat().st_size,
        "sha256": sha256_file(temporary_path),
    }
    if records is not None:
        contract["records"] = records
    return contract


def _pair_inventory_sha256(records: Sequence[WebPIIRecord]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.relative_image.encode("utf-8")):
        digest.update(record.relative_image.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record.image_bytes).encode("ascii"))
        digest.update(b"\0")
        digest.update(record.relative_label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(record.label_sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _publish_directory(temporary: Path, output_dir: Path, *, force: bool) -> None:
    if not force:
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(f"guard output already exists: {output_dir}")
        temporary.rename(output_dir)
        return
    if output_dir.is_symlink():
        raise ValueError("guard output directory may not be a symbolic link")
    if not output_dir.exists():
        temporary.rename(output_dir)
        return
    if not output_dir.is_dir():
        raise ValueError(f"guard output path is not a directory: {output_dir}")
    backup = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.backup-", dir=str(output_dir.parent)
        )
    )
    backup.rmdir()
    output_dir.rename(backup)
    try:
        temporary.rename(output_dir)
    except Exception:
        backup.rename(output_dir)
        raise
    shutil.rmtree(backup)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic WebPII train-derived calibration, guard, "
            "and eligible splits"
        )
    )
    parser.add_argument("--webpii-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--heldout-fraction", type=float, default=DEFAULT_HELDOUT_FRACTION
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_webpii_guard_split(
        args.webpii_root,
        args.output_dir,
        seed=args.seed,
        heldout_fraction=args.heldout_fraction,
        force=args.force,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
