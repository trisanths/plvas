from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from training.schema import sha256_file
from training.visual.prepare_webpii import DETECTOR_CLASSES


SPLIT_SCHEMA_VERSION = 1
SPLIT_KIND = "deterministic-synthetic-calibration-guard-split"


def _stable_digest(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def _atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path, context: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{context} is missing or unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return value


def _record_stratum(record: dict[str, Any]) -> str:
    template = record.get("template_id")
    if not isinstance(template, str) or not template:
        raise ValueError("synthetic validation record has no template")
    annotations = record.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError("synthetic validation annotations are invalid")
    source_classes = []
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise ValueError("synthetic validation annotation is invalid")
        source_class = annotation.get("source_class")
        if not isinstance(source_class, str) or not source_class:
            raise ValueError("synthetic validation source class is invalid")
        source_classes.append(source_class)
    image = record.get("image")
    if not isinstance(image, str) or not image:
        raise ValueError("synthetic validation image path is invalid")
    # Template is the primary layout stratum. Source subtype keeps SECRET and
    # SENSITIVE_FIELD calibration/guard support balanced within each template.
    return f"{template}\0{','.join(sorted(source_classes)) or 'NONE'}"


def build_synthetic_selection_split(
    synthetic_root: Path,
    output_dir: Path,
    *,
    seed: int = 1311,
) -> dict[str, Any]:
    """Split frozen synthetic validation into calibration and guard halves."""

    synthetic_root = synthetic_root.resolve()
    output_dir = output_dir.resolve()
    manifest_path = synthetic_root / "manifest.json"
    manifest = _read_json(manifest_path, "synthetic source manifest")
    if manifest.get("synthetic_only") is not True:
        raise ValueError("selection split source must be synthetic-only")
    if manifest.get("classes") != list(DETECTOR_CLASSES):
        raise ValueError("synthetic selection classes differ from detector contract")
    validation = manifest.get("splits", {}).get("validation")
    if not isinstance(validation, dict):
        raise ValueError("synthetic manifest has no validation split")
    records_relative = validation.get("records_path")
    if not isinstance(records_relative, str) or not records_relative:
        raise ValueError("synthetic validation records path is missing")
    records_path = synthetic_root / records_relative
    if (
        records_path.is_symlink()
        or not records_path.is_file()
        or sha256_file(records_path) != validation.get("records_sha256")
    ):
        raise ValueError("synthetic validation records hash mismatch")

    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_ids: set[str] = set()
    seen_images: set[Path] = set()
    record_count = 0
    try:
        lines = records_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError("synthetic validation records are not UTF-8") from exc
    for line_number, line in enumerate(lines, 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"synthetic validation record {line_number} is invalid JSON"
            ) from exc
        if not isinstance(record, dict) or record.get("split") != "validation":
            raise ValueError(
                f"synthetic validation record {line_number} has the wrong split"
            )
        record_id = record.get("id")
        image_relative = record.get("image")
        label_relative = record.get("label_file")
        if (
            not isinstance(record_id, str)
            or not record_id
            or record_id in seen_ids
            or not isinstance(image_relative, str)
            or not isinstance(label_relative, str)
        ):
            raise ValueError("synthetic validation record identity is invalid")
        image = (synthetic_root / image_relative).resolve()
        label = (synthetic_root / label_relative).resolve()
        if (
            synthetic_root not in image.parents
            or synthetic_root not in label.parents
            or image.is_symlink()
            or label.is_symlink()
            or not image.is_file()
            or not label.is_file()
            or image in seen_images
        ):
            raise ValueError("synthetic validation image-label inventory is unsafe")
        seen_ids.add(record_id)
        seen_images.add(image)
        strata[_record_stratum(record)].append(
            {"id": record_id, "image": image, "label": label, "record": record}
        )
        record_count += 1
    if record_count != validation.get("records"):
        raise ValueError("synthetic validation record count differs from manifest")

    calibration: list[dict[str, Any]] = []
    guard: list[dict[str, Any]] = []
    for stratum, records in sorted(strata.items()):
        ordered = sorted(
            records,
            key=lambda item: _stable_digest(
                seed, f"{stratum}\0{item['id']}"
            ),
        )
        # Alternating the deterministic rank produces exact or one-record-close
        # halves for every template/subtype stratum.
        calibration.extend(ordered[::2])
        guard.extend(ordered[1::2])
    calibration.sort(key=lambda item: str(item["image"]))
    guard.sort(key=lambda item: str(item["image"]))
    calibration_ids = {item["id"] for item in calibration}
    guard_ids = {item["id"] for item in guard}
    if calibration_ids & guard_ids or calibration_ids | guard_ids != seen_ids:
        raise RuntimeError("synthetic calibration and guard inventories overlap or omit data")

    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_list = output_dir / "synthetic-calibration-images.txt"
    guard_list = output_dir / "synthetic-guard-images.txt"
    _atomic_write(
        calibration_list,
        "".join(f"{item['image']}\n" for item in calibration),
    )
    _atomic_write(guard_list, "".join(f"{item['image']}\n" for item in guard))

    def dataset_yaml(name: str, image_list: Path) -> Path:
        path = output_dir / f"synthetic-{name}.yaml"
        _atomic_write(
            path,
            "\n".join(
                [
                    "path: /",
                    f"train: {image_list}",
                    f"val: {image_list}",
                    f"test: {image_list}",
                    f"nc: {len(DETECTOR_CLASSES)}",
                    "names:",
                    *[
                        f"  {index}: {class_name}"
                        for index, class_name in enumerate(DETECTOR_CLASSES)
                    ],
                    "",
                ]
            ),
        )
        return path

    calibration_yaml = dataset_yaml("calibration", calibration_list)
    guard_yaml = dataset_yaml("guard", guard_list)

    def inventory_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
        classes: Counter[str] = Counter()
        templates: Counter[str] = Counter()
        source_classes: Counter[str] = Counter()
        negatives = 0
        for item in items:
            record = item["record"]
            annotations = record.get("annotations")
            if not isinstance(annotations, list):
                raise ValueError("synthetic selection annotations are invalid")
            if not annotations:
                negatives += 1
            for annotation in annotations:
                if not isinstance(annotation, dict) or annotation.get("class") not in DETECTOR_CLASSES:
                    raise ValueError("synthetic selection annotation class is invalid")
                classes[annotation["class"]] += 1
            templates[str(record["template_id"])] += 1
            annotation_sources = sorted(
                str(annotation["source_class"]) for annotation in annotations
            )
            source_classes[",".join(annotation_sources) or "NONE"] += 1
        return {
            "records": len(items),
            "pure_negative_records": negatives,
            "class_counts": dict(sorted(classes.items())),
            "template_counts": dict(sorted(templates.items())),
            "source_class_counts": dict(sorted(source_classes.items())),
        }

    contract = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "kind": SPLIT_KIND,
        "seed": seed,
        "source": {
            "root": str(synthetic_root),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "validation_records": record_count,
            "validation_records_sha256": sha256_file(records_path),
            "synthetic_only": True,
        },
        "split_policy": (
            "template-and-source-class-stratified deterministic alternating hash rank"
        ),
        "calibration": {
            **inventory_summary(calibration),
            "image_list": str(calibration_list),
            "image_list_sha256": sha256_file(calibration_list),
            "dataset_yaml": str(calibration_yaml),
            "dataset_yaml_sha256": sha256_file(calibration_yaml),
        },
        "guard": {
            **inventory_summary(guard),
            "image_list": str(guard_list),
            "image_list_sha256": sha256_file(guard_list),
            "dataset_yaml": str(guard_yaml),
            "dataset_yaml_sha256": sha256_file(guard_yaml),
        },
        "identity_overlap": len(calibration_ids & guard_ids),
        "published_webpii_test_used": False,
        "values_persisted": False,
    }
    contract_path = output_dir / "synthetic-selection-split.json"
    _atomic_write(contract_path, json.dumps(contract, indent=2, sort_keys=True) + "\n")
    contract["manifest_path"] = str(contract_path)
    contract["manifest_sha256"] = sha256_file(contract_path)
    return contract
