from __future__ import annotations

import json
from pathlib import Path

from training.schema import sha256_file
from training.visual.build_synthetic_selection_split import (
    build_synthetic_selection_split,
)
from training.visual.prepare_webpii import DETECTOR_CLASSES


def _fixture(root: Path) -> None:
    records = []
    class_counts = {name: 0 for name in DETECTOR_CLASSES}
    for class_index, class_name in enumerate(DETECTOR_CLASSES):
        for offset in range(4):
            stem = f"validation-{class_name.lower()}-{offset}"
            image_relative = f"images/validation/{stem}.png"
            label_relative = f"labels/validation/{stem}.txt"
            image = root / image_relative
            label = root / label_relative
            image.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(stem.encode())
            label.write_text(
                f"{class_index} 0.5 0.5 0.2 0.1\n", encoding="utf-8"
            )
            class_counts[class_name] += 1
            records.append(
                {
                    "id": stem,
                    "split": "validation",
                    "template_id": f"panel.{class_name.lower()}",
                    "image": image_relative,
                    "label_file": label_relative,
                    "annotations": [
                        {"class": class_name, "source_class": class_name}
                    ],
                    "hard_negative_boxes": [],
                }
            )
    for offset in range(4):
        stem = f"validation-negative-{offset}"
        image_relative = f"images/validation/{stem}.png"
        label_relative = f"labels/validation/{stem}.txt"
        image = root / image_relative
        label = root / label_relative
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(stem.encode())
        label.write_text("", encoding="utf-8")
        records.append(
            {
                "id": stem,
                "split": "validation",
                "template_id": "panel.negative",
                "image": image_relative,
                "label_file": label_relative,
                "annotations": [],
                "hard_negative_boxes": [],
            }
        )
    records_path = root / "records/validation.jsonl"
    records_path.parent.mkdir(parents=True)
    records_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 2,
        "synthetic_only": True,
        "classes": list(DETECTOR_CLASSES),
        "splits": {
            "validation": {
                "records": len(records),
                "records_path": "records/validation.jsonl",
                "records_sha256": sha256_file(records_path),
                "class_counts": class_counts,
            }
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_synthetic_calibration_and_guard_are_balanced_disjoint_and_stable(
    tmp_path: Path,
) -> None:
    source = tmp_path / "synthetic"
    _fixture(source)
    first = build_synthetic_selection_split(source, tmp_path / "first", seed=1311)
    second = build_synthetic_selection_split(source, tmp_path / "second", seed=1311)

    assert first["calibration"]["records"] == 20
    assert first["guard"]["records"] == 20
    assert first["identity_overlap"] == 0
    assert first["calibration"]["class_counts"] == {
        name: 2 for name in DETECTOR_CLASSES
    }
    assert first["guard"]["class_counts"] == {
        name: 2 for name in DETECTOR_CLASSES
    }
    first_calibration = {
        Path(path).name
        for path in Path(first["calibration"]["image_list"])
        .read_text()
        .splitlines()
    }
    first_guard = {
        Path(path).name
        for path in Path(first["guard"]["image_list"]).read_text().splitlines()
    }
    assert first_calibration.isdisjoint(first_guard)
    assert first_calibration == {
        Path(path).name
        for path in Path(second["calibration"]["image_list"])
        .read_text()
        .splitlines()
    }
    assert first_guard == {
        Path(path).name
        for path in Path(second["guard"]["image_list"]).read_text().splitlines()
    }


def test_synthetic_split_rejects_tampered_records(tmp_path: Path) -> None:
    source = tmp_path / "synthetic"
    _fixture(source)
    records = source / "records/validation.jsonl"
    records.write_text(records.read_text() + "{}\n", encoding="utf-8")
    try:
        build_synthetic_selection_split(source, tmp_path / "out")
    except ValueError as error:
        assert "hash mismatch" in str(error)
    else:  # pragma: no cover
        raise AssertionError("tampered synthetic records were accepted")
