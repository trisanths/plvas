from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from training.schema import sha256_file
from training.visual.build_webpii_guard_split import (
    CALIBRATION_DATASET_NAME,
    CALIBRATION_IMAGES_NAME,
    DEFAULT_HELDOUT_FRACTION,
    DEFAULT_SEED,
    ELIGIBLE_DATASET_NAME,
    ELIGIBLE_IMAGES_NAME,
    GUARD_DATASET_NAME,
    GUARD_IMAGES_NAME,
    MANIFEST_NAME,
    build_webpii_guard_split,
)
from training.visual.prepare_webpii import (
    DETECTOR_CLASSES,
    WEBPII_ID,
    WEBPII_REVISION,
)


def _write_webpii(
    root: Path,
    *,
    groups_per_stratum: int = 10,
    variants: int = 2,
    all_classes: bool = False,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    companies = ("Acme", "Beta")
    page_types = ("checkout", "account")
    for stratum_index, (company, page_type) in enumerate(
        zip(companies, page_types, strict=True)
    ):
        for group_index in range(groups_per_stratum):
            source_id = f"source-{stratum_index}-{group_index:03d}"
            for variant_index in range(variants):
                variant = f"variant-{variant_index}"
                stem = f"{company}-{page_type}-{source_id}-{variant}"
                image_relative = f"images/train/{stem}.png"
                label_relative = f"labels/train/{stem}.txt"
                image_path = root / image_relative
                label_path = root / label_relative
                image_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(f"fixture-{stem}".encode())
                negative = (
                    not all_classes
                    and (group_index + variant_index + stratum_index) % 11 == 0
                )
                if all_classes:
                    class_indices = tuple(range(len(DETECTOR_CLASSES)))
                    label_path.write_text(
                        "".join(
                            f"{index} 0.5 0.5 0.2 0.1\n" for index in class_indices
                        ),
                        encoding="utf-8",
                    )
                elif negative:
                    class_indices: tuple[int, ...] = ()
                    label_path.write_text("", encoding="utf-8")
                else:
                    class_indices = (
                        (group_index + variant_index + stratum_index)
                        % len(DETECTOR_CLASSES),
                    )
                    label_path.write_text(
                        f"{class_indices[0]} 0.5 0.5 0.2 0.1\n",
                        encoding="utf-8",
                    )
                rows.append(
                    {
                        "id": stem,
                        "published_split": "train",
                        "source_id": source_id,
                        "variant": variant,
                        "company": company,
                        "page_type": page_type,
                        "width": 100,
                        "height": 100,
                        "image": image_relative,
                        "label_file": label_relative,
                        "annotations": [
                            {
                                "class": DETECTOR_CLASSES[index],
                                "source_key": "PII_FIXTURE",
                            }
                            for index in class_indices
                        ],
                        "hard_negative_boxes": [],
                    }
                )
    records_path = root / "records/train.jsonl"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    class_counts = Counter(
        annotation["class"]
        for row in rows
        for annotation in row["annotations"]  # type: ignore[index]
    )
    companies_count = Counter(str(row["company"]) for row in rows)
    page_types_count = Counter(str(row["page_type"]) for row in rows)
    manifest = {
        "schema_version": 1,
        "dataset": WEBPII_ID,
        "revision": WEBPII_REVISION,
        "license": "Apache-2.0",
        "published_splits_preserved": True,
        "test_used_for_checkpoint_selection": False,
        "metadata_only": False,
        "classes": list(DETECTOR_CLASSES),
        "splits": {
            "train": {
                "records": len(rows),
                "annotations": sum(len(row["annotations"]) for row in rows),  # type: ignore[arg-type]
                "negative_images": sum(not row["annotations"] for row in rows),
                "class_counts": dict(sorted(class_counts.items())),
                "companies": dict(sorted(companies_count.items())),
                "page_types": dict(sorted(page_types_count.items())),
                "records_path": "records/train.jsonl",
                "records_sha256": sha256_file(records_path),
            },
            "test": {
                "records": 17,
                "records_path": "records/test.jsonl",
                "records_sha256": "f" * 64,
            },
        },
        "values_persisted": False,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return rows


def _persist_manifest(root: Path, manifest: dict[str, object]) -> None:
    records_path = root / "records/train.jsonl"
    manifest["splits"]["train"]["records_sha256"] = sha256_file(records_path)  # type: ignore[index]
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_guard_split_is_deterministic_stratified_disjoint_and_exhaustive(
    tmp_path: Path,
) -> None:
    source = tmp_path / "webpii"
    rows = _write_webpii(source)
    output = tmp_path / "guard"

    result = build_webpii_guard_split(source, output)
    first_bytes = {
        name: (output / name).read_bytes()
        for name in (
            CALIBRATION_IMAGES_NAME,
            GUARD_IMAGES_NAME,
            ELIGIBLE_IMAGES_NAME,
            CALIBRATION_DATASET_NAME,
            GUARD_DATASET_NAME,
            ELIGIBLE_DATASET_NAME,
            MANIFEST_NAME,
        )
    }
    repeated = build_webpii_guard_split(source, output, force=True)
    assert first_bytes == {name: (output / name).read_bytes() for name in first_bytes}
    assert result["manifest_sha256"] == repeated["manifest_sha256"]

    calibration_paths = (output / CALIBRATION_IMAGES_NAME).read_text().splitlines()
    guard_paths = (output / GUARD_IMAGES_NAME).read_text().splitlines()
    eligible_paths = (output / ELIGIBLE_IMAGES_NAME).read_text().splitlines()
    assert len(calibration_paths) + len(guard_paths) + len(eligible_paths) == len(rows)
    assert set(calibration_paths).isdisjoint(guard_paths)
    assert set(calibration_paths).isdisjoint(eligible_paths)
    assert set(guard_paths).isdisjoint(eligible_paths)
    expected_paths = {(source / str(row["image"])).resolve().as_posix() for row in rows}
    assert (
        set(calibration_paths) | set(guard_paths) | set(eligible_paths)
        == expected_paths
    )

    by_path = {
        (source / str(row["image"])).resolve().as_posix(): (
            row["company"],
            row["page_type"],
            row["source_id"],
        )
        for row in rows
    }
    calibration_groups = {by_path[path] for path in calibration_paths}
    guard_groups = {by_path[path] for path in guard_paths}
    eligible_groups = {by_path[path] for path in eligible_paths}
    assert len(calibration_groups) == 1
    assert len(guard_groups) == 1
    assert calibration_groups.isdisjoint(guard_groups)
    assert calibration_groups.isdisjoint(eligible_groups)
    assert guard_groups.isdisjoint(eligible_groups)
    assert {group[:2] for group in calibration_groups | guard_groups} == {
        ("Acme", "checkout"),
        ("Beta", "account"),
    }
    for group in calibration_groups | guard_groups:
        assigned = calibration_paths if group in calibration_groups else guard_paths
        assert sum(by_path[path] == group for path in assigned) == 2
        assert all(by_path[path] != group for path in eligible_paths)

    persisted = json.loads((output / MANIFEST_NAME).read_text())
    assert persisted["configuration"]["seed"] == DEFAULT_SEED
    assert persisted["configuration"]["heldout_fraction"] == DEFAULT_HELDOUT_FRACTION
    assert persisted["selection"]["groups"] == {
        "total": 20,
        "heldout": 2,
        "calibration": 1,
        "guard": 1,
        "eligible": 18,
        "pairwise_overlap": {
            "calibration_guard": 0,
            "calibration_eligible": 0,
            "guard_eligible": 0,
        },
    }
    assert persisted["invariants"]["identity_disjoint"] is True
    assert persisted["invariants"]["records_exhaustive"] is True
    assert persisted["invariants"]["raw_pii_values_persisted"] is False
    assert persisted["values_persisted"] is False
    assert persisted["published_test"] == {
        "excluded": True,
        "used_for_selection": False,
        "emitted_in_dataset_yaml": False,
        "declared_records": 17,
    }
    for contract in persisted["outputs"].values():
        path = Path(contract["path"])
        assert contract["sha256"] == sha256_file(path)
        assert contract["bytes"] == path.stat().st_size

    for name, validation in (
        (CALIBRATION_DATASET_NAME, CALIBRATION_IMAGES_NAME),
        (GUARD_DATASET_NAME, GUARD_IMAGES_NAME),
        (ELIGIBLE_DATASET_NAME, None),
    ):
        dataset = yaml.safe_load((output / name).read_text())
        assert dataset["path"] == str(output.resolve())
        assert dataset["train"] == ELIGIBLE_IMAGES_NAME
        if validation is None:
            assert "val" not in dataset
        else:
            assert dataset["val"] == validation
        assert "test" not in dataset
        assert dataset["names"] == {
            index: class_name for index, class_name in enumerate(DETECTOR_CLASSES)
        }


def test_guard_manifest_has_complete_class_support_company_and_page_stats(
    tmp_path: Path,
) -> None:
    source = tmp_path / "webpii"
    _write_webpii(source, groups_per_stratum=20, variants=3)
    result = build_webpii_guard_split(source, tmp_path / "guard")

    statistics = result["statistics"]
    for split in ("all", "calibration", "guard", "eligible"):
        stats = statistics[split]
        assert list(stats["class_instances"]) == list(DETECTOR_CLASSES)
        assert list(stats["class_image_support"]) == list(DETECTOR_CLASSES)
        assert set(stats["companies"]) <= {"Acme", "Beta"}
        assert set(stats["page_types"]) <= {"checkout", "account"}
        assert all(
            {"company", "page_type", "records", "groups"} <= set(item)
            for item in stats["company_page_strata"]
        )
    for class_name in DETECTOR_CLASSES:
        assert statistics["all"]["class_instances"][class_name] == (
            statistics["calibration"]["class_instances"][class_name]
            + statistics["guard"]["class_instances"][class_name]
            + statistics["eligible"]["class_instances"][class_name]
        )
        assert statistics["all"]["class_image_support"][class_name] == (
            statistics["calibration"]["class_image_support"][class_name]
            + statistics["guard"]["class_image_support"][class_name]
            + statistics["eligible"]["class_image_support"][class_name]
        )


def test_calibration_and_guard_both_keep_all_class_support_when_feasible(
    tmp_path: Path,
) -> None:
    source = tmp_path / "webpii"
    _write_webpii(source, groups_per_stratum=20, variants=2, all_classes=True)
    result = build_webpii_guard_split(source, tmp_path / "guard")

    coverage = result["selection"]["class_coverage"]
    assert coverage["calibration_missing_classes"] == []
    assert coverage["guard_missing_classes"] == []
    assert coverage["dual_all_class_support_achieved"] is True
    assert all(
        coverage["calibration_group_support"][name] > 0
        and coverage["guard_group_support"][name] > 0
        for name in DETECTOR_CLASSES
    )


def test_guard_split_rejects_records_sha_mismatch_without_touching_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "webpii"
    _write_webpii(source)
    output = tmp_path / "guard"
    build_webpii_guard_split(source, output)
    before = (output / MANIFEST_NAME).read_bytes()

    records_path = source / "records/train.jsonl"
    records_path.write_text(records_path.read_text() + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_webpii_guard_split(source, output, force=True)
    assert (output / MANIFEST_NAME).read_bytes() == before


def test_guard_split_rejects_published_test_record_in_train_contract(
    tmp_path: Path,
) -> None:
    source = tmp_path / "webpii"
    _write_webpii(source)
    records_path = source / "records/train.jsonl"
    rows = records_path.read_text().splitlines()
    first = json.loads(rows[0])
    first["published_split"] = "test"
    rows[0] = json.dumps(first, sort_keys=True, separators=(",", ":"))
    records_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    manifest = json.loads((source / "manifest.json").read_text())
    _persist_manifest(source, manifest)

    with pytest.raises(ValueError, match="not published train data"):
        build_webpii_guard_split(source, tmp_path / "guard")


def test_guard_split_rejects_unpaired_inventory_file(tmp_path: Path) -> None:
    source = tmp_path / "webpii"
    _write_webpii(source)
    extra = source / "labels/train/unpaired.txt"
    extra.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="not exhaustive over label files"):
        build_webpii_guard_split(source, tmp_path / "guard")


def test_guard_split_rejects_symbolic_link_pair(tmp_path: Path) -> None:
    source = tmp_path / "webpii"
    rows = _write_webpii(source)
    image = source / str(rows[0]["image"])
    target = image.with_name("target.png")
    image.rename(target)
    image.symlink_to(target)

    with pytest.raises(ValueError, match="symbolic link"):
        build_webpii_guard_split(source, tmp_path / "guard")


def test_guard_split_rejects_manifest_count_even_with_updated_hash(
    tmp_path: Path,
) -> None:
    source = tmp_path / "webpii"
    _write_webpii(source)
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["splits"]["train"]["records"] += 1
    _persist_manifest(source, manifest)

    with pytest.raises(ValueError, match="count differs from manifest"):
        build_webpii_guard_split(source, tmp_path / "guard")
