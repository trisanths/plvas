from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from training.visual.build_replay_train_list import (
    DEFAULT_SOURCE_QUOTA,
    build_replay_train_list,
    materialize_replay_training_stage,
)
from training.visual.prepare_webpii import DETECTOR_CLASSES
from training.visual.train_detector import validate_composed_dataset_contract


def _source(root: Path, name: str, class_ids: list[int | None]) -> Path:
    source = root / name
    images = source / "images/train"
    labels = source / "labels/train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for index, class_id in enumerate(class_ids):
        stem = f"{name}-{index:03d}"
        (images / f"{stem}.png").write_bytes(b"fake-png-" + bytes([index % 251]))
        label = "" if class_id is None else f"{class_id} 0.5 0.5 0.25 0.25\n"
        (labels / f"{stem}.txt").write_text(label, encoding="utf-8")
    (source / "manifest.json").write_text(
        json.dumps({"source": name, "classes": list(DETECTOR_CLASSES)}) + "\n",
        encoding="utf-8",
    )
    return source


def _fixtures(tmp_path: Path) -> dict[str, Path]:
    # WebPII supplies full class coverage; synthetic has multiple rare examples
    # so every constructed logical block can receive an explicit rare replay.
    webpii = _source(
        tmp_path,
        "webpii",
        list(range(len(DETECTOR_CLASSES))) * 4 + [None] * 4,
    )
    synthetic = _source(
        tmp_path,
        "synthetic",
        list(range(len(DETECTOR_CLASSES))) * 3 + [None] * 3,
    )
    supplemental = _source(
        tmp_path,
        "supplemental_ats",
        [0, 1, 2, 3, 7, 7, 7, None] * 2,
    )
    return {
        "webpii": webpii,
        "synthetic": synthetic,
        "supplemental_ats": supplemental,
    }


def _class_ids(image: Path) -> set[int]:
    label = Path(
        str(image)
        .replace("/images/train/", "/labels/train/")
    ).with_suffix(".txt")
    return {
        int(line.split()[0])
        for line in label.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def test_replay_list_is_deterministic_quota_exact_and_rare_safe(
    tmp_path: Path,
) -> None:
    sources = _fixtures(tmp_path / "sources")
    first = build_replay_train_list(
        sources,
        tmp_path / "first",
        logical_batches=8,
        seed=1311,
    )
    second = build_replay_train_list(
        sources,
        tmp_path / "second",
        logical_batches=8,
        seed=1311,
    )

    first_lines = Path(first["output"]["train_list"]).read_text().splitlines()
    second_lines = Path(second["output"]["train_list"]).read_text().splitlines()
    assert [Path(line).name for line in first_lines] == [
        Path(line).name for line in second_lines
    ]
    assert first["output"]["sha256"] == second["output"]["sha256"]
    assert len(first_lines) == 8 * 64
    assert first["selected"]["source_counts"] == {
        source: quota * 8 for source, quota in sorted(DEFAULT_SOURCE_QUOTA.items())
    }
    assert first["logical_block_contract"]["all_blocks_validated"] is True
    assert first["configuration_sha256"] == second["configuration_sha256"]
    weights = first["original"]["normalized_inverse_sqrt_class_weights"]
    assert weights["SECRET"] > weights["ADDRESS"]
    assert weights["CVC"] > weights["ADDRESS"]
    assert weights["SENSITIVE_IMAGE"] > weights["ADDRESS"]

    different_seed = build_replay_train_list(
        sources,
        tmp_path / "different-seed",
        logical_batches=8,
        seed=1312,
    )
    assert different_seed["output"]["sha256"] != first["output"]["sha256"]

    for offset in range(0, len(first_lines), 64):
        block = [Path(line) for line in first_lines[offset : offset + 64]]
        source_counts = Counter(
            "webpii"
            if "/webpii/" in str(image)
            else "synthetic"
            if "/synthetic/" in str(image)
            else "supplemental_ats"
            for image in block
        )
        assert source_counts == Counter(DEFAULT_SOURCE_QUOTA)
        synthetic_classes = {
            class_id
            for image in block
            if "/synthetic/" in str(image)
            for class_id in _class_ids(image)
        }
        assert {
            DETECTOR_CLASSES.index("CVC"),
            DETECTOR_CLASSES.index("SECRET"),
            DETECTOR_CLASSES.index("SENSITIVE_IMAGE"),
        } <= synthetic_classes


def test_replay_manifest_preserves_pairing_and_provenance(tmp_path: Path) -> None:
    sources = _fixtures(tmp_path / "sources")
    result = build_replay_train_list(
        sources,
        tmp_path / "out",
        logical_batches=8,
        seed=7,
    )
    manifest_path = Path(result["manifest_path"])
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert persisted["consumer_contract"]["format"] == "ultralytics-image-list"
    assert persisted["consumer_contract"]["physical_batch_guarantee"] is False
    assert persisted["output"]["records"] == 512
    assert set(persisted["sources"]) == {
        "webpii",
        "synthetic",
        "supplemental_ats",
    }
    assert all(
        len(source["pair_inventory_sha256"]) == 64
        and len(source["manifest"]["sha256"]) == 64
        for source in persisted["sources"].values()
    )
    for image_value in Path(result["output"]["train_list"]).read_text().splitlines():
        image = Path(image_value)
        label = Path(
            str(image).replace("/images/train/", "/labels/train/")
        ).with_suffix(".txt")
        assert image.is_file()
        assert label.is_file()


def test_replay_rejects_bad_quota_and_malformed_labels(tmp_path: Path) -> None:
    sources = _fixtures(tmp_path / "sources")
    with pytest.raises(ValueError, match="does not cover every training image"):
        build_replay_train_list(
            sources,
            tmp_path / "partial-coverage",
            logical_batches=1,
        )
    with pytest.raises(ValueError, match="sum to batch_size"):
        build_replay_train_list(
            sources,
            tmp_path / "bad-quota",
            logical_batches=1,
            source_quota={
                "webpii": 24,
                "synthetic": 31,
                "supplemental_ats": 8,
            },
        )

    bad_label = next((sources["webpii"] / "labels/train").glob("*.txt"))
    bad_label.write_text("99 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown class"):
        build_replay_train_list(
            sources,
            tmp_path / "bad-label",
            logical_batches=1,
        )


def test_replay_counts_repeated_instances_and_auto_covers_every_source(
    tmp_path: Path,
) -> None:
    sources = _fixtures(tmp_path / "sources")
    repeated = next((sources["webpii"] / "labels/train").glob("*.txt"))
    class_id = int(repeated.read_text().split()[0])
    repeated.write_text(
        f"{class_id} 0.3 0.3 0.1 0.1\n{class_id} 0.7 0.7 0.1 0.1\n",
        encoding="utf-8",
    )
    result = build_replay_train_list(sources, tmp_path / "out")

    # The original class contract counts boxes, not merely per-image presence.
    assert result["original"]["class_counts"][DETECTOR_CLASSES[class_id]] == 10
    assert result["selected"]["coverage_ratio_by_source"] == {
        "supplemental_ats": 1.0,
        "synthetic": 1.0,
        "webpii": 1.0,
    }


def test_materialized_replay_contract_is_deterministic_and_tamper_evident(
    tmp_path: Path,
) -> None:
    import yaml

    root = tmp_path / "stage"
    sources = _fixtures(root)
    (sources["synthetic"] / "images/validation").mkdir(parents=True)
    (sources["webpii"] / "images/test").mkdir(parents=True)
    dataset_path = root / "dataset.yaml"
    dataset = {
        "path": "/",
        "train": [
            str(sources["webpii"] / "images/train"),
            str(sources["synthetic"] / "images/train"),
            str(sources["supplemental_ats"] / "images/train"),
        ],
        "val": str(sources["synthetic"] / "images/validation"),
        "test": str(sources["webpii"] / "images/test"),
        "nc": len(DETECTOR_CLASSES),
        "names": list(DETECTOR_CLASSES),
    }
    dataset_path.write_text(yaml.safe_dump(dataset), encoding="utf-8")
    source_path = root / "source_manifest.json"
    source_document = {
        "schema_version": 2,
        "published_splits_preserved": True,
        "test_used_for_checkpoint_selection": False,
        "selection_data": "screen-native-synthetic-validation",
        "test_source": "WebPII/test",
        "classes": list(DETECTOR_CLASSES),
        "dataset_yaml": str(dataset_path),
        "sources": {
            source: {"manifest": str(source_root / "manifest.json")}
            for source, source_root in sources.items()
        },
    }
    source_path.write_text(json.dumps(source_document), encoding="utf-8")
    staging_path = root / "staging_manifest.json"
    staging_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_contract_sha256": "a" * 64,
                "local_dataset_yaml": str(dataset_path),
                "local_source_manifest": str(source_path),
            }
        ),
        encoding="utf-8",
    )
    stage = {
        "root": str(root),
        "dataset_yaml": str(dataset_path),
        "source_manifest": str(source_path),
        "staging_manifest": str(staging_path),
    }

    first = materialize_replay_training_stage(stage)
    first_hashes = (
        first["replay_manifest_sha256"],
        first["replay_train_list_sha256"],
    )
    second = materialize_replay_training_stage(stage)
    assert first_hashes == (
        second["replay_manifest_sha256"],
        second["replay_train_list_sha256"],
    )
    derived_dataset = yaml.safe_load(Path(second["dataset_yaml"]).read_text())
    derived_source = json.loads(Path(second["source_manifest"]).read_text())
    assert validate_composed_dataset_contract(
        derived_dataset, derived_source, Path(second["source_manifest"])
    ) == {index: name for index, name in enumerate(DETECTOR_CLASSES)}

    replay_list = Path(second["replay_train_list"])
    replay_list.write_text(replay_list.read_text() + "/tmp/escape.png\n")
    try:
        validate_composed_dataset_contract(
            derived_dataset, derived_source, Path(second["source_manifest"])
        )
    except RuntimeError as error:
        assert "SHA-256 mismatch" in str(error)
    else:  # pragma: no cover - fail-closed assertion
        raise AssertionError("tampered replay list was accepted")
