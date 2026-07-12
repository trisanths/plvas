from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from training.schema import sha256_file
from training.visual.calibrate_detector_runtime import (
    CALIBRATION_POLICY,
    canonical_json_sha256,
)
from training.visual.build_runtime_selection_stage import (
    materialize_runtime_selection_stage,
)
from training.visual.prepare_webpii import DETECTOR_CLASSES
from training.visual.diagnose_detector_curves import HIGH_RECALL_THRESHOLDS
from training.visual.runtime_selection_evaluator import (
    CalibratedRuntimeSelectionEvaluator,
    load_validated_runtime_selection_contract,
    validate_replay_against_runtime_selection,
)
from training.visual.train_detector import _validate_replay_training_contract


def _artifact(path: Path, records: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if records is not None:
        result["records"] = records
    return result


def _write_list(path: Path, images: list[Path]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{image.resolve()}\n" for image in images), encoding="utf-8")
    return _artifact(path, len(images))


def _write_yaml(path: Path, image_list: Path) -> Path:
    document = {
        "path": "/",
        "train": str(image_list.resolve()),
        "val": str(image_list.resolve()),
        "test": str(image_list.resolve()),
        "nc": len(DETECTOR_CLASSES),
        "names": {index: name for index, name in enumerate(DETECTOR_CLASSES)},
    }
    path.write_text(yaml.safe_dump(document, sort_keys=True), encoding="utf-8")
    return path


def _fixture(
    root: Path, *, webpii_guard_uses_test: bool = False
) -> tuple[Path, dict[str, Path]]:
    webpii = root / "webpii"
    synthetic = root / "synthetic"
    for directory in (
        webpii / "images/train",
        webpii / "images/test",
        synthetic / "images/validation",
    ):
        directory.mkdir(parents=True)
    images = {
        "webpii_calibration": webpii / "images/train/calibration.png",
        "webpii_guard": webpii
        / ("images/test/guard.png" if webpii_guard_uses_test else "images/train/guard.png"),
        "webpii_eligible": webpii / "images/train/eligible.png",
        "synthetic_calibration": synthetic / "images/validation/calibration.png",
        "synthetic_guard": synthetic / "images/validation/guard.png",
    }
    for name, image in images.items():
        image.write_bytes(name.encode())
    webpii_source_manifest = webpii / "manifest.json"
    synthetic_source_manifest = synthetic / "manifest.json"
    webpii_source_manifest.write_text('{"source":"webpii"}\n', encoding="utf-8")
    synthetic_source_manifest.write_text(
        '{"source":"synthetic"}\n', encoding="utf-8"
    )

    selection = root / "selection"
    lists = {
        "webpii_calibration": _write_list(
            selection / "webpii-calibration.txt", [images["webpii_calibration"]]
        ),
        "webpii_guard": _write_list(
            selection / "webpii-guard.txt", [images["webpii_guard"]]
        ),
        "webpii_eligible": _write_list(
            selection / "webpii-eligible.txt", [images["webpii_eligible"]]
        ),
        "synthetic_calibration": _write_list(
            selection / "synthetic-calibration.txt",
            [images["synthetic_calibration"]],
        ),
        "synthetic_guard": _write_list(
            selection / "synthetic-guard.txt", [images["synthetic_guard"]]
        ),
    }
    webpii_split_manifest = selection / "webpii-split.json"
    webpii_split_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "webpii-train-derived-identity-disjoint-guard-split",
                "source": {
                    "root": str(webpii.resolve()),
                    "manifest": {
                        "path": str(webpii_source_manifest.resolve()),
                        "sha256": sha256_file(webpii_source_manifest),
                    },
                },
                "published_test": {
                    "excluded": True,
                    "used_for_selection": False,
                    "emitted_in_dataset_yaml": False,
                    "declared_records": 1,
                },
                "invariants": {
                    "identity_disjoint": True,
                    "group_overlap": 0,
                    "record_overlap": 0,
                    "groups_exhaustive": True,
                    "records_exhaustive": True,
                    "published_test_excluded": True,
                },
                "outputs": {
                    "calibration_images": lists["webpii_calibration"],
                    "guard_images": lists["webpii_guard"],
                    "eligible_images": lists["webpii_eligible"],
                },
                "values_persisted": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    synthetic_split_manifest = selection / "synthetic-split.json"
    synthetic_split_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "deterministic-synthetic-calibration-guard-split",
                "source": {
                    "root": str(synthetic.resolve()),
                    "manifest": str(synthetic_source_manifest.resolve()),
                    "manifest_sha256": sha256_file(synthetic_source_manifest),
                },
                "calibration": {
                    "records": 1,
                    "image_list": lists["synthetic_calibration"]["path"],
                    "image_list_sha256": lists["synthetic_calibration"]["sha256"],
                },
                "guard": {
                    "records": 1,
                    "image_list": lists["synthetic_guard"]["path"],
                    "image_list_sha256": lists["synthetic_guard"]["sha256"],
                },
                "identity_overlap": 0,
                "published_webpii_test_used": False,
                "values_persisted": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    combined_list = selection / "combined-calibration.txt"
    combined_artifact = _write_list(
        combined_list,
        sorted(
            [images["webpii_calibration"], images["synthetic_calibration"]]
        ),
    )
    combined_yaml = _write_yaml(selection / "combined.yaml", combined_list)
    synthetic_guard_list = Path(lists["synthetic_guard"]["path"])
    webpii_guard_list = Path(lists["webpii_guard"]["path"])
    synthetic_guard_yaml = _write_yaml(
        selection / "synthetic-guard.yaml", synthetic_guard_list
    )
    webpii_guard_yaml = _write_yaml(
        selection / "webpii-guard.yaml", webpii_guard_list
    )
    partitions = {
        "webpii": {
            "kind": "webpii-published-train-partitions",
            "source_root": str(webpii.resolve()),
            "allowed_image_root": str((webpii / "images/train").resolve()),
            "forbidden_test_root": str((webpii / "images/test").resolve()),
            "split_manifest": _artifact(webpii_split_manifest),
            "calibration": lists["webpii_calibration"],
            "guard": lists["webpii_guard"],
            "eligible": lists["webpii_eligible"],
        },
        "synthetic": {
            "kind": "synthetic-validation-partitions",
            "source_root": str(synthetic.resolve()),
            "allowed_image_root": str(
                (synthetic / "images/validation").resolve()
            ),
            "split_manifest": _artifact(synthetic_split_manifest),
            "calibration": lists["synthetic_calibration"],
            "guard": lists["synthetic_guard"],
        },
    }
    immutable = {
        "schema_version": 2,
        "seed": 1311,
        "calibration_policy": CALIBRATION_POLICY,
        "frozen_seed_threshold_profile": HIGH_RECALL_THRESHOLDS,
        "published_webpii_test_used": False,
        "webpii_split_manifest_sha256": sha256_file(webpii_split_manifest),
        "synthetic_split_manifest_sha256": sha256_file(synthetic_split_manifest),
        "combined_calibration_images_sha256": combined_artifact["sha256"],
        "synthetic_guard_images_sha256": lists["synthetic_guard"]["sha256"],
        "webpii_guard_images_sha256": lists["webpii_guard"]["sha256"],
        "webpii_eligible_images_sha256": lists["webpii_eligible"]["sha256"],
    }
    contract = {
        "schema_version": 2,
        "kind": "calibrated-runtime-selection-contract",
        "classes": list(DETECTOR_CLASSES),
        "immutable_inputs": immutable,
        "contract_sha256": canonical_json_sha256(immutable),
        "calibration": {
            "dataset_yaml": str(combined_yaml.resolve()),
            "dataset_yaml_sha256": sha256_file(combined_yaml),
            "image_list": combined_artifact["path"],
            "image_list_sha256": combined_artifact["sha256"],
            "records": combined_artifact["records"],
        },
        "guards": {
            "synthetic-guard": {
                "dataset_yaml": str(synthetic_guard_yaml.resolve()),
                "dataset_yaml_sha256": sha256_file(synthetic_guard_yaml),
                "image_list": lists["synthetic_guard"]["path"],
                "image_list_sha256": lists["synthetic_guard"]["sha256"],
                "records": 1,
            },
            "webpii-guard": {
                "dataset_yaml": str(webpii_guard_yaml.resolve()),
                "dataset_yaml_sha256": sha256_file(webpii_guard_yaml),
                "image_list": lists["webpii_guard"]["path"],
                "image_list_sha256": lists["webpii_guard"]["sha256"],
                "records": 1,
            },
        },
        "selection_inventory": {
            "synthetic-guard": lists["synthetic_guard"],
            "webpii-guard": lists["webpii_guard"],
        },
        "source_manifests": {
            "webpii": {
                "path": str(webpii_split_manifest.resolve()),
                "sha256": sha256_file(webpii_split_manifest),
            },
            "synthetic": {
                "path": str(synthetic_split_manifest.resolve()),
                "sha256": sha256_file(synthetic_split_manifest),
            },
        },
        "partitions": partitions,
        "training_eligibility": {
            "webpii_images": lists["webpii_eligible"]["path"],
            "webpii_images_sha256": lists["webpii_eligible"]["sha256"],
            "records": 1,
        },
        "published_test_used": False,
        "values_persisted": False,
    }
    contract_path = selection / "runtime-selection-contract.json"
    contract_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return contract_path, images


def test_validated_selection_contract_proves_exact_non_test_partitions(
    tmp_path: Path,
) -> None:
    contract, images = _fixture(tmp_path)
    provenance = load_validated_runtime_selection_contract(contract)

    assert provenance["webpii"]["guard"] == {images["webpii_guard"].resolve()}
    assert provenance["webpii"]["eligible"] == {
        images["webpii_eligible"].resolve()
    }
    CalibratedRuntimeSelectionEvaluator(
        contract,
        device="cpu",
        batch=1,
        workers=0,
        output_dir=tmp_path / "audit",
    )


def test_materialized_selection_stage_passes_strict_provenance_validation(
    tmp_path: Path,
) -> None:
    from training.tests.test_visual_synthetic_selection_split import (
        _fixture as write_synthetic,
    )
    from training.tests.test_visual_webpii_guard_split import _write_webpii

    webpii = tmp_path / "materialized/webpii"
    synthetic = tmp_path / "materialized/synthetic"
    _write_webpii(webpii, all_classes=True)
    (webpii / "images/test").mkdir(parents=True)
    write_synthetic(synthetic)
    result = materialize_runtime_selection_stage(
        webpii_root=webpii,
        synthetic_root=synthetic,
        output_dir=tmp_path / "materialized/selection",
    )

    provenance = load_validated_runtime_selection_contract(
        Path(result["contract_path"])
    )
    assert len(provenance["calibration_images"]) == (
        result["webpii"]["statistics"]["calibration"]["records"]
        + result["synthetic"]["calibration"]["records"]
    )


def test_selection_contract_rejects_fully_hash_bound_webpii_test_leak(
    tmp_path: Path,
) -> None:
    contract, _images = _fixture(tmp_path, webpii_guard_uses_test=True)

    with pytest.raises(RuntimeError, match="approved non-test split"):
        load_validated_runtime_selection_contract(contract)


def test_selection_contract_rejects_yaml_bypass_and_incomplete_union(
    tmp_path: Path,
) -> None:
    contract_path, images = _fixture(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    guard_yaml = Path(contract["guards"]["webpii-guard"]["dataset_yaml"])
    bypass = _write_list(
        tmp_path / "selection/bypass-test.txt",
        [tmp_path / "webpii/images/test/unseen.png"],
    )
    Path(bypass["path"]).parent.mkdir(parents=True, exist_ok=True)
    guard_document = yaml.safe_load(guard_yaml.read_text(encoding="utf-8"))
    guard_document["val"] = bypass["path"]
    guard_yaml.write_text(yaml.safe_dump(guard_document), encoding="utf-8")
    contract["guards"]["webpii-guard"]["dataset_yaml_sha256"] = sha256_file(
        guard_yaml
    )
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(RuntimeError, match="bypasses its bound image list"):
        load_validated_runtime_selection_contract(contract_path)

    contract_path, images = _fixture(tmp_path / "second")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    combined = Path(contract["calibration"]["image_list"])
    combined.write_text(
        f"{images['synthetic_calibration'].resolve()}\n", encoding="utf-8"
    )
    contract["calibration"]["image_list_sha256"] = sha256_file(combined)
    contract["calibration"]["records"] = 1
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(RuntimeError, match="exact partition union"):
        load_validated_runtime_selection_contract(contract_path)


def test_replay_must_be_eligible_and_disjoint_from_selection_holdouts(
    tmp_path: Path,
) -> None:
    contract, images = _fixture(tmp_path)
    provenance = load_validated_runtime_selection_contract(contract)

    validate_replay_against_runtime_selection(
        [images["webpii_eligible"]], provenance
    )
    with pytest.raises(RuntimeError, match="selection holdout"):
        validate_replay_against_runtime_selection(
            [images["webpii_guard"]], provenance
        )
    with pytest.raises(RuntimeError, match="omits an eligible"):
        validate_replay_against_runtime_selection([], provenance)


def test_training_replay_contract_enforces_runtime_selection_eligibility(
    tmp_path: Path,
) -> None:
    contract_path, images = _fixture(tmp_path)
    supplemental = tmp_path / "supplemental_ats"
    (supplemental / "images/train").mkdir(parents=True)
    (supplemental / "manifest.json").write_text("{}\n", encoding="utf-8")
    source_manifest_path = tmp_path / "source-manifest.json"

    def source_document(replay_root: Path, image: Path) -> dict[str, Any]:
        replay_root.mkdir(parents=True)
        replay_list = replay_root / "train.txt"
        replay_list.write_text(f"{image.resolve()}\n", encoding="utf-8")
        configuration: dict[str, Any] = {}
        replay_manifest = replay_root / "replay.json"
        replay_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "deterministic-source-class-aware-replay-list",
                    "classes": list(DETECTOR_CLASSES),
                    "configuration": configuration,
                    "configuration_sha256": "c" * 64,
                    "consumer_contract": {"physical_batch_guarantee": False},
                    "output": {
                        "train_list": str(replay_list.resolve()),
                        "records": 1,
                        "sha256": sha256_file(replay_list),
                    },
                }
            ),
            encoding="utf-8",
        )
        return {
            "training_replay": {
                "schema_version": 1,
                "kind": "deterministic-source-class-aware-replay-list",
                "physical_batch_guarantee": False,
                "manifest": str(replay_manifest.resolve()),
                "manifest_sha256": sha256_file(replay_manifest),
                "train_list": str(replay_list.resolve()),
                "train_list_sha256": sha256_file(replay_list),
                "records": 1,
                "configuration": configuration,
                "configuration_sha256": "c" * 64,
                "runtime_selection_contract_sha256": sha256_file(contract_path),
            },
            "runtime_selection": {
                "contract": str(contract_path.resolve()),
                "contract_sha256": sha256_file(contract_path),
                "webpii_train_eligible_records": 1,
            },
            "sources": {
                "webpii": {
                    "manifest": str((tmp_path / "webpii/manifest.json").resolve())
                },
                "synthetic": {
                    "manifest": str((tmp_path / "synthetic/manifest.json").resolve())
                },
                "supplemental_ats": {
                    "manifest": str((supplemental / "manifest.json").resolve())
                },
            },
        }

    eligible = source_document(tmp_path / "eligible-replay", images["webpii_eligible"])
    source_manifest_path.write_text(json.dumps(eligible), encoding="utf-8")
    _validate_replay_training_contract(
        Path(eligible["training_replay"]["train_list"]),
        eligible,
        source_manifest_path,
    )

    foreign_webpii = tmp_path / "foreign-webpii"
    foreign_webpii.mkdir()
    (foreign_webpii / "manifest.json").write_text("{}\n", encoding="utf-8")
    mismatched_sources = source_document(
        tmp_path / "mismatched-source-replay", images["webpii_eligible"]
    )
    mismatched_sources["sources"]["webpii"]["manifest"] = str(
        (foreign_webpii / "manifest.json").resolve()
    )
    source_manifest_path.write_text(
        json.dumps(mismatched_sources), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="bound to the composed sources"):
        _validate_replay_training_contract(
            Path(mismatched_sources["training_replay"]["train_list"]),
            mismatched_sources,
            source_manifest_path,
        )

    holdout = source_document(tmp_path / "holdout-replay", images["webpii_guard"])
    source_manifest_path.write_text(json.dumps(holdout), encoding="utf-8")
    with pytest.raises(RuntimeError, match="selection holdout"):
        _validate_replay_training_contract(
            Path(holdout["training_replay"]["train_list"]),
            holdout,
            source_manifest_path,
        )

    ineligible_image = tmp_path / "webpii/images/train/ineligible.png"
    ineligible_image.write_bytes(b"ineligible")
    ineligible = source_document(tmp_path / "ineligible-replay", ineligible_image)
    source_manifest_path.write_text(json.dumps(ineligible), encoding="utf-8")
    with pytest.raises(RuntimeError, match="ineligible WebPII"):
        _validate_replay_training_contract(
            Path(ineligible["training_replay"]["train_list"]),
            ineligible,
            source_manifest_path,
        )
