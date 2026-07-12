from __future__ import annotations

import argparse
import copy
import contextlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from training.visual.prepare_webpii import DETECTOR_CLASSES
from training.visual.project_classifier_checkpoint import CLASSIFIER_TENSOR_NAMES
from training.visual.runtime_safety import (
    REQUIRED_SELECTION_SLICES,
    RUNTIME_SELECTION_POLICY_NAME,
    canonical_json_sha256,
    consider_runtime_selection_candidate,
    runtime_selection_score,
    validate_runtime_selection_candidate,
)
from training.visual.train_detector import (
    CALIBRATED_RUNTIME_SELECTION_SOURCE,
    SAFE_FINE_TUNE_POLICY,
    SAFETY_SELECTION_POLICY,
    SCREENSHOT_AUGMENTATION_POLICY,
    SEED_INCUMBENT_NAME,
    STABLE_OPTIMIZER_POLICY,
    STABLE_OPTIMIZER_POLICY_NAME,
    build_training_resume_contract,
    evaluate_first_epoch_canary,
    evaluation_report,
    load_visual_resume_state,
    prepare_supplemental_test_yaml,
    persist_projected_fp32_ema_checkpoint,
    sha256_file,
    train,
    validate_composed_dataset_contract,
)


NAMES = {index: name for index, name in enumerate(DETECTOR_CLASSES)}
RUNTIME_HASHES = {
    "runtime": "1" * 64,
    "geometry": "2" * 64,
    "calibration": "3" * 64,
    "selection": "4" * 64,
}


def class_values(value: float) -> dict[str, float]:
    return {name: value for name in DETECTOR_CLASSES}


def class_counts(value: int) -> dict[str, int]:
    return {name: value for name in DETECTOR_CLASSES}


def runtime_slice_metrics(
    *,
    checkpoint_sha256: str,
    epoch: int,
    profile_sha256: str,
    inventory_sha256: str,
    privacy: float,
    compatible: float,
    map50: float = 0.95,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "metric_basis": RUNTIME_SELECTION_POLICY_NAME,
        "epoch": epoch,
        "runtime_contract_sha256": RUNTIME_HASHES["runtime"],
        "checkpoint_sha256": checkpoint_sha256,
        "image_inventory_sha256": inventory_sha256,
        "runtime_geometry_sha256": RUNTIME_HASHES["geometry"],
        "threshold_profile_sha256": profile_sha256,
        "map50": map50,
        "per_class_ap50": class_values(map50),
        "per_class_ap50_95": class_values(map50 - 0.1),
        "per_class_fixed_profile_precision": class_values(0.9),
        "per_class_fixed_profile_recall": class_values(0.95),
        "per_class_privacy_coverage_recall": class_values(privacy),
        "per_class_compatible_coverage_recall": class_values(compatible),
        "privacy_coverage_recall": privacy,
        "compatible_coverage_recall": compatible,
        "irrecoverable_misses": 0,
        "per_class_irrecoverable_misses": class_counts(0),
        "per_class_support": class_counts(100),
        "hard_negative_images": 100,
        "hard_negative_images_with_masks": 5,
        "truth": 900,
        "images": 1000,
        "predictions": 1000,
        "preprofile_saturated_images": 0,
    }


def runtime_selection_candidate(
    *,
    checkpoint_sha256: str,
    epoch: int,
    privacy: float,
    compatible: float | None = None,
    map50: float = 0.95,
    profile_value: float = 0.1,
) -> dict[str, object]:
    compatible = privacy - 0.02 if compatible is None else compatible
    threshold_profile = class_values(profile_value)
    profile_sha256 = canonical_json_sha256(threshold_profile)
    return {
        "schema_version": 1,
        "metric_basis": RUNTIME_SELECTION_POLICY_NAME,
        "epoch": epoch,
        "checkpoint_sha256": checkpoint_sha256,
        "calibration_contract_sha256": RUNTIME_HASHES["calibration"],
        "selection_inventory_contract_sha256": RUNTIME_HASHES["selection"],
        "threshold_profile_sha256": profile_sha256,
        "threshold_profile": threshold_profile,
        "calibration_passed": True,
        "calibration_evidence": {
            "kind": "deterministic-test-calibration",
            "passed": True,
            "profile_sha256": profile_sha256,
        },
        "slices": {
            name: runtime_slice_metrics(
                checkpoint_sha256=checkpoint_sha256,
                epoch=epoch,
                profile_sha256=profile_sha256,
                inventory_sha256=str(index + 5) * 64,
                privacy=privacy,
                compatible=compatible,
                map50=map50,
            )
            for index, name in enumerate(REQUIRED_SELECTION_SLICES)
        },
    }


class FakeRuntimeSelectionEvaluator:
    def __init__(
        self,
        *,
        seed_privacy: float = 0.90,
        candidate_privacy: float = 0.92,
    ) -> None:
        self.seed_privacy = seed_privacy
        self.candidate_privacy = candidate_privacy
        self.initialized = False

    def initialize_seed(self, checkpoint: Path) -> dict[str, object]:
        self.initialized = True
        return validate_runtime_selection_candidate(
            runtime_selection_candidate(
                checkpoint_sha256=sha256_file(checkpoint),
                epoch=-1,
                privacy=self.seed_privacy,
            )
        )

    def evaluate_candidate(
        self, checkpoint: Path, *, epoch: int
    ) -> dict[str, object]:
        if not self.initialized:
            raise AssertionError("candidate evaluated before frozen seed")
        return validate_runtime_selection_candidate(
            runtime_selection_candidate(
                checkpoint_sha256=sha256_file(checkpoint),
                epoch=epoch,
                privacy=self.candidate_privacy,
            )
        )


def fake_runtime_diagnostic_modules() -> tuple[dict[str, types.ModuleType], list]:
    calls: list[tuple[object, ...]] = []
    curves = types.ModuleType("training.visual.diagnose_detector_curves")
    curves.HIGH_RECALL_THRESHOLDS = class_values(0.1)
    runtime = types.ModuleType("training.visual.diagnose_detector_runtime")

    def resolve_dataset_images(dataset_yaml: Path, split: str) -> list[Path]:
        calls.append(("resolve", Path(dataset_yaml), split))
        return [Path(f"/fixture/{Path(dataset_yaml).name}-{split}.png")]

    def evaluate_checkpoint(
        checkpoint: Path,
        images: list[Path],
        **kwargs: object,
    ) -> tuple[dict[str, object], dict[str, object]]:
        calls.append(("evaluate", Path(checkpoint), tuple(images), kwargs))
        per_class = {
            name: {
                "coverage_recall": 0.95,
                "class_compatible_coverage_recall": 0.94,
                "support": 100,
            }
            for name in DETECTOR_CLASSES
        }
        return (
            {
                "image_inventory": {
                    "ordered_absolute_paths_sha256": "a" * 64,
                    "records": len(images),
                },
                "profiles": {
                    "high-recall": {
                        "per_class": per_class,
                        "truth": 900,
                        "coverage_recall": 0.95,
                        "hard_negative_mask_rate": 0.01,
                        "irrecoverable_misses": 0,
                        "preprofile_saturated_images": 0,
                    }
                },
            },
            {"passed": True},
        )

    runtime.resolve_dataset_images = resolve_dataset_images
    runtime.evaluate_checkpoint = evaluate_checkpoint
    return (
        {
            "training.visual.diagnose_detector_curves": curves,
            "training.visual.diagnose_detector_runtime": runtime,
        },
        calls,
    )


def write_runtime_selection_contract(root: Path) -> Path:
    stage = root / "runtime-selection"
    stage.mkdir(parents=True)
    artifacts: dict[str, dict[str, object]] = {}
    for name in ("calibration", *REQUIRED_SELECTION_SLICES):
        image = stage / f"{name}.png"
        image.write_bytes(f"runtime-selection-{name}".encode())
        image_list = stage / f"{name}-images.txt"
        image_list.write_text(f"{image.resolve()}\n", encoding="utf-8")
        dataset_yaml = stage / f"{name}.yaml"
        dataset_yaml.write_text(
            f"path: /\ntrain: {image_list.resolve()}\n"
            f"val: {image_list.resolve()}\ntest: {image_list.resolve()}\n",
            encoding="utf-8",
        )
        artifacts[name] = {
            "dataset_yaml": str(dataset_yaml.resolve()),
            "dataset_yaml_sha256": sha256_file(dataset_yaml),
            "image_list": str(image_list.resolve()),
            "image_list_sha256": sha256_file(image_list),
            "records": 1,
        }
    immutable_inputs = {
        "schema_version": 1,
        "seed": 1311,
        "published_webpii_test_used": False,
        "calibration_images_sha256": artifacts["calibration"][
            "image_list_sha256"
        ],
        "guard_images_sha256": {
            name: artifacts[name]["image_list_sha256"]
            for name in REQUIRED_SELECTION_SLICES
        },
    }
    selection_inventory = {
        name: {
            "path": artifacts[name]["image_list"],
            "sha256": artifacts[name]["image_list_sha256"],
            "records": 1,
        }
        for name in REQUIRED_SELECTION_SLICES
    }
    contract = {
        "schema_version": 1,
        "kind": "calibrated-runtime-selection-contract",
        "classes": list(DETECTOR_CLASSES),
        "immutable_inputs": immutable_inputs,
        "contract_sha256": canonical_json_sha256(immutable_inputs),
        "calibration": artifacts["calibration"],
        "guards": {
            name: artifacts[name] for name in REQUIRED_SELECTION_SLICES
        },
        "selection_inventory": selection_inventory,
        "published_test_used": False,
        "values_persisted": False,
    }
    path = stage / "runtime-selection-contract.json"
    path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


class FakeSafeFineTuneController:
    def __init__(self) -> None:
        self.last_report = {
            "passed": True,
            "event": "fixture",
            "trainable_parameters": ["model.23.cv3.fixture.weight"],
        }
        self.event_counts = {
            "on_pretrain_routine_end": 1,
            "on_train_batch_start": 1,
        }

    @staticmethod
    def training_overrides() -> dict[str, int]:
        return {"freeze": 23}


def write_supplemental(root: Path) -> dict[str, object]:
    records: list[str] = []
    for index in range(200):
        image = root / f"images/test/{index:03d}.png"
        label = root / f"labels/test/{index:03d}.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(f"image-{index}".encode())
        label.write_text("", encoding="utf-8")
        records.append(
            json.dumps(
                {
                    "id": f"test-{index}",
                    "split": "test",
                    "image": image.relative_to(root).as_posix(),
                    "image_sha256": sha256_file(image),
                    "label_file": label.relative_to(root).as_posix(),
                    "label_sha256": sha256_file(label),
                },
                sort_keys=True,
            )
        )
    (root / "images/train").mkdir(parents=True)
    records_path = root / "records/test.jsonl"
    records_path.parent.mkdir(parents=True)
    records_path.write_text("\n".join(records) + "\n", encoding="utf-8")
    content_hashes = {
        "images_aggregate_sha256": "1" * 64,
        "labels_aggregate_sha256": "2" * 64,
    }
    manifest = {
        "schema_version": 2,
        "supplemental_only": True,
        "training_use": "train-split-only",
        "test_used_for_checkpoint_selection": False,
        "values_persisted": False,
        "unmapped_source_labels_fail_closed": True,
        "classes": list(DETECTOR_CLASSES),
        "splits": {
            "train": {"records": 800},
            "test": {
                "records": 200,
                "records_path": "records/test.jsonl",
                "records_sha256": sha256_file(records_path),
            },
        },
        "content_hashes": content_hashes,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "training_split_records": 800,
        "test_excluded_from_training_and_selection": True,
        **content_hashes,
    }


def composed_fixture(
    root: Path, supplemental_source: dict[str, object] | None = None
) -> tuple[dict[str, object], dict[str, object], Path]:
    webpii = root / "webpii"
    synthetic = root / "synthetic"
    composed = root / "composed"
    for source_root in (webpii, synthetic, composed):
        source_root.mkdir()
        (source_root / "manifest.json").write_text("{}\n", encoding="utf-8")
    dataset = {
        "path": "/",
        "train": [str(webpii / "images/train")],
        "val": str(synthetic / "images/validation"),
        "test": str(webpii / "images/test"),
        "nc": len(DETECTOR_CLASSES),
        "names": list(DETECTOR_CLASSES),
    }
    sources: dict[str, object] = {
        "webpii": {"manifest": str(webpii / "manifest.json")},
        "synthetic": {"manifest": str(synthetic / "manifest.json")},
    }
    if supplemental_source is not None:
        sources["supplemental_ats"] = supplemental_source
    runtime_selection_contract = write_runtime_selection_contract(root)
    source_manifest = {
        "schema_version": 2 if supplemental_source is not None else 1,
        "published_splits_preserved": True,
        "test_used_for_checkpoint_selection": False,
        "selection_data": CALIBRATED_RUNTIME_SELECTION_SOURCE,
        "test_source": "WebPII/test",
        "classes": list(DETECTOR_CLASSES),
        "sources": sources,
        "runtime_selection": {
            "kind": "calibrated-runtime-selection-contract",
            "contract": str(runtime_selection_contract),
            "contract_sha256": sha256_file(runtime_selection_contract),
            "published_test_used": False,
        },
    }
    return dataset, source_manifest, composed / "manifest.json"


def resume_fixture(
    root: Path,
    *,
    progress_schema: int = 4,
    checkpoint_epochs: int | None = None,
    total_epochs: int = 4,
) -> tuple[argparse.Namespace, dict[str, object], object]:
    run_id = "resume-run"
    local_stage = root / "ephemeral" / run_id
    local_stage.mkdir(parents=True)
    dataset_yaml = local_stage / "dataset.yaml"
    source_manifest = local_stage / "source_manifest.json"
    staging_manifest = local_stage / "staging_manifest.json"
    runtime_selection_contract = write_runtime_selection_contract(local_stage)
    dataset_yaml.write_text("names: fixture\n", encoding="utf-8")
    source_manifest.write_text(
        json.dumps(
            {
                "classes": "fixture",
                "runtime_selection": {
                    "kind": "calibrated-runtime-selection-contract",
                    "contract": str(runtime_selection_contract),
                    "contract_sha256": sha256_file(runtime_selection_contract),
                    "published_test_used": False,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    staging_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_contract_sha256": "a" * 64,
                "local_dataset_yaml_sha256": sha256_file(dataset_yaml),
                "local_source_manifest_sha256": sha256_file(source_manifest),
            }
        ),
        encoding="utf-8",
    )

    output = root / run_id / "visual/training"
    output.mkdir(parents=True)
    base_checkpoint = output / "frozen-seed.pt"
    base_checkpoint.write_bytes(b"frozen-seed")
    persisted = {
        "dataset_yaml": output / "staged-dataset.yaml",
        "source_manifest": output / "staged-source-manifest.json",
        "staging_manifest": output / "dataset-staging-manifest.json",
        "runtime_selection_contract": (
            output / "staged-runtime-selection-contract.json"
        ),
    }
    for source, destination in (
        (dataset_yaml, persisted["dataset_yaml"]),
        (source_manifest, persisted["source_manifest"]),
        (staging_manifest, persisted["staging_manifest"]),
        (
            runtime_selection_contract,
            persisted["runtime_selection_contract"],
        ),
    ):
        destination.write_bytes(source.read_bytes())
    data_staging = {
        f"{name}_{field}": (
            str(path) if field == "path" else sha256_file(path)
        )
        for name, path in persisted.items()
        for field in ("path", "sha256")
    }
    args = argparse.Namespace(
        dataset_yaml=dataset_yaml,
        source_manifest=source_manifest,
        local_staging_manifest=staging_manifest,
        runtime_selection_contract=runtime_selection_contract,
        data_staging=data_staging,
        output_dir=output,
        base_checkpoint=base_checkpoint,
        run_id=run_id,
        attempt_id="default",
        attempt_manifest_sha256=None,
        epochs=total_epochs,
        image_size=640,
        batch_size=64,
        seed=1311,
        runtime_selection_evaluator=FakeRuntimeSelectionEvaluator(
            seed_privacy=0.90,
            candidate_privacy=0.915,
        ),
    )
    contract = build_training_resume_contract(args, NAMES)
    weights = output / "ultralytics-run/weights"
    weights.mkdir(parents=True)
    last = weights / "last.pt"
    best = weights / "best.pt"
    safety_best = output / "safety-best.pt"
    last.write_bytes(b"durable-last")
    best.write_bytes(b"durable-best")
    safety_best.write_bytes(last.read_bytes())

    def checkpoint_record(path: Path) -> dict[str, object]:
        return {
            "path": str(path),
            "exists": True,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    seed_decision, selection_state = consider_runtime_selection_candidate(
        None,
        runtime_selection_candidate(
            checkpoint_sha256=sha256_file(base_checkpoint),
            epoch=-1,
            privacy=0.90,
        ),
    )
    assert selection_state is not None
    seed_incumbent = {
        **seed_decision["candidate"],
        "checkpoint": str(base_checkpoint),
        "checkpoint_sha256": sha256_file(base_checkpoint),
    }
    history: list[dict[str, object]] = []
    for epoch, privacy in ((0, 0.92), (1, 0.915)):
        metrics = runtime_selection_candidate(
            checkpoint_sha256=sha256_file(last),
            epoch=epoch,
            privacy=privacy,
        )
        decision, next_state = consider_runtime_selection_candidate(
            selection_state,
            metrics,
        )
        if next_state is not None:
            selection_state = next_state
        record = {
            "epoch": epoch,
            "metric_basis": RUNTIME_SELECTION_POLICY_NAME,
            "runtime_selection_metrics": decision["candidate"],
            "selected": decision["selected"],
            "selection_reason": decision["reason"],
            "selection_deltas": decision["deltas"],
            "selection_gates": decision["gates"],
            "canary_gate": None,
        }
        if epoch == 0:
            record["canary_gate"] = evaluate_first_epoch_canary(
                seed_incumbent, record
            )
        history.append(record)
    incumbent = selection_state["incumbent"]
    runtime_profile = output / "safety-best-thresholds.json"
    runtime_profile.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "metric_basis": RUNTIME_SELECTION_POLICY_NAME,
                "checkpoint_sha256": incumbent["checkpoint_sha256"],
                "threshold_profile": incumbent["threshold_profile"],
                "threshold_profile_sha256": incumbent[
                    "threshold_profile_sha256"
                ],
                "calibration_contract_sha256": incumbent[
                    "calibration_contract_sha256"
                ],
                "selection_inventory_contract_sha256": incumbent[
                    "selection_inventory_contract_sha256"
                ],
                "calibration_evidence": incumbent["calibration_evidence"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    progress = {
        "schema_version": progress_schema,
        "epoch": 1,
        "selection_policy": SAFETY_SELECTION_POLICY,
        "best_score": list(runtime_selection_score(incumbent)),
        "seed_incumbent": seed_incumbent,
        "selection_state": selection_state,
        "history": history,
        "resume_contract": contract,
        "checkpoints": {
            "last": checkpoint_record(last),
            "best": checkpoint_record(best),
            "safety_best": checkpoint_record(safety_best),
            "runtime_profile": checkpoint_record(runtime_profile),
        },
    }
    (output / "safety-selection-progress.json").write_text(
        json.dumps(progress), encoding="utf-8"
    )
    (output / SEED_INCUMBENT_NAME).write_text(
        json.dumps(seed_incumbent), encoding="utf-8"
    )
    train_args = {
        "epochs": checkpoint_epochs if checkpoint_epochs is not None else total_epochs,
        "data": str(dataset_yaml),
        "save_dir": str(output / "ultralytics-run"),
        "imgsz": args.image_size,
        "batch": args.batch_size,
        "seed": args.seed,
        "freeze": 23,
        **STABLE_OPTIMIZER_POLICY,
        **SCREENSHOT_AUGMENTATION_POLICY,
    }

    class FakeTorch:
        @staticmethod
        def load(
            path: str, *, map_location: str, weights_only: bool
        ) -> dict[str, object]:
            if (
                Path(path) != last
                or map_location != "cpu"
                or weights_only is not False
            ):
                raise AssertionError("resume loaded an unexpected checkpoint")
            return {
                "epoch": 1,
                "optimizer": {"state": {}, "param_groups": []},
                "scaler": {},
                "ema": object(),
                "train_args": train_args,
            }

    return args, contract, FakeTorch


class DetectorEvaluationContractTests(unittest.TestCase):
    def test_projected_fp32_checkpoint_restores_every_frozen_tensor(self) -> None:
        class FakeDType:
            def __init__(self, name: str) -> None:
                self.name = name

            def __str__(self) -> str:
                return self.name

        class FakeTensor:
            def __init__(self, values: list[float], dtype: str = "torch.float32"):
                self.values = list(values)
                self.shape = (len(values),)
                self.dtype = FakeDType(dtype)

            def detach(self):
                return self

            def cpu(self):
                return self

            def numel(self) -> int:
                return len(self.values)

            def equal(self, other) -> bool:
                return (
                    self.values == other.values
                    and str(self.dtype) == str(other.dtype)
                )

            def copy_(self, other):
                self.values = list(other.values)
                self.dtype = FakeDType(str(other.dtype))
                return self

            def is_floating_point(self) -> bool:
                return str(self.dtype).startswith("torch.float")

            def __getitem__(self, index: int):
                return FakeTensorRow(self, index)

        class FakeTensorRow:
            def __init__(self, parent: FakeTensor, index: int) -> None:
                self.parent = parent
                self.index = index
                self.shape = ()

            @property
            def dtype(self):
                return self.parent.dtype

            def detach(self):
                return self

            def cpu(self):
                return self

            def equal(self, other) -> bool:
                return (
                    self.parent.values[self.index]
                    == other.parent.values[other.index]
                    and str(self.dtype) == str(other.dtype)
                )

            def copy_(self, other):
                self.parent.values[self.index] = other.parent.values[other.index]
                return self

        class FakeModel:
            def __init__(self, state: dict[str, FakeTensor]) -> None:
                self.state = state

            def state_dict(self):
                return self.state

            def float(self):
                for tensor in self.state.values():
                    if tensor.is_floating_point():
                        tensor.dtype = FakeDType("torch.float32")
                return self

            def cpu(self):
                return self

        class FakeTorch:
            stored = None
            initial = {
                "epoch": 0,
                "optimizer": {"state": {}},
                "scaler": {},
                "ema": object(),
            }

            @classmethod
            def load(cls, _path, *, map_location, weights_only):
                self.assertEqual(map_location, "cpu")
                self.assertFalse(weights_only)
                return copy.deepcopy(cls.stored or cls.initial)

            @classmethod
            def save(cls, value, path):
                cls.stored = copy.deepcopy(value)
                Path(path).write_bytes(b"projected-fp32-checkpoint")

            @staticmethod
            def no_grad():
                return contextlib.nullcontext()

        reference_state = {"model.0.weight": FakeTensor([1.0])}
        live_state = {
            "model.0.weight": FakeTensor([9.0], "torch.float16")
        }
        for name in CLASSIFIER_TENSOR_NAMES:
            reference_state[name] = FakeTensor(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            )
            live_state[name] = FakeTensor(
                [10.0, 1.0, 2.0, 99.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "torch.float16",
            )
        reference = FakeModel(reference_state)
        live_ema = FakeModel(live_state)
        trainer = SimpleNamespace(
            epoch=0, ema=SimpleNamespace(ema=live_ema)
        )
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "last.pt"
            checkpoint.write_bytes(b"ultralytics-fp16-checkpoint")
            audit = persist_projected_fp32_ema_checkpoint(
                reference,
                trainer,
                checkpoint,
                trainable_parameter_names=CLASSIFIER_TENSOR_NAMES,
                protected_class_rows=(3,),
                torch_module=FakeTorch,
            )

        persisted = FakeTorch.stored["ema"].state_dict()
        self.assertEqual(persisted["model.0.weight"].values, [1.0])
        for name in CLASSIFIER_TENSOR_NAMES:
            self.assertEqual(
                persisted[name].values,
                [10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            )
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["changed_frozen_tensors"], [])
        self.assertEqual(
            audit["changed_trainable_tensors"], sorted(CLASSIFIER_TENSOR_NAMES)
        )
        self.assertEqual(audit["protected_classes"], ["ADDRESS"])
        self.assertEqual(audit["protected_row_mismatches"], [])
        self.assertEqual(audit["non_fp32_floating_tensors"], [])

    def test_classification_only_optimizer_policy_is_pinned(self) -> None:
        self.assertEqual(
            STABLE_OPTIMIZER_POLICY_NAME,
            "safe-output-classifier-adamw-v5-fp32-trust-region",
        )
        self.assertEqual(STABLE_OPTIMIZER_POLICY["optimizer"], "AdamW")
        self.assertEqual(STABLE_OPTIMIZER_POLICY["lr0"], 1e-7)
        self.assertEqual(STABLE_OPTIMIZER_POLICY["weight_decay"], 0.0)
        self.assertEqual(STABLE_OPTIMIZER_POLICY["cls_pw"], 0.0)
        self.assertEqual(STABLE_OPTIMIZER_POLICY["warmup_epochs"], 1.0)
        self.assertEqual(
            SAFE_FINE_TUNE_POLICY["protected_classifier_classes"], ["ADDRESS"]
        )
        self.assertEqual(SAFE_FINE_TUNE_POLICY["protected_classifier_rows"], [3])
        self.assertEqual(
            len(SAFE_FINE_TUNE_POLICY["trainable_parameters"]), 6
        )
        self.assertEqual(
            SAFE_FINE_TUNE_POLICY["checkpoint_ema_precision"], "float32"
        )

    def test_first_epoch_canary_rejects_runtime_or_ap_collapse(self) -> None:
        seed = runtime_selection_candidate(
            checkpoint_sha256="8" * 64,
            epoch=-1,
            privacy=0.95,
        )
        candidate = runtime_selection_candidate(
            checkpoint_sha256="9" * 64,
            epoch=0,
            privacy=0.95,
        )
        safe_record = {
            "runtime_selection_metrics": candidate,
            "selection_gates": {"all_hard_gates": True},
        }
        self.assertTrue(evaluate_first_epoch_canary(seed, safe_record)["passed"])

        collapsed = copy.deepcopy(safe_record)
        for summary in collapsed["runtime_selection_metrics"]["slices"].values():
            summary["per_class_privacy_coverage_recall"]["PHONE"] = 0.70
        result = evaluate_first_epoch_canary(seed, collapsed)
        self.assertFalse(result["passed"])
        self.assertFalse(
            result["gates"][
                "seed_synthetic_guard_individual_privacy_recall_drop"
            ]
        )

        ap_collapse = copy.deepcopy(safe_record)
        for summary in ap_collapse["runtime_selection_metrics"]["slices"].values():
            summary["map50"] = 0.90
            summary["per_class_ap50"]["ADDRESS"] = 0.90
        ap_result = evaluate_first_epoch_canary(seed, ap_collapse)
        self.assertFalse(ap_result["passed"])
        self.assertFalse(
            ap_result["gates"]["seed_synthetic_guard_map50_drop"]
        )

    def test_modal_visual_resume_flag_uses_spawn_and_get(self) -> None:
        modal_source = (
            Path(__file__).resolve().parents[1] / "modal_app.py"
        ).read_text(encoding="utf-8")
        self.assertIn("visual_resume: bool = False", modal_source)
        self.assertIn('visual_attempt: str = "default"', modal_source)
        self.assertIn('visual_seed_checkpoint: str = ""', modal_source)
        self.assertIn('visual_seed_sha256: str = ""', modal_source)
        self.assertIn("attempt_output_dir(paths[\"root\"], attempt_id)", modal_source)
        self.assertIn("copy_workers=VISUAL_STAGE_COPY_WORKERS", modal_source)
        self.assertIn("cpu=VISUAL_TRAIN_CPUS", modal_source)
        self.assertIn("workers=VISUAL_DATALOADER_WORKERS", modal_source)
        self.assertIn("batch_size=VISUAL_TRAIN_BATCH_SIZE", modal_source)
        self.assertIn('VISUAL_TRAIN_GPU = "L4"', modal_source)
        self.assertIn("VISUAL_TRAIN_CPUS = 16.0", modal_source)
        self.assertIn("VISUAL_DATALOADER_WORKERS = 16", modal_source)
        self.assertIn("VISUAL_TRAIN_BATCH_SIZE = 64", modal_source)
        self.assertIn("resume: bool = False", modal_source)
        self.assertIn("visual_train_remote.spawn(", modal_source)
        self.assertIn("result[\"visual_train\"] = visual_call.get()", modal_source)

    def test_resume_restores_safety_history_from_only_persisted_last(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args, contract, torch_module = resume_fixture(Path(temporary))
            self.assertNotIn("workers", contract)
            state = load_visual_resume_state(args, contract, torch_module)
            self.assertEqual(
                state["last_path"],
                args.output_dir / "ultralytics-run/weights/last.pt",
            )
            self.assertEqual(len(state["history"]), 2)
            self.assertEqual(
                state["best_key"], (0.92, 0.92, 0.9, 0.9, 0.95, -1.0)
            )
            self.assertEqual(state["selection_state"]["incumbent"]["epoch"], 0)

    def test_resume_rejects_tampered_checkpoint_and_epoch_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args, contract, torch_module = resume_fixture(Path(temporary))
            last = args.output_dir / "ultralytics-run/weights/last.pt"
            last.write_bytes(b"tampered-xxx")
            with self.assertRaisesRegex(RuntimeError, "last.*SHA-256"):
                load_visual_resume_state(args, contract, torch_module)

        with tempfile.TemporaryDirectory() as temporary:
            args, contract, torch_module = resume_fixture(
                Path(temporary), checkpoint_epochs=5
            )
            with self.assertRaisesRegex(RuntimeError, "total epochs"):
                load_visual_resume_state(args, contract, torch_module)

    def test_resume_rejects_jointly_tampered_seed_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args, contract, torch_module = resume_fixture(Path(temporary))
            seed_path = args.output_dir / SEED_INCUMBENT_NAME
            progress_path = args.output_dir / "safety-selection-progress.json"
            seed = json.loads(seed_path.read_text(encoding="utf-8"))
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            seed["checkpoint_sha256"] = "f" * 64
            progress["seed_incumbent"] = seed
            seed_path.write_text(json.dumps(seed), encoding="utf-8")
            progress_path.write_text(json.dumps(progress), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "immutable seed"):
                load_visual_resume_state(args, contract, torch_module)

    def test_legacy_progress_is_rejected_after_policy_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args, contract, torch_module = resume_fixture(
                Path(temporary), progress_schema=1
            )
            with self.assertRaisesRegex(RuntimeError, "legacy visual progress"):
                load_visual_resume_state(args, contract, torch_module)

    def test_failed_first_epoch_canary_is_never_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args, contract, torch_module = resume_fixture(Path(temporary))
            progress_path = args.output_dir / "safety-selection-progress.json"
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            progress["history"][0]["canary_gate"] = {"passed": False}
            progress_path.write_text(json.dumps(progress), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "failed first-epoch"):
                load_visual_resume_state(args, contract, torch_module)

    def test_pretraining_budget_pause_restarts_without_loading_last(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args, contract, _torch_module = resume_fixture(Path(temporary))
            progress_path = args.output_dir / "safety-selection-progress.json"
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            seed = progress["seed_incumbent"]
            seed_decision, seed_state = consider_runtime_selection_candidate(
                None, seed
            )
            self.assertIsNotNone(seed_state)
            incumbent = seed_state["incumbent"]
            progress.update(
                {
                    "status": "budget-paused-before-training",
                    "epoch": -1,
                    "best_score": list(runtime_selection_score(incumbent)),
                    "selection_state": seed_state,
                    "history": [],
                    "resume_allowed": True,
                }
            )
            progress_path.write_text(json.dumps(progress), encoding="utf-8")

            class NoCheckpointLoad:
                @staticmethod
                def load(*_args, **_kwargs):
                    raise AssertionError("pre-training restart loaded last.pt")

            state = load_visual_resume_state(args, contract, NoCheckpointLoad)
            self.assertTrue(seed_decision["selected"])
            self.assertTrue(state["restart_before_training"])
            self.assertFalse(state["training_complete"])
            self.assertEqual(state["history"], [])
            self.assertIsNone(state["last_path"])

    def test_final_epoch_progress_recovers_manifest_without_retraining(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, _contract, _torch_loader = resume_fixture(root, total_epochs=2)
            args.base_license = "AGPL-3.0-only"
            args.base_source = "fixture"
            args.commercial_license_approved = False
            args.resume = True
            args.device = "cpu"
            args.workers = 0
            args.patience = args.epochs
            args.checkpoint_commit_hook = None

            # Ultralytics 8.4.92 rewrites these during final_eval(). Recovery
            # must not trust their pre-finalization progress hashes.
            weights = args.output_dir / "ultralytics-run/weights"
            (weights / "last.pt").write_bytes(b"stripped-last")
            (weights / "best.pt").write_bytes(b"stripped-best")

            class FakeYOLO:
                loaded: list[str] = []
                training_calls: list[dict[str, object]] = []
                validation_calls: list[dict[str, object]] = []

                def __init__(self, source: str) -> None:
                    self.source = source
                    self.loaded.append(source)

                def train(self, **kwargs: object) -> None:
                    self.training_calls.append(kwargs)
                    raise AssertionError("finalize-only recovery retrained the model")

                def val(self, **kwargs: object) -> SimpleNamespace:
                    self.validation_calls.append(kwargs)
                    return SimpleNamespace(
                        names=NAMES,
                        box=SimpleNamespace(
                            p=[0.9] * len(NAMES),
                            r=[0.85] * len(NAMES),
                            ap_class_index=list(NAMES),
                        ),
                        results_dict={"metrics/mAP50(B)": 0.5},
                    )

            torch = types.ModuleType("torch")
            torch.__version__ = "fixture-torch"
            torch.cuda = SimpleNamespace(is_available=lambda: False)
            torch.load = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("finalize-only recovery loaded last.pt")
            )
            yaml = types.ModuleType("yaml")
            yaml.safe_load = lambda _content: {}
            ultralytics = types.ModuleType("ultralytics")
            ultralytics.YOLO = FakeYOLO
            runtime_modules, runtime_calls = fake_runtime_diagnostic_modules()

            with (
                patch.dict(
                    sys.modules,
                    {
                        "torch": torch,
                        "yaml": yaml,
                        "ultralytics": ultralytics,
                        **runtime_modules,
                    },
                ),
                patch(
                    "training.visual.train_detector.validate_composed_dataset_contract",
                    return_value=NAMES,
                ),
                patch(
                    "training.visual.train_detector.prepare_supplemental_test_yaml",
                    return_value=None,
                ),
                patch(
                    "training.visual.train_detector.importlib.metadata.version",
                    return_value="fixture-ultralytics",
                ),
            ):
                manifest = train(args)

            self.assertEqual(FakeYOLO.training_calls, [])
            self.assertTrue(manifest["training_attempt"]["finalize_only_recovery"])
            self.assertEqual(len(manifest["history"]), 2)
            safety_best = str(args.output_dir / "safety-best.pt")
            self.assertEqual(
                [call["name"] for call in FakeYOLO.validation_calls],
                [
                    "safety-best-validation",
                    "frozen-base-webpii-test",
                    "safety-best-webpii-test",
                ],
            )
            self.assertIn(safety_best, FakeYOLO.loaded)
            self.assertEqual(
                [call[0] for call in runtime_calls],
                ["resolve", "evaluate", "evaluate"],
            )
            self.assertTrue((args.output_dir / "training_manifest.json").is_file())
            invocation = json.loads(
                (args.output_dir / "training-invocation-state.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(invocation["status"], "completed")
            self.assertTrue(invocation["finalize_only_recovery"])

    def test_train_resumes_with_optimizer_state_and_restored_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, _contract, torch_loader = resume_fixture(root)
            commits: list[str] = []
            args.base_license = "AGPL-3.0-only"
            args.base_source = "fixture"
            args.commercial_license_approved = False
            args.resume = True
            args.device = "cpu"
            args.workers = 0
            args.patience = args.epochs
            args.checkpoint_commit_hook = lambda: commits.append("commit")

            class FakeYOLO:
                loaded: list[str] = []
                training_calls: list[dict[str, object]] = []

                def __init__(self, source: str) -> None:
                    self.source = source
                    self.model = SimpleNamespace()
                    self.callbacks: dict[str, list[object]] = {}
                    self.loaded.append(source)

                def add_callback(self, event: str, callback: object) -> None:
                    self.callbacks.setdefault(event, []).append(callback)

                def train(self, **kwargs: object) -> SimpleNamespace:
                    self.training_calls.append(kwargs)
                    weights = args.output_dir / "ultralytics-run/weights"
                    last = weights / "last.pt"
                    best = weights / "best.pt"
                    last.write_bytes(b"resumed-last")
                    best.write_bytes(b"resumed-aggregate-best")
                    trainer = SimpleNamespace(
                        epoch=2,
                        last=str(last),
                        best=str(best),
                        validator=SimpleNamespace(
                            metrics=SimpleNamespace(
                                box=SimpleNamespace(
                                    p=[0.75] * len(NAMES),
                                    r=[0.65] * len(NAMES),
                                    ap_class_index=list(NAMES),
                                )
                            )
                        ),
                        model=SimpleNamespace(names=NAMES),
                        metrics={"metrics/mAP50(B)": 0.5},
                        safe_finetune_audit={
                            "passed": True,
                            "trainable_parameters": [
                                "model.23.cv3.fixture.weight"
                            ],
                        },
                    )
                    for callback in self.callbacks.get(
                        "on_pretrain_routine_end", []
                    ):
                        callback(trainer)
                    callbacks = self.callbacks.get("on_model_save", [])
                    if not callbacks:
                        raise AssertionError("resume callback was not installed")
                    for callback in callbacks:
                        callback(trainer)
                    return SimpleNamespace(results_dict={"train/box_loss": 0.25})

                def val(self, **_kwargs: object) -> SimpleNamespace:
                    return SimpleNamespace(
                        names=NAMES,
                        box=SimpleNamespace(
                            p=[0.9] * len(NAMES),
                            r=[0.85] * len(NAMES),
                            ap_class_index=list(NAMES),
                        ),
                        results_dict={"metrics/mAP50(B)": 0.5},
                    )

            numpy = types.ModuleType("numpy")
            numpy.mean = lambda values: sum(values) / len(values)
            torch = types.ModuleType("torch")
            torch.__version__ = "fixture-torch"
            torch.cuda = SimpleNamespace(is_available=lambda: False)
            torch.load = torch_loader.load
            yaml = types.ModuleType("yaml")
            yaml.safe_load = lambda _content: {}
            ultralytics = types.ModuleType("ultralytics")
            ultralytics.YOLO = FakeYOLO
            runtime_modules, runtime_calls = fake_runtime_diagnostic_modules()

            with (
                patch.dict(
                    sys.modules,
                    {
                        "numpy": numpy,
                        "torch": torch,
                        "yaml": yaml,
                        "ultralytics": ultralytics,
                        **runtime_modules,
                    },
                ),
                patch(
                    "training.visual.train_detector.validate_composed_dataset_contract",
                    return_value=NAMES,
                ),
                patch(
                    "training.visual.train_detector.prepare_supplemental_test_yaml",
                    return_value=None,
                ),
                patch(
                    "training.visual.train_detector.importlib.metadata.version",
                    return_value="fixture-ultralytics",
                ),
                patch(
                    "training.visual.safe_finetune.wire_safe_finetune_callbacks",
                    return_value=FakeSafeFineTuneController(),
                ),
                patch(
                    "training.visual.safe_finetune.validate_seed_state_parity",
                    return_value={"passed": True},
                ),
                patch(
                    "training.visual.safe_finetune.validate_frozen_state_parity",
                    return_value={"passed": True},
                ),
                patch(
                    "training.visual.train_detector.persist_projected_fp32_ema_checkpoint",
                    side_effect=lambda *_args, **_kwargs: {
                        "passed": True,
                        "checkpoint_sha256": sha256_file(
                            args.output_dir / "ultralytics-run/weights/last.pt"
                        ),
                    },
                ),
            ):
                manifest = train(args)

            self.assertEqual(len(FakeYOLO.training_calls), 1)
            training_call = FakeYOLO.training_calls[0]
            self.assertTrue(training_call["resume"])
            self.assertTrue(training_call["save"])
            self.assertEqual(training_call["save_period"], 1)
            self.assertEqual(training_call["epochs"], 4)
            expected_last = args.output_dir / "ultralytics-run/weights/last.pt"
            self.assertIn(str(expected_last), FakeYOLO.loaded)
            self.assertEqual(len(manifest["history"]), 3)
            self.assertFalse(manifest["history"][-1]["selected"])
            self.assertEqual(
                manifest["best_checkpoint"]["score"],
                [0.92, 0.92, 0.9, 0.9, 0.95, -1.0],
            )
            persistence = manifest["checkpoint_persistence"]
            self.assertTrue(persistence["resume_supported"])
            self.assertTrue(persistence["resumed"])
            self.assertEqual(persistence["resumed_from_epoch"], 1)
            self.assertEqual(persistence["committed_epochs"], 3)
            self.assertEqual(persistence["committed_epochs_this_invocation"], 1)
            self.assertEqual(commits, ["commit"])
            self.assertEqual(
                [call[0] for call in runtime_calls],
                ["resolve", "evaluate", "evaluate"],
            )
            progress = json.loads(
                (args.output_dir / "safety-selection-progress.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(progress["schema_version"], 4)
            self.assertEqual(progress["resume_contract"], persistence["resume_contract"])

    def test_composed_contract_requires_exact_published_test_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, source_manifest, manifest_path = composed_fixture(root)
            self.assertEqual(
                validate_composed_dataset_contract(
                    dataset, source_manifest, manifest_path
                ),
                NAMES,
            )

            missing_test = dict(dataset)
            del missing_test["test"]
            with self.assertRaisesRegex(RuntimeError, "must include.*published test"):
                validate_composed_dataset_contract(
                    missing_test, source_manifest, manifest_path
                )

            wrong_test = dict(dataset)
            wrong_test["test"] = str(root / "other/images/test")
            with self.assertRaisesRegex(RuntimeError, "composed WebPII"):
                validate_composed_dataset_contract(
                    wrong_test, source_manifest, manifest_path
                )

    def test_evaluation_report_aligns_and_reports_missing_test_classes(self) -> None:
        result = SimpleNamespace(
            names=NAMES,
            box=SimpleNamespace(
                p=[0.9, 0.8],
                r=[0.7, 0.6],
                ap_class_index=[0, 4],
            ),
            results_dict={"metrics/mAP50(B)": 0.25},
        )
        report = evaluation_report(result, NAMES, "missing_test_classes")
        self.assertEqual(report["aggregate"]["metrics/mAP50(B)"], 0.25)
        self.assertEqual(report["per_class"]["NAME"]["recall"], 0.7)
        self.assertEqual(report["per_class"]["CVC"]["precision"], 0.0)
        self.assertIn("CVC", report["missing_test_classes"])
        self.assertEqual(report["missing_classes"], report["missing_test_classes"])
        self.assertEqual(report["minimum_secret_class_recall"], 0.0)

    def test_supplemental_test_requires_exactly_200_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            supplemental = root / "supplemental"
            supplemental.mkdir()
            source = write_supplemental(supplemental)
            manifest_path = supplemental / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["splits"]["test"]["records"] = 199
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            source["manifest_sha256"] = sha256_file(manifest_path)
            output = root / "output"
            output.mkdir()
            with self.assertRaisesRegex(RuntimeError, "exactly 200 records"):
                prepare_supplemental_test_yaml(
                    {"sources": {"supplemental_ats": source}},
                    root / "composed/manifest.json",
                    output,
                    NAMES,
                )

    def test_train_compares_frozen_and_selected_checkpoints_on_both_tests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            supplemental = root / "supplemental"
            supplemental.mkdir()
            supplemental_source = write_supplemental(supplemental)
            dataset, source_manifest, source_manifest_path = composed_fixture(
                root, supplemental_source
            )
            source_manifest_path.write_text(
                json.dumps(source_manifest), encoding="utf-8"
            )
            dataset_yaml = root / "composed/dataset.yaml"
            dataset_yaml.write_text("fixture: true\n", encoding="utf-8")
            base_checkpoint = root / "base.pt"
            base_checkpoint.write_bytes(b"base-checkpoint")
            output = root / "output"
            last_checkpoint = output / "ultralytics-run/weights/last.pt"
            best_checkpoint = output / "ultralytics-run/weights/best.pt"
            commit_calls: list[str] = []

            class FakeYOLO:
                loaded: list[str] = []
                validation_calls: list[dict[str, object]] = []
                training_calls: list[dict[str, object]] = []

                def __init__(self, source: str) -> None:
                    self.source = source
                    self.model = SimpleNamespace()
                    self.callbacks: dict[str, list[object]] = {}
                    self.loaded.append(source)

                def add_callback(self, event: str, callback: object) -> None:
                    self.callbacks.setdefault(event, []).append(callback)

                def train(self, **kwargs: object) -> SimpleNamespace:
                    self.training_calls.append(kwargs)
                    last_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    last_checkpoint.write_bytes(b"selected-checkpoint")
                    best_checkpoint.write_bytes(b"aggregate-best-checkpoint")
                    metrics = SimpleNamespace(
                        box=SimpleNamespace(
                            p=[0.7] * len(NAMES),
                            r=[0.8] * len(NAMES),
                            ap_class_index=list(NAMES),
                        )
                    )
                    trainer = SimpleNamespace(
                        epoch=0,
                        last=str(last_checkpoint),
                        best=str(best_checkpoint),
                        validator=SimpleNamespace(metrics=metrics),
                        model=SimpleNamespace(names=NAMES),
                        metrics={"metrics/mAP50(B)": 0.4},
                        safe_finetune_audit={
                            "passed": True,
                            "trainable_parameters": [
                                "model.23.cv3.fixture.weight"
                            ],
                        },
                    )
                    for callback in self.callbacks.get(
                        "on_pretrain_routine_end", []
                    ):
                        callback(trainer)
                    callbacks = self.callbacks.get("on_model_save", [])
                    if not callbacks:
                        raise AssertionError("training callback was not installed")
                    for callback in callbacks:
                        callback(trainer)
                    return SimpleNamespace(results_dict={"train/box_loss": 0.5})

                def val(self, **kwargs: object) -> SimpleNamespace:
                    self.validation_calls.append(kwargs)
                    name = kwargs["name"]
                    precision = 0.9
                    recall = 0.85
                    if name == "frozen-seed-validation":
                        indices = list(NAMES)
                        aggregate = {"metrics/mAP50(B)": 0.3}
                        precision = 0.6
                        recall = 0.6
                    elif name == "frozen-base-webpii-test":
                        indices = [index for index in NAMES if index != 5]
                        aggregate = {"metrics/mAP50(B)": 0.1}
                    elif name == "frozen-base-supplemental-test":
                        indices = [index for index in NAMES if index != 6]
                        aggregate = {"metrics/mAP50(B)": 0.2}
                    elif name == "safety-best-validation":
                        indices = list(NAMES)
                        aggregate = {"metrics/mAP50(B)": 0.4}
                    elif name == "safety-best-webpii-test":
                        indices = [index for index in NAMES if index != 5]
                        aggregate = {"metrics/mAP50(B)": 0.5}
                    elif name == "safety-best-supplemental-test":
                        indices = [index for index in NAMES if index != 6]
                        aggregate = {"metrics/mAP50(B)": 0.6}
                    else:
                        raise AssertionError(name)
                    return SimpleNamespace(
                        names=NAMES,
                        box=SimpleNamespace(
                            p=[precision] * len(indices),
                            r=[recall] * len(indices),
                            ap_class_index=indices,
                        ),
                        results_dict=aggregate,
                    )

            numpy = types.ModuleType("numpy")
            numpy.mean = lambda values: sum(values) / len(values)
            torch = types.ModuleType("torch")
            torch.__version__ = "fixture-torch"
            torch.cuda = SimpleNamespace(is_available=lambda: False)
            yaml = types.ModuleType("yaml")
            yaml.safe_load = lambda _content: dataset
            ultralytics = types.ModuleType("ultralytics")
            ultralytics.YOLO = FakeYOLO
            runtime_modules, runtime_calls = fake_runtime_diagnostic_modules()

            args = argparse.Namespace(
                dataset_yaml=dataset_yaml,
                source_manifest=source_manifest_path,
                runtime_selection_contract=Path(
                    source_manifest["runtime_selection"]["contract"]
                ),
                runtime_selection_evaluator=FakeRuntimeSelectionEvaluator(),
                base_checkpoint=base_checkpoint,
                base_license="AGPL-3.0-only",
                base_source="fixture",
                commercial_license_approved=False,
                output_dir=output,
                epochs=1,
                image_size=640,
                batch_size=4,
                device="cpu",
                workers=0,
                patience=1,
                seed=1311,
                data_staging={"fixture": True},
                checkpoint_commit_hook=lambda: commit_calls.append("commit"),
                enforce_canary=False,
            )
            with (
                patch.dict(
                    sys.modules,
                    {
                        "numpy": numpy,
                        "torch": torch,
                        "yaml": yaml,
                        "ultralytics": ultralytics,
                        **runtime_modules,
                    },
                ),
                patch(
                    "training.visual.train_detector.importlib.metadata.version",
                    return_value="fixture-ultralytics",
                ),
                patch(
                    "training.visual.safe_finetune.wire_safe_finetune_callbacks",
                    return_value=FakeSafeFineTuneController(),
                ),
                patch(
                    "training.visual.safe_finetune.validate_seed_state_parity",
                    return_value={"passed": True},
                ),
                patch(
                    "training.visual.safe_finetune.validate_frozen_state_parity",
                    return_value={"passed": True},
                ),
                patch(
                    "training.visual.train_detector.persist_projected_fp32_ema_checkpoint",
                    side_effect=lambda *_args, **_kwargs: {
                        "passed": True,
                        "checkpoint_sha256": sha256_file(last_checkpoint),
                    },
                ),
            ):
                manifest = train(args)

            self.assertEqual(
                [call["name"] for call in FakeYOLO.validation_calls],
                [
                    "safety-best-validation",
                    "frozen-base-webpii-test",
                    "frozen-base-supplemental-test",
                    "safety-best-webpii-test",
                    "safety-best-supplemental-test",
                ],
            )
            self.assertEqual(
                [call["split"] for call in FakeYOLO.validation_calls],
                ["val", "test", "test", "test", "test"],
            )
            self.assertEqual(manifest["schema_version"], 4)
            self.assertEqual(commit_calls, ["commit", "commit"])
            self.assertEqual(manifest["checkpoint_persistence"]["committed_epochs"], 1)
            self.assertTrue(manifest["checkpoint_persistence"]["commit_hook_enabled"])
            self.assertEqual(manifest["data"]["staging"], {"fixture": True})
            self.assertEqual(manifest["training_config"]["workers"], 0)
            progress = json.loads(
                (output / "safety-selection-progress.json").read_text(encoding="utf-8")
            )
            self.assertTrue(progress["checkpoints"]["last"]["exists"])
            self.assertTrue(progress["checkpoints"]["best"]["exists"])
            self.assertTrue(progress["checkpoints"]["safety_best"]["exists"])
            self.assertAlmostEqual(manifest["best_checkpoint"]["score"][0], 0.92)
            self.assertEqual(len(FakeYOLO.training_calls), 1)
            training_call = FakeYOLO.training_calls[0]
            self.assertEqual(
                {
                    key: training_call[key]
                    for key in SCREENSHOT_AUGMENTATION_POLICY
                },
                SCREENSHOT_AUGMENTATION_POLICY,
            )
            self.assertEqual(
                {
                    key: training_call[key] for key in STABLE_OPTIMIZER_POLICY
                },
                STABLE_OPTIMIZER_POLICY,
            )
            self.assertEqual(training_call["fliplr"], 0.0)
            self.assertEqual(training_call["flipud"], 0.0)
            self.assertEqual(training_call["mosaic"], 0.0)
            self.assertEqual(training_call["degrees"], 0.0)
            self.assertEqual(training_call["translate"], 0.02)
            self.assertEqual(training_call["scale"], 0.05)
            self.assertTrue(training_call["save"])
            self.assertEqual(training_call["save_period"], 1)
            self.assertFalse(training_call["resume"])
            self.assertEqual(
                manifest["training_config"]["augmentation_policy"][
                    "ultralytics_overrides"
                ],
                SCREENSHOT_AUGMENTATION_POLICY,
            )
            baseline = manifest["baseline_published_test_evaluation"]
            self.assertEqual(
                baseline["timing"], "post-selection-release-baseline"
            )
            self.assertEqual(baseline["checkpoint_status"], "frozen")
            self.assertFalse(baseline["used_for_checkpoint_selection"])
            self.assertEqual(baseline["aggregate"]["metrics/mAP50(B)"], 0.1)
            baseline_supplemental = manifest["baseline_supplemental_test_evaluation"]
            self.assertFalse(baseline_supplemental["used_for_checkpoint_selection"])
            self.assertEqual(
                baseline_supplemental["aggregate"]["metrics/mAP50(B)"], 0.2
            )
            published = manifest["published_test_evaluation"]
            self.assertEqual(published["timing"], "post-selection")
            self.assertFalse(published["used_for_checkpoint_selection"])
            self.assertEqual(published["aggregate"]["metrics/mAP50(B)"], 0.5)
            self.assertIn("CVC", published["missing_test_classes"])
            self.assertEqual(published["per_class"]["CVC"]["recall"], 0.0)
            supplemental_result = manifest["supplemental_test_evaluation"]
            self.assertFalse(supplemental_result["used_for_checkpoint_selection"])
            self.assertIn("SECRET", supplemental_result["missing_test_classes"])
            self.assertEqual(
                [call[0] for call in runtime_calls],
                [
                    "resolve",
                    "evaluate",
                    "evaluate",
                    "resolve",
                    "evaluate",
                    "evaluate",
                ],
            )
            self.assertTrue(
                manifest["published_test_runtime_evaluation"]["release_gate"][
                    "passed"
                ]
            )
            self.assertTrue((output / "supplemental-test.yaml").is_file())
            persisted = json.loads(
                (output / "training_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(persisted, manifest)

    def test_expired_deadline_runs_no_test_inference_or_epoch_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, source_manifest, source_manifest_path = composed_fixture(root)
            source_manifest_path.write_text(
                json.dumps(source_manifest), encoding="utf-8"
            )
            dataset_yaml = root / "composed/dataset.yaml"
            dataset_yaml.write_text("fixture: true\n", encoding="utf-8")
            base_checkpoint = root / "base.pt"
            base_checkpoint.write_bytes(b"base-checkpoint")
            output = root / "output"
            commits: list[str] = []
            clock = [111.0]

            class FakeYOLO:
                training_calls: list[dict[str, object]] = []

                def __init__(self, source: str) -> None:
                    self.source = source

                def train(self, **kwargs: object) -> None:
                    self.training_calls.append(kwargs)
                    raise AssertionError("epoch zero started after the deadline")

                def val(self, **kwargs: object) -> SimpleNamespace:
                    raise AssertionError(
                        f"test inference ran before the budget gate: {kwargs}"
                    )

            torch = types.ModuleType("torch")
            torch.__version__ = "fixture-torch"
            torch.cuda = SimpleNamespace(is_available=lambda: False)
            yaml = types.ModuleType("yaml")
            yaml.safe_load = lambda _content: dataset
            ultralytics = types.ModuleType("ultralytics")
            ultralytics.YOLO = FakeYOLO
            args = argparse.Namespace(
                dataset_yaml=dataset_yaml,
                source_manifest=source_manifest_path,
                runtime_selection_contract=Path(
                    source_manifest["runtime_selection"]["contract"]
                ),
                runtime_selection_evaluator=FakeRuntimeSelectionEvaluator(),
                base_checkpoint=base_checkpoint,
                base_license="AGPL-3.0-only",
                base_source="fixture",
                commercial_license_approved=False,
                output_dir=output,
                epochs=2,
                image_size=640,
                batch_size=4,
                device="cpu",
                workers=0,
                patience=2,
                seed=1311,
                data_staging={"fixture": True},
                checkpoint_commit_hook=lambda: commits.append("commit"),
                deadline_monotonic=110.0,
                enforce_canary=True,
            )

            with (
                patch.dict(
                    sys.modules,
                    {"torch": torch, "yaml": yaml, "ultralytics": ultralytics},
                ),
                patch(
                    "training.visual.train_detector.time.monotonic",
                    side_effect=lambda: clock[0],
                ),
            ):
                paused = train(args)

            self.assertEqual(FakeYOLO.training_calls, [])
            self.assertEqual(paused["status"], "budget-paused-before-training")
            self.assertTrue(paused["resume_allowed"])
            self.assertEqual(commits, ["commit", "commit"])
            progress = json.loads(
                (output / "safety-selection-progress.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(progress["status"], "budget-paused-before-training")
            self.assertEqual(progress["epoch"], -1)
            self.assertEqual(progress["history"], [])
            self.assertTrue(progress["resume_allowed"])
            self.assertTrue((output / "safety-best.pt").is_file())
            self.assertFalse((output / "ultralytics-run/weights/last.pt").exists())


if __name__ == "__main__":
    unittest.main()
