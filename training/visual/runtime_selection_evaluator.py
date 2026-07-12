from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import yaml

from training.schema import sha256_file
from training.visual.calibrate_detector_runtime import (
    CALIBRATION_POLICY,
    calibrate_threshold_profile,
    canonical_json_sha256,
)
from training.visual.diagnose_detector_curves import (
    HIGH_RECALL_THRESHOLDS,
    summarize_curves,
)
from training.visual.diagnose_detector_runtime import (
    evaluate_checkpoint,
    runtime_contract,
)
from training.visual.prepare_webpii import DETECTOR_CLASSES
from training.visual.runtime_safety import (
    REQUIRED_SELECTION_SLICES,
    RUNTIME_SELECTION_POLICY_NAME,
    summarize_runtime_candidate,
    validate_runtime_selection_candidate,
)


def _read_json(path: Path, context: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{context} is missing or unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{context} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{context} must contain an object")
    return value


def _validated_artifact(
    contract: dict[str, Any], path_key: str, hash_key: str, context: str
) -> Path:
    value = contract.get(path_key)
    expected = contract.get(hash_key)
    if not isinstance(value, str) or not isinstance(expected, str):
        raise RuntimeError(f"{context} path or hash is missing")
    path = Path(value)
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
        raise RuntimeError(f"{context} failed hash verification")
    return path


def _read_image_list(path: Path) -> list[Path]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeError("selection image list is not valid UTF-8") from exc
    raw_images = [Path(line) for line in lines]
    images = [path.resolve() for path in raw_images]
    if (
        not images
        or any(not line for line in lines)
        or any(not path.is_absolute() for path in raw_images)
        or len(set(images)) != len(images)
        or any(path.is_symlink() or not path.is_file() for path in images)
    ):
        raise RuntimeError("selection image inventory is empty, duplicated, or unsafe")
    return images


def _validated_root(value: Any, expected_suffix: tuple[str, ...], context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{context} is missing")
    raw = Path(value)
    root = raw.resolve()
    if (
        not raw.is_absolute()
        or raw.is_symlink()
        or not root.is_dir()
        or (
            expected_suffix
            and root.parts[-len(expected_suffix) :] != expected_suffix
        )
    ):
        raise RuntimeError(f"{context} is unsafe or points at the wrong split")
    return root


def _require_images_under(
    images: Iterable[Path], allowed_root: Path, context: str
) -> None:
    for image in images:
        if image == allowed_root or not image.is_relative_to(allowed_root):
            raise RuntimeError(f"{context} escapes its approved non-test split")


def _validate_dataset_yaml(
    dataset_yaml: Path, image_list: Path, context: str
) -> None:
    try:
        dataset = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise RuntimeError(f"{context} is not valid UTF-8 YAML") from exc
    if not isinstance(dataset, dict):
        raise RuntimeError(f"{context} must contain a mapping")
    root = Path(str(dataset.get("path", dataset_yaml.parent)))
    if not root.is_absolute():
        root = (dataset_yaml.parent / root).resolve()
    value = dataset.get("val")
    values = value if isinstance(value, list) else [value]
    if len(values) != 1 or not isinstance(values[0], str):
        raise RuntimeError(f"{context} val must contain exactly one path")
    path = Path(values[0])
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if resolved != image_list.resolve():
        raise RuntimeError(f"{context} val bypasses its bound image list")
    expected_names = {index: name for index, name in enumerate(DETECTOR_CLASSES)}
    raw_names = dataset.get("names")
    if isinstance(raw_names, list):
        names = {index: str(name) for index, name in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        try:
            names = {int(index): str(name) for index, name in raw_names.items()}
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{context} class contract is invalid") from exc
    else:
        names = {}
    if dataset.get("nc") != len(DETECTOR_CLASSES) or names != expected_names:
        raise RuntimeError(f"{context} class contract changed")


def _validate_artifact_binding(
    declared: Any,
    generated: Any,
    *,
    context: str,
    records_required: bool,
) -> Path:
    if not isinstance(declared, dict) or not isinstance(generated, dict):
        raise RuntimeError(f"{context} provenance is missing")
    keys = ("path", "bytes", "sha256") + (("records",) if records_required else ())
    if any(declared.get(key) != generated.get(key) for key in keys):
        raise RuntimeError(f"{context} differs from its hashed split manifest")
    path = Path(str(declared.get("path", "")))
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != declared.get("bytes")
        or sha256_file(path) != declared.get("sha256")
    ):
        raise RuntimeError(f"{context} failed artifact verification")
    return path.resolve()


def _validate_source_manifest_binding(
    split_manifest: dict[str, Any], source_root: Path, *, synthetic: bool
) -> None:
    source = split_manifest.get("source")
    if not isinstance(source, dict) or source.get("root") != str(source_root):
        raise RuntimeError("selection split source root changed")
    if synthetic:
        path_value = source.get("manifest")
        expected_sha256 = source.get("manifest_sha256")
    else:
        manifest = source.get("manifest")
        path_value = manifest.get("path") if isinstance(manifest, dict) else None
        expected_sha256 = manifest.get("sha256") if isinstance(manifest, dict) else None
    if not isinstance(path_value, str) or not isinstance(expected_sha256, str):
        raise RuntimeError("selection split source manifest binding is missing")
    path = Path(path_value)
    if (
        path.resolve() != (source_root / "manifest.json").resolve()
        or path.is_symlink()
        or not path.is_file()
        or sha256_file(path) != expected_sha256
    ):
        raise RuntimeError("selection split source manifest hash changed")


def load_validated_runtime_selection_contract(path: Path) -> dict[str, Any]:
    """Load a selection contract and prove every image came from approved splits."""

    contract = _read_json(path, "runtime selection contract")
    if (
        contract.get("schema_version") != 2
        or contract.get("kind") != "calibrated-runtime-selection-contract"
        or contract.get("classes") != list(DETECTOR_CLASSES)
        or contract.get("published_test_used") is not False
        or contract.get("values_persisted") is not False
    ):
        raise RuntimeError("runtime selection contract is unsupported or unsafe")
    immutable_inputs = contract.get("immutable_inputs")
    if (
        not isinstance(immutable_inputs, dict)
        or immutable_inputs.get("schema_version") != 2
        or immutable_inputs.get("published_webpii_test_used") is not False
        or contract.get("contract_sha256")
        != canonical_json_sha256(immutable_inputs)
    ):
        raise RuntimeError("runtime selection immutable-input hash mismatch")
    partitions = contract.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) != {"webpii", "synthetic"}:
        raise RuntimeError("runtime selection partition provenance is missing")

    webpii = partitions["webpii"]
    synthetic = partitions["synthetic"]
    if (
        not isinstance(webpii, dict)
        or webpii.get("kind") != "webpii-published-train-partitions"
        or not isinstance(synthetic, dict)
        or synthetic.get("kind") != "synthetic-validation-partitions"
    ):
        raise RuntimeError("runtime selection partition kind changed")
    webpii_root = _validated_root(webpii.get("source_root"), (), "WebPII source root")
    synthetic_root = _validated_root(
        synthetic.get("source_root"), (), "synthetic source root"
    )
    webpii_allowed = _validated_root(
        webpii.get("allowed_image_root"),
        ("images", "train"),
        "WebPII allowed image root",
    )
    webpii_test = _validated_root(
        webpii.get("forbidden_test_root"),
        ("images", "test"),
        "WebPII forbidden test root",
    )
    synthetic_allowed = _validated_root(
        synthetic.get("allowed_image_root"),
        ("images", "validation"),
        "synthetic allowed image root",
    )
    if (
        webpii_allowed != (webpii_root / "images/train").resolve()
        or webpii_test != (webpii_root / "images/test").resolve()
        or synthetic_allowed != (synthetic_root / "images/validation").resolve()
    ):
        raise RuntimeError("runtime selection approved source roots changed")

    webpii_manifest_path = _validate_artifact_binding(
        webpii.get("split_manifest"),
        webpii.get("split_manifest"),
        context="WebPII split manifest",
        records_required=False,
    )
    synthetic_manifest_path = _validate_artifact_binding(
        synthetic.get("split_manifest"),
        synthetic.get("split_manifest"),
        context="synthetic split manifest",
        records_required=False,
    )
    webpii_manifest = _read_json(webpii_manifest_path, "WebPII split manifest")
    synthetic_manifest = _read_json(
        synthetic_manifest_path, "synthetic split manifest"
    )
    published_test = webpii_manifest.get("published_test")
    webpii_invariants = webpii_manifest.get("invariants")
    if (
        webpii_manifest.get("schema_version") != 1
        or webpii_manifest.get("kind")
        != "webpii-train-derived-identity-disjoint-guard-split"
        or not isinstance(published_test, dict)
        or set(published_test)
        != {
            "excluded",
            "used_for_selection",
            "emitted_in_dataset_yaml",
            "declared_records",
        }
        or published_test.get("excluded") is not True
        or published_test.get("used_for_selection") is not False
        or published_test.get("emitted_in_dataset_yaml") is not False
        or not isinstance(webpii_invariants, dict)
        or webpii_invariants.get("published_test_excluded") is not True
        or webpii_invariants.get("identity_disjoint") is not True
        or webpii_invariants.get("group_overlap") != 0
        or webpii_invariants.get("record_overlap") != 0
        or webpii_invariants.get("groups_exhaustive") is not True
        or webpii_invariants.get("records_exhaustive") is not True
        or webpii_manifest.get("values_persisted") is not False
    ):
        raise RuntimeError("WebPII split manifest does not prove test exclusion")
    if (
        synthetic_manifest.get("schema_version") != 1
        or synthetic_manifest.get("kind")
        != "deterministic-synthetic-calibration-guard-split"
        or synthetic_manifest.get("published_webpii_test_used") is not False
        or synthetic_manifest.get("identity_overlap") != 0
        or synthetic_manifest.get("values_persisted") is not False
    ):
        raise RuntimeError("synthetic split manifest is unsupported or unsafe")
    _validate_source_manifest_binding(webpii_manifest, webpii_root, synthetic=False)
    _validate_source_manifest_binding(synthetic_manifest, synthetic_root, synthetic=True)

    webpii_outputs = webpii_manifest.get("outputs")
    if not isinstance(webpii_outputs, dict):
        raise RuntimeError("WebPII split outputs are missing")
    webpii_lists: dict[str, list[Path]] = {}
    for name, output_name in (
        ("calibration", "calibration_images"),
        ("guard", "guard_images"),
        ("eligible", "eligible_images"),
    ):
        list_path = _validate_artifact_binding(
            webpii.get(name),
            webpii_outputs.get(output_name),
            context=f"WebPII {name} partition",
            records_required=True,
        )
        images = _read_image_list(list_path)
        if len(images) != webpii[name].get("records"):
            raise RuntimeError(f"WebPII {name} partition count changed")
        _require_images_under(images, webpii_allowed, f"WebPII {name} partition")
        if any(image.is_relative_to(webpii_test) for image in images):
            raise RuntimeError("WebPII published test leaked into runtime selection")
        webpii_lists[name] = images
    webpii_sets = {name: set(images) for name, images in webpii_lists.items()}
    if (
        webpii_sets["calibration"] & webpii_sets["guard"]
        or webpii_sets["calibration"] & webpii_sets["eligible"]
        or webpii_sets["guard"] & webpii_sets["eligible"]
    ):
        raise RuntimeError("WebPII selection partitions overlap")

    synthetic_lists: dict[str, list[Path]] = {}
    for name in ("calibration", "guard"):
        generated = synthetic_manifest.get(name)
        list_path = _validate_artifact_binding(
            synthetic.get(name),
            {
                "path": generated.get("image_list") if isinstance(generated, dict) else None,
                "bytes": Path(str(generated.get("image_list", ""))).stat().st_size
                if isinstance(generated, dict)
                and Path(str(generated.get("image_list", ""))).is_file()
                else None,
                "sha256": generated.get("image_list_sha256")
                if isinstance(generated, dict)
                else None,
                "records": generated.get("records")
                if isinstance(generated, dict)
                else None,
            },
            context=f"synthetic {name} partition",
            records_required=True,
        )
        images = _read_image_list(list_path)
        if len(images) != synthetic[name].get("records"):
            raise RuntimeError(f"synthetic {name} partition count changed")
        _require_images_under(images, synthetic_allowed, f"synthetic {name} partition")
        synthetic_lists[name] = images
    if set(synthetic_lists["calibration"]) & set(synthetic_lists["guard"]):
        raise RuntimeError("synthetic selection partitions overlap")

    calibration = contract.get("calibration")
    guards = contract.get("guards")
    if not isinstance(calibration, dict) or not isinstance(guards, dict):
        raise RuntimeError("runtime selection calibration or guard inputs are missing")
    calibration_yaml = _validated_artifact(
        calibration,
        "dataset_yaml",
        "dataset_yaml_sha256",
        "runtime calibration YAML",
    )
    calibration_list = _validated_artifact(
        calibration,
        "image_list",
        "image_list_sha256",
        "runtime calibration image list",
    )
    calibration_images = _read_image_list(calibration_list)
    expected_calibration = sorted(
        webpii_lists["calibration"] + synthetic_lists["calibration"]
    )
    if (
        calibration_images != expected_calibration
        or calibration.get("records") != len(expected_calibration)
    ):
        raise RuntimeError("combined calibration is not the exact partition union")
    _validate_dataset_yaml(calibration_yaml, calibration_list, "runtime calibration YAML")

    if set(guards) != set(REQUIRED_SELECTION_SLICES):
        raise RuntimeError("runtime selection guard slice contract changed")
    guard_inputs: dict[str, dict[str, Any]] = {}
    expected_guards = {
        "synthetic-guard": (synthetic["guard"], synthetic_lists["guard"]),
        "webpii-guard": (webpii["guard"], webpii_lists["guard"]),
    }
    for name in REQUIRED_SELECTION_SLICES:
        guard = guards[name]
        partition, expected_images = expected_guards[name]
        if not isinstance(guard, dict):
            raise RuntimeError(f"runtime selection {name} contract is invalid")
        image_list = _validated_artifact(
            guard,
            "image_list",
            "image_list_sha256",
            f"runtime selection {name} image list",
        )
        dataset_yaml = _validated_artifact(
            guard,
            "dataset_yaml",
            "dataset_yaml_sha256",
            f"runtime selection {name} YAML",
        )
        images = _read_image_list(image_list)
        if (
            guard.get("image_list") != partition.get("path")
            or guard.get("image_list_sha256") != partition.get("sha256")
            or guard.get("records") != partition.get("records")
            or images != expected_images
        ):
            raise RuntimeError(f"runtime selection {name} bypasses its partition")
        _validate_dataset_yaml(dataset_yaml, image_list, f"runtime selection {name} YAML")
        guard_inputs[name] = {
            "dataset_yaml": dataset_yaml,
            "image_list": image_list,
            "images": images,
        }

    eligibility = contract.get("training_eligibility")
    if (
        not isinstance(eligibility, dict)
        or eligibility.get("webpii_images") != webpii["eligible"].get("path")
        or eligibility.get("webpii_images_sha256")
        != webpii["eligible"].get("sha256")
        or eligibility.get("records") != webpii["eligible"].get("records")
    ):
        raise RuntimeError("WebPII training eligibility bypasses its partition")
    selection_inventory = contract.get("selection_inventory")
    if (
        not isinstance(selection_inventory, dict)
        or selection_inventory.get("synthetic-guard") != synthetic["guard"]
        or selection_inventory.get("webpii-guard") != webpii["guard"]
    ):
        raise RuntimeError("runtime selection inventory differs from its partitions")
    source_manifests = contract.get("source_manifests")
    expected_source_manifests = {
        "webpii": {
            "path": webpii["split_manifest"]["path"],
            "sha256": webpii["split_manifest"]["sha256"],
        },
        "synthetic": {
            "path": synthetic["split_manifest"]["path"],
            "sha256": synthetic["split_manifest"]["sha256"],
        },
    }
    if source_manifests != expected_source_manifests:
        raise RuntimeError("runtime selection split-manifest provenance changed")
    seed = immutable_inputs.get("seed")
    if type(seed) is not int:
        raise RuntimeError("runtime selection seed is invalid")
    expected_immutable_inputs = {
        "schema_version": 2,
        "seed": seed,
        "calibration_policy": CALIBRATION_POLICY,
        "frozen_seed_threshold_profile": HIGH_RECALL_THRESHOLDS,
        "webpii_split_manifest_sha256": webpii["split_manifest"]["sha256"],
        "synthetic_split_manifest_sha256": synthetic["split_manifest"]["sha256"],
        "combined_calibration_images_sha256": sha256_file(calibration_list),
        "synthetic_guard_images_sha256": synthetic["guard"]["sha256"],
        "webpii_guard_images_sha256": webpii["guard"]["sha256"],
        "webpii_eligible_images_sha256": webpii["eligible"]["sha256"],
        "published_webpii_test_used": False,
    }
    if immutable_inputs != expected_immutable_inputs:
        raise RuntimeError("runtime selection immutable inputs are not provenance-exact")

    return {
        "contract": contract,
        "contract_file_sha256": sha256_file(path),
        "calibration_yaml": calibration_yaml,
        "calibration_image_list": calibration_list,
        "calibration_images": calibration_images,
        "guard_inputs": guard_inputs,
        "webpii": {
            "source_root": webpii_root,
            "allowed_image_root": webpii_allowed,
            "forbidden_test_root": webpii_test,
            "calibration": frozenset(webpii_sets["calibration"]),
            "guard": frozenset(webpii_sets["guard"]),
            "eligible": frozenset(webpii_sets["eligible"]),
        },
        "synthetic": {
            "source_root": synthetic_root,
            "allowed_image_root": synthetic_allowed,
            "calibration": frozenset(synthetic_lists["calibration"]),
            "guard": frozenset(synthetic_lists["guard"]),
        },
    }


def validate_replay_against_runtime_selection(
    replay_images: Iterable[Path], provenance: dict[str, Any]
) -> None:
    """Prove every replayed WebPII image is eligible and no holdout is replayed."""

    webpii = provenance.get("webpii")
    if not isinstance(webpii, dict):
        raise RuntimeError("validated runtime selection lacks WebPII provenance")
    allowed_root = webpii["allowed_image_root"]
    eligible = set(webpii["eligible"])
    holdout = set(webpii["calibration"]) | set(webpii["guard"])
    replayed_webpii: set[Path] = set()
    for raw_image in replay_images:
        image = Path(raw_image).resolve()
        if image in holdout:
            raise RuntimeError("detector replay contains a runtime-selection holdout")
        if image.is_relative_to(allowed_root):
            replayed_webpii.add(image)
    if not replayed_webpii <= eligible:
        raise RuntimeError("detector replay contains an ineligible WebPII train image")
    if replayed_webpii != eligible:
        raise RuntimeError("detector replay omits an eligible WebPII train image")


class CalibratedRuntimeSelectionEvaluator:
    """Produce one fail-closed operational summary per detector checkpoint."""

    def __init__(
        self,
        selection_contract_path: Path,
        *,
        device: str,
        batch: int,
        workers: int,
        output_dir: Path,
    ) -> None:
        self.contract_path = selection_contract_path
        provenance = load_validated_runtime_selection_contract(
            selection_contract_path
        )
        self.contract = provenance["contract"]
        self.contract_sha256 = provenance["contract_file_sha256"]
        declared_sha = self.contract.get("contract_sha256")
        # The on-disk document cannot self-hash. Its declared contract hash is
        # instead the canonical hash of the immutable inputs block.
        immutable_inputs = self.contract.get("immutable_inputs")
        if (
            not isinstance(immutable_inputs, dict)
            or declared_sha != canonical_json_sha256(immutable_inputs)
        ):
            raise RuntimeError("runtime selection immutable-input hash mismatch")
        self.calibration_contract_sha256 = declared_sha
        selection_inventory = self.contract.get("selection_inventory")
        if not isinstance(selection_inventory, dict):
            raise RuntimeError("runtime selection inventory contract is missing")
        self.selection_inventory_contract_sha256 = canonical_json_sha256(
            selection_inventory
        )
        self.calibration_yaml = provenance["calibration_yaml"]
        self.calibration_image_list = provenance["calibration_image_list"]
        self.guard_inputs = provenance["guard_inputs"]
        self.device = str(device)
        self.batch = int(batch)
        self.workers = int(workers)
        if self.batch <= 0 or self.workers < 0:
            raise ValueError("runtime selection batch or workers are invalid")
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._seed_calibration_box: Any | None = None
        self._seed_names: dict[int, str] | None = None

    @staticmethod
    def _normalize_names(value: Any) -> dict[int, str]:
        if isinstance(value, list):
            names = {index: str(name) for index, name in enumerate(value)}
        elif isinstance(value, dict):
            names = {int(index): str(name) for index, name in value.items()}
        else:
            raise RuntimeError("runtime selection evaluation has no class names")
        expected = {index: name for index, name in enumerate(DETECTOR_CLASSES)}
        if names != expected:
            raise RuntimeError("runtime selection detector classes changed")
        return names

    def _validate_checkpoint(self, checkpoint: Path) -> tuple[Path, str]:
        checkpoint = checkpoint.resolve()
        if checkpoint.is_symlink() or not checkpoint.is_file():
            raise FileNotFoundError(f"runtime selection checkpoint is unsafe: {checkpoint}")
        return checkpoint, sha256_file(checkpoint)

    def _val(
        self,
        checkpoint: Path,
        dataset_yaml: Path,
        *,
        name: str,
    ) -> Any:
        from ultralytics import YOLO

        return YOLO(str(checkpoint)).val(
            data=str(dataset_yaml),
            split="val",
            imgsz=640,
            batch=self.batch,
            device=self.device,
            workers=self.workers,
            plots=False,
            verbose=False,
            save_json=False,
            project=str(self.output_dir),
            name=name,
            exist_ok=True,
        )

    @staticmethod
    def _map50(evaluation: Any) -> float:
        results = getattr(evaluation, "results_dict", None)
        value = results.get("metrics/mAP50(B)") if isinstance(results, dict) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError("runtime selection evaluation has no mAP50")
        normalized = float(value)
        if not 0.0 <= normalized <= 1.0:
            raise RuntimeError("runtime selection mAP50 is invalid")
        return normalized

    def _evaluate_guards(
        self,
        checkpoint: Path,
        checkpoint_sha256: str,
        *,
        epoch: int,
        profile: dict[str, float],
        calibration_evidence: dict[str, Any],
    ) -> dict[str, Any]:
        profile_sha256 = canonical_json_sha256(profile)
        slices: dict[str, Any] = {}
        for slice_name in REQUIRED_SELECTION_SLICES:
            inputs = self.guard_inputs[slice_name]
            evaluation = self._val(
                checkpoint,
                inputs["dataset_yaml"],
                name=(
                    f"runtime-{checkpoint_sha256[:12]}-epoch-{epoch}-"
                    f"{slice_name}-curves"
                ),
            )
            names = self._normalize_names(getattr(evaluation, "names", None))
            box = getattr(evaluation, "box", None)
            if box is None:
                raise RuntimeError(f"runtime selection {slice_name} has no box metrics")
            curves = summarize_curves(box, names, thresholds=profile)
            profiles = {"high-recall": profile}
            runtime_report, _truth_status = evaluate_checkpoint(
                checkpoint,
                inputs["images"],
                device=self.device,
                batch=self.batch,
                threshold_profiles=profiles,
            )
            if runtime_report.get("checkpoint_sha256") != checkpoint_sha256:
                raise RuntimeError("runtime selection checkpoint changed during audit")
            slices[slice_name] = summarize_runtime_candidate(
                epoch=epoch,
                map50=self._map50(evaluation),
                curve_report=curves,
                runtime_report=runtime_report,
                runtime_contract=runtime_contract(profiles),
            )
        return validate_runtime_selection_candidate(
            {
                "schema_version": 1,
                "metric_basis": RUNTIME_SELECTION_POLICY_NAME,
                "epoch": epoch,
                "checkpoint_sha256": checkpoint_sha256,
                "calibration_contract_sha256": (
                    self.calibration_contract_sha256
                ),
                "selection_inventory_contract_sha256": (
                    self.selection_inventory_contract_sha256
                ),
                "threshold_profile_sha256": profile_sha256,
                "threshold_profile": profile,
                "calibration_passed": True,
                "calibration_evidence": calibration_evidence,
                "slices": slices,
            }
        )

    def initialize_seed(self, checkpoint: Path) -> dict[str, Any]:
        checkpoint, checkpoint_sha256 = self._validate_checkpoint(checkpoint)
        calibration = self._val(
            checkpoint,
            self.calibration_yaml,
            name=f"runtime-seed-{checkpoint_sha256[:12]}-calibration-curves",
        )
        self._seed_names = self._normalize_names(
            getattr(calibration, "names", None)
        )
        self._seed_calibration_box = getattr(calibration, "box", None)
        if self._seed_calibration_box is None:
            raise RuntimeError("runtime seed calibration has no box metrics")
        profile = dict(HIGH_RECALL_THRESHOLDS)
        profile_sha256 = canonical_json_sha256(profile)
        return self._evaluate_guards(
            checkpoint,
            checkpoint_sha256,
            epoch=-1,
            profile=profile,
            calibration_evidence={
                "kind": "frozen-seed-high-recall-profile",
                "passed": True,
                "profile_sha256": profile_sha256,
                "calibration_policy_sha256": canonical_json_sha256(
                    CALIBRATION_POLICY
                ),
            },
        )

    def evaluate_candidate(self, checkpoint: Path, *, epoch: int) -> dict[str, Any]:
        if self._seed_calibration_box is None or self._seed_names is None:
            raise RuntimeError("runtime seed must be initialized before candidates")
        checkpoint, checkpoint_sha256 = self._validate_checkpoint(checkpoint)
        calibration = self._val(
            checkpoint,
            self.calibration_yaml,
            name=(
                f"runtime-{checkpoint_sha256[:12]}-epoch-{epoch}-"
                "calibration-curves"
            ),
        )
        names = self._normalize_names(getattr(calibration, "names", None))
        if names != self._seed_names:
            raise RuntimeError("candidate calibration class contract changed")
        candidate_box = getattr(calibration, "box", None)
        if candidate_box is None:
            raise RuntimeError("candidate calibration has no box metrics")
        calibration_report = calibrate_threshold_profile(
            self._seed_calibration_box,
            candidate_box,
            names,
        )
        if calibration_report.get("passed") is not True:
            failed = [
                name
                for name, values in calibration_report.get("per_class", {}).items()
                if values.get("passed") is not True
            ]
            raise RuntimeError(
                "candidate threshold calibration failed for: " + ", ".join(failed)
            )
        profile = calibration_report["calibrated_profile"]
        return self._evaluate_guards(
            checkpoint,
            checkpoint_sha256,
            epoch=epoch,
            profile=profile,
            calibration_evidence={
                "kind": "per-class-recall-preserving-calibration",
                "passed": True,
                "profile_sha256": calibration_report[
                    "calibrated_profile_sha256"
                ],
                "report": calibration_report,
            },
        )
