from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from training.visual.compose_dataset import DETECTOR_CLASSES
from training.visual.diagnose_detector_curves import HIGH_RECALL_THRESHOLDS


BALANCED_THRESHOLDS = {
    "NAME": 0.25,
    "EMAIL": 0.25,
    "PHONE": 0.25,
    "ADDRESS": 0.25,
    "CARD_NUMBER": 0.20,
    "CVC": 0.03,
    "SECRET": 0.20,
    "SENSITIVE_FIELD": 0.20,
    "SENSITIVE_IMAGE": 0.50,
}
THRESHOLD_PROFILES = {
    "high-recall": HIGH_RECALL_THRESHOLDS,
    "balanced": BALANCED_THRESHOLDS,
}
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VISUAL_PROPOSAL_COVERAGE = 0.90
NMS_IOU_THRESHOLD = 0.70
MAXIMUM_DETECTIONS = 300
PADDING_FRACTION = 0.04
LOW_CONFIDENCE_FLOOR = 0.005
PREPROFILE_NMS_IOU_THRESHOLD = 0.999
PREPROFILE_MAXIMUM_DETECTIONS = 2000


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def runtime_contract(
    threshold_profiles: dict[str, dict[str, float]] = THRESHOLD_PROFILES,
) -> dict[str, Any]:
    return {
        "best_class_only": True,
        "threshold_profiles": threshold_profiles,
        "nms_iou_threshold": NMS_IOU_THRESHOLD,
        "maximum_detections": MAXIMUM_DETECTIONS,
        "padding_fraction": PADDING_FRACTION,
        "visual_proposal_coverage": VISUAL_PROPOSAL_COVERAGE,
        "low_confidence_floor": LOW_CONFIDENCE_FLOOR,
        "preprofile_nms_iou_threshold": PREPROFILE_NMS_IOU_THRESHOLD,
        "preprofile_maximum_detections": PREPROFILE_MAXIMUM_DETECTIONS,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_dataset_images(dataset_yaml: Path, split: str) -> list[Path]:
    if dataset_yaml.is_symlink() or not dataset_yaml.is_file():
        raise FileNotFoundError(f"dataset YAML is missing or unsafe: {dataset_yaml}")
    dataset = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    if not isinstance(dataset, dict) or split not in dataset:
        raise ValueError(f"dataset YAML has no {split} split")
    root = Path(str(dataset.get("path", dataset_yaml.parent)))
    if not root.is_absolute():
        root = (dataset_yaml.parent / root).resolve()
    raw_values = dataset[split]
    values = raw_values if isinstance(raw_values, list) else [raw_values]
    images: list[Path] = []
    for value in values:
        path = Path(str(value))
        if not path.is_absolute():
            path = (root / path).resolve()
        if path.suffix.lower() == ".txt":
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"dataset image list is missing or unsafe: {path}")
            for line in path.read_text(encoding="utf-8").splitlines():
                image = Path(line)
                images.append(
                    image.resolve() if image.is_absolute() else (root / image).resolve()
                )
        elif path.is_dir() and not path.is_symlink():
            images.extend(
                candidate.resolve()
                for candidate in path.rglob("*")
                if candidate.is_file()
                and not candidate.is_symlink()
                and candidate.suffix.lower() in IMAGE_SUFFIXES
            )
        else:
            raise ValueError(f"dataset split path is not a directory or list: {path}")
    images = sorted(images)
    if (
        not images
        or len(images) != len(set(images))
        or any(
            image.is_symlink()
            or not image.is_file()
            or not label_path_for_image(image).is_file()
            for image in images
        )
    ):
        raise ValueError("dataset image-label inventory is incomplete or duplicated")
    return images


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    candidates = [index for index, value in enumerate(parts) if value == "images"]
    if len(candidates) != 1:
        raise ValueError(f"image path has no unique images directory: {image_path}")
    parts[candidates[0]] = "labels"
    return Path(*parts).with_suffix(".txt")


def load_truth(label_path: Path, width: int, height: int) -> list[dict[str, Any]]:
    truth: list[dict[str, Any]] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"invalid YOLO label at {label_path}:{line_number}")
        class_id = int(values[0])
        center_x, center_y, box_width, box_height = map(float, values[1:])
        if class_id not in range(len(DETECTOR_CLASSES)):
            raise ValueError(f"invalid class at {label_path}:{line_number}")
        truth.append(
            {
                "class_id": class_id,
                "label": DETECTOR_CLASSES[class_id],
                "box": np.array(
                    [
                        (center_x - box_width / 2) * width,
                        (center_y - box_height / 2) * height,
                        (center_x + box_width / 2) * width,
                        (center_y + box_height / 2) * height,
                    ],
                    dtype=float,
                ),
            }
        )
    return truth


def pad_and_clip_boxes(
    boxes: np.ndarray, width: int, height: int
) -> np.ndarray:
    if not len(boxes):
        return boxes.reshape(0, 4)
    output = boxes.astype(float, copy=True)
    padding = np.maximum(
        2.0,
        np.minimum(output[:, 2] - output[:, 0], output[:, 3] - output[:, 1])
        * PADDING_FRACTION,
    )
    output[:, 0] = np.clip(output[:, 0] - padding, 0, width)
    output[:, 1] = np.clip(output[:, 1] - padding, 0, height)
    output[:, 2] = np.clip(output[:, 2] + padding, 0, width)
    output[:, 3] = np.clip(output[:, 3] + padding, 0, height)
    return output


def intersection_over_truth(boxes: np.ndarray, truth: np.ndarray) -> np.ndarray:
    if not len(boxes):
        return np.empty(0, dtype=float)
    top_left = np.maximum(boxes[:, :2], truth[:2])
    bottom_right = np.minimum(boxes[:, 2:], truth[2:])
    intersection = np.prod(np.maximum(0.0, bottom_right - top_left), axis=1)
    truth_area = max(0.0, truth[2] - truth[0]) * max(0.0, truth[3] - truth[1])
    if truth_area <= 0:
        return np.zeros(len(boxes), dtype=float)
    return intersection / truth_area


def intersection_over_union(one: np.ndarray, many: np.ndarray) -> np.ndarray:
    if not len(many):
        return np.empty(0, dtype=float)
    top_left = np.maximum(one[:2], many[:, :2])
    bottom_right = np.minimum(one[2:], many[:, 2:])
    intersection = np.prod(np.maximum(0.0, bottom_right - top_left), axis=1)
    one_area = max(0.0, one[2] - one[0]) * max(0.0, one[3] - one[1])
    many_area = np.maximum(0.0, many[:, 2] - many[:, 0]) * np.maximum(
        0.0, many[:, 3] - many[:, 1]
    )
    union = one_area + many_area - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )


def runtime_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    *,
    maximum_detections: int = MAXIMUM_DETECTIONS,
    iou_threshold: float = NMS_IOU_THRESHOLD,
) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    selected: list[int] = []
    for index in order.tolist():
        same_class = [
            previous for previous in selected if class_ids[previous] == class_ids[index]
        ]
        if same_class and np.any(
            intersection_over_union(boxes[index], boxes[np.asarray(same_class)])
            > iou_threshold
        ):
            continue
        selected.append(index)
        if len(selected) >= maximum_detections:
            break
    return np.asarray(selected, dtype=int)


def select_profile(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    thresholds: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    accepted = np.array(
        [
            score >= thresholds[DETECTOR_CLASSES[int(class_id)]]
            for score, class_id in zip(scores, class_ids)
        ],
        dtype=bool,
    )
    boxes = boxes[accepted]
    scores = scores[accepted]
    class_ids = class_ids[accepted]
    kept = runtime_nms(boxes, scores, class_ids)
    return boxes[kept], scores[kept], class_ids[kept]


def _new_summary() -> dict[str, Any]:
    return {
        "images": 0,
        "hard_negative_images": 0,
        "hard_negative_images_with_masks": 0,
        "predictions": 0,
        "truth": 0,
        "covered": 0,
        "class_compatible_covered": 0,
        "irrecoverable_misses": 0,
        "preprofile_saturated_images": 0,
        "per_class": {
            name: {
                "support": 0,
                "covered": 0,
                "class_compatible_covered": 0,
                "irrecoverable_misses": 0,
            }
            for name in DETECTOR_CLASSES
        },
    }


def _finalize_summary(summary: dict[str, Any]) -> dict[str, Any]:
    summary = json.loads(json.dumps(summary))
    summary["coverage_recall"] = (
        summary["covered"] / summary["truth"] if summary["truth"] else 1.0
    )
    summary["class_compatible_coverage_recall"] = (
        summary["class_compatible_covered"] / summary["truth"]
        if summary["truth"]
        else 1.0
    )
    summary["predictions_per_image"] = (
        summary["predictions"] / summary["images"] if summary["images"] else 0.0
    )
    summary["hard_negative_mask_rate"] = (
        summary["hard_negative_images_with_masks"]
        / summary["hard_negative_images"]
        if summary["hard_negative_images"]
        else 0.0
    )
    for values in summary["per_class"].values():
        values["coverage_recall"] = (
            values["covered"] / values["support"] if values["support"] else 1.0
        )
        values["class_compatible_coverage_recall"] = (
            values["class_compatible_covered"] / values["support"]
            if values["support"]
            else 1.0
        )
    return summary


def evaluate_checkpoint(
    checkpoint: Path,
    image_paths: list[Path],
    *,
    device: str,
    batch: int,
    threshold_profiles: dict[str, dict[str, float]] = THRESHOLD_PROFILES,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    from ultralytics import YOLO

    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint is missing or unsafe: {checkpoint}")
    if type(batch) is not int or batch <= 0:
        raise ValueError("runtime evaluation batch must be positive")
    if not isinstance(threshold_profiles, dict) or not threshold_profiles:
        raise ValueError("runtime threshold profiles are missing")
    for profile, thresholds in threshold_profiles.items():
        if (
            not isinstance(profile, str)
            or not profile
            or not isinstance(thresholds, dict)
            or set(thresholds) != set(DETECTOR_CLASSES)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0.0 <= float(value) <= 1.0
                for value in thresholds.values()
            )
        ):
            raise ValueError("runtime threshold profile is invalid")
    resolved_images = [path.resolve() for path in image_paths]
    expected_results = set(resolved_images)
    if (
        not resolved_images
        or len(expected_results) != len(resolved_images)
        or any(path.is_symlink() or not path.is_file() for path in resolved_images)
    ):
        raise ValueError("runtime image inventory is empty, duplicated, or unsafe")
    inventory_text = "".join(f"{path}\n" for path in resolved_images)
    inventory_sha256 = hashlib.sha256(inventory_text.encode()).hexdigest()

    summaries = {name: _new_summary() for name in threshold_profiles}
    truth_status: dict[str, dict[str, Any]] = {}
    model = YOLO(str(checkpoint))
    checkpoint_sha256 = sha256_file(checkpoint)
    source_manifest = Path(
        f"/tmp/plva-runtime-validation-{checkpoint_sha256[:16]}-{os.getpid()}.txt"
    )
    source_manifest.write_text(
        inventory_text,
        encoding="utf-8",
    )
    results = model.predict(
        # A Python list is treated as one in-memory image batch by Ultralytics
        # and would allocate all validation tensors at once. A text manifest
        # uses its streaming file loader and honors the requested batch size.
        source=str(source_manifest),
        stream=True,
        imgsz=640,
        batch=batch,
        device=device,
        conf=LOW_CONFIDENCE_FLOOR,
        iou=PREPROFILE_NMS_IOU_THRESHOLD,
        max_det=PREPROFILE_MAXIMUM_DETECTIONS,
        agnostic_nms=False,
        rect=False,
        verbose=False,
        save=False,
    )
    seen_results: set[Path] = set()
    for result in results:
        image_path = Path(result.path)
        resolved_result = image_path.resolve()
        if resolved_result not in expected_results or resolved_result in seen_results:
            raise RuntimeError("runtime predictor returned an unexpected image inventory")
        seen_results.add(resolved_result)
        height, width = map(int, result.orig_shape)
        truth = load_truth(label_path_for_image(image_path), width, height)
        if result.boxes is None:
            raw_boxes = np.empty((0, 4), dtype=float)
            raw_scores = np.empty(0, dtype=float)
            raw_classes = np.empty(0, dtype=int)
        else:
            raw_boxes = result.boxes.xyxy.cpu().numpy()
            raw_scores = result.boxes.conf.cpu().numpy()
            raw_classes = result.boxes.cls.cpu().numpy().astype(int)
        raw_boxes = pad_and_clip_boxes(raw_boxes, width, height)
        saturated = len(raw_boxes) >= PREPROFILE_MAXIMUM_DETECTIONS

        profile_predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for profile, thresholds in threshold_profiles.items():
            boxes, scores, classes = select_profile(
                raw_boxes, raw_scores, raw_classes, thresholds
            )
            profile_predictions[profile] = (boxes, scores, classes)
            summary = summaries[profile]
            summary["images"] += 1
            summary["predictions"] += len(boxes)
            if saturated:
                summary["preprofile_saturated_images"] += 1
            if not truth:
                summary["hard_negative_images"] += 1
                if len(boxes):
                    summary["hard_negative_images_with_masks"] += 1

        relative = image_path.as_posix().split("/images/validation/", 1)[-1]
        for truth_index, annotation in enumerate(truth):
            key = f"{relative}#{truth_index}"
            status: dict[str, Any] = {
                "class": annotation["label"],
                "profiles": {},
            }
            raw_coverages = intersection_over_truth(raw_boxes, annotation["box"])
            low_covering = np.where(raw_coverages >= VISUAL_PROPOSAL_COVERAGE)[0]
            status["low_confidence_covering"] = [
                {
                    "class": DETECTOR_CLASSES[int(raw_classes[index])],
                    "score": float(raw_scores[index]),
                }
                for index in low_covering.tolist()
            ]
            irrecoverable = not len(low_covering)
            for profile, (boxes, _scores, classes) in profile_predictions.items():
                summary = summaries[profile]
                values = summary["per_class"][annotation["label"]]
                summary["truth"] += 1
                values["support"] += 1
                if irrecoverable:
                    summary["irrecoverable_misses"] += 1
                    values["irrecoverable_misses"] += 1
                coverages = intersection_over_truth(boxes, annotation["box"])
                covering = np.where(coverages >= VISUAL_PROPOSAL_COVERAGE)[0]
                covered = bool(len(covering))
                compatible = bool(
                    len(covering)
                    and np.any(classes[covering] == annotation["class_id"])
                )
                if covered:
                    summary["covered"] += 1
                    values["covered"] += 1
                if compatible:
                    summary["class_compatible_covered"] += 1
                    values["class_compatible_covered"] += 1
                status["profiles"][profile] = {
                    "covered": covered,
                    "class_compatible_covered": compatible,
                    "covering_classes": sorted(
                        {
                            DETECTOR_CLASSES[int(classes[index])]
                            for index in covering.tolist()
                        }
                    ),
                }
            truth_status[key] = status

    source_manifest.unlink(missing_ok=True)
    if seen_results != expected_results:
        raise RuntimeError("runtime predictor did not evaluate the complete image inventory")

    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "image_inventory": {
            "images": len(resolved_images),
            "ordered_absolute_paths_sha256": inventory_sha256,
        },
        "runtime_contract": runtime_contract(threshold_profiles),
        "runtime_contract_sha256": canonical_json_sha256(
            runtime_contract(threshold_profiles)
        ),
        "profiles": {
            name: _finalize_summary(summary) for name, summary in summaries.items()
        },
    }
    return report, truth_status


def compare_status(
    seed: dict[str, dict[str, Any]], candidate: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    if set(seed) != set(candidate):
        raise ValueError("seed and candidate truth inventories differ")
    comparison: dict[str, Any] = {"profiles": {}}
    profiles = set()
    for status in seed.values():
        profiles.update(status.get("profiles", {}))
    if not profiles:
        raise ValueError("runtime comparison has no threshold profiles")
    for profile in sorted(profiles):
        lost = []
        gained = []
        lost_compatible = []
        attribution = Counter()
        for key in sorted(seed):
            seed_profile = seed[key]["profiles"][profile]
            candidate_profile = candidate[key]["profiles"][profile]
            if seed_profile["covered"] and not candidate_profile["covered"]:
                lost.append(key)
                if candidate[key]["low_confidence_covering"]:
                    attribution["below_profile_threshold"] += 1
                else:
                    attribution["localization_or_preprofile_nms"] += 1
            if not seed_profile["covered"] and candidate_profile["covered"]:
                gained.append(key)
            if (
                seed_profile["class_compatible_covered"]
                and not candidate_profile["class_compatible_covered"]
            ):
                lost_compatible.append(key)
                if candidate_profile["covered"]:
                    attribution["wrong_top1_but_still_redacted"] += 1
        comparison["profiles"][profile] = {
            "lost_privacy_coverage": len(lost),
            "gained_privacy_coverage": len(gained),
            "net_privacy_coverage": len(gained) - len(lost),
            "lost_class_compatible_coverage": len(lost_compatible),
            "loss_attribution": dict(sorted(attribution.items())),
            "lost_privacy_examples": lost[:100],
            "gained_privacy_examples": gained[:100],
        }
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare checkpoints with PLVA's top-1 runtime detector policy"
    )
    parser.add_argument("--dataset-yaml", type=Path, required=True)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    image_paths = resolve_dataset_images(args.dataset_yaml, "val")

    seed_report, seed_status = evaluate_checkpoint(
        args.seed, image_paths, device=args.device, batch=args.batch
    )
    candidate_report, candidate_status = evaluate_checkpoint(
        args.candidate, image_paths, device=args.device, batch=args.batch
    )
    document = {
        "schema_version": 1,
        "kind": "plva-runtime-top1-fixed-profile-detector-comparison",
        "dataset_yaml": str(args.dataset_yaml),
        "validation_images": len(image_paths),
        "runtime_contract": runtime_contract(),
        "checkpoints": {"seed": seed_report, "candidate": candidate_report},
        "candidate_minus_seed": compare_status(seed_status, candidate_status),
    }
    serialized = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.partial")
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(args.output)
    print(serialized, end="")


if __name__ == "__main__":
    main()
