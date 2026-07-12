from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import modal

from training.modal_app import (
    VISUAL_STAGE_COPY_WORKERS,
    VISUAL_TRAIN_BATCH_SIZE,
    VISUAL_TRAIN_CPUS,
    VISUAL_TRAIN_MEMORY_MIB,
    cache_volume,
    run_paths,
    run_volume,
    validate_run_id,
    visual_image,
)


APP_NAME = "plva-visual-projection-audit"
PROJECTION_AUDIT_GPU = "L4"
_OUTPUT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.json")

app = modal.App(APP_NAME, image=visual_image)


def _parse_candidates(value: str) -> list[dict[str, str]]:
    try:
        candidates = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("candidate configuration must be valid JSON") from exc
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidate configuration must be a non-empty list")
    normalized: list[dict[str, str]] = []
    labels: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != {
            "label",
            "path",
            "sha256",
        }:
            raise ValueError("each candidate requires label, path, and sha256")
        label = candidate["label"]
        path = candidate["path"]
        sha256 = candidate["sha256"]
        if (
            not isinstance(label, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", label)
            or label in labels
            or not isinstance(path, str)
            or not path.startswith("/vol/")
            or not isinstance(sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", sha256)
        ):
            raise ValueError("candidate configuration is unsafe or duplicated")
        labels.add(label)
        normalized.append({"label": label, "path": path, "sha256": sha256})
    return normalized


@app.function(
    image=visual_image,
    gpu=PROJECTION_AUDIT_GPU,
    cpu=VISUAL_TRAIN_CPUS,
    memory=VISUAL_TRAIN_MEMORY_MIB,
    timeout=3 * 60 * 60,
    retries=2,
    volumes={"/vol": run_volume, "/cache": cache_volume},
)
def projection_audit_remote(
    run_id: str,
    seed_checkpoint: str,
    expected_seed_sha256: str,
    candidates_json: str,
    output_name: str,
) -> dict[str, Any]:
    from training.schema import sha256_file
    from training.visual.audit_checkpoint_projections import (
        CandidateSpec,
        audit_checkpoint_projections,
    )

    validate_run_id(run_id)
    if (
        not seed_checkpoint.startswith("/vol/")
        or not re.fullmatch(r"[0-9a-f]{64}", expected_seed_sha256)
        or _OUTPUT_NAME.fullmatch(output_name) is None
    ):
        raise ValueError("projection audit seed or output configuration is unsafe")
    candidate_values = _parse_candidates(candidates_json)
    candidates = tuple(
        CandidateSpec(
            label=value["label"],
            checkpoint=Path(value["path"]),
            expected_sha256=value["sha256"],
        )
        for value in candidate_values
    )
    paths = run_paths(run_id)
    output = paths["root"] / "diagnostics" / output_name
    result = audit_checkpoint_projections(
        dataset_yaml=paths["visual_dataset"] / "dataset.yaml",
        composed_manifest=paths["visual_dataset"] / "manifest.json",
        source_roots={
            "webpii": paths["visual_webpii"],
            "synthetic": paths["visual_synthetic"],
            "supplemental_ats": paths["visual_supplemental"],
        },
        stage_root=Path("/tmp/plva-v9-projection-runtime-audit"),
        seed_checkpoint=Path(seed_checkpoint),
        candidates=candidates,
        output=output,
        expected_seed_sha256=expected_seed_sha256,
        evaluation_output_dir=Path("/tmp/plva-v9-projection-runtime-evaluations"),
        device="0",
        batch=64,
        workers=16,
        replay_seed=1311,
        replay_batch_size=VISUAL_TRAIN_BATCH_SIZE,
        copy_workers=VISUAL_STAGE_COPY_WORKERS,
        force_stage=True,
    )
    run_volume.commit()
    return {
        "gpu": PROJECTION_AUDIT_GPU,
        "output": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
        "summary": result["summary"],
        "published_test_used": result["published_test_used"],
    }


@app.local_entrypoint()
def main(
    run_id: str,
    seed_checkpoint: str,
    expected_seed_sha256: str,
    candidates_json: str,
    output_name: str = "v9-projection-runtime-audit.json",
) -> None:
    result = projection_audit_remote.remote(
        run_id,
        seed_checkpoint,
        expected_seed_sha256,
        candidates_json,
        output_name,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
