from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from training.schema import sha256_file
from training.visual import audit_checkpoint_projections as audit_module
from training.visual.audit_checkpoint_projections import (
    CandidateSpec,
    audit_checkpoint_projections,
    main,
    parse_candidate_specs,
)


def _inputs(root: Path) -> dict[str, Any]:
    dataset = root / "dataset.yaml"
    dataset.write_text("names: []\n", encoding="utf-8")
    manifest = root / "source-manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    roots = {
        "webpii": root / "webpii",
        "synthetic": root / "synthetic",
        "supplemental_ats": root / "supplemental-ats",
    }
    for source_root in roots.values():
        source_root.mkdir()
    seed = root / "seed.pt"
    first = root / "first.pt"
    second = root / "second.pt"
    seed.write_bytes(b"immutable-seed")
    first.write_bytes(b"first-projection")
    second.write_bytes(b"second-projection")
    return {
        "dataset": dataset,
        "manifest": manifest,
        "roots": roots,
        "seed": seed,
        "first": first,
        "second": second,
    }


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch, *, stage_root: Path
) -> dict[str, Any]:
    calls: dict[str, Any] = {
        "stage": [],
        "replay": [],
        "evaluator_init": [],
        "seed": [],
        "candidates": [],
        "consider": [],
        "canary": [],
    }

    def fake_stage(
        dataset_yaml: Path,
        composed_manifest: Path,
        roots: dict[str, Path],
        destination: Path,
        *,
        force: bool,
        copy_workers: int,
    ) -> dict[str, Any]:
        destination.mkdir(parents=True, exist_ok=True)
        calls["stage"].append(
            (dataset_yaml, composed_manifest, roots, destination, force, copy_workers)
        )
        return {
            "root": str(destination),
            "dataset_yaml": str(destination / "dataset.yaml"),
            "source_manifest": str(destination / "source-manifest.json"),
            "staging_manifest": str(destination / "staging-manifest.json"),
            "metadata": {"test_used_for_checkpoint_selection": False},
        }

    def fake_replay(stage: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        calls["replay"].append((stage, kwargs))
        contract = stage_root / "selection/runtime-selection-contract.json"
        contract.parent.mkdir(parents=True, exist_ok=True)
        contract.write_text('{"published_test_used":false}\n', encoding="utf-8")
        return {
            **stage,
            "mode": "verified-local-stage-with-deterministic-replay",
            "runtime_selection_contract": str(contract.resolve()),
            "runtime_selection_contract_sha256": sha256_file(contract),
            "runtime_selection": {"published_test_used": False},
        }

    class FakeEvaluator:
        def __init__(
            self,
            contract: Path,
            *,
            device: str,
            batch: int,
            workers: int,
            output_dir: Path,
        ) -> None:
            calls["evaluator_init"].append(
                (contract, device, batch, workers, output_dir)
            )

        def initialize_seed(self, checkpoint: Path) -> dict[str, Any]:
            calls["seed"].append(checkpoint)
            return {
                "checkpoint_sha256": sha256_file(checkpoint),
                "projection": "seed",
            }

        def evaluate_candidate(self, checkpoint: Path, *, epoch: int) -> dict[str, Any]:
            calls["candidates"].append((checkpoint, epoch))
            return {
                "checkpoint_sha256": sha256_file(checkpoint),
                "projection": checkpoint.stem,
                "epoch": epoch,
            }

    def fake_consider(
        state: dict[str, Any] | None, metrics: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        calls["consider"].append((state, metrics))
        if state is None:
            return (
                {
                    "valid": True,
                    "selected": True,
                    "reason": "initial-calibrated-runtime-reference",
                    "gates": {"all_hard_gates": True},
                },
                {"incumbent": "seed", "history": []},
            )
        selected = metrics["projection"] == "first"
        next_state = {
            "incumbent": metrics["projection"] if selected else state["incumbent"],
            "history": [*state["history"], metrics["projection"]],
        }
        return (
            {
                "valid": True,
                "selected": selected,
                "reason": "selected" if selected else "rejected",
                "gates": {"all_hard_gates": True},
            },
            next_state,
        )

    def fake_canary(seed: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
        calls["canary"].append((seed, record))
        return {
            "passed": record["selection_gates"]["all_hard_gates"],
            "candidate": record["runtime_selection_metrics"]["projection"],
        }

    monkeypatch.setattr(audit_module, "stage_composed_dataset", fake_stage)
    monkeypatch.setattr(audit_module, "materialize_replay_training_stage", fake_replay)
    monkeypatch.setattr(
        audit_module, "CalibratedRuntimeSelectionEvaluator", FakeEvaluator
    )
    monkeypatch.setattr(
        audit_module, "consider_runtime_selection_candidate", fake_consider
    )
    monkeypatch.setattr(
        audit_module,
        "evaluate_first_epoch_calibrated_runtime_canary",
        fake_canary,
    )
    return calls


def test_audit_stages_once_and_evaluates_ordered_projections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path)
    stage_root = tmp_path / "stage"
    output = tmp_path / "audit.json"
    calls = _install_fakes(monkeypatch, stage_root=stage_root)

    result = audit_checkpoint_projections(
        dataset_yaml=inputs["dataset"],
        composed_manifest=inputs["manifest"],
        source_roots=inputs["roots"],
        stage_root=stage_root,
        seed_checkpoint=inputs["seed"],
        candidates=(
            CandidateSpec("first", inputs["first"], sha256_file(inputs["first"])),
            CandidateSpec("second", inputs["second"], sha256_file(inputs["second"])),
        ),
        output=output,
        expected_seed_sha256=sha256_file(inputs["seed"]),
        device="cpu",
        batch=3,
        workers=0,
    )

    assert len(calls["stage"]) == 1
    assert len(calls["replay"]) == 1
    assert calls["replay"][0][1]["runtime_selection"] is True
    assert len(calls["evaluator_init"]) == 1
    assert calls["evaluator_init"][0][1:4] == ("cpu", 3, 0)
    assert calls["seed"] == [inputs["seed"].resolve()]
    assert calls["candidates"] == [
        (inputs["first"].resolve(), 0),
        (inputs["second"].resolve(), 0),
    ]
    assert len(calls["consider"]) == 3
    assert len(calls["canary"]) == 2
    assert (
        calls["consider"][1][0]
        == calls["consider"][2][0]
        == {
            "incumbent": "seed",
            "history": [],
        }
    )
    assert all(
        canary_seed == calls["consider"][0][1]
        for canary_seed, _record in calls["canary"]
    )

    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted == result
    assert result["published_test_used"] is False
    assert result["runtime_selection"]["candidate_comparison_mode"] == (
        "order-independent-fresh-immutable-seed-state"
    )
    assert result["runtime_selection"]["invariants"] == {
        "single_evaluator_instance": True,
        "single_seed_evaluation": True,
        "candidate_order_independent": True,
        "each_candidate_considered_from_immutable_seed_state": True,
        "each_canary_compared_to_immutable_seed": True,
    }
    seed_state_sha256 = result["runtime_selection"]["seed"]["selection_state_sha256"]
    assert {
        record["selection_input_state_sha256"]
        for record in result["runtime_selection"]["candidates"]
    } == {seed_state_sha256}
    assert result["summary"] == {
        "candidate_count": 2,
        "all_canaries_passed": True,
        "candidate_order_independent": True,
        "selected_candidate_labels": ["first"],
        "candidates": [
            {
                "label": "first",
                "sha256": sha256_file(inputs["first"]),
                "selected": True,
                "reason": "selected",
                "canary_passed": True,
            },
            {
                "label": "second",
                "sha256": sha256_file(inputs["second"]),
                "selected": False,
                "reason": "rejected",
                "canary_passed": True,
            },
        ],
    }
    payload = dict(result)
    payload_sha256 = payload.pop("payload_sha256")
    assert payload_sha256 == audit_module.canonical_json_sha256(payload)
    assert not list(tmp_path.glob(".audit.json.partial-*"))


def test_main_prints_only_concise_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inputs = _inputs(tmp_path)
    stage_root = tmp_path / "stage"
    output = tmp_path / "audit.json"
    _install_fakes(monkeypatch, stage_root=stage_root)

    assert (
        main(
            [
                "--dataset-yaml",
                str(inputs["dataset"]),
                "--composed-manifest",
                str(inputs["manifest"]),
                "--webpii-root",
                str(inputs["roots"]["webpii"]),
                "--synthetic-root",
                str(inputs["roots"]["synthetic"]),
                "--ats-root",
                str(inputs["roots"]["supplemental_ats"]),
                "--stage-root",
                str(stage_root),
                "--seed-checkpoint",
                str(inputs["seed"]),
                "--expected-seed-sha256",
                sha256_file(inputs["seed"]),
                "--candidate",
                f"first={inputs['first']}",
                "--expected-candidate-sha256",
                f"first={sha256_file(inputs['first'])}",
                "--output",
                str(output),
                "--device",
                "cpu",
            ]
        )
        == 0
    )

    stdout = json.loads(capsys.readouterr().out)
    full = json.loads(output.read_text(encoding="utf-8"))
    assert stdout["candidate_count"] == 1
    assert stdout["candidates"] == full["summary"]["candidates"]
    assert stdout["output"]["sha256"] == sha256_file(output)
    assert "staging" not in stdout
    assert "runtime_selection" in full


def test_candidate_decisions_and_canaries_are_order_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path)
    stage_root = tmp_path / "stage-order-independent"

    def run(labels: tuple[str, ...], suffix: str) -> dict[str, Any]:
        _install_fakes(monkeypatch, stage_root=stage_root)
        checkpoints = {"first": inputs["first"], "second": inputs["second"]}
        return audit_checkpoint_projections(
            dataset_yaml=inputs["dataset"],
            composed_manifest=inputs["manifest"],
            source_roots=inputs["roots"],
            stage_root=stage_root,
            seed_checkpoint=inputs["seed"],
            candidates=tuple(
                CandidateSpec(label, checkpoints[label]) for label in labels
            ),
            output=tmp_path / f"audit-{suffix}.json",
            device="cpu",
        )

    forward = run(("first", "second"), "forward")
    reverse = run(("second", "first"), "reverse")

    def outcomes(result: dict[str, Any]) -> dict[str, Any]:
        return {
            record["label"]: {
                "metrics": record["runtime_selection_metrics"],
                "decision": record["selection_decision"],
                "canary": record["calibrated_runtime_canary"],
                "baseline": record["selection_input_state_sha256"],
            }
            for record in result["runtime_selection"]["candidates"]
        }

    assert outcomes(forward) == outcomes(reverse)
    assert forward["summary"]["candidate_order_independent"] is True
    assert reverse["summary"]["candidate_order_independent"] is True


@pytest.mark.parametrize(
    ("candidates", "hashes", "message"),
    [
        (["one=/a.pt", "one=/b.pt"], [], "duplicate candidate label"),
        (["one=/a.pt"], ["two=" + "a" * 64], "no matching candidate"),
        (
            ["one=/a.pt"],
            ["one=" + "a" * 64, "one=" + "b" * 64],
            "duplicate expected candidate",
        ),
        (["seed=/a.pt"], [], "reserved"),
    ],
)
def test_candidate_assignments_fail_closed(
    candidates: list[str], hashes: list[str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_candidate_specs(candidates, hashes)


def test_hash_mismatch_fails_before_dataset_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path)
    staged = False

    def forbidden_stage(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal staged
        staged = True
        raise AssertionError("stage must not run")

    monkeypatch.setattr(audit_module, "stage_composed_dataset", forbidden_stage)
    with pytest.raises(RuntimeError, match="seed checkpoint SHA-256 mismatch"):
        audit_checkpoint_projections(
            dataset_yaml=inputs["dataset"],
            composed_manifest=inputs["manifest"],
            source_roots=inputs["roots"],
            stage_root=tmp_path / "stage",
            seed_checkpoint=inputs["seed"],
            candidates=(CandidateSpec("first", inputs["first"]),),
            output=tmp_path / "audit.json",
            expected_seed_sha256="0" * 64,
        )
    assert staged is False


def test_symlink_checkpoint_is_rejected_before_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path)
    linked = tmp_path / "linked.pt"
    linked.symlink_to(inputs["first"])
    staged = False

    def forbidden_stage(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal staged
        staged = True
        raise AssertionError("stage must not run")

    monkeypatch.setattr(audit_module, "stage_composed_dataset", forbidden_stage)
    with pytest.raises(RuntimeError, match="missing or unsafe"):
        audit_checkpoint_projections(
            dataset_yaml=inputs["dataset"],
            composed_manifest=inputs["manifest"],
            source_roots=inputs["roots"],
            stage_root=tmp_path / "stage",
            seed_checkpoint=inputs["seed"],
            candidates=(CandidateSpec("linked", linked),),
            output=tmp_path / "audit.json",
        )
    assert staged is False
