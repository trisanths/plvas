from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.schema import sha256_file
from training.visual.prepare_webpii import DETECTOR_CLASSES
from training.visual.train_detector import (
    OTHER_SENSITIVE_CLASSES,
    SECRET_VISUAL_CLASSES,
    STABLE_OPTIMIZER_POLICY,
    consider_safety_candidate,
    require_valid_safety_decision,
)
from training.visual.training_attempt import (
    attempt_artifact_dir,
    attempt_output_dir,
    initialize_seeded_attempt,
    load_seeded_attempt,
    resolve_frozen_seed,
    seeded_attempt_start_mode,
    validate_seeded_attempt_invocation,
)


def class_map(
    secret: float, other: float, overrides: dict[str, float] | None = None
) -> dict[str, float]:
    values = {
        name: secret if name in SECRET_VISUAL_CLASSES else other
        for name in DETECTOR_CLASSES
    }
    values.update(overrides or {})
    return values


def initialize_state(
    recalls: dict[str, float] | None = None,
    precisions: dict[str, float] | None = None,
) -> dict:
    decision, state = consider_safety_candidate(
        None,
        epoch=-1,
        recall_by_class=recalls or class_map(0.5, 0.8),
        precision_by_class=precisions
        or {name: 0.8 for name in DETECTOR_CLASSES},
        missing_classes=[],
    )
    if not decision["selected"] or state is None:
        raise AssertionError(decision)
    return state


class ConstrainedSafetySelectionTests(unittest.TestCase):
    def test_human_epoch_five_cannot_displace_epoch_two(self) -> None:
        epoch_two_recalls = class_map(0.5830700553, 0.9273324774)
        state = initialize_state(
            epoch_two_recalls,
            {name: 0.8556403503 for name in DETECTOR_CLASSES},
        )
        decision, next_state = consider_safety_candidate(
            state,
            epoch=5,
            recall_by_class=class_map(0.5924606303, 0.7004927363),
            precision_by_class={
                name: 0.8975670572 for name in DETECTOR_CLASSES
            },
            missing_classes=[],
        )
        self.assertFalse(decision["selected"])
        self.assertIs(next_state, state)
        self.assertAlmostEqual(
            decision["deltas"]["secret_recall_minimum"], 0.009390575
        )
        self.assertAlmostEqual(
            decision["deltas"]["other_sensitive_recall_mean"], -0.2268397411
        )

    def test_exact_meaningful_gain_and_drop_boundaries_pass(self) -> None:
        state = initialize_state()
        decision, next_state = consider_safety_candidate(
            state,
            epoch=0,
            recall_by_class=class_map(0.51, 0.77),
            precision_by_class={name: 0.75 for name in DETECTOR_CLASSES},
            missing_classes=[],
        )
        self.assertTrue(decision["selected"])
        self.assertEqual(decision["reason"], "selected-meaningful-secret-improvement")
        self.assertIsNot(next_state, state)

        decision, _ = consider_safety_candidate(
            state,
            epoch=0,
            recall_by_class=class_map(0.51, 0.769999),
            precision_by_class={name: 0.75 for name in DETECTOR_CLASSES},
            missing_classes=[],
        )
        self.assertFalse(decision["selected"])
        self.assertEqual(decision["reason"], "rejected-regression-guard")

    def test_tie_band_and_precision_paths_require_recall_safety(self) -> None:
        state = initialize_state()
        decision, _ = consider_safety_candidate(
            state,
            epoch=0,
            recall_by_class=class_map(0.509, 0.81),
            precision_by_class={name: 0.8 for name in DETECTOR_CLASSES},
            missing_classes=[],
        )
        self.assertTrue(decision["selected"])
        self.assertEqual(
            decision["reason"], "selected-tie-band-other-sensitive-improvement"
        )

        decision, _ = consider_safety_candidate(
            state,
            epoch=0,
            recall_by_class=class_map(0.5, 0.8),
            precision_by_class={name: 0.81 for name in DETECTOR_CLASSES},
            missing_classes=[],
        )
        self.assertTrue(decision["selected"])
        self.assertEqual(
            decision["reason"],
            "selected-precision-tiebreak-without-recall-regression",
        )

        regressed = class_map(0.5, 0.8, {OTHER_SENSITIVE_CLASSES[0]: 0.799})
        decision, _ = consider_safety_candidate(
            state,
            epoch=0,
            recall_by_class=regressed,
            precision_by_class={name: 0.81 for name in DETECTOR_CLASSES},
            missing_classes=[],
        )
        self.assertFalse(decision["selected"])

    def test_individual_secret_guard_and_high_water_prevent_ratcheting(self) -> None:
        seed = class_map(
            0.5,
            0.8,
            {"CVC": 0.8, "SECRET": 0.8},
        )
        state = initialize_state(seed)
        candidate = class_map(
            0.51,
            0.8,
            {"CVC": 0.78, "SECRET": 0.8},
        )
        decision, _ = consider_safety_candidate(
            state,
            epoch=0,
            recall_by_class=candidate,
            precision_by_class={name: 0.8 for name in DETECTOR_CLASSES},
            missing_classes=[],
        )
        self.assertFalse(decision["selected"])
        self.assertFalse(decision["gates"]["individual_secret_drop"])

        state = initialize_state()
        first, state = consider_safety_candidate(
            state,
            epoch=0,
            recall_by_class=class_map(0.51, 0.77),
            precision_by_class={name: 0.8 for name in DETECTOR_CLASSES},
            missing_classes=[],
        )
        self.assertTrue(first["selected"])
        assert state is not None
        second, unchanged = consider_safety_candidate(
            state,
            epoch=1,
            recall_by_class=class_map(0.52, 0.74),
            precision_by_class={name: 0.8 for name in DETECTOR_CLASSES},
            missing_classes=[],
        )
        self.assertFalse(second["selected"])
        self.assertFalse(second["gates"]["other_mean_drop_from_high_water"])
        self.assertIs(unchanged, state)

    def test_missing_and_nonfinite_metrics_abort_fail_closed(self) -> None:
        state = initialize_state()
        decision, unchanged = consider_safety_candidate(
            state,
            epoch=0,
            recall_by_class=class_map(0.6, 0.8),
            precision_by_class={name: 0.8 for name in DETECTOR_CLASSES},
            missing_classes=["CVC"],
        )
        self.assertFalse(decision["valid"])
        self.assertIs(unchanged, state)
        with self.assertRaisesRegex(RuntimeError, "incomplete or non-finite"):
            require_valid_safety_decision(decision)

        invalid = class_map(0.6, 0.8, {"SECRET": float("nan")})
        decision, unchanged = consider_safety_candidate(
            state,
            epoch=0,
            recall_by_class=invalid,
            precision_by_class={name: 0.8 for name in DETECTOR_CLASSES},
            missing_classes=[],
        )
        self.assertFalse(decision["valid"])
        self.assertIs(unchanged, state)


class FrozenSeedAttemptTests(unittest.TestCase):
    @staticmethod
    def _staged_metadata(root: Path) -> dict[str, str]:
        stage = root / "local-stage"
        stage.mkdir()
        values = {
            "dataset_yaml": stage / "dataset.yaml",
            "source_manifest": stage / "source-manifest.json",
            "staging_manifest": stage / "staging-manifest.json",
            "replay_manifest": stage / "replay-manifest.json",
            "replay_train_list": stage / "replay-list.txt",
            "runtime_selection_contract": stage / "selection.json",
        }
        for index, path in enumerate(values.values()):
            path.write_text(f"stage-{index}\n", encoding="utf-8")
        return {name: str(path) for name, path in values.items()}

    def test_attempt_paths_are_isolated(self) -> None:
        root = Path("/vol/runs/example")
        self.assertEqual(attempt_output_dir(root, "default"), root / "visual/training")
        self.assertEqual(
            attempt_output_dir(root, "stable"), root / "visual/training-stable"
        )
        self.assertEqual(
            attempt_artifact_dir(root, "stable"), root / "visual/artifacts-stable"
        )

    def test_frozen_seed_is_pinned_copied_and_resume_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot_root = root / "snapshots"
            snapshot_root.mkdir()
            source = snapshot_root / "aggregate-best-epoch2.pt"
            source.write_bytes(b"frozen-epoch-two")
            license_path = root / "detector-base-license.txt"
            license_path.write_text("AGPL-3.0-only\n", encoding="utf-8")
            inherited = root / "detector-base-provenance.json"
            inherited.write_text('{"source":"v2"}\n', encoding="utf-8")
            resolved = resolve_frozen_seed(
                str(source), volume_root=root, snapshot_root=snapshot_root
            )
            attempt_root = root / "runs/run/visual/training-stable"
            result = initialize_seeded_attempt(
                attempt_root=attempt_root,
                run_id="run",
                attempt_id="stable",
                source_checkpoint=resolved,
                expected_sha256=sha256_file(source),
                seed_provenance={"snapshot_id": "epoch2-safety"},
                optimizer_policy=STABLE_OPTIMIZER_POLICY,
                license_path=license_path,
                inherited_provenance_path=inherited,
            )
            self.assertEqual(result["checkpoint"].read_bytes(), source.read_bytes())
            self.assertEqual(result["checkpoint_sha256"], sha256_file(source))
            self.assertTrue(result["manifest"]["resume_allowed"])
            with self.assertRaises(FileExistsError):
                initialize_seeded_attempt(
                    attempt_root=attempt_root,
                    run_id="run",
                    attempt_id="stable",
                    source_checkpoint=resolved,
                    expected_sha256=sha256_file(source),
                    seed_provenance={},
                    optimizer_policy=STABLE_OPTIMIZER_POLICY,
                    license_path=license_path,
                )

            (attempt_root / "frozen-seed.pt").write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "byte count|SHA-256"):
                load_seeded_attempt(
                    attempt_root=attempt_root,
                    run_id="run",
                    attempt_id="stable",
                    optimizer_policy=STABLE_OPTIMIZER_POLICY,
                    license_path=license_path,
                )

    def test_seed_hash_mismatch_fails_before_attempt_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshots = root / "snapshots"
            snapshots.mkdir()
            seed = snapshots / "seed.pt"
            seed.write_bytes(b"seed")
            license_path = root / "license.txt"
            license_path.write_text("AGPL-3.0-only\n", encoding="utf-8")
            attempt = root / "runs/run/visual/training-stable"
            with self.assertRaisesRegex(RuntimeError, "CLI pin"):
                initialize_seeded_attempt(
                    attempt_root=attempt,
                    run_id="run",
                    attempt_id="stable",
                    source_checkpoint=seed,
                    expected_sha256="0" * 64,
                    seed_provenance={},
                    optimizer_policy=STABLE_OPTIMIZER_POLICY,
                    license_path=license_path,
                )
            self.assertFalse(attempt.exists())

    def test_pre_epoch_attempt_atomically_binds_stage_and_restarts_from_seed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshots = root / "snapshots"
            snapshots.mkdir()
            seed = snapshots / "seed.pt"
            seed.write_bytes(b"seed-checkpoint")
            license_path = root / "license.txt"
            license_path.write_text("AGPL-3.0-only\n", encoding="utf-8")
            stage = self._staged_metadata(root)
            attempt_root = root / "runs/run/visual/training-safe"
            provenance = {"snapshot": "safe", "purpose": "test"}

            attempt = initialize_seeded_attempt(
                attempt_root=attempt_root,
                run_id="run",
                attempt_id="safe",
                source_checkpoint=seed,
                expected_sha256=sha256_file(seed),
                seed_provenance=provenance,
                optimizer_policy=STABLE_OPTIMIZER_POLICY,
                license_path=license_path,
                staged_metadata=stage,
            )

            self.assertEqual(
                seeded_attempt_start_mode(attempt_root, attempt),
                "fresh-from-seed",
            )
            self.assertEqual(
                set(attempt["staged_metadata"]),
                {
                    f"{name}_{field}"
                    for name in stage
                    for field in ("path", "sha256")
                },
            )
            validate_seeded_attempt_invocation(
                attempt,
                source_checkpoint=seed,
                expected_sha256=sha256_file(seed),
                seed_provenance=provenance,
            )

            (attempt_root / "safety-selection-progress.json").write_text(
                json.dumps({"schema_version": 4}), encoding="utf-8"
            )
            self.assertEqual(
                seeded_attempt_start_mode(attempt_root, attempt), "resume"
            )

    def test_pre_epoch_recovery_rejects_mutable_or_mismatched_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshots = root / "snapshots"
            snapshots.mkdir()
            seed = snapshots / "seed.pt"
            seed.write_bytes(b"seed-checkpoint")
            license_path = root / "license.txt"
            license_path.write_text("AGPL-3.0-only\n", encoding="utf-8")
            attempt_root = root / "runs/run/visual/training-safe"
            attempt = initialize_seeded_attempt(
                attempt_root=attempt_root,
                run_id="run",
                attempt_id="safe",
                source_checkpoint=seed,
                expected_sha256=sha256_file(seed),
                seed_provenance={"snapshot": "safe"},
                optimizer_policy=STABLE_OPTIMIZER_POLICY,
                license_path=license_path,
                staged_metadata=self._staged_metadata(root),
            )

            (attempt_root / "ultralytics-run").mkdir()
            with self.assertRaisesRegex(RuntimeError, "disposable training"):
                seeded_attempt_start_mode(attempt_root, attempt)
            (attempt_root / "ultralytics-run").rmdir()

            with self.assertRaisesRegex(RuntimeError, "fresh invocation"):
                validate_seeded_attempt_invocation(
                    attempt,
                    source_checkpoint=seed,
                    expected_sha256=sha256_file(seed),
                    seed_provenance={"snapshot": "different"},
                )

            provenance_path = attempt_root / "frozen-seed-provenance.json"
            document = json.loads(provenance_path.read_text(encoding="utf-8"))
            document["declared_provenance"] = {"snapshot": "tampered"}
            provenance_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "provenance bytes changed"):
                load_seeded_attempt(
                    attempt_root=attempt_root,
                    run_id="run",
                    attempt_id="safe",
                    optimizer_policy=STABLE_OPTIMIZER_POLICY,
                    license_path=license_path,
                )

    def test_pre_epoch_recovery_discards_only_verified_seed_audit_outputs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshots = root / "snapshots"
            snapshots.mkdir()
            seed = snapshots / "seed.pt"
            seed.write_bytes(b"seed-checkpoint")
            license_path = root / "license.txt"
            license_path.write_text("AGPL-3.0-only\n", encoding="utf-8")
            attempt_root = root / "runs/run/visual/training-safe"
            attempt = initialize_seeded_attempt(
                attempt_root=attempt_root,
                run_id="run",
                attempt_id="safe",
                source_checkpoint=seed,
                expected_sha256=sha256_file(seed),
                seed_provenance={"snapshot": "safe"},
                optimizer_policy=STABLE_OPTIMIZER_POLICY,
                license_path=license_path,
                staged_metadata=self._staged_metadata(root),
            )
            disposable = [
                attempt_root / "runtime-selection-evaluation",
                attempt_root / "frozen-base-webpii-test",
                attempt_root / "frozen-base-supplemental-test",
            ]
            for directory in disposable:
                directory.mkdir()
                (directory / "audit.json").write_text("{}\n", encoding="utf-8")
            for name in (
                "safety-best.pt",
                "safety-best-thresholds.json",
                "seed-incumbent.json",
                "supplemental-test.yaml",
            ):
                path = attempt_root / name
                path.write_text("pre-epoch\n", encoding="utf-8")
                disposable.append(path)

            self.assertEqual(
                seeded_attempt_start_mode(
                    attempt_root,
                    attempt,
                    discard_pre_epoch_outputs=True,
                ),
                "fresh-from-seed",
            )
            self.assertTrue(all(not path.exists() for path in disposable))
            self.assertEqual(
                sha256_file(attempt_root / "frozen-seed.pt"),
                attempt["checkpoint_sha256"],
            )

    def test_pre_epoch_recovery_refuses_checkpoint_or_epoch_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshots = root / "snapshots"
            snapshots.mkdir()
            seed = snapshots / "seed.pt"
            seed.write_bytes(b"seed-checkpoint")
            license_path = root / "license.txt"
            license_path.write_text("AGPL-3.0-only\n", encoding="utf-8")
            attempt_root = root / "runs/run/visual/training-safe"
            attempt = initialize_seeded_attempt(
                attempt_root=attempt_root,
                run_id="run",
                attempt_id="safe",
                source_checkpoint=seed,
                expected_sha256=sha256_file(seed),
                seed_provenance={"snapshot": "safe"},
                optimizer_policy=STABLE_OPTIMIZER_POLICY,
                license_path=license_path,
                staged_metadata=self._staged_metadata(root),
            )
            weights = attempt_root / "ultralytics-run/weights"
            weights.mkdir(parents=True)
            (weights / "last.pt").write_bytes(b"optimizer-checkpoint")
            with self.assertRaisesRegex(RuntimeError, "checkpoint or optimizer"):
                seeded_attempt_start_mode(
                    attempt_root,
                    attempt,
                    discard_pre_epoch_outputs=True,
                )

            (weights / "last.pt").unlink()
            results = attempt_root / "ultralytics-run/results.csv"
            results.write_text("epoch,loss\n0,1.0\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "epoch history"):
                seeded_attempt_start_mode(
                    attempt_root,
                    attempt,
                    discard_pre_epoch_outputs=True,
                )


if __name__ == "__main__":
    unittest.main()
