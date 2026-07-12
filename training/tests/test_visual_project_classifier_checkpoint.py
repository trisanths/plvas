from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.visual.prepare_webpii import DETECTOR_CLASSES
from training.visual.project_classifier_checkpoint import (
    CLASSIFIER_TENSOR_NAMES,
    audit_classifier_projection,
    canonical_json_sha256,
    main,
    project_classifier_state_dict,
    project_detector_checkpoints,
    sha256_file,
)


class FakeRow:
    def __init__(self, parent: FakeTensor, index: int) -> None:
        self.parent = parent
        self.index = index
        self.shape = parent.shape[1:]
        self.dtype = parent.dtype

    @property
    def value(self) -> int:
        return self.parent.values[self.index]

    def detach(self) -> FakeRow:
        return self

    def cpu(self) -> FakeRow:
        return self

    def equal(self, other: object) -> bool:
        return (
            isinstance(other, FakeRow)
            and self.shape == other.shape
            and self.dtype == other.dtype
            and self.value == other.value
        )

    def copy_(self, other: FakeRow) -> FakeRow:
        self.parent.values[self.index] = other.value
        return self


class FakeTensor:
    def __init__(
        self,
        values: list[int],
        shape: tuple[int, ...],
        dtype: str = "float32",
    ) -> None:
        self.values = list(values)
        self.shape = shape
        self.dtype = dtype

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def clone(self) -> FakeTensor:
        return FakeTensor(self.values, self.shape, self.dtype)

    def equal(self, other: object) -> bool:
        return (
            isinstance(other, FakeTensor)
            and self.values == other.values
            and self.shape == other.shape
            and self.dtype == other.dtype
        )

    def __getitem__(self, index: int) -> FakeRow:
        return FakeRow(self, index)

    def to_json(self) -> dict[str, object]:
        return {
            "values": self.values,
            "shape": list(self.shape),
            "dtype": self.dtype,
        }

    @classmethod
    def from_json(cls, value: dict[str, object]) -> FakeTensor:
        return cls(
            list(value["values"]),  # type: ignore[arg-type]
            tuple(value["shape"]),  # type: ignore[arg-type]
            str(value["dtype"]),
        )


def make_state() -> dict[str, FakeTensor]:
    state = {
        "model.0.conv.weight": FakeTensor([1, 2], (2,)),
        "model.23.cv2.0.2.weight": FakeTensor([3, 4], (2,)),
        "model.23.cv3.0.0.conv.weight": FakeTensor([5, 6], (2,)),
    }
    for scale in range(3):
        state[f"model.23.cv3.{scale}.2.weight"] = FakeTensor(
            [100 * scale + row for row in range(len(DETECTOR_CLASSES))],
            (len(DETECTOR_CLASSES), 8 + scale, 1, 1),
        )
        state[f"model.23.cv3.{scale}.2.bias"] = FakeTensor(
            [200 * scale + row for row in range(len(DETECTOR_CLASSES))],
            (len(DETECTOR_CLASSES),),
        )
    return state


def clone_state(state: dict[str, FakeTensor]) -> dict[str, FakeTensor]:
    return {name: tensor.clone() for name, tensor in state.items()}


def changed_candidate(seed: dict[str, FakeTensor]) -> dict[str, FakeTensor]:
    candidate = clone_state(seed)
    candidate["model.0.conv.weight"].values[0] = 999
    candidate["model.23.cv3.0.0.conv.weight"].values[1] = 998
    candidate["model.23.cv3.0.2.weight"].values[0] += 50
    candidate["model.23.cv3.1.2.bias"].values[1] += 60
    candidate["model.23.cv3.2.2.weight"].values[4] += 70
    candidate["model.23.cv3.2.2.bias"].values[4] += 80
    return candidate


def assert_tensor_equal(left: FakeTensor, right: FakeTensor) -> None:
    assert left.equal(right)


def test_pure_projection_reverts_frozen_and_restores_protected_rows() -> None:
    seed = make_state()
    candidate = changed_candidate(seed)
    seed_before = clone_state(seed)
    candidate_before = clone_state(candidate)

    projected = project_classifier_state_dict(
        seed,
        candidate,
        class_names={str(i): name for i, name in enumerate(DETECTOR_CLASSES)},
        protected_class_names=("CARD_NUMBER",),
    )

    assert_tensor_equal(projected["model.0.conv.weight"], seed["model.0.conv.weight"])
    assert_tensor_equal(
        projected["model.23.cv3.0.0.conv.weight"],
        seed["model.23.cv3.0.0.conv.weight"],
    )
    assert (
        projected["model.23.cv3.0.2.weight"].values[0]
        == candidate["model.23.cv3.0.2.weight"].values[0]
    )
    assert (
        projected["model.23.cv3.1.2.bias"].values[1]
        == candidate["model.23.cv3.1.2.bias"].values[1]
    )
    card_row = DETECTOR_CLASSES.index("CARD_NUMBER")
    assert (
        projected["model.23.cv3.2.2.weight"].values[card_row]
        == seed["model.23.cv3.2.2.weight"].values[card_row]
    )
    assert (
        projected["model.23.cv3.2.2.bias"].values[card_row]
        == seed["model.23.cv3.2.2.bias"].values[card_row]
    )

    report = audit_classifier_projection(
        seed,
        candidate,
        projected,
        protected_class_names=("CARD_NUMBER",),
    )
    assert report["passed"] is True
    assert report["allowed_change_observed"] is True
    assert {row["class_name"] for row in report["changed_allowed_rows"]} == {
        "NAME",
        "EMAIL",
    }
    assert report["candidate_changed_frozen_tensors"] == [
        "model.0.conv.weight",
        "model.23.cv3.0.0.conv.weight",
    ]
    for name in seed:
        assert_tensor_equal(seed[name], seed_before[name])
        assert_tensor_equal(candidate[name], candidate_before[name])
        assert projected[name] is not seed[name]
        assert projected[name] is not candidate[name]


def test_projection_requires_an_unprotected_allowed_change() -> None:
    seed = make_state()
    with pytest.raises(RuntimeError, match="no allowed classifier row change"):
        project_classifier_state_dict(seed, clone_state(seed))

    protected_only = clone_state(seed)
    protected_only["model.23.cv3.0.2.bias"].values[4] += 1
    with pytest.raises(RuntimeError, match="no allowed classifier row change"):
        project_classifier_state_dict(
            seed,
            protected_only,
            protected_class_names=("CARD_NUMBER",),
        )


def test_projection_validates_exact_keys_shapes_dtypes_and_names() -> None:
    seed = make_state()
    candidate = changed_candidate(seed)

    missing = clone_state(candidate)
    del missing[CLASSIFIER_TENSOR_NAMES[0]]
    with pytest.raises(ValueError, match="keys differ"):
        project_classifier_state_dict(seed, missing)

    wrong_shape = clone_state(candidate)
    wrong_shape[CLASSIFIER_TENSOR_NAMES[0]].shape = (9, 99, 1, 1)
    with pytest.raises(ValueError, match="shape"):
        project_classifier_state_dict(seed, wrong_shape)

    wrong_dtype = clone_state(candidate)
    wrong_dtype[CLASSIFIER_TENSOR_NAMES[1]].dtype = "float16"
    with pytest.raises(ValueError, match="dtype"):
        project_classifier_state_dict(seed, wrong_dtype)

    extra_seed = clone_state(seed)
    extra_candidate = clone_state(candidate)
    extra_seed["model.23.cv3.3.2.weight"] = FakeTensor([1] * 9, (9, 2, 1, 1))
    extra_candidate["model.23.cv3.3.2.weight"] = FakeTensor([1] * 9, (9, 2, 1, 1))
    with pytest.raises(ValueError, match="exact projection boundary"):
        project_classifier_state_dict(extra_seed, extra_candidate)

    wrong_names = list(DETECTOR_CLASSES)
    wrong_names[0], wrong_names[1] = wrong_names[1], wrong_names[0]
    with pytest.raises(ValueError, match="exactly match"):
        project_classifier_state_dict(seed, candidate, class_names=wrong_names)
    with pytest.raises(ValueError, match="unknown protected"):
        project_classifier_state_dict(
            seed, candidate, protected_class_names=("NOT_A_CLASS",)
        )
    with pytest.raises(ValueError, match="duplicates"):
        project_classifier_state_dict(
            seed,
            candidate,
            protected_class_names=("CVC", "CVC"),
        )


def test_audit_detects_frozen_protected_and_candidate_row_tampering() -> None:
    seed = make_state()
    candidate = changed_candidate(seed)
    projected = project_classifier_state_dict(
        seed, candidate, protected_class_names=("CARD_NUMBER",)
    )

    projected["model.0.conv.weight"].values[0] += 1
    projected["model.23.cv3.2.2.bias"].values[4] += 1
    projected["model.23.cv3.1.2.bias"].values[1] += 1
    report = audit_classifier_projection(
        seed,
        candidate,
        projected,
        protected_class_names=("CARD_NUMBER",),
        strict=False,
    )
    assert report["passed"] is False
    assert report["frozen_tensor_mismatches"] == ["model.0.conv.weight"]
    assert report["protected_row_mismatches"] == [
        {
            "tensor": "model.23.cv3.2.2.bias",
            "row_index": 4,
            "class_name": "CARD_NUMBER",
        }
    ]
    assert report["candidate_row_mismatches"] == [
        {
            "tensor": "model.23.cv3.1.2.bias",
            "row_index": 1,
            "class_name": "EMAIL",
        }
    ]
    with pytest.raises(RuntimeError, match="frozen tensors"):
        audit_classifier_projection(
            seed,
            candidate,
            projected,
            protected_class_names=("CARD_NUMBER",),
        )


class FakeModel:
    def __init__(self, names: dict[str, str], state: dict[str, FakeTensor]) -> None:
        self.names = names
        self.state = clone_state(state)

    def state_dict(self) -> dict[str, FakeTensor]:
        return self.state

    def load_state_dict(
        self, state: dict[str, FakeTensor], *, strict: bool
    ) -> SimpleNamespace:
        assert strict is True
        self.state = clone_state(state)
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])


class FakeYOLO:
    corrupt_saved_frozen_tensor = False

    def __init__(self, path: str) -> None:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        state = {
            name: FakeTensor.from_json(tensor)
            for name, tensor in value["state"].items()
        }
        self.model = FakeModel(value["names"], state)

    def save(self, path: str) -> None:
        state = clone_state(self.model.state)
        if self.corrupt_saved_frozen_tensor:
            state["model.0.conv.weight"].values[0] += 123
        value = {
            "names": self.model.names,
            "state": {name: tensor.to_json() for name, tensor in state.items()},
        }
        Path(path).write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def write_fake_checkpoint(
    path: Path,
    state: dict[str, FakeTensor],
    *,
    names: tuple[str, ...] = DETECTOR_CLASSES,
) -> None:
    value = {
        "names": {str(index): name for index, name in enumerate(names)},
        "state": {name: tensor.to_json() for name, tensor in state.items()},
    }
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def test_cli_ultralytics_save_reloads_audits_and_emits_hashed_provenance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    seed_path = tmp_path / "seed.pt"
    candidate_path = tmp_path / "candidate.pt"
    output_path = tmp_path / "projected.pt"
    provenance_path = tmp_path / "projection.json"
    seed = make_state()
    candidate = changed_candidate(seed)
    write_fake_checkpoint(seed_path, seed)
    write_fake_checkpoint(candidate_path, candidate)

    exit_code = main(
        [
            "--seed-checkpoint",
            str(seed_path),
            "--candidate-checkpoint",
            str(candidate_path),
            "--output-checkpoint",
            str(output_path),
            "--provenance",
            str(provenance_path),
            "--protected-class",
            "CARD_NUMBER",
            "--expected-seed-sha256",
            sha256_file(seed_path),
            "--expected-candidate-sha256",
            sha256_file(candidate_path),
        ],
        yolo_factory=FakeYOLO,
    )

    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["audit_passed"] is True
    assert printed["protected_classes"] == ["CARD_NUMBER"]
    assert printed["projected_checkpoint"]["sha256"] == sha256_file(output_path)
    assert printed["provenance"]["sha256"] == sha256_file(provenance_path)

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    payload = dict(provenance)
    payload_sha256 = payload.pop("payload_sha256")
    assert payload_sha256 == canonical_json_sha256(payload)
    assert provenance["policy_sha256"] == canonical_json_sha256(provenance["policy"])
    assert provenance["inputs"]["seed_checkpoint"]["sha256"] == sha256_file(seed_path)
    assert provenance["inputs"]["rejected_candidate_checkpoint"][
        "sha256"
    ] == sha256_file(candidate_path)
    assert provenance["output"]["projected_checkpoint"]["sha256"] == sha256_file(
        output_path
    )
    assert provenance["in_memory_audit"]["passed"] is True
    assert provenance["reloaded_checkpoint_audit"]["passed"] is True

    reloaded = FakeYOLO(str(output_path))
    output_state = reloaded.model.state_dict()
    report = audit_classifier_projection(
        seed,
        candidate,
        output_state,
        protected_class_names=("CARD_NUMBER",),
    )
    assert report["passed"] is True


def test_reload_audit_blocks_corrupt_checkpoint_publication(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.pt"
    candidate_path = tmp_path / "candidate.pt"
    output_path = tmp_path / "projected.pt"
    provenance_path = tmp_path / "projection.json"
    write_fake_checkpoint(seed_path, make_state())
    write_fake_checkpoint(candidate_path, changed_candidate(make_state()))
    FakeYOLO.corrupt_saved_frozen_tensor = True
    try:
        with pytest.raises(RuntimeError, match="frozen tensors"):
            project_detector_checkpoints(
                seed_checkpoint=seed_path,
                candidate_checkpoint=candidate_path,
                output_checkpoint=output_path,
                provenance_path=provenance_path,
                yolo_factory=FakeYOLO,
            )
    finally:
        FakeYOLO.corrupt_saved_frozen_tensor = False
    assert not output_path.exists()
    assert not provenance_path.exists()


def test_checkpoint_projection_refuses_name_or_hash_drift(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.pt"
    candidate_path = tmp_path / "candidate.pt"
    output_path = tmp_path / "projected.pt"
    provenance_path = tmp_path / "projection.json"
    write_fake_checkpoint(seed_path, make_state())
    wrong_names = tuple(reversed(DETECTOR_CLASSES))
    write_fake_checkpoint(
        candidate_path, changed_candidate(make_state()), names=wrong_names
    )
    with pytest.raises(ValueError, match="exactly match"):
        project_detector_checkpoints(
            seed_checkpoint=seed_path,
            candidate_checkpoint=candidate_path,
            output_checkpoint=output_path,
            provenance_path=provenance_path,
            yolo_factory=FakeYOLO,
        )

    write_fake_checkpoint(candidate_path, changed_candidate(make_state()))
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        project_detector_checkpoints(
            seed_checkpoint=seed_path,
            candidate_checkpoint=candidate_path,
            output_checkpoint=output_path,
            provenance_path=provenance_path,
            expected_candidate_sha256="0" * 64,
            yolo_factory=FakeYOLO,
        )
