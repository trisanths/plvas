from __future__ import annotations

import unittest
from types import SimpleNamespace

from training.visual.safe_finetune import (
    CALLBACK_EVENTS,
    FROZEN_LAYER_INDICES,
    HEAD_LAYER_INDEX,
    SafeFineTuneController,
    apply_safe_finetune_controls,
    safe_finetune_training_overrides,
    validate_frozen_state_parity,
    validate_seed_state_parity,
    wire_safe_finetune_callbacks,
)


class FakeParameter:
    def __init__(
        self,
        elements: int,
        requires_grad: bool = True,
        shape: tuple[int, ...] | None = None,
    ) -> None:
        self.elements = elements
        self.requires_grad = requires_grad
        self.shape = shape or (elements,)
        self.hooks: list = []

    def numel(self) -> int:
        return self.elements

    def requires_grad_(self, value: bool) -> FakeParameter:
        self.requires_grad = value
        return self

    def register_hook(self, hook):
        self.hooks.append(hook)
        return SimpleNamespace(remove=lambda: None)


class FakeBatchNorm:
    def __init__(self, width: int) -> None:
        self.training = True
        self.weight = FakeParameter(width)
        self.bias = FakeParameter(width)

    def eval(self) -> FakeBatchNorm:
        self.training = False
        return self

    def train(self) -> FakeBatchNorm:
        self.training = True
        return self


class FakeDetector:
    def __init__(self, *, unsafe_auxiliary: bool = False) -> None:
        self.backbone_bn = FakeBatchNorm(8)
        self.head_bn = FakeBatchNorm(4)
        self.parameters = [
            ("model.0.conv.weight", FakeParameter(24)),
            ("model.0.bn.weight", self.backbone_bn.weight),
            ("model.0.bn.bias", self.backbone_bn.bias),
            ("model.22.cv2.conv.weight", FakeParameter(32)),
            ("model.23.cv2.0.conv.weight", FakeParameter(48)),
            ("model.23.cv2.0.bn.weight", self.head_bn.weight),
            ("model.23.cv2.0.bn.bias", self.head_bn.bias),
            ("model.23.cv3.0.0.conv.weight", FakeParameter(40)),
            ("model.23.cv3.0.2.weight", FakeParameter(36, shape=(9, 4, 1, 1))),
            ("model.23.cv3.0.2.bias", FakeParameter(9, shape=(9,))),
            ("model.23.cv3.1.2.weight", FakeParameter(36, shape=(9, 4, 1, 1))),
            ("model.23.cv3.1.2.bias", FakeParameter(9, shape=(9,))),
            ("model.23.cv3.2.2.weight", FakeParameter(36, shape=(9, 4, 1, 1))),
            ("model.23.cv3.2.2.bias", FakeParameter(9, shape=(9,))),
        ]
        if unsafe_auxiliary:
            self.parameters.append(("auxiliary.weight", FakeParameter(2)))

    def named_parameters(self):
        return iter(self.parameters)

    def named_modules(self):
        return iter(
            [
                ("", self),
                ("model.0.bn", self.backbone_bn),
                ("model.23.cv2.0.bn", self.head_bn),
            ]
        )


class FakeYOLO:
    def __init__(self) -> None:
        self.callbacks: dict[str, list] = {}

    def add_callback(self, event: str, callback) -> None:
        self.callbacks.setdefault(event, []).append(callback)


class FakeStateTensor:
    def __init__(self, values: tuple[int, ...], dtype: str = "float32") -> None:
        self.values = values
        self.shape = (len(values),)
        self.dtype = dtype

    def detach(self):
        return self

    def cpu(self):
        return self

    def numel(self) -> int:
        return len(self.values)

    def equal(self, other) -> bool:
        return self.values == other.values and self.dtype == other.dtype


class FakeGradient:
    def __init__(self, values: list[int]) -> None:
        self.values = list(values)

    def clone(self) -> FakeGradient:
        return FakeGradient(self.values)

    def __setitem__(self, indices, value: int) -> None:
        for index in indices if isinstance(indices, list) else [indices]:
            self.values[index] = value


class FakeStateModel:
    def __init__(
        self,
        values: tuple[int, ...],
        *,
        classifier: tuple[int, ...] = (4, 5),
        classifier_buffer: tuple[int, ...] = (7, 8),
    ) -> None:
        self.values = values
        self.classifier = classifier
        self.classifier_buffer = classifier_buffer

    def state_dict(self):
        return {
            "model.0.weight": FakeStateTensor(self.values),
            "model.23.cv3.0.2.weight": FakeStateTensor(self.classifier),
            "model.23.cv3.0.bn.running_mean": FakeStateTensor(
                self.classifier_buffer
            ),
        }


def is_fake_batchnorm(module) -> bool:
    return isinstance(module, FakeBatchNorm)


class SafeFineTuneApplicationTests(unittest.TestCase):
    def test_seed_state_parity_fails_closed_before_first_step(self) -> None:
        report = validate_seed_state_parity(
            FakeStateModel((1, 2, 3)), FakeStateModel((1, 2, 3))
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["compared_elements"], 7)

        with self.assertRaisesRegex(RuntimeError, "tensors differ"):
            validate_seed_state_parity(
                FakeStateModel((1, 2, 3)), FakeStateModel((1, 2, 4))
            )

    def test_post_epoch_parity_allows_only_classifier_changes(self) -> None:
        report = validate_frozen_state_parity(
            FakeStateModel((1, 2, 3), classifier=(4, 5)),
            FakeStateModel((1, 2, 3), classifier=(4, 6)),
            trainable_parameter_names=("model.23.cv3.0.2.weight",),
        )
        self.assertTrue(report["passed"])
        self.assertEqual(
            report["changed_trainable_tensors"], ["model.23.cv3.0.2.weight"]
        )
        self.assertTrue(report["trainable_change_observed"])
        no_op = validate_frozen_state_parity(
            FakeStateModel((1, 2, 3)),
            FakeStateModel((1, 2, 3)),
            trainable_parameter_names=("model.23.cv3.0.2.weight",),
        )
        self.assertTrue(no_op["passed"])
        self.assertFalse(no_op["trainable_change_observed"])
        with self.assertRaisesRegex(RuntimeError, "frozen tensors changed"):
            validate_frozen_state_parity(
                FakeStateModel((1, 2, 3)),
                FakeStateModel((1, 2, 4)),
                trainable_parameter_names=("model.23.cv3.0.2.weight",),
            )
        with self.assertRaisesRegex(RuntimeError, "frozen tensors changed"):
            validate_frozen_state_parity(
                FakeStateModel((1, 2, 3), classifier_buffer=(7, 8)),
                FakeStateModel((1, 2, 3), classifier_buffer=(7, 9)),
                trainable_parameter_names=("model.23.cv3.0.2.weight",),
            )

    def test_trains_only_classifier_and_freezes_regression_and_batchnorm(self) -> None:
        model = FakeDetector()
        report = apply_safe_finetune_controls(
            model,
            event="fixture",
            batchnorm_predicate=is_fake_batchnorm,
        )
        parameters = dict(model.named_parameters())

        self.assertFalse(parameters["model.0.conv.weight"].requires_grad)
        self.assertFalse(parameters["model.22.cv2.conv.weight"].requires_grad)
        self.assertFalse(parameters["model.23.cv2.0.conv.weight"].requires_grad)
        self.assertFalse(parameters["model.23.cv3.0.0.conv.weight"].requires_grad)
        self.assertTrue(parameters["model.23.cv3.0.2.weight"].requires_grad)
        self.assertFalse(parameters["model.23.cv2.0.bn.weight"].requires_grad)
        self.assertFalse(parameters["model.23.cv2.0.bn.bias"].requires_grad)
        self.assertFalse(model.backbone_bn.training)
        self.assertFalse(model.head_bn.training)

        self.assertTrue(report["passed"])
        self.assertEqual(report["event"], "fixture")
        self.assertEqual(report["counts"]["batchnorm_modules"], 2)
        self.assertEqual(report["counts"]["batchnorm_affine_tensors"], 4)
        self.assertEqual(report["counts"]["frozen_backbone_tensors"], 4)
        self.assertEqual(report["counts"]["trainable_tensors_after"], 6)
        self.assertEqual(report["counts"]["trainable_elements_after"], 135)
        self.assertEqual(report["counts"]["frozen_head_tensors"], 4)
        self.assertEqual(report["counts"]["unexpected_trainable_tensors"], 0)
        self.assertEqual(report["violations"], [])

    def test_fail_closed_on_trainable_parameter_outside_head(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            "parameters outside the approved final class-output convolutions",
        ):
            apply_safe_finetune_controls(
                FakeDetector(unsafe_auxiliary=True),
                batchnorm_predicate=is_fake_batchnorm,
            )

    def test_non_strict_report_exposes_layout_violations(self) -> None:
        model = FakeDetector(unsafe_auxiliary=True)
        report = apply_safe_finetune_controls(
            model,
            batchnorm_predicate=is_fake_batchnorm,
            strict=False,
        )
        self.assertFalse(report["passed"])
        self.assertEqual(
            report["unexpected_trainable_parameters"], ["auxiliary.weight"]
        )


class SafeFineTuneCallbackTests(unittest.TestCase):
    def test_override_freezes_exactly_layers_zero_through_twenty_two(self) -> None:
        self.assertEqual(FROZEN_LAYER_INDICES, tuple(range(23)))
        self.assertEqual(HEAD_LAYER_INDEX, 23)
        self.assertEqual(safe_finetune_training_overrides(), {"freeze": 23})

    def test_callbacks_reapply_batchnorm_after_model_train(self) -> None:
        yolo = FakeYOLO()
        reports: list[dict] = []
        controller = wire_safe_finetune_callbacks(
            yolo,
            batchnorm_predicate=is_fake_batchnorm,
            report_sink=reports.append,
        )
        model = FakeDetector()
        trainer = SimpleNamespace(model=model)

        self.assertEqual(set(yolo.callbacks), set(CALLBACK_EVENTS))
        yolo.callbacks["on_pretrain_routine_end"][0](trainer)
        pretrain_mask_audit = controller.last_report[
            "protected_gradient_mask_audit"
        ]
        self.assertTrue(pretrain_mask_audit["passed"])
        self.assertEqual(pretrain_mask_audit["protected_classes"], ["ADDRESS"])
        self.assertEqual(pretrain_mask_audit["protected_rows"], [3])
        self.assertEqual(pretrain_mask_audit["hook_handles"], 6)
        output_weight = dict(model.named_parameters())[
            "model.23.cv3.0.2.weight"
        ]
        masked = output_weight.hooks[0](FakeGradient(list(range(9))))
        self.assertEqual(masked.values, [0, 1, 2, 0, 4, 5, 6, 7, 8])
        model.backbone_bn.train()
        model.head_bn.train()
        model.head_bn.weight.requires_grad_(True)
        dict(model.named_parameters())["model.22.cv2.conv.weight"].requires_grad_(True)
        self.assertTrue(model.head_bn.training)

        yolo.callbacks["on_train_batch_start"][0](trainer)
        self.assertFalse(model.backbone_bn.training)
        self.assertFalse(model.head_bn.training)
        self.assertFalse(model.head_bn.weight.requires_grad)
        self.assertFalse(
            dict(model.named_parameters())["model.22.cv2.conv.weight"].requires_grad
        )
        self.assertEqual(
            controller.event_counts,
            {"on_pretrain_routine_end": 1, "on_train_batch_start": 1},
        )
        self.assertEqual(len(reports), 2)
        self.assertEqual(
            controller.last_report["protected_gradient_mask_audit"][
                "hook_handles"
            ],
            6,
        )
        self.assertEqual(len(output_weight.hooks), 1)
        self.assertIs(trainer.safe_finetune_audit, controller.last_report)
        self.assertEqual(controller.last_report["event"], "on_train_batch_start")

    def test_controller_cannot_be_wired_twice(self) -> None:
        controller = SafeFineTuneController(batchnorm_predicate=is_fake_batchnorm)
        controller.wire(FakeYOLO())
        with self.assertRaisesRegex(RuntimeError, "already wired"):
            controller.wire(FakeYOLO())


if __name__ == "__main__":
    unittest.main()
