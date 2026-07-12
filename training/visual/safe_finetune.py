"""Fail-closed controls for a conservative Ultralytics detector canary.

The pinned detector uses layers ``model.0`` through ``model.22`` as its
backbone/neck and ``model.23`` as its detection head. Only the three final
``model.23.cv3.<scale>.2`` class-output convolutions are allowed to train. The
shared classification features, box regression, DFL, backbone/neck, and every
BatchNorm tensor stay frozen. Ultralytics calls ``model.train()`` at the start
of every epoch, which puts BatchNorm modules back into training mode. The
callback installed here therefore reapplies the complete policy at the start
of every training batch, after that call.

Typical use::

    controls = wire_safe_finetune_callbacks(yolo)
    yolo.train(..., **controls.training_overrides())

``training_overrides`` makes Ultralytics freeze layers before it constructs the
optimizer.  The callbacks independently enforce and audit the same boundary,
and also freeze every BatchNorm affine parameter in the head.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from training.visual.prepare_webpii import DETECTOR_CLASSES
from training.visual.project_classifier_checkpoint import CLASSIFIER_TENSOR_NAMES


HEAD_LAYER_INDEX = 23
FROZEN_LAYER_INDICES = tuple(range(HEAD_LAYER_INDEX))
TRAINABLE_PARAMETER_NAMES = CLASSIFIER_TENSOR_NAMES
PROTECTED_CLASSIFIER_CLASSES = ("ADDRESS",)
PROTECTED_CLASSIFIER_ROWS = tuple(
    DETECTOR_CLASSES.index(name) for name in PROTECTED_CLASSIFIER_CLASSES
)
CALLBACK_EVENTS = (
    "on_pretrain_routine_end",
    "on_train_batch_start",
)
AUDIT_SCHEMA_VERSION = 1

_LAYER_PARAMETER = re.compile(r"^model\.(\d+)(?:\.|$)")
_TRAINABLE_PARAMETERS = frozenset(TRAINABLE_PARAMETER_NAMES)
BatchNormPredicate = Callable[[Any], bool]
ReportSink = Callable[[dict[str, Any]], None]


def validate_seed_state_parity(
    reference_model: Any,
    candidate_model: Any,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Prove the trainer starts from the immutable seed before its first step."""

    reference = _unwrap_model(reference_model)
    candidate = _unwrap_model(candidate_model)
    if not callable(getattr(reference, "state_dict", None)) or not callable(
        getattr(candidate, "state_dict", None)
    ):
        raise TypeError("seed parity models must expose state_dict()")
    reference_state = reference.state_dict()
    candidate_state = candidate.state_dict()
    reference_keys = list(reference_state)
    candidate_keys = list(candidate_state)
    missing = sorted(set(reference_keys) - set(candidate_keys))
    unexpected = sorted(set(candidate_keys) - set(reference_keys))
    mismatched: list[str] = []
    elements = 0
    for name in sorted(set(reference_keys) & set(candidate_keys)):
        expected = reference_state[name].detach().cpu()
        actual = candidate_state[name].detach().cpu()
        elements += int(expected.numel())
        same_shape = tuple(expected.shape) == tuple(actual.shape)
        same_dtype = str(expected.dtype) == str(actual.dtype)
        equal = bool(expected.equal(actual)) if same_shape and same_dtype else False
        if not equal:
            mismatched.append(name)
    violations: list[str] = []
    if missing:
        violations.append("trainer model is missing seed tensors")
    if unexpected:
        violations.append("trainer model has unexpected tensors")
    if mismatched:
        violations.append("trainer model tensors differ from the immutable seed")
    report = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "kind": "immutable-seed-pre-step-state-parity",
        "reference_tensors": len(reference_keys),
        "candidate_tensors": len(candidate_keys),
        "compared_elements": elements,
        "missing_tensors": missing,
        "unexpected_tensors": unexpected,
        "mismatched_tensors": mismatched,
        "violations": violations,
        "passed": not violations,
    }
    if strict and violations:
        raise RuntimeError("seed parity check failed: " + "; ".join(violations))
    return report


def validate_frozen_state_parity(
    reference_model: Any,
    candidate_model: Any,
    *,
    trainable_parameter_names: tuple[str, ...],
    strict: bool = True,
) -> dict[str, Any]:
    """Prove an epoch changed only the explicitly approved classifier tensors."""

    if not trainable_parameter_names or any(
        not name for name in trainable_parameter_names
    ):
        raise ValueError("frozen-state parity requires exact trainable names")
    approved_trainable = set(trainable_parameter_names)
    reference = _unwrap_model(reference_model)
    candidate = _unwrap_model(candidate_model)
    if not callable(getattr(reference, "state_dict", None)) or not callable(
        getattr(candidate, "state_dict", None)
    ):
        raise TypeError("frozen-state parity models must expose state_dict()")
    reference_state = reference.state_dict()
    candidate_state = candidate.state_dict()
    missing = sorted(set(reference_state) - set(candidate_state))
    unexpected = sorted(set(candidate_state) - set(reference_state))
    frozen_mismatches: list[str] = []
    trainable_mismatches: list[str] = []
    frozen_elements = 0
    for name in sorted(set(reference_state) & set(candidate_state)):
        expected = reference_state[name].detach().cpu()
        actual = candidate_state[name].detach().cpu()
        same_shape = tuple(expected.shape) == tuple(actual.shape)
        same_dtype = str(expected.dtype) == str(actual.dtype)
        equal = bool(expected.equal(actual)) if same_shape and same_dtype else False
        trainable = name in approved_trainable
        if trainable:
            if not equal:
                trainable_mismatches.append(name)
            continue
        frozen_elements += int(expected.numel())
        if not equal:
            frozen_mismatches.append(name)
    violations: list[str] = []
    if missing:
        violations.append("candidate is missing reference tensors")
    if unexpected:
        violations.append("candidate has unexpected tensors")
    if frozen_mismatches:
        violations.append("frozen tensors changed during class-output-only training")
    report = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "kind": "class-output-only-frozen-state-parity",
        "approved_trainable_parameters": sorted(approved_trainable),
        "reference_tensors": len(reference_state),
        "candidate_tensors": len(candidate_state),
        "compared_frozen_elements": frozen_elements,
        "changed_trainable_tensors": trainable_mismatches,
        "trainable_change_observed": bool(trainable_mismatches),
        "missing_tensors": missing,
        "unexpected_tensors": unexpected,
        "changed_frozen_tensors": frozen_mismatches,
        "violations": violations,
        "passed": not violations,
    }
    if strict and violations:
        raise RuntimeError("frozen-state parity failed: " + "; ".join(violations))
    return report


def _default_batchnorm_predicate() -> BatchNormPredicate:
    try:
        from torch.nn.modules.batchnorm import _BatchNorm
    except ImportError as error:  # pragma: no cover - exercised in the image
        raise RuntimeError(
            "safe fine-tune controls require PyTorch or an explicit batchnorm_predicate"
        ) from error
    return lambda module: isinstance(module, _BatchNorm)


def _unwrap_model(model: Any) -> Any:
    """Unwrap torch.compile/DDP wrappers without descending into YOLO.model."""

    seen: set[int] = set()
    current = model
    while id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "_orig_mod"):
            current = current._orig_mod
            continue
        if hasattr(current, "module"):
            current = current.module
            continue
        break
    return current


def _layer_index(parameter_name: str) -> int | None:
    match = _LAYER_PARAMETER.match(parameter_name)
    return int(match.group(1)) if match else None


def _is_trainable_parameter(parameter_name: str) -> bool:
    return parameter_name in _TRAINABLE_PARAMETERS


def _parameter_elements(parameter: Any) -> int:
    numel = getattr(parameter, "numel", None)
    if not callable(numel):
        raise TypeError("model parameters must expose numel()")
    return int(numel())


def _freeze(parameter: Any) -> None:
    requires_grad = getattr(parameter, "requires_grad_", None)
    if not callable(requires_grad):
        raise TypeError("model parameters must expose requires_grad_()")
    requires_grad(False)


def _protected_row_gradient_hook(gradient: Any) -> Any:
    """Zero ADDRESS gradients without mutating the incoming gradient tensor."""

    clone = getattr(gradient, "clone", None)
    if not callable(clone):
        raise TypeError("classifier gradients must support clone()")
    masked = clone()
    masked[list(PROTECTED_CLASSIFIER_ROWS)] = 0
    return masked


def safe_finetune_training_overrides() -> dict[str, int]:
    """Return the pre-optimizer Ultralytics override for head-only training."""

    # Ultralytics interprets an integer N as range(N), so 23 freezes 0..22.
    return {"freeze": HEAD_LAYER_INDEX}


def apply_safe_finetune_controls(
    model: Any,
    *,
    event: str = "manual",
    batchnorm_predicate: BatchNormPredicate | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Apply the final-class-output/BatchNorm policy and return a count audit.

    Args:
        model: An Ultralytics detection model, optionally DDP/compile wrapped.
        event: Callback or caller label recorded in the report.
        batchnorm_predicate: Test seam or custom BatchNorm classifier.  The
            default recognizes every PyTorch ``_BatchNorm`` subtype.
        strict: Raise if the model layout or postcondition is unsafe.

    Returns:
        A JSON-serializable report containing parameter/module counts and all
        postcondition violations.
    """

    root = _unwrap_model(model)
    if not callable(getattr(root, "named_parameters", None)):
        raise TypeError("model must expose named_parameters()")
    if not callable(getattr(root, "named_modules", None)):
        raise TypeError("model must expose named_modules()")

    predicate = batchnorm_predicate or _default_batchnorm_predicate()
    parameters = list(root.named_parameters())
    modules = list(root.named_modules())
    before_trainable = {
        id(parameter): bool(parameter.requires_grad) for _, parameter in parameters
    }

    layer_indices_present: set[int] = set()
    frozen_backbone_names: list[str] = []
    frozen_head_names: list[str] = []
    for name, parameter in parameters:
        layer_index = _layer_index(name)
        if layer_index is not None:
            layer_indices_present.add(layer_index)
        if layer_index in FROZEN_LAYER_INDICES:
            frozen_backbone_names.append(name)
            _freeze(parameter)
        elif layer_index == HEAD_LAYER_INDEX and not _is_trainable_parameter(name):
            frozen_head_names.append(name)
            _freeze(parameter)

    batchnorm_modules: list[tuple[str, Any]] = []
    batchnorm_parameter_ids: set[int] = set()
    batchnorm_training_before = 0
    for name, module in modules:
        if not predicate(module):
            continue
        batchnorm_modules.append((name, module))
        batchnorm_training_before += int(bool(getattr(module, "training", False)))
        module.eval()
        for attribute in ("weight", "bias"):
            parameter = getattr(module, attribute, None)
            if parameter is None:
                continue
            batchnorm_parameter_ids.add(id(parameter))
            _freeze(parameter)

    trainable_after = [
        (name, parameter)
        for name, parameter in parameters
        if bool(parameter.requires_grad)
    ]
    head_parameters = [
        (name, parameter)
        for name, parameter in parameters
        if _layer_index(name) == HEAD_LAYER_INDEX
    ]
    classification_parameters = [
        (name, parameter)
        for name, parameter in head_parameters
        if _is_trainable_parameter(name)
    ]
    classification_names = {name for name, _parameter in classification_parameters}
    missing_classification_parameters = sorted(
        set(TRAINABLE_PARAMETER_NAMES) - classification_names
    )
    trainable_after_names = {name for name, _parameter in trainable_after}
    disabled_classification_parameters = sorted(
        set(TRAINABLE_PARAMETER_NAMES) - trainable_after_names
    )
    unexpected_trainable = [
        name
        for name, parameter in trainable_after
        if not _is_trainable_parameter(name) or id(parameter) in batchnorm_parameter_ids
    ]
    trainable_batchnorm = [
        name
        for name, parameter in parameters
        if id(parameter) in batchnorm_parameter_ids and bool(parameter.requires_grad)
    ]
    training_batchnorm = [
        name
        for name, module in batchnorm_modules
        if bool(getattr(module, "training", False))
    ]

    violations: list[str] = []
    if not frozen_backbone_names:
        violations.append("no model.0-model.22 parameters were found")
    if not head_parameters:
        violations.append("no model.23 head parameters were found")
    if missing_classification_parameters:
        violations.append("final model.23.cv3 class-output inventory is incomplete")
    if disabled_classification_parameters:
        violations.append("final model.23.cv3 class-output parameters are not trainable")
    if not frozen_head_names:
        violations.append("no box-regression head parameters were frozen")
    if not batchnorm_modules:
        violations.append("no BatchNorm modules were found")
    if unexpected_trainable:
        violations.append(
            "parameters outside the approved final class-output convolutions "
            "remain trainable"
        )
    if trainable_batchnorm:
        violations.append("BatchNorm affine parameters remain trainable")
    if training_batchnorm:
        violations.append("BatchNorm modules remain in training mode")

    def tensor_count(items: list[tuple[str, Any]]) -> int:
        return len(items)

    def element_count(items: list[tuple[str, Any]]) -> int:
        return sum(_parameter_elements(parameter) for _, parameter in items)

    newly_frozen = [
        (name, parameter)
        for name, parameter in parameters
        if before_trainable[id(parameter)] and not bool(parameter.requires_grad)
    ]
    batchnorm_parameters = [
        (name, parameter)
        for name, parameter in parameters
        if id(parameter) in batchnorm_parameter_ids
    ]
    report: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "event": event,
        "policy": {
            "head_layer_index": HEAD_LAYER_INDEX,
            "frozen_layer_indices": list(FROZEN_LAYER_INDICES),
            "trainable_parameters": list(TRAINABLE_PARAMETER_NAMES),
            "protected_classifier_classes": list(PROTECTED_CLASSIFIER_CLASSES),
            "protected_classifier_rows": list(PROTECTED_CLASSIFIER_ROWS),
            "protected_gradient_policy": "zero-before-optimizer-step",
            "shared_classification_features": "frozen",
            "box_regression": "frozen",
            "batchnorm_running_stats": "frozen-via-eval",
            "batchnorm_affine_parameters": "frozen",
            "required_batch_callback": "on_train_batch_start",
        },
        "layer_indices_present": sorted(layer_indices_present),
        "counts": {
            "modules": len(modules),
            "parameter_tensors": tensor_count(parameters),
            "parameter_elements": element_count(parameters),
            "trainable_tensors_before": sum(before_trainable.values()),
            "trainable_tensors_after": tensor_count(trainable_after),
            "trainable_elements_after": element_count(trainable_after),
            "newly_frozen_tensors": tensor_count(newly_frozen),
            "newly_frozen_elements": element_count(newly_frozen),
            "frozen_backbone_tensors": len(frozen_backbone_names),
            "frozen_head_tensors": len(frozen_head_names),
            "frozen_head_elements": element_count(
                [
                    (name, parameter)
                    for name, parameter in parameters
                    if name in set(frozen_head_names)
                ]
            ),
            "head_parameter_tensors": tensor_count(head_parameters),
            "classification_parameter_tensors": tensor_count(
                classification_parameters
            ),
            "batchnorm_modules": len(batchnorm_modules),
            "batchnorm_modules_training_before": batchnorm_training_before,
            "batchnorm_modules_training_after": len(training_batchnorm),
            "batchnorm_affine_tensors": tensor_count(batchnorm_parameters),
            "trainable_batchnorm_affine_tensors": len(trainable_batchnorm),
            "unexpected_trainable_tensors": len(unexpected_trainable),
        },
        "unexpected_trainable_parameters": unexpected_trainable,
        "missing_trainable_parameters": missing_classification_parameters,
        "disabled_trainable_parameters": disabled_classification_parameters,
        "trainable_parameters": [name for name, _parameter in trainable_after],
        "trainable_batchnorm_parameters": trainable_batchnorm,
        "training_batchnorm_modules": training_batchnorm,
        "violations": violations,
        "passed": not violations,
    }
    if strict and violations:
        raise RuntimeError("unsafe fine-tune controls: " + "; ".join(violations))
    return report


class SafeFineTuneController:
    """Wire and account for the safe fine-tune callbacks on a YOLO object."""

    def __init__(
        self,
        *,
        batchnorm_predicate: BatchNormPredicate | None = None,
        report_sink: ReportSink | None = None,
        strict: bool = True,
    ) -> None:
        self.batchnorm_predicate = batchnorm_predicate
        self.report_sink = report_sink
        self.strict = strict
        self.last_report: dict[str, Any] | None = None
        self.event_counts = {event: 0 for event in CALLBACK_EVENTS}
        self._wired_object_id: int | None = None
        self._gradient_mask_parameter_ids: dict[str, int] = {}
        self._gradient_mask_handles: list[Any] = []

    def _ensure_protected_gradient_masks(self, model: Any) -> dict[str, Any]:
        root = _unwrap_model(model)
        parameters = dict(root.named_parameters())
        violations: list[str] = []
        registered_now: list[str] = []
        for name in TRAINABLE_PARAMETER_NAMES:
            parameter = parameters.get(name)
            if parameter is None:
                violations.append(f"protected-gradient parameter is missing: {name}")
                continue
            raw_shape = getattr(parameter, "shape", None)
            try:
                shape = tuple(int(value) for value in raw_shape)
            except (TypeError, ValueError):
                shape = ()
            if not shape or shape[0] != len(DETECTOR_CLASSES):
                violations.append(f"protected-gradient class rows changed: {name}")
                continue
            if not bool(getattr(parameter, "requires_grad", False)):
                violations.append(f"protected-gradient parameter is frozen: {name}")
                continue
            existing_id = self._gradient_mask_parameter_ids.get(name)
            if existing_id is not None:
                if existing_id != id(parameter):
                    violations.append(
                        f"protected-gradient parameter identity changed: {name}"
                    )
                continue
            register_hook = getattr(parameter, "register_hook", None)
            if not callable(register_hook):
                violations.append(f"protected-gradient hook is unavailable: {name}")
                continue
            handle = register_hook(_protected_row_gradient_hook)
            self._gradient_mask_parameter_ids[name] = id(parameter)
            self._gradient_mask_handles.append(handle)
            registered_now.append(name)
        missing_registered = sorted(
            set(TRAINABLE_PARAMETER_NAMES) - set(self._gradient_mask_parameter_ids)
        )
        if missing_registered:
            violations.append("protected-gradient hooks are incomplete")
        return {
            "schema_version": 1,
            "kind": "protected-classifier-row-gradient-mask",
            "protected_classes": list(PROTECTED_CLASSIFIER_CLASSES),
            "protected_rows": list(PROTECTED_CLASSIFIER_ROWS),
            "expected_parameter_names": list(TRAINABLE_PARAMETER_NAMES),
            "registered_parameter_names": sorted(
                self._gradient_mask_parameter_ids
            ),
            "registered_now": sorted(registered_now),
            "missing_registered_parameters": missing_registered,
            "hook_handles": len(self._gradient_mask_handles),
            "violations": violations,
            "passed": not violations,
        }

    def training_overrides(self) -> dict[str, int]:
        """Return overrides that freeze the backbone before optimizer build."""

        return safe_finetune_training_overrides()

    def apply(self, trainer: Any, event: str) -> dict[str, Any]:
        """Apply controls to ``trainer.model`` and record the latest audit."""

        if not hasattr(trainer, "model"):
            raise TypeError("Ultralytics callback object must expose trainer.model")
        report = apply_safe_finetune_controls(
            trainer.model,
            event=event,
            batchnorm_predicate=self.batchnorm_predicate,
            strict=self.strict,
        )
        gradient_mask_audit = self._ensure_protected_gradient_masks(trainer.model)
        report["protected_gradient_mask_audit"] = gradient_mask_audit
        if gradient_mask_audit["passed"] is not True:
            report["violations"].append(
                "protected classifier-row gradient masks were not proven"
            )
            report["passed"] = False
            if self.strict:
                raise RuntimeError(
                    "unsafe fine-tune controls: protected classifier-row "
                    "gradient masks were not proven"
                )
        self.last_report = report
        self.event_counts[event] = self.event_counts.get(event, 0) + 1
        # Expose the latest audit on the trainer for logging/persistence without
        # retaining one large report for every batch.
        trainer.safe_finetune_audit = report
        if self.report_sink is not None:
            self.report_sink(report)
        return report

    def wire(self, yolo: Any) -> SafeFineTuneController:
        """Attach callbacks to a YOLO wrapper and return this controller."""

        if not callable(getattr(yolo, "add_callback", None)):
            raise TypeError("YOLO object must expose add_callback(event, callback)")
        if self._wired_object_id is not None:
            raise RuntimeError("safe fine-tune callbacks are already wired")
        self._wired_object_id = id(yolo)
        for event in CALLBACK_EVENTS:
            yolo.add_callback(
                event,
                lambda trainer, event=event: self.apply(trainer, event),
            )
        return self


def wire_safe_finetune_callbacks(
    yolo: Any,
    *,
    batchnorm_predicate: BatchNormPredicate | None = None,
    report_sink: ReportSink | None = None,
    strict: bool = True,
) -> SafeFineTuneController:
    """Create and wire a :class:`SafeFineTuneController` to ``yolo``."""

    return SafeFineTuneController(
        batchnorm_predicate=batchnorm_predicate,
        report_sink=report_sink,
        strict=strict,
    ).wire(yolo)
