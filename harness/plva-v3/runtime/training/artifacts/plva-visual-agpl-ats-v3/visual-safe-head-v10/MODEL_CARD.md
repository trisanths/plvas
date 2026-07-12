# PLVA fine-grained visual detector

Status: trained-unpublished; not approved for release.

This detector proposes class-aware sensitive regions for PLVA's local
visual path. OCR plus semantic detection remains an independent safety path.

## Artifact

- ONNX: `detector.onnx` (10,428,657 bytes)
- SHA-256: `2ed3f25c9bee375dc1683cf0ffa2044374b6bc35dd89a465a7dce6451ce8b928`
- Opset: 17
- Training epochs represented: 35
- Runtime thresholds: `safety-best-thresholds.json`
- Runtime profile SHA-256: `919da6455fc93ffb8ef929eeb5ded012b102c97a083b421cd0b5226b8be8b90e`
- Quantization: gated; FP32 remains the only exported artifact

## Operational selection metrics

- Mean privacy coverage: 0.975990
- Mean class-compatible coverage: 0.965281
- Minimum secret privacy coverage: 0.946821
- Metric basis: calibrated per-class thresholds with PLVA top-1, padding, and NMS

## Framework diagnostics (not checkpoint-selection metrics)

- mAP50: 0.000000
- mAP50-95: 0.000000
- Precision: 0.000000
- Recall: 0.000000

These measurements validate the training/export path only; they are not
release-quality evidence.

## License and release status

The source checkpoint is recorded as `AGPL-3.0-only`. AGPL
development output cannot ship in a closed-source product without an
Ultralytics Enterprise license. Cross-runtime parity, frozen holdout
quality gates, dataset-license review, and WebGPU evidence also remain
required.

Static Conv INT8 is not emitted automatically. A separate QDQ candidate
must pass size, frozen holdout recall, full-output numerical, and pinned
Node/WASM parity gates before it can be promoted.
