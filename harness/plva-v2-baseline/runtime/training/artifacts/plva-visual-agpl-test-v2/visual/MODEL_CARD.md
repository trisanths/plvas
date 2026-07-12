# PLVA fine-grained visual detector

Status: trained-unpublished; not approved for release.

This detector proposes class-aware sensitive regions for PLVA's local 
visual path. OCR plus semantic detection remains an independent safety path.

## Artifact

- ONNX: `detector.onnx` (10,428,657 bytes)
- SHA-256: `450ee07452d618eb3159d565f1787b17a8cabbfee3e72686e932460be576cc2e`
- Opset: 17
- Training epochs represented: 20

## Smoke metrics

- mAP50: 0.953329
- mAP50-95: 0.784975
- Precision: 0.958430
- Recall: 0.873571
- Minimum secret-class recall: 0.000000

These measurements validate the training/export path only; they are not 
release-quality evidence.

## License and release status

The source checkpoint is recorded as `AGPL-3.0-only`. AGPL 
development output cannot ship in a closed-source product without an 
Ultralytics Enterprise license. Cross-runtime parity, frozen holdout 
quality gates, dataset-license review, and WebGPU evidence also remain 
required.
