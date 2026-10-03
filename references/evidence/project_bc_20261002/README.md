# Frozen B/C reference evidence

From the repository root, verify the canonical public package directly:

```powershell
python references/evidence/project_bc_20261002/scripts/verify_reference_run.py --verify-package references/evidence/project_bc_20261002
```

Run from this package directory with Python 3.9 or later (standard library only):

```powershell
python scripts/verify_reference_run.py --verify-package .
```

This verifies all raw-question scores, frozen identities, stage/source hashes and recomputes the report. The blind worksheet is blank and has not been reviewed; the identity mapping is sealed separately.

Python versions can differ in the last bits of floating-point summation. Only named derived cost/time/ratio floats use finite absolute and relative tolerances of 1e-12. All original artifact SHA-256 hashes, identities, booleans, integer counts, scores and configuration remain exact. Actual AI-assisted subset judgments and their crosscheck are preserved in run/ai_assisted_review.json and run/ai_review_crosscheck.json; human review remains not_reviewed.

Model and adapter weights are excluded. `omitted_weights.json` records observed adapter file hashes, which are historical receipts, not an independent verification of omitted bytes. To repeat training, install the recorded PyTorch/Transformers runtime, run `scripts/prepare_reference_model.py`, then `python post_train/experiments/project_bc_reference.py --model-dir study_runs/models/Qwen2.5-0.5B-Instruct --config run/config.resolved.json --device cuda --output-dir study_runs/repeated_bc_reference`. The preparation script downloads the exact revision and verifies all model file hashes in `post_train/experiments/reference_model.json`. No original ignored run directory is needed for rescoring.
