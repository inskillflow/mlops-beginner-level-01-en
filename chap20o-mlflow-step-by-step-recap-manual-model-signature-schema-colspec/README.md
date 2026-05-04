# chap20o - Step-by-step recap: manual model signature (Schema + ColSpec + ModelSignature)

Lesson: [`../20o-practical-work-15o-mlflow-step-by-step-recap-manual-model-signature-schema-colspec.md`](../20o-practical-work-15o-mlflow-step-by-step-recap-manual-model-signature-schema-colspec.md).

## What's new vs chap20n

- Replace `infer_signature(test_x, preds)` with a full manual block:
  - `ColSpec("double", "column_name")` for each of the 11 feature columns
  - `Schema([ColSpec, ...])` for inputs and outputs
  - `ModelSignature(inputs=..., outputs=...)` for the complete contract
- Replace `test_x.head(5)` with a hardcoded **dict of arrays** as `input_example`
- Output type is `"long"` (not `"double"`) — a deliberate choice to declare integer quality scores

## Quick run

```bash
docker compose up -d --build mlflow
docker compose run --rm trainer --alpha 0.7 --l1_ratio 0.7
```

Open http://localhost:5000 -> experiment `experiment_signature`:
- model/signature.json: 11 input columns (double) + 1 output (long)
- model/input_example.json: 5 hardcoded representative rows

## Key difference vs infer_signature (chap 20n)

| | infer_signature (20n) | Manual (20o) |
|---|---|---|
| Output type | `double` (from numpy) | `long` (your choice) |
| Needs data | Yes | No |
| Lines of code | 1 | ~15 |

## Tear down

```bash
docker compose down
docker compose down -v
```
