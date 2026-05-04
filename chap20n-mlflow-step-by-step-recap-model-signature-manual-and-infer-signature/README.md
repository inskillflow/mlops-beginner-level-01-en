# chap20n - Step-by-step recap: model signature (manual + infer_signature)

Lesson: [`../20n-practical-work-15n-mlflow-step-by-step-recap-model-signature-manual-and-infer-signature.md`](../20n-practical-work-15n-mlflow-step-by-step-recap-model-signature-manual-and-infer-signature.md).

## What's new vs chap20m

- `mlflow.sklearn.autolog(log_models=False, log_model_signatures=False, log_input_examples=False)` to take control of model logging
- `infer_signature(test_x, predicted_qualities)` -> automatic signature in one line
- `mlflow.sklearn.log_model(lr, "elasticnet_model", signature=signature, input_example=test_x.head(5))`
- Artifacts `elasticnet_model/signature.json` and `elasticnet_model/input_example.json` appear in the UI

## Quick run

```bash
docker compose up -d --build mlflow
docker compose run --rm trainer --alpha 0.7 --l1_ratio 0.7
```

Open http://localhost:5000 -> experiment `experiment_signature` -> Artifacts tab -> expand `elasticnet_model/`:
- `signature.json` contains input columns + types
- `input_example.json` contains 5 real rows from test_x

## Approach A (manual) vs Approach B (infer_signature)

| | Approach A (manual) | Approach B (infer) |
|---|---|---|
| Code | `ModelSignature` + `Schema` + `ColSpec` | `infer_signature(test_x, preds)` |
| Maintenance | Manual | Automatic |
| Data needed | No | Yes |

See the lesson appendix for the complete Approach A code block.

## Tear down

```bash
docker compose down
docker compose down -v
```
