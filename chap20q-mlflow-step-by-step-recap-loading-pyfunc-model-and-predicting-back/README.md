# chap20q - Step-by-step recap: load a pyfunc model back and re-predict

Lesson: [`../20q-practical-work-15q-mlflow-step-by-step-recap-loading-pyfunc-model-and-predicting-back.md`](../20q-practical-work-15q-mlflow-step-by-step-recap-loading-pyfunc-model-and-predicting-back.md).

## What's new vs chap20p

- Capture `model_info = mlflow.pyfunc.log_model(...)` and use `model_info.model_uri`
  (no more hardcoded `runs:/<id>/...`)
- `loaded = mlflow.pyfunc.load_model(model_info.model_uri)`
- Re-predict on `test_x`, recompute RMSE/MAE/R2, log them as `loaded_test_*`
- `assert np.allclose(...)` sanity check that round-trip is lossless

## Quick run

```bash
docker compose up -d --build mlflow
docker compose run --rm trainer --alpha 0.4 --l1_ratio 0.4
```

Expected stdout end:
```
Before log -> RMSE=0.7785  MAE=0.6223  R2=0.1054
Logged model URI: runs:/<id>/sklearn_mlflow_pyfunc
After load  -> RMSE=0.7785  MAE=0.6223  R2=0.1054
Sanity check OK
```

Open http://localhost:5000 -> Metrics tab shows BOTH `rmse` and `loaded_test_rmse` (identical values).

## Tear down

```bash
docker compose down
docker compose down -v
```
