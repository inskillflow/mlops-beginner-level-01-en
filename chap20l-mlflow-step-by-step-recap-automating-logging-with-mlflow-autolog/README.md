# chap20l - Step-by-step recap: mlflow.autolog()

Lesson: [`../20l-practical-work-15l-mlflow-step-by-step-recap-automating-logging-with-mlflow-autolog.md`](../20l-practical-work-15l-mlflow-step-by-step-recap-automating-logging-with-mlflow-autolog.md).

## What's new vs chap20k

- One line `mlflow.autolog(log_input_examples=True)` -> MLflow auto-logs:
  - **Every** sklearn hyperparameter (12 for ElasticNet, even the ones we left default)
  - Training metrics (`training_score`, `training_mean_squared_error`, ...)
  - The fitted model with **signature** + **input example**
  - The `mlflow.autologging` system tag
- We keep manual calls only for what autolog can't do:
  - Test-set metrics (`test_rmse`, `test_mae`, `test_r2`)
  - Custom artifact (the input CSV)
  - Custom run tags

## Quick run

```bash
docker compose up -d --build mlflow
docker compose run --rm trainer --alpha 0.7 --l1_ratio 0.7
```

Open http://localhost:5000 -> experiment `experiment_autolog`:
- Parameters tab: 12 params (vs 2 manually)
- Metrics tab: training + test metrics
- Artifacts tab: `model/` (with `signature.json` + `input_example.json`) + `red-wine-quality.csv`

## Key gotchas

- Call `mlflow.autolog()` **before** `.fit()` (or it won't trigger).
- Autolog does **not** compute test/validation metrics. Do them manually.
- Autolog does **not** know about your CSVs. Use `log_artifact(s)` for them.

## Tear down

```bash
docker compose down
docker compose down -v
```
