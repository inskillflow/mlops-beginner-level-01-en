# chap20u - Step-by-step recap: registered_model_name (Model Registry)

Lesson: [`../20u-practical-work-15u-mlflow-step-by-step-recap-registered-model-name-with-mlflow-sklearn-log-model.md`](../20u-practical-work-15u-mlflow-step-by-step-recap-registered-model-name-with-mlflow-sklearn-log-model.md).

## What's new vs chap20t

- Trim the script (no pyfunc wrapper, no baseline, no thresholds, no autolog)
- Single registry trigger:
  ```python
  mlflow.sklearn.log_model(
      sk_model=lr,
      artifact_path="model",
      registered_model_name="elasticnet-api",   # <- THIS
  )
  ```
- First run -> creates `elasticnet-api` Version 1
- Each subsequent run -> bumps to Version 2, 3, ...
- Backend MUST be SQLite/PostgreSQL/MySQL (not file://) for the registry to work

## Quick run

```bash
docker compose up -d --build mlflow
docker compose run --rm trainer --alpha 0.6 --l1_ratio 0.6
docker compose run --rm trainer --alpha 0.4 --l1_ratio 0.2
```

Open http://localhost:5000:
- **Experiments tab** -> `experiment_register_model_api` -> 2 runs
- **Models tab** -> `elasticnet-api` -> Version 1 + Version 2

## Loading by name from any consumer

```python
mlflow.pyfunc.load_model("models:/elasticnet-api/2")
mlflow.pyfunc.load_model("models:/elasticnet-api/Production")
mlflow.pyfunc.load_model("models:/elasticnet-api@champion")
```

## Tear down

```bash
docker compose down
docker compose down -v
```
