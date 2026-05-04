# chap20r - Step-by-step recap: mlflow.evaluate default regressor

Lesson: [`../20r-practical-work-15r-mlflow-step-by-step-recap-mlflow-evaluate-default-regressor.md`](../20r-practical-work-15r-mlflow-step-by-step-recap-mlflow-evaluate-default-regressor.md).

## What's new vs chap20q

- Replace the manual re-predict + log_metrics block with one call:
  ```python
  result = mlflow.evaluate(
      model_info.model_uri,
      test,                      # full DataFrame (features + label)
      targets="quality",
      model_type="regressor",
      evaluators=["default"],
  )
  ```
- 9+ regression metrics auto-logged: rmse, mse, mae, mape, r2, max_error, sum_on_target, mean_on_target, example_count
- Auto-generated artifact: `eval_results_table.json` (row-level predictions vs targets)
- Returns an `EvaluationResult` object: `result.metrics`, `result.artifacts`, `result.tables`

## Quick run

```bash
docker compose up -d --build mlflow
docker compose run --rm trainer --alpha 0.4 --l1_ratio 0.4
```

Open http://localhost:5000 -> experiment `experiment_custom_sklearn`:
- Metrics tab: 9+ metrics from evaluate + your 3 manual ones
- Artifacts tab: `eval_results_table.json` (row-by-row analysis)

## Use case: CI/CD promotion gate

```python
result = mlflow.evaluate(model_uri, test, targets="quality", model_type="regressor")
assert result.metrics["root_mean_squared_error"] < 0.85, "Model regressed!"
```

## Tear down

```bash
docker compose down
docker compose down -v
```
