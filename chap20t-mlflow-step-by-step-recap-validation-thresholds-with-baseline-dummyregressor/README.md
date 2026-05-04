# chap20t - Step-by-step recap: validation thresholds + baseline DummyRegressor

Lesson: [`../20t-practical-work-15t-mlflow-step-by-step-recap-validation-thresholds-with-baseline-dummyregressor.md`](../20t-practical-work-15t-mlflow-step-by-step-recap-validation-thresholds-with-baseline-dummyregressor.md).

## What's new vs chap20s

- Train + log a baseline `DummyRegressor` (predicts the mean of the training target)
- `SklearnWrapper.__init__(self, artifact_name)` parameterised so the same class wraps both candidate and baseline
- Build `validation_thresholds = {"mean_squared_error": MetricThreshold(threshold=0.6, min_absolute_change=0.1, min_relative_change=0.05, greater_is_better=False)}`
- Pass `validation_thresholds=...` AND `baseline_model=baseline_uri` to `mlflow.evaluate`
- `mlflow.evaluate` raises `ModelValidationFailedException` when any threshold is violated -> CI/CD-friendly gate

## Quick run -- pass case

```bash
docker compose up -d --build mlflow
docker compose run --rm trainer --alpha 0.4 --l1_ratio 0.4
```

End of stdout:
```
VALIDATION PASSED. Candidate cleared all thresholds vs baseline.
```

## Force a failure

```bash
docker compose run --rm trainer --alpha 5.0 --l1_ratio 1.0
```

ElasticNet over-regularised barely beats the dummy -> exception:
```
VALIDATION FAILED:
Model validation failed for the following thresholds:
  Metric 'mean_squared_error': absolute change ... < required 0.1
```

## Tear down

```bash
docker compose down
docker compose down -v
```
