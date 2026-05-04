# chap20s - Step-by-step recap: mlflow.evaluate with custom metrics and scatter artifact

Lesson: [`../20s-practical-work-15s-mlflow-step-by-step-recap-mlflow-evaluate-custom-metrics-and-scatter-artifact.md`](../20s-practical-work-15s-mlflow-step-by-step-recap-mlflow-evaluate-custom-metrics-and-scatter-artifact.md).

## What's new vs chap20r

- 2 custom metrics built with `make_metric`:
  - `squared_diff_plus_one` -> `sum((pred - target + 1)**2)` (lower is better)
  - `sum_on_target_divided_by_two` -> derived from `builtin_metrics["sum_on_target"]` (higher is better)
- 1 custom artifact: `prediction_target_scatter` -> matplotlib PNG (predictions vs targets with `y=x` line)
- Wired in via `extra_metrics=[...]` and `custom_artifacts=[...]` to `mlflow.evaluate`
- New experiment name: `experiment_model_evaluation`
- New dependency: `matplotlib==3.9.2` (uses headless `Agg` backend)

## Note on the deprecated `custom_metrics=` kwarg

The original snippet uses `custom_metrics=`. We use **`extra_metrics=`** instead -- since MLflow 2.4 that's the recommended name (`custom_metrics=` still works but emits a `DeprecationWarning`).

## Quick run

```bash
docker compose up -d --build mlflow
docker compose run --rm trainer --alpha 0.7 --l1_ratio 0.7
```

Open http://localhost:5000 -> experiment `experiment_model_evaluation`:
- Metrics tab: built-in regression metrics + `squared_diff_plus_one` + `sum_on_target_divided_by_two`
- Artifacts tab: `eval_results_table.json` + **`example_scatter_plot.png`** (click to preview)

## Tear down

```bash
docker compose down
docker compose down -v
```
