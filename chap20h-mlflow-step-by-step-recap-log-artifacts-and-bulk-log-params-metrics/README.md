# chap20h - Step-by-step recap: log_artifacts + bulk log_params/log_metrics

Lesson: [`../20h-practical-work-15h-mlflow-step-by-step-recap-log-artifacts-with-log-params-and-log-metrics-bulk-versions.md`](../20h-practical-work-15h-mlflow-step-by-step-recap-log-artifacts-with-log-params-and-log-metrics-bulk-versions.md).

## What's new vs chap20g

- `mlflow.log_artifacts("data/")` -> log the whole folder
- `mlflow.log_params({...})` and `mlflow.log_metrics({...})` -> bulk versions
- `mlflow.get_artifact_uri()` -> print where files are stored
- Save `train.csv` and `test.csv` to `data/` so we have something to log

## Quick run

```bash
docker compose up -d --build mlflow
docker compose run --rm trainer --alpha 0.5 --l1_ratio 0.3
ls data/   # red-wine-quality.csv + train.csv + test.csv
```

Open http://localhost:5000 -> experiment `experiment_4` -> Artifacts tab.

## Tear down

```bash
docker compose down
docker compose down -v
```
