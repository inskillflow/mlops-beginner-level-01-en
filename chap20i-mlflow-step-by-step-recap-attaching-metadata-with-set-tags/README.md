# chap20i - Step-by-step recap: attaching metadata with set_tags

Lesson: [`../20i-practical-work-15i-mlflow-step-by-step-recap-attaching-metadata-to-runs-with-set-tags.md`](../20i-practical-work-15i-mlflow-step-by-step-recap-attaching-metadata-to-runs-with-set-tags.md).

## What's new vs chap20h

- `mlflow.set_tags({...})` to attach searchable metadata to the run
- Print the run's user-defined tags from `last_active_run().data.tags`

## Quick run

```bash
docker compose up -d --build mlflow
docker compose run --rm trainer --alpha 0.5 --l1_ratio 0.3
```

Open http://localhost:5000 -> experiment `experiment_4` -> latest run -> Tags tab.

## Filter runs by tag (UI search bar)

```text
tags.`release.version` = "2.0"
tags.`release.candidate` LIKE "RC%"
tags.`engineering` = "ML platform" and metrics.rmse < 0.8
```

Backticks are required around dotted keys.

## Tear down

```bash
docker compose down
docker compose down -v
```
