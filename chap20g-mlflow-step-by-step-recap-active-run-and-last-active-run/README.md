# chap20g - Step-by-step recap: active_run + last_active_run

The full lesson lives at [`../20g-practical-work-15g-mlflow-step-by-step-recap-active-run-and-last-active-run-with-start-end-run.md`](../20g-practical-work-15g-mlflow-step-by-step-recap-active-run-and-last-active-run-with-start-end-run.md).

## What's new vs chap20f

- Imperative style: `mlflow.start_run()` ... `mlflow.end_run()` (no `with`)
- `mlflow.active_run()` to grab the current run while it's open
- `mlflow.last_active_run()` to grab the last run AFTER `end_run()`
- Print of all experiment metadata fields

## Quick run

```bash
docker compose up -d --build mlflow
docker compose run --rm trainer --alpha 0.4 --l1_ratio 0.4
```

Open http://localhost:5000 -> experiment `experiment_2`.

## active_run vs last_active_run

| Function | While run is open | After `end_run()` |
|---|---|---|
| `mlflow.active_run()` | The current Run | `None` |
| `mlflow.last_active_run()` | The current Run | The last one (status FINISHED) |

## Tear down

```bash
docker compose down       # keep volumes
docker compose down -v    # wipe everything
```
