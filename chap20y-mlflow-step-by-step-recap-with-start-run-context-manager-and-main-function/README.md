# chap20y - Step-by-step recap: with mlflow.start_run(experiment_id=...) + main()

Lesson: [`../20y-practical-work-15y-mlflow-step-by-step-recap-with-start-run-context-manager-and-main-function.md`](../20y-practical-work-15y-mlflow-step-by-step-recap-with-start-run-context-manager-and-main-function.md).

## What's new (vs the imperative style of chap20a-20x)

- Wrap everything in a `def main(): ...` + `if __name__ == "__main__": main()`
- Use the **context manager** form:
  ```python
  with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
      ...
      mlflow.log_metrics(...)
      mlflow.sklearn.log_model(lr, "model")
  # run auto-closed here: FINISHED on success, FAILED on exception
  ```
- Pass `experiment_id` explicitly to `start_run` -- no ambiguity
- Use `mlflow.last_active_run()` both INSIDE (returns the active run) and AFTER the block (returns the just-finished run) -- handy for downstream logic
- Experiment with a space in its name: `"Project exp 1"` -- demonstrates MLflow handles it fine

## Why this skeleton matters

- **No zombie runs**: context manager ensures `mlflow.end_run()` is always called, even on exception
- **Importable**: `from train import main` is safe (nothing runs at import time)
- **Unit-testable**: tests can call `main()` with a mocked argparse

## Quick run

```bash
docker compose up -d --build mlflow
docker compose run --rm trainer --alpha 0.4 --l1_ratio 0.4
```

End of stdout:
```
After the block:
  run_id  : <id>
  status  : FINISHED
```

Open http://localhost:5000 -> `Project exp 1` -> the run with metrics + params + `model/` artifact.

## Tear down

```bash
docker compose down
docker compose down -v
```
