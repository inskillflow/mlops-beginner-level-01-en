# chap20j - Step-by-step recap: multiple runs in one experiment

Lesson: [`../20j-practical-work-15j-mlflow-step-by-step-recap-multiple-runs-in-one-experiment.md`](../20j-practical-work-15j-mlflow-step-by-step-recap-multiple-runs-in-one-experiment.md).

## What's new vs chap20i

- Helper `train_one_run(name, alpha, l1_ratio, ...)` factorises the per-run logic
- A list of `CONFIGS` + a `for` loop -> 3 runs created in `experiment_5`
- Each run gets its own `run_name` (`run1.1`, `run2.1`, `run3.1`)

## Quick run

```bash
docker compose up -d --build mlflow
docker compose run --rm trainer --alpha 0.7 --l1_ratio 0.7
```

Open http://localhost:5000 -> experiment `experiment_5` -> tick the 3 runs -> click **Compare**.

## Try a bigger grid

Edit `trainer/train.py`, extend `CONFIGS`, then:

```bash
docker compose run --rm trainer
```

In the UI search bar:

```text
metrics.rmse < 0.75
```

## Tear down

```bash
docker compose down
docker compose down -v
```
