# chap20e - Step-by-step recap: passing the tracking URI via an env var

The full lesson lives at [`../20e-practical-work-15e-mlflow-step-by-step-recap-passing-the-tracking-uri-via-environment-variable.md`](../20e-practical-work-15e-mlflow-step-by-step-recap-passing-the-tracking-uri-via-environment-variable.md).

## What's new vs chap20d

- `train.py`: `mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))`
- `docker-compose.yml`: `environment: MLFLOW_TRACKING_URI: "http://mlflow:5000"` on the trainer

## Quick run

```bash
docker compose up -d --build mlflow

docker compose run --rm trainer --alpha 0.1 --l1_ratio 0.1
docker compose run --rm trainer --alpha 0.5 --l1_ratio 0.5
docker compose run --rm trainer --alpha 0.9 --l1_ratio 0.1
```

Open http://localhost:5000 -> experiment `experiment_1` -> the 3 runs are now actually visible.

## Override the URI at runtime

```bash
docker compose run --rm \
  -e MLFLOW_TRACKING_URI=http://my-other-vm:5000 \
  trainer --alpha 0.3 --l1_ratio 0.7
```

## Tear down

```bash
docker compose down       # keep volumes
docker compose down -v    # wipe everything
```
