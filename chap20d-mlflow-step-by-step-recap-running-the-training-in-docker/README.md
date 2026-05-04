# chap20d - Step-by-step recap: running the training in a second Docker service

The full lesson lives at [`../20d-practical-work-15d-mlflow-step-by-step-recap-running-the-training-in-a-second-docker-service-with-requirements-txt.md`](../20d-practical-work-15d-mlflow-step-by-step-recap-running-the-training-in-a-second-docker-service-with-requirements-txt.md).

## What's new vs chap20c

- 2 services in `docker-compose.yml`: `mlflow` (server) + `trainer` (one-shot)
- `trainer/Dockerfile` with `requirements.txt` + `ENTRYPOINT ["python", "train.py"]`
- CLI args forwarded directly: `docker compose run --rm trainer --alpha 0.1 --l1_ratio 0.1`

## Quick run

```bash
docker compose up -d --build mlflow

docker compose run --rm trainer --alpha 0.1 --l1_ratio 0.1
docker compose run --rm trainer --alpha 0.5 --l1_ratio 0.5
docker compose run --rm trainer --alpha 0.9 --l1_ratio 0.1
```

Then open http://localhost:5000

## Why is the UI empty?

Because `train.py` does NOT call `set_tracking_uri(...)`. The runs are written to `file:///code/mlruns` inside the trainer container - which is destroyed by `--rm`.

The fix is the topic of [chap 20e](../20e-practical-work-15e-mlflow-step-by-step-recap-passing-the-tracking-uri-via-environment-variable.md).

## Tear down

```bash
docker compose down       # keep volumes
docker compose down -v    # wipe everything
```
