# chap20a - Step-by-step recap: Hello MLflow basics

The full lesson lives at [`../20a-practical-work-15a-mlflow-step-by-step-recap-hello-mlflow-basics.md`](../20a-practical-work-15a-mlflow-step-by-step-recap-hello-mlflow-basics.md).

## Quick run

```bash
docker compose up --build
```

Open http://localhost:5000

## Log a first run from the host

```bash
# host venv with mlflow==2.16.2
python hello_mlflow.py
```

Refresh the UI -> experiment `hello_mlflow` appears with one run.

## Tear down

```bash
docker compose down       # keep runs
docker compose down -v    # wipe everything
```
