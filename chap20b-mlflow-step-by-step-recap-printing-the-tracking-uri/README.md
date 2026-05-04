# chap20b - Step-by-step recap: print the tracking URI

The full lesson lives at [`../20b-practical-work-15b-mlflow-step-by-step-recap-printing-the-tracking-uri.md`](../20b-practical-work-15b-mlflow-step-by-step-recap-printing-the-tracking-uri.md).

## What's new vs chap20a

A single line:

```python
print("Tracking URI:", mlflow.get_tracking_uri())
```

## Quick run

```bash
docker compose up --build
python hello_mlflow.py    # from your host venv
```

Expected output:

```
Tracking URI: http://localhost:5000
Done. Open http://localhost:5000 to see your run.
```

## Tear down

```bash
docker compose down       # keep runs
docker compose down -v    # wipe everything
```
