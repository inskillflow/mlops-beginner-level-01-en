# chap20m - Step-by-step recap: PostgreSQL backend store + S3 for production

Lesson: [`../20m-practical-work-15m-mlflow-step-by-step-recap-postgresql-backend-store-and-s3-artifacts.md`](../20m-practical-work-15m-mlflow-step-by-step-recap-postgresql-backend-store-and-s3-artifacts.md).

## What's new vs all previous chapters

- A `postgres:16-alpine` service added to the compose stack
- MLflow's `--backend-store-uri` now uses `postgresql://...` (not `sqlite:///`)
- `psycopg2-binary` added to the mlflow Dockerfile
- Artifacts still on a named Docker volume (same as before)

## Quick run

```bash
docker compose up -d --build
docker compose run --rm trainer --alpha 0.5 --l1_ratio 0.5
```

## Verify PostgreSQL is used

```bash
# List MLflow's tables
docker compose exec postgres psql -U mlflowuser -d mlflowdb -c "\dt"

# Query last 5 runs
docker compose exec postgres psql -U mlflowuser -d mlflowdb \
  -c "SELECT run_uuid, name, status FROM runs ORDER BY start_time DESC LIMIT 5;"
```

## Production: PostgreSQL + S3

See Section 8 of the lesson for the full compose snippet and the IAM policy.

| Config | Backend | Artifacts | Use for |
|---|---|---|---|
| SQLite + local dir | `sqlite:///...` | `./mlruns` | Dev (chapters 20a-20l) |
| **PostgreSQL + volume** | `postgresql://...` | named volume | **This chapter** |
| PostgreSQL + S3 | `postgresql://...` | `s3://bucket` | Production |

## Tear down

```bash
docker compose down          # keep postgres-data + mlflow-artifacts
docker compose down -v       # wipe EVERYTHING (all runs lost)
```
