<a id="top"></a>

# Chapter 20m — Step-by-step recap: replacing SQLite with **PostgreSQL** (backend store) — and a look at S3 for production

## Table of Contents

| # | Section |
|---|---|
| 1 | [Objective](#section-1) |
| 2 | [What we change vs all previous chapters](#section-2) |
| 3 | [The three MLflow storage layers](#section-3) |
| 4 | [SQLite vs PostgreSQL — when to upgrade](#section-4) |
| 5 | [Project structure](#section-5) |
| 6 | [The code](#section-6) |
| 7 | [Run it, verify PostgreSQL is actually used](#section-7) |
| 8 | [Production variant: PostgreSQL + S3](#section-8) |
| 9 | [Tear down](#section-9) |
| 10 | [Recap and next chapter](#section-10) |

---

<a id="section-1"></a>

## 1. Objective

All previous chapters (20a → 20l) used **SQLite** as the MLflow backend store. That's fine for learning but SQLite breaks under concurrent writes and doesn't survive in multi-container or multi-user setups.

Today we:

1. Add a **`postgres`** service to the Docker Compose stack.
2. Tell the **`mlflow`** service to use `postgresql://...` as its `--backend-store-uri`.
3. Confirm the runs are actually stored in Postgres (not in a local file).
4. Explain how to swap the local artifact volume for **S3** when deploying to real infrastructure.

The `train.py` doesn't change at all — the switch is entirely in the compose file.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. What we change vs all previous chapters

| File | Change |
|---|---|
| `docker-compose.yml` | + new `postgres` service (image `postgres:16-alpine`). MLflow's `CMD` now points to `postgresql://...`. MLflow image also needs `psycopg2-binary`. |
| `mlflow/Dockerfile` | Add `psycopg2-binary==2.9.10` to the `pip install` line. |
| `trainer/train.py` | **Unchanged**. Still reads `MLFLOW_TRACKING_URI` from env. |

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. The three MLflow storage layers

MLflow stores information in three distinct places:

```
┌────────────────────────────────────────────────────────────────┐
│                        MLflow server                           │
│                                                                │
│  ┌──────────────────────┐    ┌──────────────────────────────┐ │
│  │  Backend store       │    │  Artifact store              │ │
│  │  (metadata)          │    │  (binary files)              │ │
│  │                      │    │                              │ │
│  │  • Experiment names  │    │  • model.pkl                 │ │
│  │  • Run IDs, names    │    │  • train.csv, test.csv       │ │
│  │  • Parameters        │    │  • plots (.png)              │ │
│  │  • Metrics           │    │  • signature.json            │ │
│  │  • Tags              │    │  • input_example.json        │ │
│  │                      │    │                              │ │
│  │  Dev:   SQLite       │    │  Dev:   named volume         │ │
│  │  Prod:  PostgreSQL   │    │  Prod:  S3 / GCS / ADLS      │ │
│  └──────────────────────┘    └──────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

They are **independent**: you can mix any backend store with any artifact store.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. SQLite vs PostgreSQL — when to upgrade

| Criterion | SQLite | PostgreSQL |
|---|---|---|
| Setup | Zero deps — just a file | Needs a DB server |
| Concurrent writes | Fails under load (file-level lock) | MVCC — handles many writers |
| Multi-container | Risky (shared file over volume) | First-class — TCP socket |
| Multi-user | Not designed for it | Yes |
| Migrations / upgrades | Manual | Handled by MLflow (`mlflow db upgrade`) |
| Production | **No** | **Yes** |
| Chapter 20a → 20l | ✓ (perfect for learning) | — |
| **This chapter** | — | ✓ |

> [!IMPORTANT]
> Run `mlflow db upgrade --url <backend-store-uri>` whenever you upgrade MLflow's version on an existing PostgreSQL database. It applies schema migrations automatically. Never skip this step — a version mismatch will crash the server at startup.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. Project structure

```text
chap20m-mlflow-step-by-step-recap-postgresql-backend-store-and-s3-artifacts/
├── README.md
├── docker-compose.yml          ← adds postgres service, changes mlflow CMD
├── data/
│   └── red-wine-quality.csv
├── mlflow/
│   └── Dockerfile              ← adds psycopg2-binary
└── trainer/
    ├── Dockerfile
    ├── requirements.txt
    └── train.py                ← unchanged from chap 20l
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-6"></a>

## 6. The code

### 6.1 `mlflow/Dockerfile`

```dockerfile
FROM python:3.12-slim
WORKDIR /mlflow
RUN pip install --no-cache-dir \
        mlflow==2.16.2 \
        psycopg2-binary==2.9.10
EXPOSE 5000
CMD ["mlflow", "server", \
     "--backend-store-uri", "postgresql://mlflowuser:mlflowpassword@postgres:5432/mlflowdb", \
     "--default-artifact-root", "/mlflow/mlruns", \
     "--host", "0.0.0.0", "--port", "5000"]
```

The only difference from chap 20a→20l Dockerfiles: `psycopg2-binary` is added and the `CMD` uses a `postgresql://` URI.

### 6.2 `docker-compose.yml`

```yaml
services:

  # ─── NEW: PostgreSQL backend store ────────────────────────────────────────
  postgres:
    image: postgres:16-alpine
    container_name: postgres-recap-20m
    environment:
      POSTGRES_DB:       mlflowdb
      POSTGRES_USER:     mlflowuser
      POSTGRES_PASSWORD: mlflowpassword
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - recap-net
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U mlflowuser -d mlflowdb"]
      interval: 5s
      timeout: 3s
      retries: 10

  # ─── MLflow server (now talking to Postgres) ──────────────────────────────
  mlflow:
    build:
      context: ./mlflow
    image: mlops/mlflow-pg-recap:latest
    container_name: mlflow-recap-20m
    ports:
      - "5000:5000"
    volumes:
      - mlflow-artifacts:/mlflow/mlruns
    networks:
      - recap-net
    restart: unless-stopped
    depends_on:
      postgres:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "python", "-c",
             "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:5000').status==200 else 1)"]
      interval: 10s
      timeout: 5s
      retries: 5

  # ─── Trainer (unchanged) ──────────────────────────────────────────────────
  trainer:
    build:
      context: ./trainer
    image: mlops/trainer-recap:latest
    container_name: trainer-recap-20m
    environment:
      MLFLOW_TRACKING_URI: "http://mlflow:5000"
    volumes:
      - ./data:/code/data
    networks:
      - recap-net
    depends_on:
      mlflow:
        condition: service_healthy

volumes:
  postgres-data:
  mlflow-artifacts:

networks:
  recap-net:
    driver: bridge
```

### 6.3 `trainer/train.py` — unchanged from chap 20l

Use any recent `train.py` from 20g→20l. The trainer talks to `http://mlflow:5000`; it has no idea the backend store has changed.

### 6.4 `trainer/Dockerfile` and `trainer/requirements.txt` — unchanged

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-7"></a>

## 7. Run it, verify PostgreSQL is actually used

```bash
cd chap20m-mlflow-step-by-step-recap-postgresql-backend-store-and-s3-artifacts
docker compose up -d --build
docker compose run --rm trainer --alpha 0.5 --l1_ratio 0.5
```

Open [http://localhost:5000](http://localhost:5000) → experiment `experiment_autolog` (or whatever experiment your `train.py` targets) → 1 run logged.

Now confirm the data is in Postgres (not in any SQLite file):

```bash
docker compose exec postgres psql -U mlflowuser -d mlflowdb -c "\dt"
```

Expected output:

```text
                   List of relations
 Schema |            Name            | Type  |   Owner
--------+----------------------------+-------+------------
 public | alembic_version            | table | mlflowuser
 public | experiment_tags            | table | mlflowuser
 public | experiments                | table | mlflowuser
 public | input_tags                 | table | mlflowuser
 public | inputs                     | table | mlflowuser
 public | latest_metrics             | table | mlflowuser
 public | metrics                    | table | mlflowuser
 public | model_version_tags         | table | mlflowuser
 public | model_versions             | table | mlflowuser
 public | params                     | table | mlflowuser
 public | registered_model_tags      | table | mlflowuser
 public | registered_models          | table | mlflowuser
 public | run_tags                   | table | mlflowuser
 public | runs                       | table | mlflowuser
 public | tags                       | table | mlflowuser
```

These are MLflow's own tables. Query the runs directly:

```bash
docker compose exec postgres psql -U mlflowuser -d mlflowdb \
  -c "SELECT run_uuid, name, status, lifecycle_stage FROM runs ORDER BY start_time DESC LIMIT 5;"
```

> [!TIP]
> You can use any Postgres GUI (pgAdmin, TablePlus, DBeaver…) by connecting to `localhost:5432` with `mlflowuser` / `mlflowpassword` / `mlflowdb`. No port is published by default — add `ports: ["5432:5432"]` under the `postgres` service if you want external access.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-8"></a>

## 8. Production variant: PostgreSQL + S3

When you move to real infrastructure, you'll typically keep PostgreSQL as the backend store and replace the local Docker volume with **AWS S3** for artifacts. The two command-line recipes:

### 8.1 SQLite (learning, local only)

```bash
mlflow server \
  --backend-store-uri  sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 127.0.0.1 \
  --port 5000
```

### 8.2 PostgreSQL + S3 (production)

```bash
mlflow server \
  --backend-store-uri  postgresql://user:password@postgres:5432/mlflowdb \
  --default-artifact-root  s3://my-mlflow-bucket/artifacts \
  --host 0.0.0.0 \
  --port 5000 \
  --no-serve-artifacts
```

`--no-serve-artifacts` tells MLflow not to proxy artifact downloads — the trainer and any client that needs the model fetches directly from S3 using their own AWS credentials. MLflow only stores the URI pointer in Postgres.

### 8.3 Docker Compose snippet for the S3 variant

Add these environment variables to the `mlflow` service:

```yaml
mlflow:
  environment:
    AWS_ACCESS_KEY_ID:     "${AWS_ACCESS_KEY_ID}"
    AWS_SECRET_ACCESS_KEY: "${AWS_SECRET_ACCESS_KEY}"
    AWS_DEFAULT_REGION:    "us-east-1"
  command:
    - mlflow
    - server
    - --backend-store-uri
    - postgresql://mlflowuser:mlflowpassword@postgres:5432/mlflowdb
    - --default-artifact-root
    - s3://my-mlflow-bucket/artifacts
    - --host
    - "0.0.0.0"
    - --port
    - "5000"
    - --no-serve-artifacts
```

And add the same AWS env vars to the `trainer` service so it can upload artifacts to S3:

```yaml
trainer:
  environment:
    MLFLOW_TRACKING_URI:   "http://mlflow:5000"
    AWS_ACCESS_KEY_ID:     "${AWS_ACCESS_KEY_ID}"
    AWS_SECRET_ACCESS_KEY: "${AWS_SECRET_ACCESS_KEY}"
    AWS_DEFAULT_REGION:    "us-east-1"
```

Put the real keys in a `.env` file at the compose project root (never hard-code them). Docker Compose picks it up automatically:

```bash
# .env  (DO NOT commit this file)
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
```

### 8.4 IAM policy for the S3 bucket

The IAM user (or role) that runs MLflow needs at minimum:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:ListBucket",
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject"
      ],
      "Resource": [
        "arn:aws:s3:::my-mlflow-bucket",
        "arn:aws:s3:::my-mlflow-bucket/*"
      ]
    }
  ]
}
```

### 8.5 Summary table

| Config | Backend store | Artifact store | Concurrency | Use for |
|---|---|---|---|---|
| SQLite + local dir | `sqlite:///mlflow.db` | `./mlruns` | Single user | Local dev |
| **PostgreSQL + local volume** | `postgresql://...` | named Docker volume | Multi-container | **This chapter** |
| PostgreSQL + S3 | `postgresql://...` | `s3://bucket/...` | Multi-user / cloud | Production |

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-9"></a>

## 9. Tear down

```bash
docker compose down          # keep postgres-data + mlflow-artifacts
docker compose down -v       # wipe everything (all data gone!)
```

> [!WARNING]
> `docker compose down -v` deletes the `postgres-data` volume — all runs, experiments and registered models stored in Postgres are permanently lost.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-10"></a>

## 10. Recap and next chapter

| Layer | This chapter | Previous chapters |
|---|---|---|
| Backend store | **PostgreSQL** (production-grade) | SQLite (learning) |
| Artifact store | Named Docker volume | Named Docker volume |
| Trainer code | Unchanged | Unchanged |
| Verification | `psql` direct query | UI only |

Next: **[chapter 20n](./20n-practical-work-15n-mlflow-step-by-step-recap-model-signature-manual-and-infer-signature.md)** — add a **model signature** so MLflow knows the exact data types and shapes the model expects. Two approaches: **manual** (`ModelSignature`, `Schema`, `ColSpec`) and **automatic** (`infer_signature`).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 20m — PostgreSQL backend store + S3 for production</strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>
