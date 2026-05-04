<a id="top"></a>

# Chapter 20aa — Step-by-step recap: the MLflow CLI (`mlflow doctor`, `mlflow artifacts`, `mlflow db upgrade`, `mlflow experiments`, `mlflow runs`)

## Table of Contents

| # | Section |
|---|---|
| 1 | [Objective](#section-1) |
| 2 | [Where to run these commands? The `cli` service pattern](#section-2) |
| 3 | [Project structure](#section-3) |
| 4 | [Seeding some data so the commands have something to show](#section-4) |
| 5 | [`mlflow doctor` — installation health check](#section-5) |
| 6 | [`mlflow experiments` — create / rename / delete / restore / search / csv](#section-6) |
| 7 | [`mlflow runs` — list / describe / delete / restore](#section-7) |
| 8 | [`mlflow artifacts` — list / download / log-artifacts](#section-8) |
| 9 | [`mlflow db upgrade` — apply schema migrations](#section-9) |
| 10 | [Tips: `MLFLOW_TRACKING_URI`, output formats, scripting](#section-10) |
| 11 | [Tear down](#section-11) |
| 12 | [Recap and next chapter](#section-12) |

---

<a id="section-1"></a>

## 1. Objective

Up to chap 20z everything was **Python code talking to MLflow**. But MLflow ships with a full **command-line interface** that does most of what the Python SDK does — and a few admin things the SDK doesn't. The CLI is what you reach for to:

- Diagnose a misbehaving installation: `mlflow doctor`.
- Browse / mutate the registry from the terminal: `mlflow experiments ...`, `mlflow runs ...`.
- Pull artifacts to disk for inspection or sharing: `mlflow artifacts download ...`.
- Apply DB migrations after upgrading MLflow itself: `mlflow db upgrade ...`.
- Drive everything from shell scripts and CI pipelines.

Today's chapter walks through the entire essential CLI surface, all run from a small **`cli` Docker service** that talks to our existing MLflow tracking server.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. Where to run these commands? The `cli` service pattern

The MLflow CLI is just `pip install mlflow`. Every container we built so far already has it. But the cleanest pattern in a multi-service setup is a **dedicated `cli` service**:

- Built from a tiny image with only `mlflow`.
- Has `MLFLOW_TRACKING_URI=http://mlflow:5000` baked in via env var.
- Mounts a `cli_artifact/` directory on the host — handy for `mlflow artifacts download` outputs.
- No `ENTRYPOINT` — you spawn arbitrary commands with `docker compose run --rm cli mlflow ...` (or open a shell with `docker compose run --rm --entrypoint sh cli`).

This way the same workflow that runs locally also runs in CI: `docker compose run --rm cli mlflow runs list --experiment-id 1`.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. Project structure

```text
chap20aa-mlflow-step-by-step-recap-mlflow-cli-doctor-artifacts-experiments-runs/
├── README.md
├── docker-compose.yml
├── data/
│   └── red-wine-quality.csv
├── mlflow/
│   └── Dockerfile
├── trainer/                ← seeds 1-2 runs so we have something to inspect
│   ├── Dockerfile
│   ├── requirements.txt
│   └── train.py
└── cli/                    ← the CLI sandbox
    ├── Dockerfile
    └── requirements.txt
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. Seeding some data so the commands have something to show

Spin everything up and produce a couple of runs first:

```bash
cd chap20aa-mlflow-step-by-step-recap-mlflow-cli-doctor-artifacts-experiments-runs
docker compose up -d --build mlflow

docker compose run --rm trainer --alpha 0.4 --l1_ratio 0.4
docker compose run --rm trainer --alpha 0.6 --l1_ratio 0.6
```

Now we have an experiment (`experiment_cli_demo`) with 2 runs and one `model/` artifact in each. Note one of the run ids — we'll use it below as `<RUN_ID>`. Find it in the UI ([http://localhost:5000](http://localhost:5000)) or via `mlflow runs list` (next sections).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. `mlflow doctor` — installation health check

Reports the Python / MLflow / system versions, the configured tracking URI, the artifact backend, the registry status, and any known incompatibilities.

```bash
docker compose run --rm cli mlflow doctor
```

Sample output:

```text
System information: Linux #1 SMP Tue ...
Python version: 3.12.7
MLflow version: 2.16.2
MLflow module location: /usr/local/lib/python3.12/site-packages/mlflow/__init__.py
Tracking URI: http://mlflow:5000
Registry URI: http://mlflow:5000
MLflow environment variables:
  MLFLOW_TRACKING_URI: http://mlflow:5000
MLflow dependencies:
  Flask: 3.0.3
  alembic: 1.13.2
  click: 8.1.7
  ...
```

To redact env vars before posting in a bug report:

```bash
docker compose run --rm cli mlflow doctor --mask-envs
```

> [!TIP]
> `mlflow doctor` is the first thing to run when something feels off. 9 times out of 10 it surfaces the problem (wrong tracking URI, mismatched MLflow version on client vs server, missing optional dep…).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-6"></a>

## 6. `mlflow experiments` — create / rename / delete / restore / search / csv

### 6.1 Create a new experiment

```bash
docker compose run --rm cli mlflow experiments create --experiment-name cli_experiment
```

Output: `Created experiment 'cli_experiment' with id <NEW_EXP_ID>`. Remember that id (or use the UI / `experiments search` below to find it).

### 6.2 Search experiments — including deleted

The `--view` flag accepts `active_only` (default), `deleted_only`, or `all`.

```bash
docker compose run --rm cli mlflow experiments search --view all
```

Output (table):

```text
Experiment Id    Name                    Artifact Location                Lifecycle Stage
0                Default                 mlflow-artifacts:/0              active
1                experiment_cli_demo     mlflow-artifacts:/1              active
2                cli_experiment          mlflow-artifacts:/2              active
```

### 6.3 Rename an experiment

```bash
docker compose run --rm cli mlflow experiments rename \
  --experiment-id 2 --new-name test1
```

### 6.4 Delete + restore

`delete` is a soft-delete: the experiment is hidden from the UI's "Active" tab but its data and runs are preserved. `restore` reverses it.

```bash
docker compose run --rm cli mlflow experiments delete  --experiment-id 2
docker compose run --rm cli mlflow experiments restore --experiment-id 2
```

After delete, `experiments search --view active_only` won't list it; `experiments search --view all` will, with `Lifecycle Stage = deleted`.

> [!IMPORTANT]
> Permanent deletion is a separate operation. Use `MlflowClient().delete_experiment(...)` then a server-side `mlflow gc` to actually free the storage.

### 6.5 Export an experiment to CSV

```bash
docker compose run --rm cli mlflow experiments csv \
  --experiment-id 1 --filename /artifacts/test.csv
```

(Adjust `/artifacts/...` to a path that's mounted in the `cli` container — e.g. `/artifacts` if you mount `./cli_artifact:/artifacts`.) The CSV has one row per run with all params and metrics — perfect for an Excel review or a quick `pandas.read_csv` analysis.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-7"></a>

## 7. `mlflow runs` — list / describe / delete / restore

### 7.1 List runs of an experiment

```bash
docker compose run --rm cli mlflow runs list --experiment-id 1 --view all
```

Output:

```text
Run ID                                Name                      Status     ...
8a4f...d1                             luxuriant-mole-512        FINISHED
3b2e...c7                             dapper-otter-009          FINISHED
```

### 7.2 Describe a single run (full JSON dump)

```bash
docker compose run --rm cli mlflow runs describe --run-id 8a4fd1...
```

You get the run's `info`, `data.params`, `data.metrics`, `data.tags`, `inputs`, etc. as JSON. Pipe through `jq` to filter:

```bash
docker compose run --rm cli sh -c \
  "mlflow runs describe --run-id 8a4fd1... | jq '.data.metrics'"
```

(Add `jq` to the `cli` image if you want this — `apt-get install -y jq` in the Dockerfile.)

### 7.3 Soft-delete + restore

```bash
docker compose run --rm cli mlflow runs delete  --run-id 8a4fd1...
docker compose run --rm cli mlflow runs restore --run-id 8a4fd1...
```

Same lifecycle model as experiments: deleted runs are hidden in the UI but still in the DB, and `runs list --view all` shows them.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-8"></a>

## 8. `mlflow artifacts` — list / download / log-artifacts

### 8.1 List a run's artifacts

```bash
docker compose run --rm cli mlflow artifacts list --run-id 8a4fd1...
```

Output:

```text
File path                       File size
model/MLmodel                   542
model/conda.yaml                184
model/python_env.yaml            89
model/requirements.txt           58
model/model.pkl                 814
```

Drill into a sub-folder by passing `--artifact-path`:

```bash
docker compose run --rm cli mlflow artifacts list \
  --run-id 8a4fd1... --artifact-path model
```

### 8.2 Download artifacts to a local folder

```bash
docker compose run --rm cli mlflow artifacts download \
  --run-id 8a4fd1... --dst-path /artifacts/cli_artifact
```

The `cli` service mounts `./cli_artifact` from the host as `/artifacts/cli_artifact`, so after the command you'll find a full copy of the run's artifact directory on your host machine. Useful for sharing a model with a non-MLflow user, or for diffing two model versions side-by-side.

### 8.3 Upload arbitrary artifacts to an existing run

```bash
docker compose run --rm cli mlflow artifacts log-artifacts \
  --run-id 8a4fd1... \
  --local-dir /artifacts/cli_artifact \
  --artifact-path cli_artifact
```

This goes the other way — pushing a local folder into the run under the sub-path `cli_artifact`. Handy when an external script produced reports that should live alongside the model in the same run.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-9"></a>

## 9. `mlflow db upgrade` — apply schema migrations

Whenever you bump MLflow on the server (e.g. `2.13` → `2.16`), the underlying SQL schema may have changed. Migrations are **not** applied automatically; you must run:

```bash
docker compose run --rm cli mlflow db upgrade sqlite:///mlflow.db
```

…with the **same backend store URI the server uses**. In our setup the server uses `sqlite:////mlflow/database/mlflow.db`, so:

```bash
docker compose run --rm \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  -v mlflow-db:/mlflow/database \
  cli mlflow db upgrade sqlite:////mlflow/database/mlflow.db
```

For PostgreSQL the URI looks like `postgresql://user:pass@host:5432/dbname`. Behind the scenes MLflow uses Alembic to apply pending revisions — same idiom as Django/Flask migrations.

> [!IMPORTANT]
> Always **back up the DB** (or a snapshot of the volume) before running `db upgrade` in production. Migrations are forward-only.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-10"></a>

## 10. Tips: `MLFLOW_TRACKING_URI`, output formats, scripting

- Most commands honour the `MLFLOW_TRACKING_URI` env var. Set it once in the shell (`export MLFLOW_TRACKING_URI=http://localhost:5000`) and skip the `--tracking-uri` flag.
- `mlflow runs list` and `mlflow experiments search` accept `--output-format json` (or `csv`) for parsing in scripts. Example:

  ```bash
  docker compose run --rm cli sh -c \
    "mlflow runs list --experiment-id 1 --view all --output-format json | jq '.[].info.run_id'"
  ```

- The CLI exits non-zero on errors → safe to chain with `&&` in shell scripts.
- Help is always one flag away: `mlflow <subcommand> --help`. Try `mlflow models --help` (chap 25/26 will use it for serving).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-11"></a>

## 11. Tear down

```bash
docker compose down
docker compose down -v        # also drops mlflow-db / mlflow-artifacts / shared volumes
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-12"></a>

## 12. Recap and next chapter

The MLflow CLI mirrors most of the Python SDK and adds a few admin essentials:

| Family | Useful commands |
|---|---|
| Diagnostics | `mlflow doctor [--mask-envs]` |
| Experiments | `create`, `rename`, `delete`, `restore`, `search --view all`, `csv` |
| Runs | `list --view all`, `describe`, `delete`, `restore` |
| Artifacts | `list`, `download --dst-path`, `log-artifacts --local-dir --artifact-path` |
| DB | `db upgrade <backend-store-uri>` |

Combine with `--output-format json` + `jq` to script anything the UI does.

This wraps up the **20a → 20aa** recap series. The trainer side is fully covered. The next chapter (21) leaves the trainer and starts the **deployment** half of the course: serving the registered pyfunc through FastAPI and consuming it from a Streamlit app.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 20aa — the MLflow CLI</strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>
