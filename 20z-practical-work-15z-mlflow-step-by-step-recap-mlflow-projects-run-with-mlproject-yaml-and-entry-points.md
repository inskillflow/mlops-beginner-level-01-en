<a id="top"></a>

# Chapter 20z — Step-by-step recap: packaging training as an `MLproject` and launching it with `mlflow.projects.run(...)`

## Table of Contents

| # | Section |
|---|---|
| 1 | [Objective](#section-1) |
| 2 | [What we add today vs chap 20y](#section-2) |
| 3 | [What is an `MLproject`?](#section-3) |
| 4 | [Anatomy of the `MLproject` YAML — entry points and parameters](#section-4) |
| 5 | [Choosing an env manager: `local`, `virtualenv`, `conda`, `docker`](#section-5) |
| 6 | [The Python launcher: `mlflow.projects.run(...)`](#section-6) |
| 7 | [Project structure](#section-7) |
| 8 | [The code](#section-8) |
| 9 | [Run it three ways](#section-9) |
| 10 | [Tear down](#section-10) |
| 11 | [Recap and next chapter](#section-11) |

---

<a id="section-1"></a>

## 1. Objective

So far our trainer scripts were launched directly: `python train.py --alpha 0.4`. Today we put the same `train.py` behind a thin **packaging layer** — a single YAML file called `MLproject` — and launch it through the **MLflow Projects** API:

```python
mlflow.projects.run(
    uri=".",
    entry_point="ElasticNet",
    parameters={"alpha": 0.3, "l1_ratio": 0.3},
    experiment_name="Project exp 1",
    env_manager="local",
)
```

What does that buy us?

- A **standard, declarative way** to expose multiple training pipelines from one repository (each `entry_point` is one runnable command).
- **Strongly-typed parameters** with defaults — MLflow validates them before launching.
- **Reproducible environments** — the YAML can pin a `python_env`, a `conda.yaml`, or a Docker image. MLflow re-creates that environment on demand.
- **Same launcher** whether you run locally, on a remote machine via SSH, on Databricks, on Kubernetes, or as a sub-call from another script.

Think of `MLproject` as the **`Makefile` of MLflow**: it tells MLflow how to run your training, what knobs to expose, and what env it needs.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. What we add today vs chap 20y

| Diff | What |
|---|---|
| Add `trainer/MLproject` (no extension) | The packaging contract. |
| Add `trainer/python_env.yaml` | Reproducible env declaration (optional, used by `env_manager="virtualenv"`). |
| Add `trainer/run_project.py` | A tiny launcher that calls `mlflow.projects.run(...)`. |
| `train.py` itself is **unchanged** from chap 20y | The packaging is purely additive. |
| Use `env_manager="local"` from inside the container | Skips env creation since the container already has the deps. |

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. What is an `MLproject`?

A YAML file (literally named `MLproject`, no extension) at the **root of a directory** turns that directory into a **runnable MLflow project**. The file declares:

1. A `name` (free text).
2. An **environment specification** (`python_env`, `conda_env`, or `docker_env`).
3. One or more **entry points** — each is a named command + its parameters.

When you call `mlflow run <path-or-git-url>`, MLflow:

- clones the directory (or pulls it from git),
- creates the declared env (or reuses one),
- runs the entry point as a subprocess with the parameters substituted in,
- captures everything under an MLflow run.

Because the spec is declarative, the **exact same project** can be launched from your laptop, a CI runner, or `mlflow run git://github.com/team/repo --version v2.1`.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. Anatomy of the `MLproject` YAML — entry points and parameters

```yaml
name: chap20z-elasticnet-project

# Reproducible env. Pick ONE of: python_env, conda_env, docker_env.
python_env: python_env.yaml

entry_points:
  ElasticNet:
    parameters:
      alpha:    { type: float, default: 0.4 }
      l1_ratio: { type: float, default: 0.4 }
    command: "python train.py --alpha {alpha} --l1_ratio {l1_ratio}"

  main:
    parameters:
      alpha:    { type: float, default: 0.4 }
      l1_ratio: { type: float, default: 0.4 }
    command: "python train.py --alpha {alpha} --l1_ratio {l1_ratio}"
```

What each piece does:

| Field | Purpose |
|---|---|
| `name` | Free-text identifier shown in logs. |
| `python_env: python_env.yaml` | The file describing how to recreate the env (Python version + pip deps). |
| `entry_points.<name>` | One named command. `main` is the **default** when no entry point is specified. |
| `parameters.<name>.type` | One of `float`, `int`, `string`, `path`, `uri`. MLflow validates the value. |
| `parameters.<name>.default` | Used when the caller doesn't pass that parameter. |
| `command` | Shell command to run. `{param}` placeholders are substituted with validated values. |

The `ElasticNet` and `main` entry points happen to do the same thing here — that's just to show how to expose **a friendly name** alongside the implicit `main` default. Real projects often have `train`, `evaluate`, `score`, `register`… each invoking a different `.py` script.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. Choosing an env manager: `local`, `virtualenv`, `conda`, `docker`

`mlflow.projects.run(..., env_manager=...)` controls how the environment for the entry point is built:

| Value | What MLflow does | Use when |
|---|---|---|
| `"local"` | Skips env creation entirely. Runs the command in the current Python interpreter. | Inside a container that already has the deps (our case). |
| `"virtualenv"` | Reads `python_env.yaml`, creates a fresh `venv`, installs the listed pip packages. | Local dev where you want isolation but don't want conda. |
| `"conda"` | Reads `conda_env`, creates a fresh conda env. | Conda-based stacks (data science laptops, Anaconda Cloud). |
| `"docker"` | Builds the `docker_env.image` and runs the entry point inside a container. | Heavy native deps (CUDA, GDAL). |

> [!IMPORTANT]
> Inside our slim Python container we use `"local"` because conda isn't installed and creating a venv-inside-a-container would be redundant. If you removed `env_manager="local"`, MLflow would default to `"virtualenv"` and try to install everything declared in `python_env.yaml` from scratch on every run — slow and pointless when the container is already complete.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-6"></a>

## 6. The Python launcher: `mlflow.projects.run(...)`

```python
import mlflow

result = mlflow.projects.run(
    uri=".",                            # local path OR git URL
    entry_point="ElasticNet",           # which entry to run; defaults to "main"
    parameters={"alpha": 0.3, "l1_ratio": 0.3},
    experiment_name="Project exp 1",
    env_manager="local",
    synchronous=True,                   # block until done; False returns a SubmittedRun
)

print(result.run_id)
print(result.get_status())
```

Equivalent CLI:

```bash
mlflow run . -e ElasticNet -P alpha=0.3 -P l1_ratio=0.3 \
             --experiment-name "Project exp 1" \
             --env-manager local
```

For a remote project (any git URL works):

```bash
mlflow run git@github.com:team/repo.git#path/to/project \
           --version v2.1 \
           -e ElasticNet -P alpha=0.3
```

Same execution model, same MLflow run as a result.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-7"></a>

## 7. Project structure

```text
chap20z-mlflow-step-by-step-recap-mlflow-projects-run-with-mlproject-yaml-and-entry-points/
├── README.md
├── docker-compose.yml
├── data/
│   └── red-wine-quality.csv
├── mlflow/
│   └── Dockerfile
└── trainer/
    ├── Dockerfile
    ├── requirements.txt
    ├── MLproject              ← packaging contract (no extension)
    ├── python_env.yaml        ← env spec (used by env_manager="virtualenv")
    ├── train.py               ← unchanged from chap 20y
    └── run_project.py         ← launcher that calls mlflow.projects.run(...)
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-8"></a>

## 8. The code

### 8.1 `trainer/MLproject` (no extension)

```yaml
name: chap20z-elasticnet-project

python_env: python_env.yaml

entry_points:
  ElasticNet:
    parameters:
      alpha:    { type: float, default: 0.4 }
      l1_ratio: { type: float, default: 0.4 }
    command: "python train.py --alpha {alpha} --l1_ratio {l1_ratio}"

  main:
    parameters:
      alpha:    { type: float, default: 0.4 }
      l1_ratio: { type: float, default: 0.4 }
    command: "python train.py --alpha {alpha} --l1_ratio {l1_ratio}"
```

### 8.2 `trainer/python_env.yaml`

```yaml
python: "3.12"
build_dependencies:
  - pip
dependencies:
  - mlflow==2.16.2
  - scikit-learn==1.5.2
  - pandas==2.2.3
  - numpy==2.1.1
```

### 8.3 `trainer/train.py` (unchanged from chap 20y)

```python
import argparse
import os
import warnings

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha",    type=float, default=0.4)
    parser.add_argument("--l1_ratio", type=float, default=0.4)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    np.random.seed(40)

    data = pd.read_csv("data/red-wine-quality.csv")
    train, test = train_test_split(data)
    train_x = train.drop(["quality"], axis=1)
    test_x  = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y  = test[["quality"]]

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    )

    # mlflow.projects.run takes care of: set_experiment, start_run, end_run.
    # We're already INSIDE that run when train.py executes.
    lr = ElasticNet(alpha=args.alpha, l1_ratio=args.l1_ratio, random_state=42)
    lr.fit(train_x, train_y)
    rmse, mae, r2 = eval_metrics(test_y, lr.predict(test_x))

    print(f"ElasticNet (alpha={args.alpha}, l1_ratio={args.l1_ratio})")
    print(f"  RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")

    mlflow.log_params({"alpha": args.alpha, "l1_ratio": args.l1_ratio})
    mlflow.log_metrics({"rmse": rmse, "r2": r2, "mae": mae})
    mlflow.sklearn.log_model(lr, "model")


if __name__ == "__main__":
    main()
```

> [!TIP]
> When invoked through `mlflow.projects.run(...)` the active MLflow run is **created by MLflow itself** (you can read it via `mlflow.active_run()`). You should NOT call `mlflow.start_run()` from `train.py` in this mode — it would either error out or open a nested run.

### 8.4 `trainer/run_project.py` (launcher — what `docker compose run --rm trainer` executes)

```python
"""Launcher: builds and runs the MLproject in this directory."""

import os
import mlflow

PARAMETERS = {
    "alpha":    0.3,
    "l1_ratio": 0.3,
}
EXPERIMENT_NAME = "Project exp 1"
ENTRY_POINT     = "ElasticNet"


if __name__ == "__main__":
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    )
    print("Tracking URI :", mlflow.get_tracking_uri())
    print("Project URI  : .  (current directory)")
    print("Entry point  :", ENTRY_POINT)
    print("Parameters   :", PARAMETERS)

    submitted = mlflow.projects.run(
        uri=".",
        entry_point=ENTRY_POINT,
        parameters=PARAMETERS,
        experiment_name=EXPERIMENT_NAME,
        env_manager="local",     # CRITICAL inside an already-built container
        synchronous=True,
    )

    print("\nProject finished.")
    print("Run id     :", submitted.run_id)
    print("Run status :", submitted.get_status())
```

### 8.5 `trainer/Dockerfile`

```dockerfile
FROM python:3.12-slim
WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY MLproject python_env.yaml train.py run_project.py ./
ENTRYPOINT ["python", "run_project.py"]
```

### 8.6 `trainer/requirements.txt`

```text
mlflow==2.16.2
scikit-learn==1.5.2
pandas==2.2.3
numpy==2.1.1
```

### 8.7 `docker-compose.yml`

Identical to chap 20y. The trainer service mounts `./data` and points at the MLflow tracking server via `MLFLOW_TRACKING_URI`.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-9"></a>

## 9. Run it three ways

### 9.1 Via the Python launcher (default `docker compose run --rm trainer`)

```bash
cd chap20z-mlflow-step-by-step-recap-mlflow-projects-run-with-mlproject-yaml-and-entry-points
docker compose up -d --build mlflow
docker compose run --rm trainer
```

Stdout (excerpt):

```text
Tracking URI : http://mlflow:5000
Project URI  : .  (current directory)
Entry point  : ElasticNet
Parameters   : {'alpha': 0.3, 'l1_ratio': 0.3}
2026/05/04 21:42:11 INFO mlflow.projects.utils: === Created directory ... ===
2026/05/04 21:42:11 INFO mlflow.projects.backend.local: === Running command ... ===
ElasticNet (alpha=0.3, l1_ratio=0.3)
  RMSE=0.7651  MAE=0.6118  R2=0.1374
Project finished.
Run id     : 9b7d2f...c1
Run status : FINISHED
```

### 9.2 Via the MLflow CLI inside the container

The same project can be launched without our launcher script:

```bash
docker compose run --rm --entrypoint sh trainer -c \
  "mlflow run . -e ElasticNet -P alpha=0.3 -P l1_ratio=0.3 \
                --experiment-name 'Project exp 1' --env-manager local"
```

Same effect, no Python launcher needed.

### 9.3 Use the `main` entry point (the implicit default)

```bash
docker compose run --rm --entrypoint sh trainer -c \
  "mlflow run . --env-manager local"
```

`main` is invoked when no `-e` is given. Defaults from the YAML kick in (`alpha=0.4 l1_ratio=0.4`).

### 9.4 In the UI

[http://localhost:5000](http://localhost:5000) → **`Project exp 1`** → each invocation produces a new run with:

- Params: `alpha`, `l1_ratio`.
- Metrics: `rmse`, `mae`, `r2`.
- Tags: `mlflow.project.entryPoint`, `mlflow.project.backend`, `mlflow.source.name` — added automatically by `projects.run`. Hover over them in the UI: they document **how the run was launched**.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-10"></a>

## 10. Tear down

```bash
docker compose down
docker compose down -v
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-11"></a>

## 11. Recap and next chapter

Three artefacts make a directory a real MLflow project:

```text
MLproject              ← entry points + parameters (declarative)
python_env.yaml        ← env recipe (used by virtualenv/conda env managers)
train.py               ← the actual code
```

Launching it is one Python call:

```python
mlflow.projects.run(
    uri=".", entry_point="ElasticNet",
    parameters={...}, experiment_name="...",
    env_manager="local",
)
```

…or one CLI command: `mlflow run . -e ElasticNet -P alpha=0.3`. Same project runs locally, on git, on a remote host, in a Docker image — without changing the trainer code.

Next: **[chapter 20aa](./20z1-practical-work-15aa-mlflow-step-by-step-recap-mlflow-cli-doctor-artifacts-experiments-runs.md)** — meet the **MLflow CLI** (`mlflow doctor`, `mlflow artifacts list/download`, `mlflow experiments create/rename/delete/restore`, `mlflow runs list/describe/delete/restore`, `mlflow db upgrade`). The everyday admin toolkit for any MLflow installation.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 20z — mlflow.projects.run with MLproject YAML and entry points</strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>
