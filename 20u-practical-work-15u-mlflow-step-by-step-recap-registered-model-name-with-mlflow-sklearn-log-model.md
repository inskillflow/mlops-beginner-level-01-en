<a id="top"></a>

# Chapter 20u — Step-by-step recap: registering a model with `mlflow.sklearn.log_model(..., registered_model_name="elasticnet-api")`

## Table of Contents

| # | Section |
|---|---|
| 1 | [Objective](#section-1) |
| 2 | [What we add today vs chap 20t](#section-2) |
| 3 | [What is the MLflow Model Registry?](#section-3) |
| 4 | [The single argument that triggers registration: `registered_model_name`](#section-4) |
| 5 | [Backend store requirement (SQLite, PostgreSQL — but not file://)](#section-5) |
| 6 | [Project structure](#section-6) |
| 7 | [The code](#section-7) |
| 8 | [Run it, browse the Models tab](#section-8) |
| 9 | [Loading the registered model from anywhere](#section-9) |
| 10 | [Tear down](#section-10) |
| 11 | [Recap and next chapter](#section-11) |

---

<a id="section-1"></a>

## 1. Objective

Up to chap 20t every model lived inside the run that produced it. To consume one from another service (FastAPI, a batch scorer…) we had to know its `runs:/<run_id>/<artifact_path>` URI. That works, but:

- Run IDs are unfriendly identifiers.
- There is no built-in concept of "the current Production model".
- Promoting a new model means hard-coding a new run ID everywhere.

The **Model Registry** solves all three. Models are referenced by **a stable name** (`elasticnet-api`) and a **stage or alias** (`Production`, `Staging`, `Champion`, `Challenger`…). The actual artifact behind that name can be swapped without anyone changing their code.

Today we take the **smallest possible step into the registry**: add a single argument — `registered_model_name="elasticnet-api"` — to `mlflow.sklearn.log_model`. MLflow will then:

1. Log the model artifact inside the run (as before).
2. Create the registered model `elasticnet-api` if it doesn't exist.
3. Add a new **version** (1, then 2, then 3…) on every subsequent training run that uses the same name.

The Models tab in the UI will show all versions for `elasticnet-api`.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. What we add today vs chap 20t

Chap 20t was the climax of the evaluation series. To put the registry in the spotlight we **trim the script back** to a small, focused training script:

| Diff | What |
|---|---|
| Drop `pyfunc` wrapper, baseline, validation thresholds, custom metrics, scatter | All proven in 20p–20t. Today's lesson is the registry. |
| Use plain `mlflow.sklearn.log_model(lr, "model", registered_model_name="elasticnet-api")` | One argument enables the registry. |
| Experiment renamed `experiment_register_model_api` | Matches the user's snippet. |
| Stop using `autolog` | The registry argument is on `log_model` itself, no autolog magic involved. |

Everything else (Docker stack, MLflow server, env-driven `MLFLOW_TRACKING_URI`) is identical.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. What is the MLflow Model Registry?

A separate database table (alongside experiments, runs, metrics…) that maps a **stable name** to a **list of versions**, each pointing to one model artifact:

```text
Model Registry
└── elasticnet-api           (the registered model name)
    ├── Version 1            (artifact from run abc123, stage = Archived)
    ├── Version 2            (artifact from run def456, stage = Staging)
    └── Version 3            (artifact from run 789xyz, stage = Production)  ← live
```

Anyone can then load by name + stage/alias:

```python
mlflow.pyfunc.load_model("models:/elasticnet-api/Production")
```

…and never has to know which run actually produced version 3.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. The single argument that triggers registration: `registered_model_name`

```python
mlflow.sklearn.log_model(
    sk_model=lr,
    artifact_path="model",                # name of the artifact folder inside the run
    registered_model_name="elasticnet-api",   # <- THIS triggers registration
)
```

Side-effects:

1. The artifact is uploaded to the run, exactly like a normal `log_model(...)` would do.
2. MLflow checks the registry: does `elasticnet-api` exist?
   - **No** → create it, add **Version 1** pointing at this artifact.
   - **Yes** → add a new **Version N+1**.
3. The new version starts in the **`None`** stage. You move it manually (or via `MlflowClient.transition_model_version_stage(..., stage="Staging")`).

> [!IMPORTANT]
> The same kwarg works for every flavour: `mlflow.sklearn.log_model(...)`, `mlflow.pyfunc.log_model(...)`, `mlflow.tensorflow.log_model(...)`, `mlflow.pytorch.log_model(...)`. Same name, same effect.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. Backend store requirement (SQLite, PostgreSQL — but not `file://`)

The registry needs a **real database** to track model versions, stages, transitions, etc. MLflow's default `file://./mlruns` backend has **no registry**. If you use it, the call still uploads the artifact but **silently skips the registration step** (you'll see a warning).

| Backend store | Registry support |
|---|---|
| `file:///mlruns` | NO |
| `sqlite:///mlflow.db` | YES |
| `postgresql://...` | YES (production-grade) |
| `mysql://...` | YES |

Our `mlflow/Dockerfile` already uses SQLite (`--backend-store-uri sqlite:///database/mlflow.db`), so we're good.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-6"></a>

## 6. Project structure

```text
chap20u-mlflow-step-by-step-recap-registered-model-name-with-mlflow-sklearn-log-model/
├── README.md
├── docker-compose.yml
├── data/
│   └── red-wine-quality.csv
├── mlflow/
│   └── Dockerfile        (SQLite backend, supports the registry)
└── trainer/
    ├── Dockerfile
    ├── requirements.txt
    └── train.py         (single mlflow.sklearn.log_model call with registered_model_name)
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-7"></a>

## 7. The code

### 7.1 `trainer/train.py`

```python
import argparse
import logging
import os
import warnings

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--alpha",    type=float, required=False, default=0.6)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.6)
args = parser.parse_args()


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    )
    print("The set tracking URI is", mlflow.get_tracking_uri())

    exp = mlflow.set_experiment(experiment_name="experiment_register_model_api")
    print(f"Name              : {exp.name}")
    print(f"Experiment_id     : {exp.experiment_id}")
    print(f"Artifact Location : {exp.artifact_location}")

    data = pd.read_csv("data/red-wine-quality.csv")
    train, test = train_test_split(data, test_size=0.25)

    train_x = train.drop(["quality"], axis=1)
    test_x  = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y  = test[["quality"]]

    alpha, l1_ratio = args.alpha, args.l1_ratio

    mlflow.start_run()

    mlflow.set_tags({
        "engineering":       "ML platform",
        "release.candidate": "RC1",
        "release.version":   "2.0",
    })

    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)
    predicted_qualities = lr.predict(test_x)
    rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)

    print(
        f"ElasticNet (alpha={alpha}, l1_ratio={l1_ratio}): "
        f"RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}"
    )

    mlflow.log_params({"alpha": alpha, "l1_ratio": l1_ratio})
    mlflow.log_metrics({"rmse": rmse, "r2": r2, "mae": mae})

    # ===== THE REGISTRY TRIGGER =====
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="model",
        registered_model_name="elasticnet-api",
    )
    print("Model URI         :", model_info.model_uri)
    print("Registered as     :", "elasticnet-api (a new version was added)")

    mlflow.end_run()

    run = mlflow.last_active_run()
    print(f"Active run id     : {run.info.run_id}")
```

### 7.2 `docker-compose.yml`

Standard MLflow + trainer (SQLite backend, identical to chap 20q–20s):

```yaml
services:
  mlflow:
    build: ./mlflow
    ports: ["5000:5000"]
    volumes: [mlflow-db:/mlflow/database, mlflow-artifacts:/mlflow/mlruns]
    networks: [recap-net]
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:5000').status==200 else 1)"]
      interval: 10s
      timeout: 5s
      retries: 5

  trainer:
    build: ./trainer
    environment:
      MLFLOW_TRACKING_URI: "http://mlflow:5000"
    volumes: [./data:/code/data]
    networks: [recap-net]
    depends_on: { mlflow: { condition: service_healthy } }

volumes: { mlflow-db: {}, mlflow-artifacts: {} }
networks: { recap-net: { driver: bridge } }
```

### 7.3 `trainer/requirements.txt`

```text
mlflow==2.16.2
scikit-learn==1.5.2
pandas==2.2.3
numpy==2.1.1
```

(No joblib / cloudpickle needed today — `mlflow.sklearn.log_model` does the serialization itself.)

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-8"></a>

## 8. Run it, browse the Models tab

```bash
cd chap20u-mlflow-step-by-step-recap-registered-model-name-with-mlflow-sklearn-log-model
docker compose up -d --build mlflow
docker compose run --rm trainer --alpha 0.6 --l1_ratio 0.6
```

Trainer's stdout (excerpt):

```text
ElasticNet (alpha=0.6, l1_ratio=0.6): RMSE=0.7873 MAE=0.6334 R2=0.0859
Successfully registered model 'elasticnet-api'.
Created version '1' of model 'elasticnet-api'.
Model URI         : runs:/<run_id>/model
Registered as     : elasticnet-api (a new version was added)
```

Re-run a second time:

```bash
docker compose run --rm trainer --alpha 0.4 --l1_ratio 0.2
```

```text
Registered model 'elasticnet-api' already exists. Creating a new version of it...
Created version '2' of model 'elasticnet-api'.
```

In the UI ([http://localhost:5000](http://localhost:5000)):

- **Experiments tab** → `experiment_register_model_api` → 2 runs, each with their own metrics.
- **Models tab** (top of the page, next to Experiments) → **`elasticnet-api`** → 2 versions:
  ```text
  Version 1   stage=None   from run <run1_id>
  Version 2   stage=None   from run <run2_id>
  ```
  Click any version to see its source run, model schema, and "Stage" dropdown (None → Staging → Production → Archived).

> [!NOTE]
> Stages (`Staging`, `Production`, `Archived`) are deprecated since MLflow 2.9 in favour of **aliases** (e.g. `champion`, `challenger`). Both still work in MLflow 2.16; aliases are the recommended path for new projects. Either way the registration call is the same — only the post-registration transition API differs.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-9"></a>

## 9. Loading the registered model from anywhere

Once registered, every consumer can fetch the model by name + version (or stage/alias). No more `runs:/<id>` to copy around:

```python
import mlflow.pyfunc

# Pin to a specific version
m = mlflow.pyfunc.load_model("models:/elasticnet-api/2")

# Whatever is currently in the Production stage
m = mlflow.pyfunc.load_model("models:/elasticnet-api/Production")

# Whatever has the alias "champion"
m = mlflow.pyfunc.load_model("models:/elasticnet-api@champion")
```

A FastAPI service in chap 25/26 will use exactly this pattern. Promoting a new model becomes a **2-click operation in the UI** (Stage → Production) and your API instantly serves the new version after a refresh.

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

A single argument turns a logged model into a **registered model** with versioning, stages and aliases:

```python
mlflow.sklearn.log_model(lr, "model", registered_model_name="elasticnet-api")
```

Re-running the script bumps the version. Consumers load by name (`models:/elasticnet-api/Production`) and never depend on a run ID again.

Next: **[chapter 20v](./20v-practical-work-15v-mlflow-step-by-step-recap-log-model-plus-pickle-dump-and-log-artifact.md)** — log the model **the canonical MLflow way** AND save a side-car `pickle` of it via `mlflow.log_artifact("elastic-net-regression.pkl")`. Useful when a downstream consumer wants the raw `.pkl` (older code, a custom loader, a notebook…).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 20u — registering a model with `registered_model_name`</strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>
