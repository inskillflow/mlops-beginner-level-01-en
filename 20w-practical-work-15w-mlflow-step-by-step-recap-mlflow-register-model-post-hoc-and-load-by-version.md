<a id="top"></a>

# Chapter 20w — Step-by-step recap: registering with `mlflow.register_model(model_uri, name)` (post-hoc) and loading by `models:/<name>/<version>`

## Table of Contents

| # | Section |
|---|---|
| 1 | [Objective](#section-1) |
| 2 | [What we add today vs chap 20u](#section-2) |
| 3 | [`registered_model_name=` (kwarg) vs `mlflow.register_model(...)` (function)](#section-3) |
| 4 | [Anatomy of `mlflow.register_model`](#section-4) |
| 5 | [Loading the registered model with `models:/<name>/<version>`](#section-5) |
| 6 | [Project structure](#section-6) |
| 7 | [The code](#section-7) |
| 8 | [Run it, browse the Models tab, predict from the registry](#section-8) |
| 9 | [Tear down](#section-9) |
| 10 | [Recap and next chapter](#section-10) |

---

<a id="section-1"></a>

## 1. Objective

In chap 20u we registered a model in **one step** by passing `registered_model_name="elasticnet-api"` to `mlflow.sklearn.log_model(...)`. That kwarg is convenient but fuses two distinct actions: **logging the artifact** and **adding a registry entry**.

Today we use the **two-step** alternative:

1. `mlflow.sklearn.log_model(lr, "model")` — log the artifact (no registry).
2. `mlflow.register_model(model_uri="runs:/<id>/model", name="elastic-api-2")` — register that artifact under a name. Returns a `ModelVersion` object with a `.version` attribute.

Then we **immediately load it back** through the registry:

```python
loaded = mlflow.pyfunc.load_model(model_uri=f"models:/elastic-api-2/{mv.version}")
```

…and run a quick prediction sanity check.

This pattern is what production pipelines use most often: *training* and *registration* are two separate jobs. The trainer logs the model; the registration job (which may run later, after evaluation gates pass) decides which artifact gets a name.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. What we add today vs chap 20u

| Diff | What |
|---|---|
| Drop `registered_model_name=` from `log_model(...)` | Logging stays a pure logging op. |
| Read `run = mlflow.active_run()` | Get the current run id. |
| Call `mv = mlflow.register_model(model_uri=f"runs:/{run.info.run_id}/model", name="elastic-api-2")` | Explicit registration. |
| Use `mv.version` to load back | No more guessing the version number. |
| Round-trip: `pyfunc.load_model("models:/elastic-api-2/<version>")` + predict + recompute metrics | Sanity check that registry serving works. |

Everything else (Docker stack with SQLite backend, `MLFLOW_TRACKING_URI` env var, `mlflow.set_tags`, `log_params/metrics`) is identical.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. `registered_model_name=` (kwarg) vs `mlflow.register_model(...)` (function)

Both create entries in the Model Registry. The difference is **timing and coupling**:

| Aspect | `registered_model_name=` kwarg (chap 20u) | `mlflow.register_model(...)` (today) |
|---|---|---|
| When does registration happen? | Inside `log_model(...)` | After the fact, in a separate call |
| Trainer must know the registry name? | Yes | No (a separate registration job knows it) |
| Can register an existing run's artifact? | Only the one being logged right now | YES — pass any `runs:/<id>/<artifact>` |
| Can register the same artifact twice (under two names)? | No | YES — call it twice with different `name`s |
| Can register from outside the run that produced it? | No | YES — register from a notebook, a CI job, an operator script… |
| Returns a `ModelVersion` object? | No (returns the `ModelInfo` of the log) | YES (with `.version`, `.name`, `.creation_timestamp`, `.source`) |

Production pipelines almost always use the **function form** because evaluation/promotion logic lives **after** training. Quick prototyping prefers the **kwarg form** because it's a one-liner.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. Anatomy of `mlflow.register_model`

```python
import mlflow

mv = mlflow.register_model(
    model_uri="runs:/<run_id>/model",       # the artifact to register
    name="elastic-api-2",                   # the registered model name
    tags={"team": "ml-platform"},           # OPTIONAL: tags on the version
    await_registration_for=300,             # OPTIONAL: seconds to wait until READY
)

print(mv.name)             # "elastic-api-2"
print(mv.version)          # "1" (or "2", "3"... auto-incremented)
print(mv.source)           # "runs:/<run_id>/model"
print(mv.run_id)           # "<run_id>"
print(mv.current_stage)    # "None"  (use MlflowClient to transition)
print(mv.status)           # "READY"
```

Behaviour:

- If `name` doesn't exist in the registry → MLflow creates it, then adds **Version 1**.
- If `name` exists → MLflow appends a new version (Version N+1).
- The version is assigned **server-side** by the tracking store, atomically. Race-safe even with concurrent training jobs.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. Loading the registered model with `models:/<name>/<version>`

Three URI flavours work everywhere `pyfunc` does:

```python
mlflow.pyfunc.load_model("models:/elastic-api-2/1")             # specific version
mlflow.pyfunc.load_model("models:/elastic-api-2/Production")    # current stage (deprecated since 2.9)
mlflow.pyfunc.load_model("models:/elastic-api-2@champion")      # alias (recommended in 2.9+)
```

To load the version we *just* registered without hardcoding a number:

```python
mv = mlflow.register_model(...)
loaded = mlflow.pyfunc.load_model(f"models:/{mv.name}/{mv.version}")
```

> [!IMPORTANT]
> Versions are **strings**, not ints (`"1"`, `"2"`…). The URL accepts both, but `mv.version` is always a string — fine to interpolate as-is.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-6"></a>

## 6. Project structure

```text
chap20w-mlflow-step-by-step-recap-mlflow-register-model-post-hoc-and-load-by-version/
├── README.md
├── docker-compose.yml
├── data/
│   └── red-wine-quality.csv
├── mlflow/
│   └── Dockerfile        (SQLite backend, supports the registry)
└── trainer/
    ├── Dockerfile
    ├── requirements.txt
    └── train.py         (log_model + register_model + load_model)
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
import mlflow.pyfunc
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

REGISTERED_NAME = "elastic-api-2"


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
    print(
        f"Experiment details - Name: {exp.name}, "
        f"ID: {exp.experiment_id}, "
        f"Artifact Location: {exp.artifact_location}"
    )

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
        f"ElasticNet model (alpha={alpha}, l1_ratio={l1_ratio}): "
        f"RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}"
    )

    mlflow.log_params({"alpha": alpha, "l1_ratio": l1_ratio})
    mlflow.log_metrics({"rmse": rmse, "r2": r2, "mae": mae})

    # ===== Step 1: log the artifact (NO registry yet) =====
    model_info = mlflow.sklearn.log_model(lr, artifact_path="model")
    print("Model URI (logged):", model_info.model_uri)

    # ===== Step 2: register that artifact in the registry =====
    run = mlflow.active_run()
    mv = mlflow.register_model(
        model_uri=f"runs:/{run.info.run_id}/model",
        name=REGISTERED_NAME,
    )
    print(f"Registered: name={mv.name!r}  version={mv.version!r}  status={mv.status}")

    # ===== Step 3: load back through the registry by version =====
    registry_uri = f"models:/{mv.name}/{mv.version}"
    print("Loading from registry:", registry_uri)
    loaded = mlflow.pyfunc.load_model(model_uri=registry_uri)

    loaded_predictions = loaded.predict(test_x)
    l_rmse, l_mae, l_r2 = eval_metrics(test_y, loaded_predictions)
    print(
        f"Registered Model Evaluation - "
        f"RMSE_test={l_rmse:.4f}  MAE_test={l_mae:.4f}  R2_test={l_r2:.4f}"
    )

    mlflow.log_metrics({
        "registry_test_rmse": l_rmse,
        "registry_test_mae":  l_mae,
        "registry_test_r2":   l_r2,
    })

    # Sanity check: served model behaves identically
    assert np.allclose(predicted_qualities, loaded_predictions), \
        "Registered model produces different predictions!"
    print("Sanity check OK: in-process and registry-loaded predictions match.")

    mlflow.end_run()
    run = mlflow.last_active_run()
    print(f"Active run id   : {run.info.run_id}")
```

### 7.2 `docker-compose.yml`, `mlflow/Dockerfile`, `trainer/Dockerfile`

Identical to chap 20u (SQLite backend is required for the registry).

### 7.3 `trainer/requirements.txt`

```text
mlflow==2.16.2
scikit-learn==1.5.2
pandas==2.2.3
numpy==2.1.1
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-8"></a>

## 8. Run it, browse the Models tab, predict from the registry

```bash
cd chap20w-mlflow-step-by-step-recap-mlflow-register-model-post-hoc-and-load-by-version
docker compose up -d --build mlflow
docker compose run --rm trainer --alpha 0.6 --l1_ratio 0.6
```

End of stdout:

```text
ElasticNet model (alpha=0.6, l1_ratio=0.6): RMSE=0.7873 MAE=0.6334 R2=0.0859
Model URI (logged): runs:/<run_id>/model
Successfully registered model 'elastic-api-2'.
Created version '1' of model 'elastic-api-2'.
Registered: name='elastic-api-2'  version='1'  status=READY
Loading from registry: models:/elastic-api-2/1
Registered Model Evaluation - RMSE_test=0.7873 MAE_test=0.6334 R2_test=0.0859
Sanity check OK: in-process and registry-loaded predictions match.
```

Re-run a second time:

```bash
docker compose run --rm trainer --alpha 0.4 --l1_ratio 0.2
```

```text
Registered model 'elastic-api-2' already exists. Creating a new version...
Created version '2' of model 'elastic-api-2'.
Registered: name='elastic-api-2'  version='2'  status=READY
Loading from registry: models:/elastic-api-2/2
```

In the UI ([http://localhost:5000](http://localhost:5000)):

- **Experiments tab** → `experiment_register_model_api` → 2 runs, each with a fresh registry version logged in its `registry_test_*` metrics.
- **Models tab** → `elastic-api-2` → Version 1 + Version 2.
  - Click Version 2 → see `Source: runs:/<run_id>/model`, the input/output schema (auto-inferred by `log_model`), and a Stage dropdown.
- The two runs' `registry_test_rmse` should match the run-local `rmse` exactly (because we `assert np.allclose(...)`).

> [!TIP]
> To register an artifact from a *previous* run (one you didn't just create), open the UI, copy the run id, and from a notebook do:
> ```python
> mv = mlflow.register_model(
>     model_uri=f"runs:/<previous_run_id>/model",
>     name="elastic-api-2",
> )
> ```
> No re-training, no re-logging — you just give an existing artifact a registry name. That's the real super-power of the function form.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-9"></a>

## 9. Tear down

```bash
docker compose down
docker compose down -v
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-10"></a>

## 10. Recap and next chapter

The two-step registry pattern decouples training from registration:

```python
mlflow.sklearn.log_model(lr, "model")                                  # log only
mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", name)     # register later
loaded = mlflow.pyfunc.load_model(f"models:/{mv.name}/{mv.version}")   # serve from registry
```

`mv.version` is the auto-incremented version string, returned for free by `register_model`. No hardcoded `/1`.

Next: **[chapter 20x](./20x-practical-work-15x-mlflow-step-by-step-recap-registering-a-model-trained-outside-mlflow-with-cloudpickle.md)** — register a model that was trained **completely outside** MLflow: load a `.pkl` produced by some other process, then push it into the registry with `mlflow.sklearn.log_model(loaded_model, ..., serialization_format="cloudpickle", registered_model_name=...)`.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 20w — mlflow.register_model post-hoc and load by version</strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>
