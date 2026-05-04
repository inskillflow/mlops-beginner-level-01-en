<a id="top"></a>

# Chapter 20v — Step-by-step recap: combining `mlflow.sklearn.log_model` with a manual `pickle.dump` and `mlflow.log_artifact("elastic-net-regression.pkl")`

## Table of Contents

| # | Section |
|---|---|
| 1 | [Objective](#section-1) |
| 2 | [What we add today vs chap 20u](#section-2) |
| 3 | [Two ways of getting a model into the run — what's the difference?](#section-3) |
| 4 | [The `get_path_type` helper — small but useful in production](#section-4) |
| 5 | [`pickle` vs `joblib` — quick reminder](#section-5) |
| 6 | [Project structure](#section-6) |
| 7 | [The code](#section-7) |
| 8 | [Run it, browse the artifacts](#section-8) |
| 9 | [Loading the side-car pickle from another script](#section-9) |
| 10 | [Tear down](#section-10) |
| 11 | [Recap and series wrap-up](#section-11) |

---

<a id="section-1"></a>

## 1. Objective

Today's chapter is about a very common, very pragmatic pattern: **logging the same model in two formats** in the same run.

1. The **canonical MLflow flavour** with `mlflow.sklearn.log_model(lr, "model")` — gives you a directory with `MLmodel`, `model.pkl`, `conda.yaml`, `requirements.txt`, signature, etc. Everything `mlflow.pyfunc.load_model(...)` understands.
2. A **plain side-car `.pkl`** with `pickle.dump(lr, ...)` + `mlflow.log_artifact("elastic-net-regression.pkl")` — a single file readable by `pickle.load()` from **any** Python process, no MLflow required.

Why both? Because not every consumer of your model wants (or can) install MLflow. A legacy notebook, a small embedded script, a colleague who only knows `pandas` + `sklearn`… they all just want a `.pkl` they can `pickle.load()`. Having both in the same run means you cover both audiences with one training script.

We'll also introduce a tiny utility — **`get_path_type(path)`** — that classifies a path as a file, a directory, or invalid. Useful for production scripts that consume external paths (data files, model directories…).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. What we add today vs chap 20u

| Diff | What |
|---|---|
| `import pickle` | Standard library serializer. |
| Add `get_path_type(path)` helper | Returns `"file"`, `"directory"`, or `"not a valid path"`. |
| `pickle.dump(lr, open(filename, "wb"))` | Save the trained model to disk as a plain `.pkl`. |
| `mlflow.log_artifact(filename)` | Upload that file to the run's artifact store under the root. |
| Drop `registered_model_name` (chap 20u's lesson) | Today's focus is the artifact pattern. |
| New experiment name `experiment_elastic_net_mlflow` | Matches the user's snippet. |

Everything else (Docker stack, `MLFLOW_TRACKING_URI` env var, `mlflow.set_tags`, `mlflow.log_params/metrics`) is identical.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. Two ways of getting a model into the run — what's the difference?

```python
mlflow.sklearn.log_model(lr, "model")                  # canonical flavour
mlflow.log_artifact("elastic-net-regression.pkl")      # side-car pickle
```

Resulting layout in the run's artifacts:

```text
mlruns/<exp_id>/<run_id>/artifacts/
├── elastic-net-regression.pkl     ← log_artifact: a single .pkl, anyone can pickle.load it
└── model/                          ← log_model:    the MLflow flavour
    ├── MLmodel                          (declares the flavour, signature, conda env)
    ├── conda.yaml
    ├── requirements.txt
    ├── python_env.yaml
    └── model.pkl                        (same model serialized inside)
```

Side-by-side comparison:

| Capability | `log_model("model")` | `log_artifact("...pkl")` |
|---|---|---|
| Loadable with `mlflow.pyfunc.load_model("runs:/<id>/model")` | Yes | No |
| Loadable with `mlflow.pyfunc.load_model("runs:/<id>/elastic-net-regression.pkl")` | — | No |
| Loadable with plain `pickle.load(open("...pkl","rb"))` | Yes (the inner `model.pkl`) | Yes |
| Carries signature / input example | Yes | No |
| Carries conda + pip requirements | Yes | No |
| Eligible for the Model Registry (`registered_model_name=`) | Yes | No |
| Disk size for ElasticNet | ~6 KB folder | ~1 KB file |

**Rule of thumb**: always use `log_model` (it's the canonical, future-proof entry). Add `log_artifact("...pkl")` **only** when a downstream consumer specifically needs a bare pickle.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. The `get_path_type` helper — small but useful in production

```python
def get_path_type(path):
    if os.path.isabs(path) and os.path.exists(path):
        if os.path.isdir(path):
            return "directory"
        return "file"
    return "not a valid path"
```

Why bother? Production training jobs often receive paths via env vars or CLI args (path to a CSV, path to a previous model directory, path to a JSON config…). Doing **one explicit classification + log** at the start of the script saves hours of debugging when the wrong thing is mounted into the container.

> [!NOTE]
> The user's original snippet had a small bug: it returned `"file"` when the path was a directory. Our version flips the labels so `isdir → "directory"` and `isfile → "file"`, which matches what the function name suggests.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. `pickle` vs `joblib` — quick reminder

Both `pickle.dump(lr, f)` and `joblib.dump(lr, f)` produce a serialized scikit-learn model. Why pick one?

| | `pickle` (stdlib) | `joblib` |
|---|---|---|
| Bundled with Python | Yes | No (extra dependency) |
| Faster on large NumPy arrays | No | Yes (memory-mapping, parallel I/O) |
| Smaller for small models | Slightly | — |
| Format compatibility across versions | Slightly more brittle | More robust |
| Used internally by `mlflow.sklearn.log_model` | — | Yes (default) |

For a tiny ElasticNet (a few floats), `pickle` is perfectly fine and avoids one dependency. For large random forests or neural nets, prefer `joblib`. We use `pickle` here because the user's snippet does.

> [!IMPORTANT]
> A `.pkl` is **executable code on load** — `pickle.load` runs `__reduce__` methods. NEVER `pickle.load` a file you didn't produce yourself. Same caveat applies to `joblib.load`.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-6"></a>

## 6. Project structure

```text
chap20v-mlflow-step-by-step-recap-log-model-plus-pickle-dump-and-log-artifact/
├── README.md
├── docker-compose.yml
├── data/
│   └── red-wine-quality.csv
├── mlflow/
│   └── Dockerfile
└── trainer/
    ├── Dockerfile
    ├── requirements.txt
    └── train.py
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
import pickle
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
parser.add_argument("--alpha",    type=float, required=False, default=0.4)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.4)
args = parser.parse_args()


def get_path_type(path):
    """Classify a filesystem path. Useful at the top of a training script
    when you want to verify what was mounted into the container."""
    if os.path.isabs(path) and os.path.exists(path):
        if os.path.isdir(path):
            return "directory"
        return "file"
    return "not a valid path"


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

    data_csv = os.path.abspath("data/red-wine-quality.csv")
    print(f"data/red-wine-quality.csv -> {get_path_type(data_csv)}")
    print(f"/code -> {get_path_type('/code')}")

    exp = mlflow.set_experiment(experiment_name="experiment_elastic_net_mlflow")
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

    # ===== 1. Canonical MLflow flavour =====
    model_info = mlflow.sklearn.log_model(lr, artifact_path="model")
    print("Model URI (canonical) :", model_info.model_uri)

    # ===== 2. Side-car pickle =====
    pickle_filename = "elastic-net-regression.pkl"
    with open(pickle_filename, "wb") as f:
        pickle.dump(lr, f)
    mlflow.log_artifact(pickle_filename)
    print(f"Side-car pickle logged: {pickle_filename}")

    mlflow.end_run()

    run = mlflow.last_active_run()
    print(f"Active run id     : {run.info.run_id}")
```

### 7.2 `trainer/requirements.txt`

```text
mlflow==2.16.2
scikit-learn==1.5.2
pandas==2.2.3
numpy==2.1.1
```

(`pickle` is in stdlib, no entry needed.)

### 7.3 `docker-compose.yml`, `mlflow/Dockerfile`, `trainer/Dockerfile`

Identical to chap 20u.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-8"></a>

## 8. Run it, browse the artifacts

```bash
cd chap20v-mlflow-step-by-step-recap-log-model-plus-pickle-dump-and-log-artifact
docker compose up -d --build mlflow
docker compose run --rm trainer --alpha 0.4 --l1_ratio 0.4
```

Trainer's stdout (excerpt):

```text
data/red-wine-quality.csv -> file
/code -> directory
ElasticNet (alpha=0.4, l1_ratio=0.4): RMSE=0.7785  MAE=0.6223  R2=0.1054
Model URI (canonical) : runs:/<run_id>/model
Side-car pickle logged: elastic-net-regression.pkl
```

In the UI ([http://localhost:5000](http://localhost:5000)) → **`experiment_elastic_net_mlflow`** → latest run → **Artifacts** tab:

```text
artifacts/
├── elastic-net-regression.pkl       ← the side-car (just a binary)
└── model/                            ← the canonical flavour
    ├── MLmodel
    ├── conda.yaml
    ├── python_env.yaml
    ├── requirements.txt
    └── model.pkl
```

Click `elastic-net-regression.pkl` → MLflow lets you download it directly. No flavour metadata, just bytes — exactly what a `pickle.load`-based consumer needs.
Click `model/MLmodel` → human-readable YAML showing the flavour, signature, etc.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-9"></a>

## 9. Loading the side-car pickle from another script

The whole point of the side-car is that you can load it without MLflow:

```python
import pickle, urllib.request, pandas as pd

# Either via MLflow's REST API to the artifact:
url = "http://localhost:5000/get-artifact?path=elastic-net-regression.pkl&run_id=<RUN_ID>"
with urllib.request.urlopen(url) as r:
    lr = pickle.load(r)

# Or via mlflow.artifacts.download_artifacts (still uses MLflow but only as a fetch tool):
import mlflow.artifacts
local = mlflow.artifacts.download_artifacts(
    f"runs:/<RUN_ID>/elastic-net-regression.pkl",
)
with open(local, "rb") as f:
    lr = pickle.load(f)

# Predict — no mlflow.pyfunc, no model wrapper, just sklearn + pickle:
sample = pd.DataFrame([{...}])
print(lr.predict(sample))
```

Compare to loading the canonical flavour:

```python
import mlflow.pyfunc
m = mlflow.pyfunc.load_model("runs:/<RUN_ID>/model")
print(m.predict(sample))
```

Both work. Same model, two doors.

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

## 11. Recap and series wrap-up

Three lines for the dual-format pattern:

```python
mlflow.sklearn.log_model(lr, "model")                # canonical flavour
pickle.dump(lr, open("elastic-net-regression.pkl", "wb"))
mlflow.log_artifact("elastic-net-regression.pkl")    # side-car for non-MLflow consumers
```

This closes the **20a → 20v** recap series. Coverage map:

| Block | Chapters | Theme |
|---|---|---|
| Setup | 20a, 20b | `set_tracking_uri`, `set_experiment`, `start_run`, basic params/metrics |
| Pipeline | 20c | Full ElasticNet on red wine quality |
| Docker plumbing | 20d, 20e | `MLFLOW_TRACKING_URI` env var, debugging the connection |
| Experiments & runs | 20f, 20g, 20i, 20j, 20k | `create_experiment`, `active_run`, `set_tags`, multiple runs/experiments |
| Artifacts & bulk | 20h | `log_artifacts(folder)`, `log_params/metrics(dict)` |
| Autolog | 20l | `mlflow.sklearn.autolog` |
| Storage | 20m | PostgreSQL backend store + S3 artifact storage |
| Signatures | 20n, 20o | `infer_signature` and manual `Schema/ColSpec` |
| Pyfunc + custom envs | 20p | `pyfunc.log_model` with wrapper, `joblib`, `conda_env` |
| Loading & evaluating | 20q, 20r, 20s | `pyfunc.load_model`, `mlflow.evaluate` (default + custom) |
| Decision gates | 20t | `validation_thresholds` + `baseline_model` |
| Registry | 20u | `registered_model_name="elasticnet-api"` |
| Dual-format | 20v | `log_model` + `pickle.dump` + `log_artifact` |

The next major step (chapter 21+) leaves the trainer side and moves to **deployment**: serving the registered pyfunc model through FastAPI and consuming it from a Streamlit frontend.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 20v — log_model + pickle.dump + log_artifact</strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>
