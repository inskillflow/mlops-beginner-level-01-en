<a id="top"></a>

# Chapter 20x — Step-by-step recap: registering a model that was trained **outside** MLflow (load `.pkl` → `mlflow.sklearn.log_model(serialization_format="cloudpickle", registered_model_name=...)`)

## Table of Contents

| # | Section |
|---|---|
| 1 | [Objective](#section-1) |
| 2 | [What we add today vs chap 20w](#section-2) |
| 3 | [Real-world scenario: importing a "foreign" model](#section-3) |
| 4 | [Two services in the same compose: `pretrainer` then `registrar`](#section-4) |
| 5 | [`serialization_format="cloudpickle"` — what does it change?](#section-5) |
| 6 | [Project structure](#section-6) |
| 7 | [The code](#section-7) |
| 8 | [Run it: produce the pickle, then register it](#section-8) |
| 9 | [Tear down](#section-9) |
| 10 | [Recap and next chapter](#section-10) |

---

<a id="section-1"></a>

## 1. Objective

Until now every registered model came from a fresh MLflow training run. In real life that's the exception — most teams inherit at least one **pre-existing** model produced **outside** of MLflow:

- A legacy `.pkl` from a notebook nobody dares re-run.
- A model handed over by a partner / vendor / AutoML platform.
- A model trained in a Spark / SageMaker / Vertex job that doesn't speak MLflow.

The pattern is always the same:

```python
loaded = pickle.load(open("foreign_model.pkl", "rb"))      # bring it in memory
mlflow.sklearn.log_model(                                  # push it into MLflow
    sk_model=loaded,
    artifact_path="model",
    serialization_format="cloudpickle",
    registered_model_name="elastic-net-regression-outside-mlflow",
)
```

The model object isn't *trained* by MLflow but it **lands in the registry exactly like a freshly trained one** — same versioning, same `models:/<name>/<version>` URI, same downstream tooling.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. What we add today vs chap 20w

| Diff | What |
|---|---|
| Two-step Docker workflow | First service produces `elastic-net-regression.pkl` outside any MLflow run; second service picks it up and registers it. |
| `pretrainer/` service | Plain sklearn training, dumps a `.pkl` to a shared volume (no MLflow at all). |
| `registrar/` service | `pickle.load` + `mlflow.sklearn.log_model(..., serialization_format="cloudpickle", registered_model_name=...)`. |
| `serialization_format="cloudpickle"` | Explicit choice — covered below. |
| New experiment name `experiment_register_outside` | Matches the user's snippet. |

There is no training inside the registrar — just import + register.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. Real-world scenario: importing a "foreign" model

Step-by-step:

1. **Outside MLflow** (the `pretrainer` service in our setup): some legacy or third-party process produces `elastic-net-regression.pkl` and drops it on a shared filesystem (S3, NFS, Docker volume…). It knows nothing of experiments, runs, parameters.

2. **Inside MLflow** (the `registrar` service): a small bridge script runs once. It:
   - Loads the `.pkl` with the standard library `pickle`.
   - Connects to the MLflow tracking server.
   - Opens a *cosmetic* run (just to give the artifact a place to live).
   - Calls `mlflow.sklearn.log_model(...)` with `registered_model_name=` to push the artifact into the registry under a stable name.

3. From now on **every consumer** treats the model exactly like a natively-trained one:

   ```python
   m = mlflow.pyfunc.load_model("models:/elastic-net-regression-outside-mlflow/1")
   ```

> [!IMPORTANT]
> The bridge run will have **no params and no metrics** — those would be lies. Add tags instead, e.g. `mlflow.set_tag("imported_from", "vendor X / 2026-04 batch")`. That way the UI clearly shows "this run is an import, not a training".

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. Two services in the same compose: `pretrainer` then `registrar`

We model the real-world separation directly in `docker-compose.yml`:

```yaml
services:
  mlflow:        # tracking server (SQLite backend)
    ...

  pretrainer:    # NO MLflow involvement at all
    image: mlops/pretrainer:latest
    volumes:
      - shared:/shared      # writes /shared/elastic-net-regression.pkl

  registrar:     # bridge: load pickle -> MLflow registry
    image: mlops/registrar:latest
    environment:
      MLFLOW_TRACKING_URI: "http://mlflow:5000"
    volumes:
      - shared:/shared      # reads /shared/elastic-net-regression.pkl

volumes:
  shared:        # named volume the two services exchange the pickle through
```

Workflow:

```bash
docker compose up -d --build mlflow
docker compose run --rm pretrainer    # produces /shared/elastic-net-regression.pkl
docker compose run --rm registrar     # loads it and registers it
```

Each service has a different image and a totally different `requirements.txt` — exactly like a real "external producer" + "MLflow bridge".

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. `serialization_format="cloudpickle"` — what does it change?

`mlflow.sklearn.log_model` accepts two values:

| Value | What MLflow uses to write the model on disk |
|---|---|
| `"cloudpickle"` (default) | `cloudpickle.dump(model, ...)` — handles closures, lambdas, locally-defined classes. |
| `"pickle"` | `pickle.dump(model, ...)` — slightly smaller, fails on closures / locally-defined classes. |

For a vanilla `sklearn.linear_model.ElasticNet` they produce nearly identical files. The reason **cloudpickle is the default** is robustness:

- Models that include custom transformers or wrappers defined inside a function will only survive `cloudpickle`, not `pickle`.
- Cross-session / cross-machine compatibility is much better with cloudpickle.

Passing `serialization_format="cloudpickle"` here is **explicit-is-better-than-implicit** — you're stating in the code that this artifact will be re-read by potentially different Python interpreters.

> [!NOTE]
> Whatever value you pick, `mlflow.pyfunc.load_model` reads the right format automatically (it inspects the `MLmodel` YAML).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-6"></a>

## 6. Project structure

```text
chap20x-mlflow-step-by-step-recap-registering-a-model-trained-outside-mlflow-with-cloudpickle/
├── README.md
├── docker-compose.yml
├── data/
│   └── red-wine-quality.csv
├── mlflow/
│   └── Dockerfile
├── pretrainer/                  ← knows NOTHING about MLflow
│   ├── Dockerfile
│   ├── requirements.txt
│   └── train_outside_mlflow.py  ← writes /shared/elastic-net-regression.pkl
└── registrar/                   ← bridge to MLflow registry
    ├── Dockerfile
    ├── requirements.txt
    └── register_external.py     ← pickle.load + log_model + registry
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-7"></a>

## 7. The code

### 7.1 `pretrainer/train_outside_mlflow.py`

A perfectly normal sklearn script. No `import mlflow` anywhere.

```python
"""Trains an ElasticNet and dumps it to /shared/elastic-net-regression.pkl.
Runs OUTSIDE any MLflow context. Could just as well be a SageMaker job,
a notebook a colleague sent you, or a vendor's binary blob."""

import argparse
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--alpha",    type=float, default=0.4)
parser.add_argument("--l1_ratio", type=float, default=0.4)
args = parser.parse_args()


def eval_metrics(actual, pred):
    return (
        np.sqrt(mean_squared_error(actual, pred)),
        mean_absolute_error(actual, pred),
        r2_score(actual, pred),
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    data = pd.read_csv("data/red-wine-quality.csv")
    train, test = train_test_split(data, test_size=0.25)

    train_x = train.drop(["quality"], axis=1)
    test_x  = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y  = test[["quality"]]

    lr = ElasticNet(alpha=args.alpha, l1_ratio=args.l1_ratio, random_state=42)
    lr.fit(train_x, train_y)

    rmse, mae, r2 = eval_metrics(test_y, lr.predict(test_x))
    print(f"[pretrainer] ElasticNet trained: RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")

    out_dir = "/shared"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "elastic-net-regression.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(lr, f)
    print(f"[pretrainer] Wrote {out_path}")
```

### 7.2 `registrar/register_external.py`

The bridge: ~15 useful lines.

```python
"""Loads a model trained outside MLflow and registers it in the MLflow registry."""

import os
import pickle

import mlflow
import mlflow.sklearn

REGISTERED_NAME = "elastic-net-regression-outside-mlflow"
PICKLE_PATH = "/shared/elastic-net-regression.pkl"


if __name__ == "__main__":
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    )
    print("[registrar] Tracking URI:", mlflow.get_tracking_uri())

    if not os.path.exists(PICKLE_PATH):
        raise SystemExit(
            f"[registrar] {PICKLE_PATH} not found. "
            f"Run `docker compose run --rm pretrainer` first."
        )

    with open(PICKLE_PATH, "rb") as f:
        loaded_model = pickle.load(f)
    print(f"[registrar] Loaded model from {PICKLE_PATH}: {type(loaded_model).__name__}")

    exp = mlflow.set_experiment(experiment_name="experiment_register_outside")
    print(f"[registrar] Experiment: {exp.name} (id={exp.experiment_id})")

    with mlflow.start_run() as run:
        # Document the import - NO params/metrics (they would be lies)
        mlflow.set_tags({
            "imported": "true",
            "source":   "external_pickle",
            "filename": os.path.basename(PICKLE_PATH),
        })

        model_info = mlflow.sklearn.log_model(
            sk_model=loaded_model,
            artifact_path="model",
            serialization_format="cloudpickle",
            registered_model_name=REGISTERED_NAME,
        )

        print(f"[registrar] Logged model URI: {model_info.model_uri}")
        print(f"[registrar] Registered as     : {REGISTERED_NAME!r}")
        print(f"[registrar] Run id            : {run.info.run_id}")
```

### 7.3 `docker-compose.yml`

```yaml
services:
  mlflow:
    build: { context: ./mlflow }
    ports: ["5000:5000"]
    volumes:
      - mlflow-db:/mlflow/database
      - mlflow-artifacts:/mlflow/mlruns
    networks: [recap-net]
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:5000').status==200 else 1)"]
      interval: 10s
      timeout: 5s
      retries: 5

  pretrainer:
    build: { context: ./pretrainer }
    container_name: pretrainer-recap-20x
    volumes:
      - ./data:/code/data
      - shared:/shared

  registrar:
    build: { context: ./registrar }
    container_name: registrar-recap-20x
    environment:
      MLFLOW_TRACKING_URI: "http://mlflow:5000"
    volumes:
      - shared:/shared
    networks: [recap-net]
    depends_on: { mlflow: { condition: service_healthy } }

volumes:
  mlflow-db:
  mlflow-artifacts:
  shared:        # the bridge between pretrainer and registrar

networks:
  recap-net:
    driver: bridge
```

### 7.4 Requirements files

`pretrainer/requirements.txt` — **no MLflow**:

```text
scikit-learn==1.5.2
pandas==2.2.3
numpy==2.1.1
```

`registrar/requirements.txt` — adds MLflow + cloudpickle:

```text
mlflow==2.16.2
scikit-learn==1.5.2
pandas==2.2.3
numpy==2.1.1
cloudpickle==3.0.0
```

### 7.5 Dockerfiles (one per service)

```dockerfile
# pretrainer/Dockerfile
FROM python:3.12-slim
WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY train_outside_mlflow.py .
ENTRYPOINT ["python", "train_outside_mlflow.py"]
```

```dockerfile
# registrar/Dockerfile
FROM python:3.12-slim
WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY register_external.py .
ENTRYPOINT ["python", "register_external.py"]
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-8"></a>

## 8. Run it: produce the pickle, then register it

```bash
cd chap20x-mlflow-step-by-step-recap-registering-a-model-trained-outside-mlflow-with-cloudpickle
docker compose up -d --build mlflow

docker compose run --rm pretrainer --alpha 0.4 --l1_ratio 0.4
# [pretrainer] ElasticNet trained: RMSE=0.7785 MAE=0.6223 R2=0.1054
# [pretrainer] Wrote /shared/elastic-net-regression.pkl

docker compose run --rm registrar
# [registrar] Tracking URI: http://mlflow:5000
# [registrar] Loaded model from /shared/elastic-net-regression.pkl: ElasticNet
# [registrar] Experiment: experiment_register_outside (id=N)
# Successfully registered model 'elastic-net-regression-outside-mlflow'.
# Created version '1' of model 'elastic-net-regression-outside-mlflow'.
# [registrar] Logged model URI: runs:/<run_id>/model
# [registrar] Registered as     : 'elastic-net-regression-outside-mlflow'
# [registrar] Run id            : <run_id>
```

In the UI ([http://localhost:5000](http://localhost:5000)):

- **Experiments tab** → `experiment_register_outside` → 1 run with **no params, no metrics**, but tags `imported=true / source=external_pickle / filename=elastic-net-regression.pkl`. That immediately tells anyone browsing that this is an import.
- **Models tab** → `elastic-net-regression-outside-mlflow` → Version 1, sourced from `runs:/<run_id>/model`.

You can re-run `docker compose run --rm pretrainer` with different hyperparameters and `docker compose run --rm registrar` again — Version 2 will be added, etc.

### Loading the imported model from anywhere

```python
import mlflow.pyfunc, pandas as pd
m = mlflow.pyfunc.load_model("models:/elastic-net-regression-outside-mlflow/1")
print(m.predict(pd.DataFrame([{
    "fixed acidity": 7.2, "volatile acidity": 0.35, "citric acid": 0.45,
    "residual sugar": 8.5, "chlorides": 0.045, "free sulfur dioxide": 30.0,
    "total sulfur dioxide": 120.0, "density": 0.997, "pH": 3.2,
    "sulphates": 0.65, "alcohol": 9.2,
}])))
```

Indistinguishable from a model that was originally trained with MLflow. That's the point.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-9"></a>

## 9. Tear down

```bash
docker compose down
docker compose down -v        # -v also drops the `shared` volume holding the .pkl
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-10"></a>

## 10. Recap and next chapter

Three lines turn any sklearn `.pkl` into a fully-fledged registry entry:

```python
loaded = pickle.load(open("foreign.pkl", "rb"))
with mlflow.start_run():
    mlflow.sklearn.log_model(
        sk_model=loaded,
        artifact_path="model",
        serialization_format="cloudpickle",
        registered_model_name="my-foreign-model",
    )
```

`pretrainer` + `registrar` services in `docker-compose.yml` mirror the real-world separation between the model's **producer** and the **MLflow bridge**.

Next: **[chapter 20y](./20y-practical-work-15y-mlflow-step-by-step-recap-with-start-run-context-manager-and-main-function.md)** — replace the imperative `mlflow.start_run() / mlflow.end_run()` style with the idiomatic Pythonic context manager `with mlflow.start_run(experiment_id=exp.experiment_id):`, all wrapped in a clean `main()` function.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 20x — registering a model trained outside MLflow</strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>
