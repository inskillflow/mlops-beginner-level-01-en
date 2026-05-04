<a id="top"></a>

# Chapter 20c — Step-by-step recap: a first ElasticNet pipeline on red-wine-quality

## Table of Contents

| # | Section |
|---|---|
| 1 | [Objective](#section-1) |
| 2 | [What we add today vs chap 20b](#section-2) |
| 3 | [Prerequisite](#section-3) |
| 4 | [The dataset](#section-4) |
| 5 | [Project structure](#section-5) |
| 6 | [The code](#section-6) |
| 7 | [Run it, see the run + the model in the UI](#section-7) |
| 8 | [Mini exercise — a tiny grid search](#section-8) |
| 9 | [Tear down](#section-9) |
| 10 | [Recap and next chapter](#section-10) |

---

<a id="section-1"></a>

## 1. Objective

Time to graduate from the fake `accuracy = 0.92` of chapters 20a and 20b. Today we run a **real machine-learning pipeline**:

- Read a CSV (`red-wine-quality.csv`).
- Split into train / test.
- Train a scikit-learn `ElasticNet`.
- Compute three regression metrics (`rmse`, `mae`, `r2`).
- Log the **hyperparameters**, the **metrics**, **and the trained model** to MLflow with `mlflow.sklearn.log_model(...)`.

That's the same shape every later chapter will keep.

> [!IMPORTANT]
> Today's script intentionally calls `mlflow.set_tracking_uri("http://localhost:5000")` first. **Forget this line and nothing lands in the dockerized UI** — you'd find your runs in a local `./mlruns/` folder instead. We saw this exact failure mode in [chapter 20b](./20b-practical-work-15b-mlflow-step-by-step-recap-printing-the-tracking-uri.md), section 5.3.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. What we add today vs chap 20b

Three new things on top of [chap 20b](./20b-practical-work-15b-mlflow-step-by-step-recap-printing-the-tracking-uri.md):

| New | Why |
|---|---|
| `argparse` | Read `--alpha` and `--l1_ratio` from the command line so we can do many runs cheaply. |
| `pandas` + `scikit-learn` (`ElasticNet`, `train_test_split`, `mean_squared_error`, …) | Real dataset, real model, real metrics. |
| `mlflow.sklearn.log_model(lr, "mymodel")` | Persist the **trained model** (not just numbers) as an artifact attached to the run. |

Same MLflow control flow you already know: `set_tracking_uri` → `set_experiment` → `start_run` → log the params, metrics, and model.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. Prerequisite

1. **Docker Desktop** (cf. [chapter 06 §1-3](./06-practical-work-2-installing-docker-desktop-and-running-mlflow-fastapi-streamlit-with-docker-compose.md) if needed).
2. **Host venv** with these packages — `mlflow==2.16.2`, `scikit-learn==1.5.2`, `pandas==2.2.3`, `numpy==2.1.1`. A `requirements.txt` is shipped with the project for convenience.

```bash
# Linux / macOS
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```powershell
# Windows PowerShell
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. The dataset

We use the **UCI Red Wine Quality** dataset — 1 599 rows, 11 physico-chemical features, target `quality` (integer 3 → 8).

The CSV is shipped inside this chapter's project at `data/red-wine-quality.csv`, comma-separated, with this header:

```text
"fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"
7.4,0.7,0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4,5
7.8,0.88,0,2.6,0.098,25,67,0.9968,3.2,0.68,9.8,5
...
```

> [!NOTE]
> The UCI version of this dataset ships with `;` as separator. The bundled copy has been converted to `,` so that the simple `pd.read_csv("data/red-wine-quality.csv")` call works without any extra options. Same numbers, friendlier format.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. Project structure

```text
chap20c-mlflow-step-by-step-recap-elasticnet-on-red-wine-quality/
├── README.md
├── requirements.txt
├── docker-compose.yml
├── mlflow/
│   └── Dockerfile
├── data/
│   └── red-wine-quality.csv
└── train.py
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-6"></a>

## 6. The code

### 6.1 `requirements.txt` (host venv)

```text
mlflow==2.16.2
scikit-learn==1.5.2
pandas==2.2.3
numpy==2.1.1
```

### 6.2 `mlflow/Dockerfile` and `docker-compose.yml`

Identical to chapters 20a / 20b. Same single-service MLflow server on port 5000 with two persistent volumes.

### 6.3 `train.py` — the real ML pipeline

```python
import argparse
import logging
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

# Read CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=False, default=0.7)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.7)
args = parser.parse_args()


# Evaluation function
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # === DON'T FORGET: point at the running MLflow server ===
    mlflow.set_tracking_uri("http://localhost:5000")
    print("Tracking URI:", mlflow.get_tracking_uri())

    # Read the wine-quality CSV (comma-separated)
    data = pd.read_csv("data/red-wine-quality.csv")

    # Split (default 0.75 / 0.25)
    train, test = train_test_split(data)

    # Predicted column is "quality" - integer in [3, 8]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = args.alpha
    l1_ratio = args.l1_ratio

    exp = mlflow.set_experiment(experiment_name="experiment_1")

    with mlflow.start_run(experiment_id=exp.experiment_id):
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE:  %s" % mae)
        print("  R2:   %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(lr, "mymodel")
```

What was added compared to your original snippet:
- `mlflow.set_tracking_uri("http://localhost:5000")` and `print(mlflow.get_tracking_uri())` — the two lines from chapters 20a and 20b that we **never** want to forget again.
- `data/red-wine-quality.csv` is shipped with the project, so the script runs out of the box.
- The redundant `data.to_csv("data/red-wine-quality.csv", index=False)` of the original snippet has been dropped — we just read the file once.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-7"></a>

## 7. Run it, see the run + the model in the UI

### 7.1 Start the MLflow server

```bash
cd chap20c-mlflow-step-by-step-recap-elasticnet-on-red-wine-quality
docker compose up --build
```

Open [http://localhost:5000](http://localhost:5000).

### 7.2 Run the script

In another terminal (host venv activated):

```bash
python train.py --alpha 0.7 --l1_ratio 0.7
```

Output:

```text
Tracking URI: http://localhost:5000
Elasticnet model (alpha=0.700000, l1_ratio=0.700000):
  RMSE: 0.7836...
  MAE:  0.6260...
  R2:   0.1063...
```

### 7.3 Visualize in the UI

Refresh [http://localhost:5000](http://localhost:5000):

1. Click experiment **`experiment_1`** in the left sidebar.
2. The run appears with the params (`alpha`, `l1_ratio`) and the metrics (`rmse`, `mae`, `r2`).
3. Click the run, then **Artifacts** → folder **`mymodel/`**:
   ```text
   mymodel/
   ├── MLmodel              ← MLflow descriptor
   ├── conda.yaml           ← env that recreates the model
   ├── python_env.yaml      ← pip-only env
   ├── requirements.txt
   └── model.pkl            ← the actual sklearn pickle
   ```

That's a **deployable model artifact** — chapters 16 and 17 of this course showed how to load it back and serve it.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-8"></a>

## 8. Mini exercise — a tiny grid search

Run the script three times with different hyperparameters:

```bash
python train.py --alpha 0.1 --l1_ratio 0.1
python train.py --alpha 0.5 --l1_ratio 0.5
python train.py --alpha 0.9 --l1_ratio 0.9
```

In the UI, switch to the **Table** view of `experiment_1` and sort by `metrics.rmse` ascending — you've just done your first hyperparameter search, fully tracked.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-9"></a>

## 9. Tear down

```bash
docker compose down       # keep all your runs
docker compose down -v    # wipe everything (DB + artifacts)
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-10"></a>

## 10. Recap and next chapter

You've now closed the loop on the **MLflow basics** in three small steps:

| Chapter | New thing |
|---|---|
| 20a | `set_tracking_uri` + `set_experiment` + `start_run` + `log_param` + `log_metric` |
| 20b | `print(mlflow.get_tracking_uri())` — the safety net |
| **20c** | A real `ElasticNet` pipeline + `mlflow.sklearn.log_model(...)` |

Next chapters of the recap (20d, 20e, …) — **to be added** — will continue layering one concept at a time: tags, autolog, signature, registry, etc.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 20c — First ElasticNet pipeline on red-wine-quality</strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>
