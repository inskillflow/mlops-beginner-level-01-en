<a id="top"></a>

# Chapter 20j — Step-by-step recap: launching **multiple runs** in **one** experiment

## Table of Contents

| # | Section |
|---|---|
| 1 | [Objective](#section-1) |
| 2 | [What we add today vs chap 20i](#section-2) |
| 3 | [Why "many runs in one experiment"?](#section-3) |
| 4 | [Two ways to write it: copy-paste vs `for` loop](#section-4) |
| 5 | [Project structure](#section-5) |
| 6 | [The code](#section-6) |
| 7 | [Run it, compare the runs in the UI](#section-7) |
| 8 | [Bonus — nested runs (`nested=True`)](#section-8) |
| 9 | [Mini exercise — your own grid](#section-9) |
| 10 | [Tear down](#section-10) |
| 11 | [Recap and next chapter](#section-11) |

---

<a id="section-1"></a>

## 1. Objective

So far, every chapter logged **one** run per execution of `train.py`. Today we launch **three runs back-to-back inside the same experiment** (`experiment_5`), each with a different `(alpha, l1_ratio)`. The goal is to compare them side-by-side in the MLflow UI to find the best `(alpha, l1_ratio)` for our ElasticNet baseline.

Same Docker stack as before (`mlflow` + `trainer`). Only `train.py` evolves.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. What we add today vs chap 20i

| Diff | What |
|---|---|
| Wrap the per-run logic in a small helper `train_one_run(name, alpha, l1_ratio)` | DRY: train, log params, log metrics, log model, log artifacts. |
| Loop `for cfg in CONFIGS:` and call the helper | One execution → 3 runs in `experiment_5`. |
| Each run gets a meaningful `run_name` (`"run1.1"`, `"run2.1"`, `"run3.1"`) | Easy to find in the UI. |
| Print `last_active_run()` after the loop | Confirms the **last** run, even though we ran several. |

Everything else (env-var URI, multi-service Docker, bulk `log_params`/`log_metrics`, `log_artifacts("data/")`, `set_tags(...)`, `get_artifact_uri()`) is exactly as in chap 20i.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. Why "many runs in one experiment"?

An **experiment** is the project-level folder ("predict wine quality with ElasticNet"). A **run** is one attempt at solving it ("alpha=0.7, l1_ratio=0.7", "alpha=0.4, l1_ratio=0.4"…).

Putting many runs in **the same experiment** is what unlocks the MLflow UI's killer feature:

- The runs table shows them side-by-side.
- The **Compare** button → metric charts (RMSE vs alpha, etc.) and parallel-coordinate plots.
- The search bar filters within the experiment (`metrics.rmse < 0.7`, `params.alpha = "0.4"`, `tags.…`).

Splitting them across N experiments **would lose** that comparison.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. Two ways to write it: copy-paste vs `for` loop

The naïve approach (the one in many tutorials) copy-pastes the same block 3 times, changing only `alpha`, `l1_ratio` and `run_name`. It works, but every bug fix needs to be applied 3× and the file balloons to 200 lines.

A small helper + a `for` loop solves it cleanly:

```python
CONFIGS = [
    ("run1.1", args.alpha, args.l1_ratio),   # CLI-driven
    ("run2.1", 0.9,        0.9),
    ("run3.1", 0.4,        0.4),
]

for name, alpha, l1 in CONFIGS:
    train_one_run(name, alpha, l1)
```

**Same MLflow output, half the lines, one bug to fix.** This is the pattern we use today.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. Project structure

```text
chap20j-mlflow-step-by-step-recap-multiple-runs-in-one-experiment/
├── README.md
├── docker-compose.yml
├── data/
│   └── red-wine-quality.csv
├── mlflow/
│   └── Dockerfile
└── trainer/
    ├── Dockerfile
    ├── requirements.txt
    └── train.py            ← helper + for loop
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-6"></a>

## 6. The code

### 6.1 `trainer/train.py`

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
parser.add_argument("--alpha", type=float, required=False, default=0.7)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.7)
args = parser.parse_args()


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


COMMON_TAGS = {
    "engineering":       "ML platform",
    "release.candidate": "RC1",
    "release.version":   "2.0",
}


def train_one_run(run_name, alpha, l1_ratio, train_x, train_y, test_x, test_y):
    """Train ONE ElasticNet model and log everything to MLflow under `run_name`."""
    mlflow.start_run(run_name=run_name)

    mlflow.set_tags(COMMON_TAGS)

    current = mlflow.active_run()
    print(f"\n>>> Run started: name={current.info.run_name}, id={current.info.run_id}")

    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)
    preds = lr.predict(test_x)
    rmse, mae, r2 = eval_metrics(test_y, preds)

    print(f"  ElasticNet(alpha={alpha:.2f}, l1_ratio={l1_ratio:.2f})  "
          f"RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")

    mlflow.log_params({"alpha": alpha, "l1_ratio": l1_ratio})
    mlflow.log_metrics({"rmse": rmse, "r2": r2, "mae": mae})
    mlflow.sklearn.log_model(lr, "my_new_model_1")
    mlflow.log_artifacts("data/")

    print(f"  Artifact path: {mlflow.get_artifact_uri()}")

    mlflow.end_run()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    )
    print("The set tracking URI is", mlflow.get_tracking_uri())

    exp = mlflow.set_experiment(experiment_name="experiment_5")
    print(f"Name: {exp.name}")
    print(f"Experiment_id: {exp.experiment_id}")

    # Load + split data ONCE (shared across all runs)
    data = pd.read_csv("data/red-wine-quality.csv")
    train, test = train_test_split(data)

    os.makedirs("data", exist_ok=True)
    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv", index=False)

    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    # === The 3 configs ===
    CONFIGS = [
        ("run1.1", args.alpha, args.l1_ratio),     # default = CLI args
        ("run2.1", 0.9,        0.9),
        ("run3.1", 0.4,        0.4),
    ]

    for name, alpha, l1 in CONFIGS:
        train_one_run(name, alpha, l1, train_x, train_y, test_x, test_y)

    # last_active_run() works AFTER the loop too
    run = mlflow.last_active_run()
    print(f"\nRecent active run id   : {run.info.run_id}")
    print(f"Recent active run name : {run.info.run_name}")
```

### 6.2 `docker-compose.yml`

Same skeleton as 20i, only `container_name`s change:

```yaml
services:
  mlflow:
    build: { context: ./mlflow }
    image: mlops/mlflow-recap:latest
    container_name: mlflow-recap-20j
    ports:
      - "5000:5000"
    volumes:
      - mlflow-db:/mlflow/database
      - mlflow-artifacts:/mlflow/mlruns
    networks: [recap-net]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:5000').status==200 else 1)"]
      interval: 10s
      timeout: 5s
      retries: 5

  trainer:
    build: { context: ./trainer }
    image: mlops/trainer-recap:latest
    container_name: trainer-recap-20j
    environment:
      MLFLOW_TRACKING_URI: "http://mlflow:5000"
    volumes:
      - ./data:/code/data
    networks: [recap-net]
    depends_on:
      mlflow: { condition: service_healthy }

volumes:
  mlflow-db:
  mlflow-artifacts:

networks:
  recap-net:
    driver: bridge
```

### 6.3 `mlflow/Dockerfile`, `trainer/Dockerfile`, `trainer/requirements.txt`

Identical to chap 20g/h/i.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-7"></a>

## 7. Run it, compare the runs in the UI

```bash
cd chap20j-mlflow-step-by-step-recap-multiple-runs-in-one-experiment
docker compose up -d --build mlflow
docker compose run --rm trainer --alpha 0.7 --l1_ratio 0.7
```

Trainer's stdout (3 runs in a row):

```text
The set tracking URI is http://mlflow:5000
Name: experiment_5
Experiment_id: 1

>>> Run started: name=run1.1, id=8a4f...d1
  ElasticNet(alpha=0.70, l1_ratio=0.70)  RMSE=0.78...  MAE=0.62...  R2=0.10...
  Artifact path: mlflow-artifacts:/1/8a4f...d1/artifacts

>>> Run started: name=run2.1, id=b1c3...92
  ElasticNet(alpha=0.90, l1_ratio=0.90)  RMSE=0.81...  MAE=0.65...  R2=0.05...
  Artifact path: mlflow-artifacts:/1/b1c3...92/artifacts

>>> Run started: name=run3.1, id=f7e2...10
  ElasticNet(alpha=0.40, l1_ratio=0.40)  RMSE=0.74...  MAE=0.58...  R2=0.20...
  Artifact path: mlflow-artifacts:/1/f7e2...10/artifacts

Recent active run id   : f7e2...10
Recent active run name : run3.1
```

In the UI ([http://localhost:5000](http://localhost:5000)):

1. Open **`experiment_5`** → 3 rows: `run1.1`, `run2.1`, `run3.1`.
2. **Tick all 3** → click **"Compare"** at the top of the table.
3. You get:
   - A **Parameter table** showing `alpha` / `l1_ratio` per run.
   - A **Metric table** showing `rmse`, `mae`, `r2` per run.
   - A **Scatter / Parallel-coordinates** view to spot trends.

> [!TIP]
> In the runs table, click the column header **`metrics.rmse`** to sort by RMSE — the best run jumps to the top.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-8"></a>

## 8. Bonus — nested runs (`nested=True`)

A common pattern: a **parent** run that "owns" several **children** (e.g. one parent per CV fold, children per fold/hyperparam combo). MLflow expresses this with **`nested=True`**:

```python
with mlflow.start_run(run_name="parent_sweep") as parent:
    mlflow.log_param("strategy", "grid_3_configs")
    for name, alpha, l1 in CONFIGS:
        with mlflow.start_run(run_name=name, nested=True):
            # … train + log as usual …
```

In the UI, the parent appears as a single row that expands to reveal its children. Useful when the parent itself logs a summary (e.g. "best RMSE among my children = 0.72"). For this chapter we kept it flat to focus on the loop pattern.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-9"></a>

## 9. Mini exercise — your own grid

Add **two more configs** to `CONFIGS` and re-run the trainer:

```python
CONFIGS = [
    ("run1.1", 0.7, 0.7),
    ("run2.1", 0.9, 0.9),
    ("run3.1", 0.4, 0.4),
    ("run4.1", 0.1, 0.1),   # NEW
    ("run5.1", 0.05, 0.5),  # NEW
]
```

```bash
docker compose run --rm trainer
```

In the UI:

```text
metrics.rmse < 0.75
```

Only the runs that actually beat 0.75 RMSE survive the filter. Now you can spot the winning hyperparams at a glance — that's the whole point of grouping runs in one experiment.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-10"></a>

## 10. Tear down

```bash
docker compose down       # keep volumes
docker compose down -v    # wipe everything
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-11"></a>

## 11. Recap and next chapter

You now know how to launch **N runs in one experiment** with a clean helper + `for` loop, plus the optional `nested=True` parent/child pattern.

Next: **[chapter 20k](./20k-practical-work-15k-mlflow-step-by-step-recap-multiple-experiments-comparing-elasticnet-ridge-lasso.md)** — same idea, but **across multiple experiments**, one per algorithm (`exp_multi_EL`, `exp_multi_Ridge`, `exp_multi_Lasso`), so we can compare **algorithms**, not just hyperparams.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 20j — multiple runs in one experiment</strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>
