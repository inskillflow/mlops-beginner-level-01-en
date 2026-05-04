<a id="top"></a>

# Chapter 20k — Step-by-step recap: launching **multiple experiments** (ElasticNet vs Ridge vs Lasso)

## Table of Contents

| # | Section |
|---|---|
| 1 | [Objective](#section-1) |
| 2 | [What we add today vs chap 20j](#section-2) |
| 3 | [When to split into experiments vs runs](#section-3) |
| 4 | [Building 3 experiments × 3 runs in one script](#section-4) |
| 5 | [Project structure](#section-5) |
| 6 | [The code](#section-6) |
| 7 | [Run it, compare experiments in the UI](#section-7) |
| 8 | [Bonus — switching experiments without losing the best run](#section-8) |
| 9 | [Mini exercise — add a 4th algorithm](#section-9) |
| 10 | [Tear down](#section-10) |
| 11 | [Recap and next chapter](#section-11) |

---

<a id="section-1"></a>

## 1. Objective

Chap 20j compared **hyperparameters** of one algorithm (ElasticNet) by putting many runs in a single experiment.

Today we go one level up: we want to compare **algorithms**. Three different regressors deserve three different experiments:

- `exp_multi_EL` → ElasticNet, 3 runs (`alpha`, `l1_ratio`)
- `exp_multi_Ridge` → Ridge, 3 runs (`alpha` only)
- `exp_multi_Lasso` → Lasso, 3 runs (`alpha` only)

End result: **9 runs, 3 experiments, one execution of `train.py`**.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. What we add today vs chap 20j

| Diff | What |
|---|---|
| Helper signature accepts a **factory** for the model | One helper trains ElasticNet, Ridge or Lasso depending on the lambda passed in. |
| `mlflow.set_experiment(...)` is called **once per algorithm** | Switches the "current experiment". |
| A list `EXPERIMENTS` of `(exp_name, model_factory, alphas)` drives 3 outer loops | Add a 4th algorithm → 1 line. |
| Lasso & Ridge log only `alpha`; ElasticNet logs `alpha` + `l1_ratio` | The factory carries the right hyperparams. |

Everything else (env-var URI, multi-service Docker, `set_tags`, `log_artifacts("data/")`, `last_active_run`) is identical to chap 20j.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. When to split into experiments vs runs

| Situation | Use… | Why |
|---|---|---|
| Same algorithm, different hyperparams | **Many runs in 1 experiment** | The MLflow Compare view groups them naturally. |
| Different algorithm families | **Different experiments** | They don't share the same hyperparam columns; mixing them would clutter the runs table. |
| Different datasets | **Different experiments** | Metrics aren't comparable across datasets. |
| Same model, different "production candidates" (e.g. weekly retrains) | **Same experiment, distinct tags** | `release.version`, `release.date`, etc. |

Today is case #2 → 3 experiments.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. Building 3 experiments × 3 runs in one script

The neat trick: pass a **callable** ("model factory") instead of a class. The callable takes the hyperparams that vary and returns a fitted-ready estimator:

```python
def make_elasticnet(alpha):
    return ElasticNet(alpha=alpha, l1_ratio=alpha, random_state=42), \
           {"alpha": alpha, "l1_ratio": alpha}      # the params to log

def make_ridge(alpha):
    return Ridge(alpha=alpha, random_state=42),     {"alpha": alpha}

def make_lasso(alpha):
    return Lasso(alpha=alpha, random_state=42),     {"alpha": alpha}
```

Then everything is data:

```python
EXPERIMENTS = [
    ("exp_multi_EL",    make_elasticnet),
    ("exp_multi_Ridge", make_ridge),
    ("exp_multi_Lasso", make_lasso),
]
ALPHAS = [0.7, 0.9, 0.4]

for exp_name, factory in EXPERIMENTS:
    mlflow.set_experiment(exp_name)
    for i, alpha in enumerate(ALPHAS, start=1):
        train_one_run(f"run{i}.1", factory, alpha, ...)
```

That's the whole architecture.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. Project structure

```text
chap20k-mlflow-step-by-step-recap-multiple-experiments-elasticnet-ridge-lasso/
├── README.md
├── docker-compose.yml
├── data/
│   └── red-wine-quality.csv
├── mlflow/
│   └── Dockerfile
└── trainer/
    ├── Dockerfile
    ├── requirements.txt
    └── train.py            ← 3 model factories + 2 nested for loops
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
from sklearn.linear_model import ElasticNet, Lasso, Ridge
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


# === Model factories: each returns (estimator, params_dict_to_log) ===
def make_elasticnet(alpha, l1_ratio):
    return (ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42),
            {"alpha": alpha, "l1_ratio": l1_ratio})


def make_ridge(alpha, l1_ratio):           # l1_ratio is ignored on purpose
    return (Ridge(alpha=alpha, random_state=42),
            {"alpha": alpha})


def make_lasso(alpha, l1_ratio):           # l1_ratio is ignored on purpose
    return (Lasso(alpha=alpha, random_state=42),
            {"alpha": alpha})


def train_one_run(run_name, factory, alpha, l1_ratio,
                  train_x, train_y, test_x, test_y):
    """Train ONE model (whatever the factory returns) and log it to MLflow."""
    mlflow.start_run(run_name=run_name)
    mlflow.set_tags(COMMON_TAGS)

    current = mlflow.active_run()
    print(f"  >>> {current.info.run_name}  (id={current.info.run_id})")

    estimator, params_to_log = factory(alpha, l1_ratio)
    estimator.fit(train_x, train_y)
    preds = estimator.predict(test_x)
    rmse, mae, r2 = eval_metrics(test_y, preds)

    print(f"      {type(estimator).__name__}({params_to_log})  "
          f"RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")

    mlflow.log_params(params_to_log)
    mlflow.log_metrics({"rmse": rmse, "r2": r2, "mae": mae})
    mlflow.sklearn.log_model(estimator, "my_new_model_1")
    mlflow.log_artifacts("data/")

    mlflow.end_run()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    )
    print("The set tracking URI is", mlflow.get_tracking_uri())

    # Load + split data ONCE (shared across all 9 runs)
    data = pd.read_csv("data/red-wine-quality.csv")
    train, test = train_test_split(data)

    os.makedirs("data", exist_ok=True)
    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv", index=False)

    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    # ===== Experiments × hyperparameter sweep =====
    EXPERIMENTS = [
        ("exp_multi_EL",    make_elasticnet),
        ("exp_multi_Ridge", make_ridge),
        ("exp_multi_Lasso", make_lasso),
    ]
    ALPHAS = [args.alpha, 0.9, 0.4]                # cli-driven first slot

    for exp_name, factory in EXPERIMENTS:
        print(f"\n========== Experiment: {exp_name} ==========")
        exp = mlflow.set_experiment(experiment_name=exp_name)
        print(f"  Name: {exp.name}  Experiment_id: {exp.experiment_id}")

        for i, alpha in enumerate(ALPHAS, start=1):
            train_one_run(
                run_name=f"run{i}.1",
                factory=factory,
                alpha=alpha,
                l1_ratio=args.l1_ratio,            # used only by ElasticNet
                train_x=train_x, train_y=train_y,
                test_x=test_x, test_y=test_y,
            )

    run = mlflow.last_active_run()
    print(f"\nRecent active run id   : {run.info.run_id}")
    print(f"Recent active run name : {run.info.run_name}")
```

### 6.2 `docker-compose.yml`

Same skeleton as 20j, only `container_name`s change:

```yaml
services:
  mlflow:
    build: { context: ./mlflow }
    image: mlops/mlflow-recap:latest
    container_name: mlflow-recap-20k
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
    container_name: trainer-recap-20k
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

Identical to chap 20j.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-7"></a>

## 7. Run it, compare experiments in the UI

```bash
cd chap20k-mlflow-step-by-step-recap-multiple-experiments-elasticnet-ridge-lasso
docker compose up -d --build mlflow
docker compose run --rm trainer --alpha 0.7 --l1_ratio 0.7
```

Trainer's stdout (3 sections, 3 runs each):

```text
========== Experiment: exp_multi_EL ==========
  Name: exp_multi_EL  Experiment_id: 1
  >>> run1.1  (id=8a4f...d1)
      ElasticNet({'alpha': 0.7, 'l1_ratio': 0.7})  RMSE=0.78...
  >>> run2.1  (id=b1c3...92)
      ElasticNet({'alpha': 0.9, 'l1_ratio': 0.7})  RMSE=0.81...
  >>> run3.1  (id=f7e2...10)
      ElasticNet({'alpha': 0.4, 'l1_ratio': 0.7})  RMSE=0.76...

========== Experiment: exp_multi_Ridge ==========
  ...
========== Experiment: exp_multi_Lasso ==========
  ...
```

In the UI ([http://localhost:5000](http://localhost:5000)) the **left sidebar** now shows three experiments. Click each one to see its 3 runs.

To compare **algorithms head-to-head**:

1. Open `exp_multi_EL` → note the best RMSE.
2. Open `exp_multi_Ridge` → same.
3. Open `exp_multi_Lasso` → same.
4. Whichever algorithm wins is the one to develop further.

> [!TIP]
> The MLflow UI also lets you **search across all experiments** with a filter like `metrics.rmse < 0.8`. Click the experiment-picker dropdown → "Select all" → then apply the filter.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-8"></a>

## 8. Bonus — switching experiments without losing the best run

In a real pipeline you'd:

1. Run **all 9 runs** as we did.
2. Use `MlflowClient().search_runs([exp_id_1, exp_id_2, exp_id_3], filter_string="...", order_by=["metrics.rmse ASC"])` to find the **single best** run across the three experiments.
3. **Register** that best run's model in the **Model Registry** (chapter 17 of the main course).

Sketch:

```python
from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
exp_ids = [client.get_experiment_by_name(n).experiment_id
           for n in ("exp_multi_EL", "exp_multi_Ridge", "exp_multi_Lasso")]
best = client.search_runs(exp_ids, order_by=["metrics.rmse ASC"], max_results=1)[0]
print("Best run:", best.info.run_id, "RMSE =", best.data.metrics["rmse"])
```

You don't even need to know upfront which algorithm won — the search figures it out.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-9"></a>

## 9. Mini exercise — add a 4th algorithm

Add **`HuberRegressor`** as a 4th algorithm:

```python
from sklearn.linear_model import HuberRegressor

def make_huber(alpha, l1_ratio):
    return (HuberRegressor(alpha=alpha), {"alpha": alpha})

EXPERIMENTS = [
    ("exp_multi_EL",    make_elasticnet),
    ("exp_multi_Ridge", make_ridge),
    ("exp_multi_Lasso", make_lasso),
    ("exp_multi_Huber", make_huber),    # NEW
]
```

```bash
docker compose run --rm trainer
```

A new experiment **`exp_multi_Huber`** appears in the left sidebar with 3 runs. Total now: **12 runs in 4 experiments** — and you wrote ~5 new lines.

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

You learned the **two-loop pattern** (experiments × hyperparams) and saw how a tiny "model factory" abstraction makes it painless to add algorithms.

Next: **[chapter 20l](./20l-practical-work-15l-mlflow-step-by-step-recap-automating-logging-with-mlflow-autolog.md)** — replace half of `train_one_run` with a single line: **`mlflow.autolog()`**. MLflow logs your params, metrics, model, signature and input example **for you**, just by detecting the `.fit()` call.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 20k — multiple experiments (ElasticNet vs Ridge vs Lasso)</strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>
