<a id="top"></a>

# Chapter 20g — Step-by-step recap: `active_run` + `last_active_run` (and the `start_run` / `end_run` style without `with`)

## Table of Contents

| # | Section |
|---|---|
| 1 | [Objective](#section-1) |
| 2 | [What we add today vs chap 20f](#section-2) |
| 3 | [`with start_run()` vs `start_run()` + `end_run()`](#section-3) |
| 4 | [`active_run()` vs `last_active_run()`](#section-4) |
| 5 | [Project structure](#section-5) |
| 6 | [The code](#section-6) |
| 7 | [Run it, observe both helpers](#section-7) |
| 8 | [Mini exercise — a deliberate crash](#section-8) |
| 9 | [Tear down](#section-9) |
| 10 | [Recap and next chapter](#section-10) |

---

<a id="section-1"></a>

## 1. Objective

Two new helpers today:

- **`mlflow.active_run()`** — returns the run that is **currently** active (or `None` if there isn't one). Useful when you need the `run_id` *during* training to log something extra.
- **`mlflow.last_active_run()`** — returns the **most recently active** run, even after `mlflow.end_run()` has been called. Useful when you want a final summary (`run_id`, `run_name`, status…) **after** the training block.

To make their behaviour unambiguous, today's `train.py` deliberately uses the **imperative pattern**: `mlflow.start_run()` followed (much later) by `mlflow.end_run()` — instead of the cleaner `with mlflow.start_run() as run:` we used in 20a → 20f.

Both styles work. Knowing both is essential because real codebases mix them.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. What we add today vs chap 20f

| Diff | What |
|---|---|
| Switch from `with mlflow.start_run(...):` to `mlflow.start_run()` + `mlflow.end_run()` | The "imperative" style. |
| Print **all six** experiment metadata fields (`name`, `experiment_id`, `artifact_location`, `tags`, `lifecycle_stage`, `creation_time`) | See exactly what `set_experiment` returned. |
| Add a **mid-run** `mlflow.active_run()` call to grab the `run_id` while the run is still open | The new helper #1. |
| Call `mlflow.last_active_run()` **after** `mlflow.end_run()` to print final summary | The new helper #2. |
| Model logged under the name `my_new_model_1` (matches your snippet) | Pure cosmetic. |

Everything else (env-var-based tracking URI, multi-service Docker, `requirements.txt`, CLI args via `ENTRYPOINT`) stays exactly as in chap 20e/f.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. `with start_run()` vs `start_run()` + `end_run()`

```python
# Style A — context manager (the one we used until chap 20f)
with mlflow.start_run(run_name="my_run") as run:
    print("Inside:", run.info.run_id)
    mlflow.log_param("x", 1)
# end_run() is called automatically here, even on exception

# Style B — imperative (today)
mlflow.start_run(run_name="my_run")
print("Inside:", mlflow.active_run().info.run_id)
mlflow.log_param("x", 1)
mlflow.end_run()
```

| Style | Pros | Cons |
|---|---|---|
| **A. `with`** (recommended) | Auto-closes the run on exception. No way to forget `end_run()`. Block scoping makes the run's lifetime obvious. | The `run` object is only visible inside the block. |
| **B. imperative** | The run object is visible to the rest of the function. Easier to spread training across multiple helper functions. | If your code raises before `end_run()`, the run stays open with status `RUNNING` forever (well, until you restart MLflow). |

> [!IMPORTANT]
> In production: prefer **Style A**. In notebooks / step-by-step scripts: Style B is fine and a touch more readable. Today we showcase Style B because it makes `active_run` and `last_active_run` more visible.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. `active_run()` vs `last_active_run()`

```python
mlflow.start_run(run_name="r1")
mlflow.active_run()        # → Run(r1, status=RUNNING)
mlflow.last_active_run()   # → Run(r1, status=RUNNING)
mlflow.end_run()
mlflow.active_run()        # → None (no run is currently active)
mlflow.last_active_run()   # → Run(r1, status=FINISHED)  ← still works!
```

| Function | Returns when there's an open run | Returns after `end_run()` |
|---|---|---|
| `mlflow.active_run()` | The current `Run` | **`None`** |
| `mlflow.last_active_run()` | The current `Run` | The last one (status `FINISHED`) |

That's the whole difference. `last_active_run()` is the one you call to print the final summary.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. Project structure

```text
chap20g-mlflow-step-by-step-recap-active-run-and-last-active-run/
├── README.md
├── docker-compose.yml
├── data/
│   └── red-wine-quality.csv
├── mlflow/
│   └── Dockerfile
└── trainer/
    ├── Dockerfile
    ├── requirements.txt
    └── train.py            ← imperative style + active_run/last_active_run
```

Same Docker skeleton as chap 20e/f. We drop the shared `myartifacts` volume from chap 20f because today's experiment uses the server's default artifact root.

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


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    )
    print("The set tracking URI is", mlflow.get_tracking_uri())

    exp = mlflow.set_experiment(experiment_name="experiment_2")

    # NEW: print every experiment metadata field
    print("Name              :", exp.name)
    print("Experiment_id     :", exp.experiment_id)
    print("Artifact Location :", exp.artifact_location)
    print("Tags              :", exp.tags)
    print("Lifecycle_stage   :", exp.lifecycle_stage)
    print("Creation timestamp:", exp.creation_time)

    data = pd.read_csv("data/red-wine-quality.csv")
    train, test = train_test_split(data)

    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha, l1_ratio = args.alpha, args.l1_ratio

    # === Imperative style: no `with` ===
    mlflow.start_run()                                              # NEW (imperative)

    # === active_run() works WHILE the run is open ===
    current = mlflow.active_run()                                   # NEW
    print("active_run() during training:", current.info.run_id)

    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)
    preds = lr.predict(test_x)
    rmse, mae, r2 = eval_metrics(test_y, preds)

    print("Elasticnet (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE:  %s" % mae)
    print("  R2:   %s" % r2)

    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    mlflow.sklearn.log_model(lr, "my_new_model_1")

    mlflow.end_run()                                                # NEW (imperative)

    # === active_run() returns None after end_run() ===
    print("active_run() after end_run():", mlflow.active_run())     # → None

    # === last_active_run() still works (status now FINISHED) ===
    run = mlflow.last_active_run()                                  # NEW
    print("Active run id   :", run.info.run_id)
    print("Active run name :", run.info.run_name)
    print("Final status    :", run.info.status)
```

### 6.2 `docker-compose.yml`

```yaml
services:
  mlflow:
    build:
      context: ./mlflow
    image: mlops/mlflow-recap:latest
    container_name: mlflow-recap-20g
    ports:
      - "5000:5000"
    volumes:
      - mlflow-db:/mlflow/database
      - mlflow-artifacts:/mlflow/mlruns
    networks:
      - recap-net
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:5000').status==200 else 1)"]
      interval: 10s
      timeout: 5s
      retries: 5

  trainer:
    build:
      context: ./trainer
    image: mlops/trainer-recap:latest
    container_name: trainer-recap-20g
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
  mlflow-db:
  mlflow-artifacts:

networks:
  recap-net:
    driver: bridge
```

### 6.3 `mlflow/Dockerfile`, `trainer/Dockerfile`, `trainer/requirements.txt` — unchanged

Identical to chap 20e/f.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-7"></a>

## 7. Run it, observe both helpers

```bash
cd chap20g-mlflow-step-by-step-recap-active-run-and-last-active-run
docker compose up -d --build mlflow
docker compose run --rm trainer --alpha 0.4 --l1_ratio 0.4
```

Output you should see:

```text
The set tracking URI is http://mlflow:5000
Name              : experiment_2
Experiment_id     : 1
Artifact Location : mlflow-artifacts:/1
Tags              : {}
Lifecycle_stage   : active
Creation timestamp: 1750000000000
active_run() during training: 8a4f...d1
Elasticnet (alpha=0.400000, l1_ratio=0.400000):
  RMSE: 0.78...
  MAE:  0.62...
  R2:   0.10...
active_run() after end_run(): None
Active run id   : 8a4f...d1
Active run name : <auto-generated by MLflow>
Final status    : FINISHED
```

Open [http://localhost:5000](http://localhost:5000) → experiment **`experiment_2`** → 1 run, status **FINISHED**, with the model `my_new_model_1` in its Artifacts tab.

> [!NOTE]
> Notice the run got an auto-generated `run_name` (something whimsical like `tasteful-snake-42`). MLflow auto-names runs that don't explicitly receive a `run_name=`. Pass `mlflow.start_run(run_name="my_clean_name")` to take control — that's what we did in chapters 08 and 20a.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-8"></a>

## 8. Mini exercise — a deliberate crash

Add a deliberate exception **between** `start_run()` and `end_run()`:

```python
mlflow.start_run()
1 / 0   # boom
mlflow.end_run()
```

Re-run:

```bash
docker compose run --rm trainer
```

In the MLflow UI, open the experiment. The run is there but its **status is `RUNNING`** — forever. That's the imperative-style trap: an exception skipped `end_run()`.

To clean it up, manually finalize:

```bash
docker compose run --rm trainer python -c "import os, mlflow; mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI']); mlflow.end_run()"
```

…or just use **Style A** (`with mlflow.start_run() as run:`) which closes the run automatically on exception. Lesson learned.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-9"></a>

## 9. Tear down

```bash
docker compose down       # keep the experiment + runs
docker compose down -v    # wipe everything
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-10"></a>

## 10. Recap and next chapter

You added two helpers and one new style:

- `mlflow.start_run()` + `mlflow.end_run()` — the imperative style.
- `mlflow.active_run()` — current run, or `None`.
- `mlflow.last_active_run()` — most recent run, even after `end_run()`.
- The full set of **experiment metadata fields**: `name`, `experiment_id`, `artifact_location`, `tags`, `lifecycle_stage`, `creation_time`.

Next: **[chapter 20h](./20h-practical-work-15h-mlflow-step-by-step-recap-log-artifacts-with-log-params-and-log-metrics-bulk-versions.md)** — same setup, but we add **`mlflow.log_artifacts(folder)`** to ship train/test CSVs to MLflow, and we switch to the **bulk** versions `log_params(dict)` / `log_metrics(dict)` to log many at once.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 20g — `active_run` + `last_active_run`</strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>
