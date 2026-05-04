<a id="top"></a>

# Chapter 20y — Step-by-step recap: idiomatic structure with a `main()` function and `with mlflow.start_run(experiment_id=exp.experiment_id):` context manager

## Table of Contents

| # | Section |
|---|---|
| 1 | [Objective](#section-1) |
| 2 | [What we change today vs the rest of the recap](#section-2) |
| 3 | [Why a `main()` function?](#section-3) |
| 4 | [Why the context manager `with mlflow.start_run(...)` ?](#section-4) |
| 5 | [Why pass `experiment_id` explicitly to `start_run`?](#section-5) |
| 6 | [`mlflow.last_active_run()` inside vs outside the `with` block](#section-6) |
| 7 | [Project structure](#section-7) |
| 8 | [The code](#section-8) |
| 9 | [Run it, see the run name and id printed inside the block](#section-9) |
| 10 | [Tear down](#section-10) |
| 11 | [Recap and series wrap-up](#section-11) |

---

<a id="section-1"></a>

## 1. Objective

Up to chap 20x, every script used the **imperative** style:

```python
mlflow.start_run()
...
mlflow.end_run()
```

It works, but it has two real problems:

1. **Forgot the `end_run()`?** → the run stays "active" forever until the Python process exits. Subsequent training calls in the same process will mistakenly attach metrics to it.
2. **Exception between `start_run` and `end_run`?** → the run is never closed. The UI shows it as `RUNNING` for ever.

Today's chapter introduces the **idiomatic Pythonic style** that fixes both:

```python
with mlflow.start_run(experiment_id=exp.experiment_id) as run:
    ...
    # mlflow.end_run() is called automatically when the block exits,
    # whether normally OR via an exception.
```

We also wrap the whole training in a **`main()` function** — no more "loose" code at module top level. This is what every production training script looks like.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. What we change today vs the rest of the recap

| Diff | What |
|---|---|
| Wrap everything in `def main(): ... if __name__ == "__main__": main()` | Cleaner structure, easier to test/import. |
| Replace `mlflow.start_run() / mlflow.end_run()` by `with mlflow.start_run(experiment_id=exp.experiment_id):` | Auto-close, exception-safe. |
| Pass `experiment_id` explicitly to `start_run(...)` | Removes any ambiguity about which experiment the run belongs to. |
| Print `mlflow.last_active_run()` info **inside** the with block | Same run id as the active one — confirms the API works as expected. |
| Use experiment name with a space: `"Project exp 1"` | Demonstrates that MLflow handles spaces fine (URL-encoded internally). |

No new MLflow features — same `log_params`, `log_metrics`, `log_model`. The whole point is the **structure**.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. Why a `main()` function?

A script that runs all its code at import time has 3 downsides:

1. **Cannot be imported safely** — the moment you write `from train import eval_metrics`, the whole training pipeline runs.
2. **No way to call it from another Python process** without a subprocess (no `from train import main; main()`).
3. **Globals leak everywhere** — `data`, `train_x`, `lr`, `rmse` are all module-level variables.

Wrapping in a `main()` function fixes all three:

```python
def main():
    args = parse_args()
    ...
    with mlflow.start_run(experiment_id=exp.experiment_id):
        ...

if __name__ == "__main__":
    main()
```

Now `from train import main` is safe, and unit tests can call `main()` with a mocked argparse.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. Why the context manager `with mlflow.start_run(...)` ?

A run object is a **resource** that must be closed (its end-time must be written to the DB, its status must move from `RUNNING` to `FINISHED` or `FAILED`). The context manager guarantees that:

```python
try:
    run = mlflow.start_run(...)        # opens the run
    ...
    raise RuntimeError("oops")          # something blows up
finally:
    mlflow.end_run(status="FAILED")    # the run is closed cleanly
```

…is exactly what `with mlflow.start_run(...)` does behind the scenes. With the context manager:

- Normal exit → run closed with status `FINISHED`.
- Exception → run closed with status `FAILED`, exception re-raised.

The UI immediately reflects the right status. No more zombie runs.

> [!IMPORTANT]
> The expression `with mlflow.start_run(...) as run:` binds `run` to the active `Run` object. You get its id with `run.info.run_id`, no need to call `mlflow.active_run()` separately.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. Why pass `experiment_id` explicitly to `start_run`?

`mlflow.start_run()` (no args) attaches to whichever experiment is "current" — set by the most recent `mlflow.set_experiment(...)` call. That's fine for short, single-experiment scripts.

In larger scripts you may juggle multiple experiments (one per algorithm, one per dataset…). To remove all ambiguity, **pin the run to a specific experiment by id**:

```python
exp = mlflow.set_experiment("Project exp 1")
print(f"Experiment_id: {exp.experiment_id}")

with mlflow.start_run(experiment_id=exp.experiment_id) as run:
    ...
```

Even if some other code later runs `mlflow.set_experiment("something else")`, this run is locked to "Project exp 1".

| `start_run()` signature | When to use |
|---|---|
| `start_run()` | Small script, single experiment, set globally. |
| `start_run(experiment_id="3")` | Multi-experiment script — explicit, race-safe. |
| `start_run(run_id="abc...")` | Resume / append to an existing run. |
| `start_run(run_name="my-run")` | Give the run a custom display name in the UI. |
| `start_run(nested=True)` | Open a child run under the currently active one (chap 20j). |
| `start_run(tags={...})` | Attach tags at run creation time. |

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-6"></a>

## 6. `mlflow.last_active_run()` inside vs outside the `with` block

The user's snippet places `mlflow.last_active_run()` **inside** the with block, which deserves a clarification:

```python
with mlflow.start_run(experiment_id=exp.experiment_id) as run:
    inner = mlflow.last_active_run()
    # inner.info.run_id == run.info.run_id   ← they're the SAME run while inside
    ...

# Once we leave the block:
outer = mlflow.last_active_run()
# outer.info.run_id == run.info.run_id       ← still the same; "last active" means "most recently active"
```

| Where | What `last_active_run()` returns |
|---|---|
| Before any run was started | `None` |
| Inside an active `with`/`start_run` block | The currently active run (= the one you opened) |
| After the run was closed | The most recently closed run (its `info.status` is now `FINISHED` or `FAILED`) |

Useful pattern: call it **right after** the with block closes to grab the run id for downstream logic (e.g. registering, alerting):

```python
with mlflow.start_run(experiment_id=exp.experiment_id):
    ...

run = mlflow.last_active_run()
print("Just finished:", run.info.run_id, run.info.status)   # FINISHED
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-7"></a>

## 7. Project structure

```text
chap20y-mlflow-step-by-step-recap-with-start-run-context-manager-and-main-function/
├── README.md
├── docker-compose.yml
├── data/
│   └── red-wine-quality.csv
├── mlflow/
│   └── Dockerfile
└── trainer/
    ├── Dockerfile
    ├── requirements.txt
    └── train.py        ← main() + with mlflow.start_run(experiment_id=...)
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-8"></a>

## 8. The code

### 8.1 `trainer/train.py`

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


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha",    type=float, required=False, default=0.4)
    parser.add_argument("--l1_ratio", type=float, required=False, default=0.4)
    return parser.parse_args()


def main():
    args = parse_args()

    warnings.filterwarnings("ignore")
    np.random.seed(40)

    data = pd.read_csv("data/red-wine-quality.csv")
    train, test = train_test_split(data)

    train_x = train.drop(["quality"], axis=1)
    test_x  = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y  = test[["quality"]]

    alpha, l1_ratio = args.alpha, args.l1_ratio

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    )
    print("The set tracking URI is", mlflow.get_tracking_uri())

    experiment = mlflow.set_experiment(experiment_name="Project exp 1")
    print("Name          :", experiment.name)
    print("Experiment_id :", experiment.experiment_id)

    # ===== The idiomatic style: context manager + explicit experiment_id =====
    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        # `run` is the active Run object. Same as mlflow.active_run() / mlflow.last_active_run() while inside.
        inner = mlflow.last_active_run()
        print("Active run_id (run)   :", run.info.run_id)
        print("Active run_id (inner) :", inner.info.run_id)
        print("Active run name       :", run.info.run_name)

        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)
        rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)

        print(f"ElasticNet model (alpha={alpha:.4f}, l1_ratio={l1_ratio:.4f})")
        print(f"  RMSE = {rmse:.4f}")
        print(f"  MAE  = {mae:.4f}")
        print(f"  R2   = {r2:.4f}")

        mlflow.log_metrics({"rmse": rmse, "r2": r2, "mae": mae})
        mlflow.log_params({"alpha": alpha, "l1_ratio": l1_ratio})

        mlflow.sklearn.log_model(lr, artifact_path="model")
        # No mlflow.end_run() needed -- the with block handles it.

    # ===== After the block: the run is FINISHED. We can still inspect it. =====
    finished = mlflow.last_active_run()
    print("After the block:")
    print("  run_id  :", finished.info.run_id)
    print("  status  :", finished.info.status)
    print("  end_time:", finished.info.end_time)


if __name__ == "__main__":
    main()
```

### 8.2 `trainer/requirements.txt`

```text
mlflow==2.16.2
scikit-learn==1.5.2
pandas==2.2.3
numpy==2.1.1
```

### 8.3 `docker-compose.yml`, `mlflow/Dockerfile`, `trainer/Dockerfile`

Identical to chap 20u (SQLite backend, single trainer service).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-9"></a>

## 9. Run it, see the run name and id printed inside the block

```bash
cd chap20y-mlflow-step-by-step-recap-with-start-run-context-manager-and-main-function
docker compose up -d --build mlflow
docker compose run --rm trainer --alpha 0.4 --l1_ratio 0.4
```

Stdout (excerpt):

```text
Name          : Project exp 1
Experiment_id : 1
Active run_id (run)   : 7c5a9b...e4
Active run_id (inner) : 7c5a9b...e4         <-- same as run, as expected
Active run name       : luxuriant-mole-512
ElasticNet model (alpha=0.4000, l1_ratio=0.4000)
  RMSE = 0.7785
  MAE  = 0.6223
  R2   = 0.1054
After the block:
  run_id  : 7c5a9b...e4
  status  : FINISHED                         <-- closed cleanly
  end_time: 1746384927000
```

In the UI ([http://localhost:5000](http://localhost:5000)) → **`Project exp 1`** → the run shows up with status FINISHED, the metrics, the params, and the `model/` artifact.

### Force an exception to see the auto-FAILED behaviour

```bash
docker compose run --rm trainer --alpha not_a_number --l1_ratio 0.4
```

`argparse` will exit before the run starts → no run is created. To force a failure **inside** the with block, you could add a `raise RuntimeError("simulated bug")` between `lr.fit` and `mlflow.log_metrics` and re-run. The UI then shows the run with status `FAILED` even though `mlflow.end_run()` was never called explicitly. That's the magic of the context manager.

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

The idiomatic skeleton every MLflow training script should follow:

```python
def main():
    mlflow.set_tracking_uri(...)
    exp = mlflow.set_experiment(...)

    with mlflow.start_run(experiment_id=exp.experiment_id) as run:
        # ...do training...
        mlflow.log_params(...)
        mlflow.log_metrics(...)
        mlflow.sklearn.log_model(...)
    # run auto-closed: status = FINISHED on success, FAILED on exception

if __name__ == "__main__":
    main()
```

Three benefits over the imperative style of chap 20a–20x:

1. Auto-close on success **and** on exception → no zombie `RUNNING` runs.
2. `experiment_id=` removes any ambiguity about which experiment the run lands in.
3. `main()` makes the script importable and unit-testable.

This wraps up the **20a → 20y** recap series. With chapters 20p–20y you now have a complete production-ready vocabulary:

- `pyfunc.log_model` with custom wrappers and conda envs (20p).
- `pyfunc.load_model` for round-trips (20q).
- `mlflow.evaluate` with default + custom metrics + custom artifacts (20r, 20s).
- `validation_thresholds` + `baseline_model` as a CI/CD gate (20t).
- Two registry styles (`registered_model_name=` kwarg vs `mlflow.register_model(...)` function) — 20u, 20w.
- Dual-format logging for non-MLflow consumers (20v).
- Importing models trained outside MLflow into the registry (20x).
- Idiomatic Python structure (20y).

Next major step (chapter 21+): leaves the trainer side and moves to **deployment** — wiring the registered model into a FastAPI service consumed by Streamlit.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 20y — context manager + main() pattern</strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>
