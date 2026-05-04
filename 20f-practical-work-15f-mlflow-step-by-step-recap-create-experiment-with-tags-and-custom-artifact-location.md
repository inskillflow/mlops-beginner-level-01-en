<a id="top"></a>

# Chapter 20f — Step-by-step recap: `create_experiment` with tags and a custom `artifact_location`

## Table of Contents

| # | Section |
|---|---|
| 1 | [Objective](#section-1) |
| 2 | [What we add today vs chap 20e](#section-2) |
| 3 | [`Path.cwd().joinpath(...).as_uri()` decoded](#section-3) |
| 4 | [The multi-container catch (and how we solve it)](#section-4) |
| 5 | [Project structure](#section-5) |
| 6 | [The code](#section-6) |
| 7 | [Run it, inspect the experiment metadata, see the artifacts](#section-7) |
| 8 | [`set_experiment` vs `create_experiment` — the comparison](#section-8) |
| 9 | [Tear down](#section-9) |
| 10 | [Recap and next chapter](#section-10) |

---

<a id="section-1"></a>

## 1. Objective

Until now we used `mlflow.set_experiment("name")` — a one-liner that creates the experiment if it doesn't exist, otherwise reuses it. Today we go one level lower with `mlflow.create_experiment(...)` to control three things `set_experiment` doesn't expose:

- a **custom `artifact_location`** (where the run's binary outputs land)
- **tags** attached to the experiment itself (`version`, `priority`, …) — searchable in the UI
- the resulting **experiment_id** returned directly

Then we read it back with `mlflow.get_experiment(exp_id)` and print every field. End of the chapter has the **side-by-side comparison** of `set_experiment` vs `create_experiment`.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. What we add today vs chap 20e

| Diff | What |
|---|---|
| `mlflow.create_experiment(name, tags, artifact_location)` | The new lower-level function. |
| `mlflow.get_experiment(exp_id)` | Read back the experiment metadata. |
| Print of `name`, `experiment_id`, `artifact_location`, `tags`, `lifecycle_stage`, `creation_time` | Verifies what was actually created. |
| `from pathlib import Path` and `.as_uri()` | Build a valid `file:///...` URI in a portable way. |
| New shared volume `myartifacts` mounted at `/mlflow/myartifacts` on **both** services | The catch explained in section 4. |

`MLFLOW_TRACKING_URI` keeps coming from the env var (the chap 20e fix). We never hard-code anything in `train.py`.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. `Path.cwd().joinpath(...).as_uri()` decoded

```python
from pathlib import Path
artifact_location = Path("/mlflow/myartifacts").as_uri()
# → "file:///mlflow/myartifacts"
```

Three steps:

| Call | Returns | Why |
|---|---|---|
| `Path("/mlflow/myartifacts")` | A `pathlib.Path` object | OS-agnostic path manipulation. |
| `.joinpath("subdir")` *(optional)* | A new `Path` joined with `subdir` | Forward-slash-safe even on Windows. |
| `.as_uri()` | A `file:///...` URI string | MLflow's `artifact_location` expects a URI, not a plain path. |

> [!NOTE]
> Your snippet used `Path.cwd().joinpath("myartifacts").as_uri()`. `Path.cwd()` returns the **current working directory** which, inside the trainer container, is `/code` (because of `WORKDIR /code` in `trainer/Dockerfile`). So that would produce `file:///code/myartifacts`. We use `/mlflow/myartifacts` instead, for the reason explained in section 4.

You can verify what `Path.cwd()` returns from any shell:

```bash
python -c "from pathlib import Path; print(Path.cwd())"
```

Other valid `artifact_location` values:

```python
"s3://my-bucket/mlflow/artifacts"         # AWS S3
"gs://my-bucket/mlflow/artifacts"         # GCS
"abfss://...@account.dfs.core.windows.net/mlflow"  # Azure ADLS Gen2
"./mlruns"                                # local relative (ill-advised in containers)
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. The multi-container catch (and how we solve it)

Here's the trap. When you call:

```python
mlflow.create_experiment(name="...", artifact_location="file:///code/myartifacts")
```

the URI `file:///code/myartifacts` is **stored in MLflow's database**. Later, when:

- the **trainer** logs an artifact → it writes to `/code/myartifacts/...` *inside the trainer container*;
- the **MLflow server** tries to list those artifacts (when you click "Artifacts" in the UI) → it looks at `/code/myartifacts/...` *inside its own container*, where nothing exists.

Result: the trainer says "logged!" but the UI shows an empty Artifacts tab.

> [!IMPORTANT]
> The fix is to put the artifacts on a **shared filesystem visible to both services**. In Docker Compose, that means a **named volume mounted on both the trainer and the mlflow server**, at the **same absolute path**. We use `/mlflow/myartifacts` and bind it via the named volume `myartifacts`.

```yaml
services:
  mlflow:
    volumes:
      - mlflow-db:/mlflow/database
      - mlflow-artifacts:/mlflow/mlruns
      - myartifacts:/mlflow/myartifacts          # NEW (shared)
  trainer:
    volumes:
      - ./data:/code/data
      - myartifacts:/mlflow/myartifacts          # NEW (shared)

volumes:
  ...
  myartifacts:                                   # NEW
```

Then in code:

```python
artifact_location = Path("/mlflow/myartifacts").as_uri()
```

Both containers see the same files at `/mlflow/myartifacts/`. The UI now shows the artifacts correctly.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. Project structure

```text
chap20f-mlflow-step-by-step-recap-create-experiment-with-tags-and-artifact-location/
├── README.md
├── docker-compose.yml          ← + named volume `myartifacts`
├── data/
│   └── red-wine-quality.csv
├── mlflow/
│   └── Dockerfile
└── trainer/
    ├── Dockerfile
    ├── requirements.txt
    └── train.py                ← create_experiment + get_experiment + Path.as_uri()
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-6"></a>

## 6. The code

### 6.1 `trainer/train.py` — `create_experiment` + `get_experiment`

```python
import argparse
import logging
import os
import warnings
from pathlib import Path

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
parser.add_argument("--alpha", type=float, required=False, default=0.5)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.5)
parser.add_argument("--exp-name", type=str, required=False,
                    default="exp_create_exp_artifact")
args = parser.parse_args()


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # === Tracking URI from env (chap 20e habit) ===
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    )
    print("Tracking URI:", mlflow.get_tracking_uri())

    # === Build a portable artifact URI ===
    artifact_root = Path("/mlflow/myartifacts")
    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_uri = artifact_root.as_uri()
    print("Artifact location URI:", artifact_uri)

    # === create_experiment (low-level) ===
    # Avoid the "experiment already exists" error on the 2nd run by
    # falling back to set_experiment if the name is already taken.
    try:
        exp_id = mlflow.create_experiment(
            name=args.exp_name,
            tags={"version": "v1", "priority": "p1"},
            artifact_location=artifact_uri,
        )
        print(f"Created experiment '{args.exp_name}' with id={exp_id}")
    except mlflow.exceptions.MlflowException as e:
        # Already exists -> fetch its id without erroring out.
        exp = mlflow.set_experiment(args.exp_name)
        exp_id = exp.experiment_id
        print(f"Experiment already exists: id={exp_id} ({e.error_code})")

    # === Read it back ===
    get_exp = mlflow.get_experiment(exp_id)
    print("Name              :", get_exp.name)
    print("Experiment_id     :", get_exp.experiment_id)
    print("Artifact Location :", get_exp.artifact_location)
    print("Tags              :", get_exp.tags)
    print("Lifecycle_stage   :", get_exp.lifecycle_stage)
    print("Creation timestamp:", get_exp.creation_time)

    # === Train + log ===
    data = pd.read_csv("data/red-wine-quality.csv")
    train, test = train_test_split(data)
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha, l1_ratio = args.alpha, args.l1_ratio

    with mlflow.start_run(experiment_id=exp_id):
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
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(lr, "model")
```

### 6.2 `docker-compose.yml` — shared volume for artifacts

```yaml
services:
  mlflow:
    build:
      context: ./mlflow
    image: mlops/mlflow-recap:latest
    container_name: mlflow-recap-20f
    ports:
      - "5000:5000"
    volumes:
      - mlflow-db:/mlflow/database
      - mlflow-artifacts:/mlflow/mlruns
      - myartifacts:/mlflow/myartifacts        # NEW shared
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
    container_name: trainer-recap-20f
    environment:
      MLFLOW_TRACKING_URI: "http://mlflow:5000"
    volumes:
      - ./data:/code/data
      - myartifacts:/mlflow/myartifacts        # NEW shared
    networks:
      - recap-net
    depends_on:
      mlflow:
        condition: service_healthy

volumes:
  mlflow-db:
  mlflow-artifacts:
  myartifacts:                                  # NEW

networks:
  recap-net:
    driver: bridge
```

### 6.3 `mlflow/Dockerfile`, `trainer/Dockerfile`, `trainer/requirements.txt` — unchanged

Identical to chap 20e.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-7"></a>

## 7. Run it, inspect the experiment metadata, see the artifacts

### 7.1 Build & start

```bash
cd chap20f-mlflow-step-by-step-recap-create-experiment-with-tags-and-artifact-location
docker compose up -d --build mlflow
```

### 7.2 First run — creates the experiment

```bash
docker compose run --rm trainer --alpha 0.4 --l1_ratio 0.4
```

Output:

```text
Tracking URI: http://mlflow:5000
Artifact location URI: file:///mlflow/myartifacts
Created experiment 'exp_create_exp_artifact' with id=1
Name              : exp_create_exp_artifact
Experiment_id     : 1
Artifact Location : file:///mlflow/myartifacts
Tags              : {'version': 'v1', 'priority': 'p1'}
Lifecycle_stage   : active
Creation timestamp: 1750000000000
Elasticnet (alpha=0.400000, l1_ratio=0.400000):
  RMSE: 0.78...
  ...
```

### 7.3 Second run — reuses the existing experiment

```bash
docker compose run --rm trainer --alpha 0.7 --l1_ratio 0.3
```

Output starts with:

```text
Experiment already exists: id=1 (RESOURCE_ALREADY_EXISTS)
Name              : exp_create_exp_artifact
...
```

This is the `try / except mlflow.exceptions.MlflowException` block doing its job — the script keeps working on the second invocation. (Without it, `create_experiment` would raise `RESOURCE_ALREADY_EXISTS` and crash.)

### 7.4 Visualize in the UI

Open [http://localhost:5000](http://localhost:5000) → experiment **`exp_create_exp_artifact`**.

- The experiment list shows the **tags** `version=v1`, `priority=p1` (Filter on `tags.version = "v1"` to find it).
- Click any run, then **Artifacts** → folder **`model/`** is there with `MLmodel`, `model.pkl`, `conda.yaml`, …
- The artifact path resolves to `file:///mlflow/myartifacts/<exp_id>/<run_id>/artifacts/model/...` — both containers see this path thanks to the shared `myartifacts` volume.

### 7.5 Inspect the volume from your host

```bash
docker compose exec mlflow ls -R /mlflow/myartifacts | head -40
```

You'll see the same tree the trainer wrote.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-8"></a>

## 8. `set_experiment` vs `create_experiment` — the comparison

Both functions can land you on an experiment. They differ in **intent**:

| Feature | `set_experiment(name)` | `create_experiment(name, tags=, artifact_location=)` |
|---|---|---|
| If experiment **doesn't exist** | Creates it (default settings only) | Creates it (you control tags + artifact_location) |
| If experiment **already exists** | Reuses it silently | Raises `MlflowException(RESOURCE_ALREADY_EXISTS)` |
| Returns | An `Experiment` object | An `experiment_id` (string) |
| Set tags at creation time? | No | Yes |
| Set custom `artifact_location`? | No | Yes |
| Best for | Application code that runs many times | One-off setup script (CI/CD, bootstrapping) |

### 8.1 The "automatic" path

```python
mlflow.set_tracking_uri("http://127.0.0.1:5000")
print("Tracking URI:", mlflow.get_tracking_uri())

exp = mlflow.set_experiment("experience_2")    # idempotent
with mlflow.start_run(experiment_id=exp.experiment_id):
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)
```

Re-runnable without try/except. **Default** artifact location, **no** tags.

### 8.2 The "explicit" path

```python
mlflow.set_tracking_uri("http://127.0.0.1:5000")
print("Tracking URI:", mlflow.get_tracking_uri())

exp_id = mlflow.create_experiment(
    name="experience_2",
    artifact_location="file:///mlflow/myartifacts",
    tags={"version": "v1"},
)
with mlflow.start_run(experiment_id=exp_id):
    ...
```

Crashes on the **second** run because `experience_2` now exists. That's why our `train.py` wraps it in a `try / except` — best of both worlds.

### 8.3 When to pick which

| Use case | Pick |
|---|---|
| Day-to-day training scripts | `set_experiment` |
| Bootstrap script run once at project setup | `create_experiment` |
| Need tags or a custom `artifact_location` | `create_experiment` (or `MlflowClient.create_experiment` + `set_experiment_tag` for finer control) |

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-9"></a>

## 9. Tear down

```bash
docker compose down       # keep volumes (DB + 2 artifact stores survive)
docker compose down -v    # wipe everything: DB + mlflow-artifacts + myartifacts
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-10"></a>

## 10. Recap and next chapter

You've now used:

- `mlflow.create_experiment(name, tags, artifact_location)` — full control.
- `mlflow.get_experiment(exp_id)` — read back metadata to verify what was created.
- `Path("/path").as_uri()` — the portable way to build a `file:///...` URI for `artifact_location`.
- The **shared-volume pattern** that makes `file://` artifact stores actually work in a multi-container setup.
- The **try / except `RESOURCE_ALREADY_EXISTS`** trick that makes `create_experiment` idempotent.

You also have the side-by-side comparison `set_experiment` vs `create_experiment` to choose the right tool for each script.

Next chapters of the recap (20g, 20h, …) — **to be added** when you give me the next snippet.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 20f — `create_experiment` with tags and a custom `artifact_location`</strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>
