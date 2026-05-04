<a id="top"></a>

# Chapter 20e — Step-by-step recap: passing the MLflow tracking URI via an environment variable

## Table of Contents

| # | Section |
|---|---|
| 1 | [Objective](#section-1) |
| 2 | [The 3 ways to set the tracking URI in Docker](#section-2) |
| 3 | [Why hostname `mlflow` and not `localhost` or an IP?](#section-3) |
| 4 | [Project structure](#section-4) |
| 5 | [The code](#section-5) |
| 6 | [Run the training, see runs land in the UI](#section-6) |
| 7 | [Override the URI at runtime (the Ops pattern)](#section-7) |
| 8 | [Tear down](#section-8) |
| 9 | [Recap and next chapter](#section-9) |

---

<a id="section-1"></a>

## 1. Objective

Fix the bug we left at the end of chapter 20d: runs were silently logged to a dead container's filesystem. Today we wire the trainer to the server **without** hard-coding any URL in `train.py`. The trick is the `MLFLOW_TRACKING_URI` environment variable, set in `docker-compose.yml`.

Your snippet's example used:

```python
mlflow.set_tracking_uri(uri="http://<IP-de-votre-VM>:5000")
```

That works, but it has two problems:

1. The IP is **baked into your code** — when your VM changes, you re-edit Python.
2. It assumes `<IP-de-votre-VM>` is reachable. **Inside Docker, services don't know about VM IPs** — they know each other by service name.

Today's fix: read the URI from the environment, default it to a service-name-based URL, and let `docker-compose.yml` decide what to inject.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. The 3 ways to set the tracking URI in Docker

| Way | When to use | What it looks like |
|---|---|---|
| **A. Hard-coded in Python** | Quick local script, one-shot. | `mlflow.set_tracking_uri("http://mlflow:5000")` |
| **B. Read from env var in Python** | Reusable script that runs in dev / CI / prod. | `mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "..."))` |
| **C. Pure env var (no Python code)** | Cleanest. MLflow auto-reads `MLFLOW_TRACKING_URI`. | `environment: MLFLOW_TRACKING_URI: "http://mlflow:5000"` in compose, **no `set_tracking_uri()` call** at all. |

> [!IMPORTANT]
> MLflow reads `MLFLOW_TRACKING_URI` automatically at import time. So if you set it in the compose file, **you don't even need `set_tracking_uri()` in your script**. Many production codebases choose option C for that reason.

We'll showcase **B** (most explicit, easiest to debug) in `train.py`, and **C** (env var only, no code change) as a one-line variant in section 7.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. Why hostname `mlflow` and not `localhost` or an IP?

Inside the Docker network, every service is reachable by **its service name** as a DNS host:

```text
trainer container ──► http://mlflow:5000  ✓     (Docker DNS resolves "mlflow")
trainer container ──► http://localhost:5000 ✗   (localhost = the trainer itself, not the server)
trainer container ──► http://192.168.x.x:5000 ✗ (an IP that doesn't exist on this network)
```

| Where the code runs | Correct URL for MLflow |
|---|---|
| Your host machine (`python train.py` from a venv) | `http://localhost:5000` |
| Another container in the same compose network (today's case) | `http://mlflow:5000` |
| A different VM / machine on your LAN | `http://<IP-of-the-VM>:5000` |

**This is exactly why we use an env var**: the URL changes with the environment, the code doesn't.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. Project structure

```text
chap20e-mlflow-step-by-step-recap-passing-tracking-uri-via-env-var/
├── README.md
├── docker-compose.yml         ← + environment: MLFLOW_TRACKING_URI: ...
├── data/
│   └── red-wine-quality.csv
├── mlflow/
│   └── Dockerfile
└── trainer/
    ├── Dockerfile
    ├── requirements.txt
    └── train.py               ← + os.getenv("MLFLOW_TRACKING_URI", ...)
```

Same files as chap 20d, two small diffs.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. The code

### 5.1 `trainer/train.py` — read the URI from env

Diff vs chap 20d (only the lines marked `# NEW`):

```python
import argparse
import logging
import os                                                     # NEW
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
parser.add_argument("--alpha", type=float, required=False, default=0.5)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.5)
args = parser.parse_args()


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # === The fix from chapter 20d ===                          # NEW
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI",            # NEW
                             "http://mlflow:5000")             # NEW
    mlflow.set_tracking_uri(tracking_uri)                       # NEW
    print("Tracking URI:", mlflow.get_tracking_uri())           # NEW

    data = pd.read_csv("data/red-wine-quality.csv")
    train, test = train_test_split(data)

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
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(lr, "mymodel")
```

The 4 new lines are the whole lesson. `os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")`:
- If the env var **is set** → use it.
- If it's **not set** → fall back to `http://mlflow:5000` (the safe default for our compose).

### 5.2 `docker-compose.yml` — inject the env var

Diff vs chap 20d (only the `environment:` block on the trainer):

```yaml
services:
  mlflow:
    build:
      context: ./mlflow
    image: mlops/mlflow-recap:latest
    container_name: mlflow-recap-20e
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
    container_name: trainer-recap-20e
    environment:                                              # NEW
      MLFLOW_TRACKING_URI: "http://mlflow:5000"               # NEW
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

### 5.3 `mlflow/Dockerfile`, `trainer/Dockerfile`, `trainer/requirements.txt` — unchanged

Identical to chap 20d.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-6"></a>

## 6. Run the training, see runs land in the UI

```bash
cd chap20e-mlflow-step-by-step-recap-passing-tracking-uri-via-env-var
docker compose up -d --build mlflow

docker compose run --rm trainer --alpha 0.1 --l1_ratio 0.1
docker compose run --rm trainer --alpha 0.5 --l1_ratio 0.5
docker compose run --rm trainer --alpha 0.9 --l1_ratio 0.1
```

Each run now prints:

```text
Tracking URI: http://mlflow:5000
Elasticnet model (alpha=0.100000, l1_ratio=0.100000):
  RMSE: 0.7931...
  ...
```

Open [http://localhost:5000](http://localhost:5000) → experiment **`experiment_1`** → **3 runs**, each with their params, metrics and the `mymodel/` artifact.

> [!NOTE]
> The runs persist across `docker compose down` because they live in the named volumes `mlflow-db` and `mlflow-artifacts` mounted on the `mlflow` service.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-7"></a>

## 7. Override the URI at runtime (the Ops pattern)

Sometimes you want a one-off run that targets a **different** server (a colleague's, a staging VM, a Databricks workspace, …). You don't change the compose, you don't change the code — you override at run time:

### 7.1 With `-e` on `docker compose run`

```bash
docker compose run --rm \
  -e MLFLOW_TRACKING_URI=http://192.168.1.42:5000 \
  trainer --alpha 0.3 --l1_ratio 0.7
```

The `-e VAR=value` flag wins over what's in the compose file. The script prints the new URI and logs there.

### 7.2 Pure env-var pattern (option C)

You can even **delete** the `set_tracking_uri()` line from `train.py` entirely. MLflow reads `MLFLOW_TRACKING_URI` from the environment all by itself at import time. Try it:

```python
# remove these lines from train.py:
# tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
# mlflow.set_tracking_uri(tracking_uri)

# keep only:
print("Tracking URI:", mlflow.get_tracking_uri())   # still prints http://mlflow:5000
```

It still works, because the env var is set in the compose file. **This is the cleanest production pattern**: zero MLflow URL anywhere in your application code.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-8"></a>

## 8. Tear down

```bash
docker compose down       # keep all your runs
docker compose down -v    # wipe everything
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-9"></a>

## 9. Recap and next chapter

You learned:

- The 3 ways to point at an MLflow server in a Docker setup (hard-coded / env-var-with-default / pure env-var).
- Why **service names** (e.g. `mlflow`) — not `localhost`, not IPs — are the right hostname inside a compose network.
- How to **override at run time** with `docker compose run --rm -e MLFLOW_TRACKING_URI=...`.

Next: **[chapter 20f](./20f-practical-work-15f-mlflow-step-by-step-recap-create-experiment-with-tags-and-custom-artifact-location.md)** — same Dockerized setup, but we replace `set_experiment` with the lower-level `create_experiment(name=, tags=, artifact_location=)` to control **where artifacts live** and to attach **searchable metadata** (`version`, `priority`, …) to the experiment itself.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 20e — Passing the tracking URI via an env var</strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>
