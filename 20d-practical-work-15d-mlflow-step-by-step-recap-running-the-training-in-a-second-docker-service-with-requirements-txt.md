<a id="top"></a>

# Chapter 20d — Step-by-step recap: running the training inside a second Docker service (with `requirements.txt`)

## Table of Contents

| # | Section |
|---|---|
| 1 | [Objective](#section-1) |
| 2 | [What we add today vs chap 20c](#section-2) |
| 3 | [Project structure](#section-3) |
| 4 | [The code](#section-4) |
| 5 | [Run the training inside Docker (with CLI args)](#section-5) |
| 6 | [Access the MLflow UI](#section-6) |
| 7 | [The bug we leave on purpose for chap 20e](#section-7) |
| 8 | [Tear down](#section-8) |
| 9 | [Recap and next chapter](#section-9) |

---

<a id="section-1"></a>

## 1. Objective

Until chap 20c, the **server** ran in Docker but the **training script** ran on your host. Today we put both in Docker:

- `mlflow` service — the tracking server (same as before).
- **`trainer` service** (new) — a Python image with `pandas`, `numpy`, `scikit-learn`, `mlflow`, plus our `train.py`.

You'll learn the canonical way to **pass CLI arguments** to a Dockerized script:

```bash
docker compose run --rm trainer --alpha 0.1 --l1_ratio 0.1
```

> [!IMPORTANT]
> The chapter intentionally **omits** `mlflow.set_tracking_uri(...)` from `train.py` (just like the snippet you posted). This will create a small "where did my run go?" mystery in section 7 — and chapter 20e will fix it the right way (via an environment variable).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. What we add today vs chap 20c

| Diff | Why |
|---|---|
| New `trainer/` folder with its own `Dockerfile` and `requirements.txt` | Reproducible Python env, frozen with the project. |
| Second service `trainer` in `docker-compose.yml` | Two services on the same Docker network. |
| `ENTRYPOINT ["python", "train.py"]` in `trainer/Dockerfile` | Lets you pass CLI args directly to `docker compose run`. |
| `data/` bind-mounted as a volume on the trainer | Edit the CSV without rebuilding the image. |
| `train.py` no longer needs the host venv | The whole pipeline is reproducible via Docker only. |

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. Project structure

```text
chap20d-mlflow-step-by-step-recap-running-the-training-in-docker/
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

Two `Dockerfile`s now. Two services in compose.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. The code

### 4.1 `trainer/requirements.txt` — the Python env, frozen

```text
mlflow==2.16.2
scikit-learn==1.5.2
pandas==2.2.3
numpy==2.1.1
```

### 4.2 `trainer/Dockerfile`

```dockerfile
FROM python:3.12-slim
WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY train.py .
ENTRYPOINT ["python", "train.py"]
```

Two new ideas:

- `RUN pip install --no-cache-dir -r requirements.txt` — the **canonical reproducible install** in Docker. Same packages, same versions, every machine, every time.
- `ENTRYPOINT ["python", "train.py"]` — when you run the container, **everything you append on the command line is forwarded as arguments to `train.py`**. That's how `docker compose run --rm trainer --alpha 0.1` works.

### 4.3 `trainer/train.py` (intentionally without `set_tracking_uri`)

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

    data = pd.read_csv("data/red-wine-quality.csv")
    train, test = train_test_split(data)

    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = args.alpha
    l1_ratio = args.l1_ratio

    print("Tracking URI (currently):", mlflow.get_tracking_uri())   # spoiler: not what you think

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

### 4.4 `mlflow/Dockerfile` — unchanged

```dockerfile
FROM python:3.12-slim
WORKDIR /mlflow
RUN pip install --no-cache-dir mlflow==2.16.2
EXPOSE 5000
CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:///database/mlflow.db", \
     "--default-artifact-root", "/mlflow/mlruns", \
     "--host", "0.0.0.0", "--port", "5000"]
```

### 4.5 `docker-compose.yml` — two services

```yaml
services:
  mlflow:
    build:
      context: ./mlflow
    image: mlops/mlflow-recap:latest
    container_name: mlflow-recap-20d
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
    container_name: trainer-recap-20d
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

A few things to notice:

- The `trainer` has **no** `command:` and **no** `restart:` — it's a one-shot service. We launch it on demand with `docker compose run`.
- `./data:/code/data` is a **bind mount** — your host folder is mirrored into the container. Edit the CSV on your host, the container sees it instantly.
- `depends_on: mlflow: service_healthy` — the trainer won't start until the MLflow server passed its healthcheck.
- Both services share the user-defined network `recap-net` so the trainer can reach the server by its DNS name (we'll need this in chap 20e).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. Run the training inside Docker (with CLI args)

### 5.1 First, build everything and start the MLflow server

```bash
cd chap20d-mlflow-step-by-step-recap-running-the-training-in-docker
docker compose up -d --build mlflow
```

`-d` = detached, so the server runs in the background. `--build` forces a rebuild if any Dockerfile or `requirements.txt` changed.

Verify:

```bash
docker compose ps
# mlflow-recap-20d    Up X seconds (healthy)
```

### 5.2 Then run the trainer with CLI arguments

```bash
docker compose run --rm trainer --alpha 0.1 --l1_ratio 0.1
```

Three things happen here:

| Token | Meaning |
|---|---|
| `docker compose run` | Spawn a one-off container of the requested service. |
| `--rm` | Delete the container as soon as the script returns (no garbage). |
| `trainer` | Which service to run. |
| `--alpha 0.1 --l1_ratio 0.1` | These tokens are appended to the `ENTRYPOINT`, so the container actually runs `python train.py --alpha 0.1 --l1_ratio 0.1`. |

You should see:

```text
Tracking URI (currently): file:///code/mlruns
Elasticnet model (alpha=0.100000, l1_ratio=0.100000):
  RMSE: 0.7931...
  MAE:  0.6271...
  R2:   0.0855...
```

### 5.3 Run a tiny grid search

```bash
docker compose run --rm trainer --alpha 0.5 --l1_ratio 0.5
docker compose run --rm trainer --alpha 0.9 --l1_ratio 0.1
docker compose run --rm trainer --alpha 0.1 --l1_ratio 0.9
```

> [!NOTE]
> Each `docker compose run --rm trainer ...` reuses the **already-built image** — startup is < 1 s after the first build. That's the whole appeal of containerising the trainer: identical Python env, identical CLI, no host venv to maintain.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-6"></a>

## 6. Access the MLflow UI

The server publishes port 5000 on your host:

| URL | What |
|---|---|
| http://localhost:5000 | MLflow UI |

Open it. **You should see no `experiment_1` and no runs.** Why? See section 7.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-7"></a>

## 7. The bug we leave on purpose for chap 20e

Look back at the trainer's first line of output:

```text
Tracking URI (currently): file:///code/mlruns
```

Without `mlflow.set_tracking_uri(...)` and without an env var, the trainer logs into a **local folder inside its own container**: `/code/mlruns`. With `--rm`, that container is destroyed at the end and your run is **lost forever**. That's why the UI on `localhost:5000` is empty.

Just to convince yourself, run **without** `--rm` once and look inside:

```bash
docker compose run trainer --alpha 0.5 --l1_ratio 0.5
docker ps -a | grep trainer-recap-20d   # find the stopped container id
docker exec -it <container_id> ls /code/mlruns/   # → meta.yaml, 0/, 1/, ...
```

The runs are there — but **inside a dead container** that nobody is reading.

> [!IMPORTANT]
> Three ways to fix this in chapter 20e:
> 1. Add `mlflow.set_tracking_uri("http://mlflow:5000")` in `train.py` (works but hard-codes a hostname).
> 2. Read the URI from `os.getenv("MLFLOW_TRACKING_URI", "...")` and pass it via Docker (the **clean** way).
> 3. Set `MLFLOW_TRACKING_URI` directly in `docker-compose.yml` so MLflow picks it up automatically (no `set_tracking_uri` line needed at all).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-8"></a>

## 8. Tear down

```bash
docker compose down       # keep the (so-far-empty) volumes
docker compose down -v    # wipe everything
```

The trainer image stays cached locally so the next chapter starts fast.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-9"></a>

## 9. Recap and next chapter

Today you learned:

- How to **freeze a Python env in Docker** with `requirements.txt` + `pip install --no-cache-dir -r ...`.
- How to **forward CLI args** to a Dockerized script via `ENTRYPOINT` + `docker compose run --rm <svc> --arg ...`.
- Why a missing `set_tracking_uri` in a Dockerized trainer is **catastrophic**: runs land inside an ephemeral container and disappear.

Next: **[chapter 20e](./20e-practical-work-15e-mlflow-step-by-step-recap-passing-the-tracking-uri-via-environment-variable.md)** — fix the bug **the production way**: `MLFLOW_TRACKING_URI` env var passed through `docker-compose.yml`, no hardcoded URL anywhere.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 20d — Running the training inside a second Docker service</strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>
