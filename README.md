<a id="top"></a>

# MLOps with MLflow, FastAPI & Streamlit — full course

A hands-on, **chapter-by-chapter** course that teaches MLOps by building one realistic stack:

```text
   Streamlit (UI)  ──►  FastAPI (REST API)  ──►  MLflow (tracking + registry)
            \____________________  Docker Compose  ____________________/
```

Every new chapter adds **one and only one** MLflow concept on top of the previous one. Same wine-quality dataset, same `ElasticNet` baseline, same Docker stack — only the focused MLflow call changes. That's how you build durable understanding.

> [!IMPORTANT]
> If you only read one section, read [§ 3 — How to navigate the course](#section-3). It explains the recommended pace, the diff-based reading habit, and the single command that runs each chapter.

---

## Table of Contents

| # | Section |
|---|---|
| 1 | [Who this course is for](#section-1) |
| 2 | [Prerequisites](#section-2) |
| 3 | [How to navigate the course](#section-3) |
| 4 | [Course map](#section-4) |
| 5 | [Detailed chapter index](#section-5) |
| 6 | [Common runtime — ports, URLs, containers](#section-6) |
| 7 | [Cheat sheet — Docker Compose](#section-7) |
| 8 | [Folder layout reference](#section-8) |
| 9 | [Troubleshooting](#section-9) |
| 10 | [Suggested learning paths](#section-10) |

---

<a id="section-1"></a>

## 1. Who this course is for

- Data scientists who already train models but don't yet **log, version, and serve** them properly.
- Backend / DevOps engineers who want to learn **what MLOps actually adds** to a normal web stack.
- Anyone preparing for a junior MLOps role and wanting a full, runnable portfolio repo.

Each chapter is self-contained: clone, `cd chapXX-...`, `docker compose up`, follow the lesson `.md`, tear down. No global state.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. Prerequisites

| | Requirement |
|---|---|
| **OS** | Windows 10/11 (with WSL 2), macOS, or Linux. Windows PowerShell or `bash` both work. |
| **Tooling** | Docker Desktop ≥ 4.30 (Windows/macOS) **or** Docker Engine + Docker Compose v2 (Linux). |
| **Memory** | 6 GB RAM available to Docker (8 GB recommended). |
| **Disk** | ~5 GB free for images, volumes, and artifacts across all chapters. |
| **Knowledge** | Python basics, `pip`, ability to read a `Dockerfile`. No prior MLflow knowledge required. |

If you don't have Docker installed yet, **start with chapter 06** — it walks you through the full Docker Desktop install, Docker concepts, and your first `docker compose up`. Once Docker is up, the **optional warm-ups [05a](./05a-practical-work-1a-installing-streamlit-and-running-a-minimal-app-with-docker.md) → [05b](./05b-practical-work-1b-installing-fastapi-and-running-a-calculator-api-with-docker.md) → [05c](./05c-practical-work-1c-installing-mlflow-server-with-docker-and-logging-a-first-run.md)** show each technology of the stack in its own single container before chapter 06 assembles them.

If you can't (or don't want to) use Docker, **chapter 05** shows the pure-`venv` path on Ubuntu 24 — but the rest of the course assumes Docker.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. How to navigate the course

### 3.1 The 4-step ritual for every chapter

For chapter `XX`:

```bash
cd chapXX-<short-name>
docker compose up --build           # or: docker compose up -d --build
# → open the lesson side-by-side: ../XX-practical-work-...md
# → open the URLs from the chapter (MLflow on :5000, FastAPI on :8000, Streamlit on :8501)
# → run the curl commands and click around the UI
docker compose down
```

### 3.2 Read with a "diff" habit

Each chapter only changes a handful of lines. The fastest way to learn is to **diff** the previous chapter against the new one:

```bash
# from inside 00-mlops-en-english/
diff -ru chap08-... chap09-...           # Linux/macOS
# or use VS Code: "Compare Folders" extension, or any GUI diff
```

You'll instantly see the few new MLflow lines that today's chapter is teaching. The lesson `.md` highlights them in a "**The lines we add today**" section near the top.

### 3.3 Don't skip the UI tour

Every chapter ends with a "Visualize in the UI" section. Looking at MLflow's UI **after** running the code is what makes the abstract Python calls click. Five minutes of clicking around saves an hour of confusion later.

### 3.4 Don't run two chapters at once

All chapters use the same ports (5000, 8000, 8501). Always `docker compose down` the previous chapter before starting the next one — otherwise the new stack won't bind its ports.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. Course map

```text
THEORY              PRACTICE — venv          PRACTICE — Docker (1 service)        PRACTICE — Docker Compose stack
─────────           ───────────────          ─────────────────────────────        ───────────────────────────────
01 What is MLOps?   05 Install MLflow        05a Streamlit minimal                06 Install Docker + run the full
02 What is MLflow?      on Ubuntu in venv    05b FastAPI calculator                   stack (Streamlit + FastAPI
03 Quiz (Q)                                  05c MLflow server                        + MLflow together)
04 Quiz (A)                                                                                       │
                                                                                                  ▼
                                                                                Then ONE MLflow concept per chapter:

                                                                                   07  set_experiment
                                                                                   08  start_run + last_active_run
                                                                                   09  log_param + log_metric
                                                                                   10  log_artifact + log_artifacts
                                                                                   11  set_tag + set_tags
                                                                                   12  loops & nested runs (grid/sweep)
                                                                                   13  mlflow.sklearn.autolog
                                                                                   14  signature + input_example
                                                                                   15  pyfunc + custom Conda env
                                                                                   16  load_model + mlflow.evaluate
                                                                                   17  Model Registry (MlflowClient)
                                                                                   18  MLflow CLI (admin & ops)
```

The **05a / 05b / 05c** chapters are optional warm-ups: each one runs a single technology in a single Docker container so you see them in isolation before chapter 06 assembles them.

Chapters 6 → 18 form the **core MLflow track**. Each one is a 30 – 60 min hands-on lab.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. Detailed chapter index

### 5.1 Theory & onboarding

| # | Title | Description |
|---|---|---|
| 01 | [Introduction — purpose & value of MLOps](./01-theoretical-notes-introduction-mlops-purpose-value.md) | Why MLOps exists, what it solves, where it sits next to DevOps. |
| 02 | [Introduction to MLflow & MLOps](./02-theoretical-notes-introduction-to-mlflow-and-mlops.md) | The 4 MLflow components, mental model. |
| 03 | [Quiz 1 — questions](./03-quiz-1-mlops-use-cases-questions.md) | Self-check on chapters 1 – 2. |
| 04 | [Quiz 1 — answers](./04-quiz-1-mlops-use-cases-answers.md) | Detailed corrections. |

### 5.2 Setup

| # | Title | What you build | Project folder |
|---|---|---|---|
| 05 | [Practical 1 — Install MLflow on Ubuntu 24 with venv](./05-practical-work-1-installing-mlflow-on-ubuntu-24-with-venv.md) | First MLflow server, no Docker. | *(no folder — runs from your terminal)* |
| 06 | [Practical 2 — Install Docker Desktop & run MLflow + FastAPI + Streamlit with Docker Compose](./06-practical-work-2-installing-docker-desktop-and-running-mlflow-fastapi-streamlit-with-docker-compose.md) | The base stack used by every later chapter. | [`chap06-mlops-stack/`](./chap06-mlops-stack/) |

#### 5.2.1 Optional warm-ups — one technology per Docker container

These three short labs let you see Streamlit, FastAPI, and MLflow **in isolation** before chapter 06 assembles them. Each one is a 5-file, single-service Docker project. Skip them if you're already comfortable with all three — go straight to chapter 06.

| # | Title | What you build | Project folder |
|---|---|---|---|
| 05a | [Practical 1a — A minimal Streamlit app, dockerized](./05a-practical-work-1a-installing-streamlit-and-running-a-minimal-app-with-docker.md) | A Streamlit page with widgets, in a single container. | [`chap05a-streamlit-minimal-with-docker/`](./chap05a-streamlit-minimal-with-docker/) |
| 05b | [Practical 1b — A FastAPI calculator API, dockerized](./05b-practical-work-1b-installing-fastapi-and-running-a-calculator-api-with-docker.md) | `/add`, `/sub`, `/mul`, `/div` with Pydantic + uvicorn. | [`chap05b-fastapi-calculator-with-docker/`](./chap05b-fastapi-calculator-with-docker/) |
| 05c | [Practical 1c — An MLflow server in Docker + a first logged run](./05c-practical-work-1c-installing-mlflow-server-with-docker-and-logging-a-first-run.md) | Dockerized MLflow tracking server + `hello_mlflow.py`. | [`chap05c-mlflow-minimal-with-docker/`](./chap05c-mlflow-minimal-with-docker/) |

### 5.3 MLflow concepts (one per chapter, all on the Docker stack)

| # | MLflow function(s) introduced | Lesson | Project folder |
|---|---|---|---|
| 07 | `mlflow.set_experiment` | [07 — Organizing runs with `set_experiment`](./07-practical-work-3-organizing-mlflow-runs-with-set-experiment-using-docker-compose.md) | [`chap07-organizing-mlflow-runs-with-set-experiment/`](./chap07-organizing-mlflow-runs-with-set-experiment/) |
| 08 | `mlflow.start_run(run_name=...)`, `mlflow.last_active_run` | [08 — Naming runs with `start_run` / `last_active_run`](./08-practical-work-4-naming-mlflow-runs-with-start-run-and-last-active-run-using-docker-compose.md) | [`chap08-naming-mlflow-runs-with-start-run-and-last-active-run/`](./chap08-naming-mlflow-runs-with-start-run-and-last-active-run/) |
| 09 | `mlflow.log_param`, `mlflow.log_metric` | [09 — Logging params and metrics](./09-practical-work-5-logging-mlflow-params-and-metrics-with-log-param-and-log-metric-using-docker-compose.md) | [`chap09-logging-params-and-metrics-with-log-param-and-log-metric/`](./chap09-logging-params-and-metrics-with-log-param-and-log-metric/) |
| 10 | `mlflow.log_artifact`, `mlflow.log_artifacts` | [10 — Saving artifacts (CSV, plot)](./10-practical-work-6-saving-mlflow-artifacts-with-log-artifact-and-log-artifacts-using-docker-compose.md) | [`chap10-saving-mlflow-artifacts-with-log-artifact-and-log-artifacts/`](./chap10-saving-mlflow-artifacts-with-log-artifact-and-log-artifacts/) |
| 11 | `mlflow.set_tag`, `mlflow.set_tags` | [11 — Tagging runs](./11-practical-work-7-tagging-mlflow-runs-with-set-tag-and-set-tags-using-docker-compose.md) | [`chap11-tagging-mlflow-runs-with-set-tag-and-set-tags/`](./chap11-tagging-mlflow-runs-with-set-tag-and-set-tags/) |
| 12 | Loops over `start_run` / `set_experiment`, `nested=True` | [12 — Multiple runs & multiple experiments](./12-practical-work-8-running-multiple-mlflow-runs-and-experiments-with-loops-using-docker-compose.md) | [`chap12-running-multiple-mlflow-runs-and-experiments/`](./chap12-running-multiple-mlflow-runs-and-experiments/) |
| 13 | `mlflow.sklearn.autolog` | [13 — Automating logging with `autolog`](./13-practical-work-9-automating-mlflow-logging-with-sklearn-autolog-using-docker-compose.md) | [`chap13-automating-mlflow-logging-with-sklearn-autolog/`](./chap13-automating-mlflow-logging-with-sklearn-autolog/) |
| 14 | `infer_signature`, `mlflow.sklearn.log_model(... signature=, input_example=)` | [14 — Saving models with signature & input example](./14-practical-work-10-saving-mlflow-models-with-signature-and-input-example-using-docker-compose.md) | [`chap14-saving-mlflow-models-with-signature-and-input-example/`](./chap14-saving-mlflow-models-with-signature-and-input-example/) |
| 15 | `mlflow.pyfunc.PythonModel`, `mlflow.pyfunc.log_model(conda_env=...)`, `joblib`, `cloudpickle` | [15 — Wrapping sklearn with `pyfunc` + custom Conda env](./15-practical-work-11-wrapping-sklearn-models-with-mlflow-pyfunc-and-custom-conda-env-using-docker-compose.md) | [`chap15-wrapping-sklearn-with-mlflow-pyfunc-and-custom-conda-env/`](./chap15-wrapping-sklearn-with-mlflow-pyfunc-and-custom-conda-env/) |
| 16 | `mlflow.pyfunc.load_model`, `mlflow.evaluate` | [16 — Loading models & running `mlflow.evaluate`](./16-practical-work-12-loading-mlflow-models-and-running-mlflow-evaluate-using-docker-compose.md) | [`chap16-loading-mlflow-models-and-running-mlflow-evaluate/`](./chap16-loading-mlflow-models-and-running-mlflow-evaluate/) |
| 17 | `mlflow.register_model`, `MlflowClient.transition_model_version_stage`, `models:/.../Production` | [17 — Model Registry & `MlflowClient`](./17-practical-work-13-versioning-mlflow-models-with-the-model-registry-and-mlflowclient-using-docker-compose.md) | [`chap17-versioning-mlflow-models-with-the-model-registry-and-mlflowclient/`](./chap17-versioning-mlflow-models-with-the-model-registry-and-mlflowclient/) |
| 18 | MLflow CLI (`doctor`, `experiments`, `runs`, `artifacts`, `models serve`, `db upgrade`, `gc`) | [18 — Managing MLflow from the command line](./18-practical-work-14-managing-mlflow-from-the-command-line-with-the-mlflow-cli-using-docker-compose.md) | [`chap18-managing-mlflow-from-the-command-line/`](./chap18-managing-mlflow-from-the-command-line/) |

> [!NOTE]
> Each `chapXX-.../` folder also has its own focused `README.md` with the local quick-start and the curl commands of that lesson.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-6"></a>

## 6. Common runtime — ports, URLs, containers

After `docker compose up`, every chapter from 06 onward exposes the same endpoints:

| Service | Container name | Host URL | Purpose |
|---|---|---|---|
| MLflow tracking + registry | `mlflow` | http://localhost:5000 | Experiments, runs, models UI |
| FastAPI | `fastapi` | http://localhost:8000 | Programmatic endpoints (`/train`, `/predict`, …) |
| FastAPI Swagger UI | `fastapi` | http://localhost:8000/docs | Try endpoints from the browser |
| Streamlit | `streamlit` | http://localhost:8501 | Friendly UI to drive FastAPI |
| MLflow CLI (chap 18 only) | `mlflow-cli` | *(no port)* | `docker compose exec cli bash` |

Inside the Docker network, services talk to each other by name (DNS):

```text
streamlit  ──►  http://fastapi:8000     (env var API_URL)
fastapi    ──►  http://mlflow:5000      (env var MLFLOW_TRACKING_URI)
```

Persistent state lives in two named volumes (kept across `docker compose down`, wiped only with `down -v`):

| Volume | What it stores |
|---|---|
| `mlflow-db` | The SQLite metadata DB (experiments, runs, registered models) |
| `mlflow-artifacts` | All artifacts: pickled models, signatures, plots, CSVs |

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-7"></a>

## 7. Cheat sheet — Docker Compose

The minimum you'll use 95 % of the time:

| Goal | Command |
|---|---|
| Build & start (foreground, see logs) | `docker compose up --build` |
| Build & start (detached) | `docker compose up -d --build` |
| Tail logs of one service | `docker compose logs -f fastapi` |
| Open a shell in a service | `docker compose exec fastapi bash` |
| Restart one service | `docker compose restart fastapi` |
| Stop, keep volumes (state preserved) | `docker compose down` |
| Stop, wipe volumes (clean slate) | `docker compose down -v` |
| Free disk space across all chapters | `docker system prune -a --volumes` |
| List running containers | `docker compose ps` |
| Inspect ports already bound | `docker compose port fastapi 8000` |

> [!IMPORTANT]
> Always `docker compose down` **before** changing chapter — otherwise the next chapter cannot bind ports 5000 / 8000 / 8501.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-8"></a>

## 8. Folder layout reference

Every `chapXX-.../` folder is structurally identical:

```text
chapXX-<descriptive-name>/
├── README.md                 ← local quick-start for THIS chapter
├── docker-compose.yml        ← 3 services (4 in chap 18)
├── mlflow/
│   └── Dockerfile            ← MLflow server image (sqlite + local artifacts)
├── fastapi/
│   ├── Dockerfile
│   ├── requirements.txt      ← changes only when a new lib is needed
│   └── app/
│       ├── main.py           ← THE file that demonstrates today's concept
│       └── (wrapper.py)      ← only from chap 15 onward
└── streamlit/
    ├── Dockerfile
    ├── requirements.txt
    └── app/
        └── app.py            ← UI to call the FastAPI endpoints
```

So when comparing two chapters, focus on:

1. `fastapi/app/main.py` — 90 % of the diff.
2. `streamlit/app/app.py` — UI follow-up.
3. `fastapi/requirements.txt` — sometimes a new dependency.
4. `docker-compose.yml` — almost never changes (chap 18 is the exception: it adds a `cli` service).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-9"></a>

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Bind for 0.0.0.0:5000 failed: port is already allocated` | Previous chapter still running. | `docker compose down` in the **previous** chapter folder. |
| MLflow UI on :5000 returns 502 / hangs | The healthcheck hasn't passed yet. | Wait 10 – 20 s. Or `docker compose logs -f mlflow`. |
| FastAPI logs `ConnectionRefusedError` to `mlflow:5000` | FastAPI booted before MLflow was healthy. | Already mitigated by `depends_on: condition: service_healthy`. If it persists: `docker compose restart fastapi`. |
| Streamlit shows `ConnectionError to http://fastapi:8000` | FastAPI crashed. | `docker compose logs -f fastapi`. Most often a Python error in your `main.py` edit. |
| Nothing changes after editing `main.py` | Image was cached. | `docker compose up --build` (rebuilds), or use a bind mount + `--reload`. |
| `mlflow.evaluate` (chap 16) errors about `shap` | Optional dep not installed. | Either ignore, or `pip install shap` in the FastAPI image. |
| `mlflow.pyfunc.load_model("models:/.../Production")` returns 404 (chap 17) | No version is in stage `Production` yet. | `POST /promote` with `stage: "Production"` first. |
| Can't reach `http://localhost:5001` from the chap 18 served model | The `cli` service in `docker-compose.yml` doesn't publish port 5001 by default. | Add `ports: ["5001:5001"]` to the `cli` service and `docker compose up -d cli`. |

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-10"></a>

## 10. Suggested learning paths

### 10.1 The full course (recommended)

Read in order: 01 → 02 → 03 → 04 → 05 → **05a → 05b → 05c** → **06 → 18**, one chapter per session. The 05a/b/c trio is optional but very low cost (~15 min each) and makes chapter 06 feel obvious.

### 10.2 Express path — "I just want to ship a model"

If you already know what tracking is and only need the productionization story:

```text
05a  Streamlit minimal                  ← only if you've never seen Streamlit
05b  FastAPI calculator                 ← only if you've never seen FastAPI
06   Stack on Docker
14   Signature + input_example          ← the model becomes deployable
15   pyfunc + Conda env                 ← wrap business rules
17   Model Registry                     ← stable models:/X/Production URI
18   CLI — mlflow models serve          ← serve it in one command
```

### 10.3 Tracking-only path — "I'm a data scientist, I just want clean experiments"

```text
05c  MLflow server in Docker            ← optional but the easiest install
06   Stack on Docker
07   set_experiment
08   start_run / last_active_run
09   log_param + log_metric
11   set_tag
12   loops & sweeps
13   autolog
```

### 10.4 If you have one weekend

```text
Saturday morning:    01 → 02 → 05a → 05b → 05c → 06
Saturday afternoon:  07 → 09
Sunday morning:      10, 11, 13
Sunday afternoon:    14, 15, 17
```

You'll come out with a runnable, production-shaped MLflow workflow.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>Ready?</strong> Open <a href="./05a-practical-work-1a-installing-streamlit-and-running-a-minimal-app-with-docker.md">chapter 05a</a> for the gentlest start, or jump straight into <a href="./06-practical-work-2-installing-docker-desktop-and-running-mlflow-fastapi-streamlit-with-docker-compose.md">chapter 06</a> for the full stack.<br/>
  <a href="#top">↑ Back to the top</a>
</p>
