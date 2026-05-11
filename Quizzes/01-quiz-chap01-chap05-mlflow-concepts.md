# Quiz - chap01 to chap05 - MLflow concepts (Multiple Choice)

50 multiple-choice questions, no answers.

Focus: **what MLflow is**, **why it exists**, **the objective of each chapter**, **MLflow concepts** (experiments, runs, params, metrics, artifacts, tags), and **comparing experiments**. NO troubleshooting questions.

> [!TIP]
> Pick **one** answer (A, B, C or D) for each question. For every question, also explain in one sentence **why the other three options are wrong**.

---

## Section 1 - What is MLflow and why does it exist?

### Question 1
What is MLflow?

A) A Python web framework for serving HTTP APIs<br/>
B) An open-source platform for managing the end-to-end ML lifecycle (tracking, models, registry, projects)<br/>
C) A cloud-only paid service for monitoring ML in production<br/>
D) A scikit-learn replacement for training models

### Question 2
Which of the following is **NOT** one of the four MLflow components?

A) Tracking<br/>
B) Projects<br/>
C) Models<br/>
D) Inference Gateway

### Question 3
The four core MLflow components are Tracking, Projects, Models, and:

A) Model Registry<br/>
B) Model Marketplace<br/>
C) Model Lab<br/>
D) Model Server

### Question 4
Which MLflow component is at the heart of chapters 01 to 05?

A) Projects<br/>
B) Model Registry<br/>
C) Tracking<br/>
D) Models

### Question 5
MLflow is best described as:

A) Framework-specific (works only with TensorFlow)<br/>
B) Framework-agnostic (works with scikit-learn, PyTorch, TF, XGBoost, etc.)<br/>
C) Language-specific (Python only, with no Java/R/REST support)<br/>
D) Cloud-specific (Azure ML only)

### Question 6
Which problem does MLflow primarily help solve?

A) Distributing model training across a GPU cluster<br/>
B) Tracking which hyperparameters produced which metrics and which artifacts<br/>
C) Writing the training code itself<br/>
D) Replacing scikit-learn as an ML library

### Question 7
A data scientist who trains 20 models per week WITHOUT MLflow typically suffers from:

A) GPU shortages<br/>
B) Lack of internet connectivity<br/>
C) Loss of traceability between hyperparameters, metrics and saved models<br/>
D) Slow training speed

### Question 8
MLflow runs on:

A) Only Linux<br/>
B) Only Windows<br/>
C) Only macOS<br/>
D) Linux, Windows and macOS (anywhere Python or Docker runs)

### Question 9
MLflow can be:

A) Used only via the paid Databricks managed version<br/>
B) Self-hosted offline on a laptop, completely free of charge<br/>
C) Used only as a SaaS hosted by MLflow Inc.<br/>
D) Used only inside Kubernetes clusters

### Question 10
Which statement about MLflow Tracking is TRUE?

A) MLflow stores only metrics, not the model itself<br/>
B) MLflow stores only the model, not the metrics or params<br/>
C) MLflow stores both the trained model (artifact) AND the params, metrics and tags that describe how it was trained<br/>
D) MLflow stores nothing on disk; it only displays values in the UI

---

## Section 2 - The objective of each chapter file

### Question 11
The objective of **chap01 (Hello MLflow basics)** is to:

A) Train a full ElasticNet pipeline with multiple metrics<br/>
B) Boot a minimal MLflow server in Docker and create one trivial run<br/>
C) Compare 10 experiments side by side<br/>
D) Deploy a model as a REST API

### Question 12
The objective of **chap02 (Print the tracking URI)** is to:

A) Refactor the codebase to support multiple URIs<br/>
B) Connect MLflow to AWS S3<br/>
C) Make the student aware of WHERE MLflow is currently writing data, before logging anything serious<br/>
D) Print the run_id of every active run

### Question 13
chap02 differs from chap01 by:

A) Adding a brand new Docker service<br/>
B) Adding a single line: `print("Tracking URI:", mlflow.get_tracking_uri())`<br/>
C) Replacing SQLite with Postgres<br/>
D) Adding a model registry

### Question 14
The objective of **chap03 (ElasticNet on red-wine-quality)** is to:

A) Run the SAME model with THREE different hyperparameter sets and compare them in the UI<br/>
B) Deploy the model to production behind a load balancer<br/>
C) Replace ElasticNet with XGBoost<br/>
D) Add a CI/CD pipeline

### Question 15
Which MLflow API calls does chap03 introduce that did NOT exist in chap01 or chap02?

A) Only `mlflow.log_metric()`<br/>
B) `mlflow.log_param()`, `mlflow.log_metric()` AND `mlflow.sklearn.log_model()`<br/>
C) Only `mlflow.log_artifact()`<br/>
D) Only `mlflow.set_experiment()`

### Question 16
The objective of **chap04 (trainer in a second Docker service)** is to:

A) Replace MLflow with TensorBoard<br/>
B) Move the training script OUT of the MLflow server container and INTO its own dedicated `trainer` service<br/>
C) Deploy MLflow on Kubernetes<br/>
D) Encrypt the SQLite database

### Question 17
The "broken-by-design" surprise of chap04 is:

A) The MLflow server fails to start<br/>
B) The training crashes with a Python error<br/>
C) The runs VANISH because the trainer writes to `file:///code/mlruns` INSIDE its own container, which then gets deleted by `--rm`<br/>
D) The Docker image cannot be built

### Question 18
The objective of **chap05 (env var)** is to:

A) Hardcode the URI inside `train.py`<br/>
B) Fix the chap04 bug by injecting `MLFLOW_TRACKING_URI` from `docker-compose.yml` and reading it with `os.getenv(...)`<br/>
C) Add a model registry<br/>
D) Add multi-user authentication

### Question 19
Why does chap05 use `os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")` instead of `mlflow.set_tracking_uri("http://mlflow:5000")` directly?

A) `os.getenv` is faster than `set_tracking_uri`<br/>
B) To follow the 12-factor app pattern (config from env, not hardcoded in source)<br/>
C) Because MLflow does not support `set_tracking_uri()`<br/>
D) Because Docker forbids hardcoded URLs in Python files

### Question 20
Why is the fix introduced as a SEPARATE chapter (chap05) instead of being merged directly into chap04?

A) Because the fix requires a different Python version<br/>
B) Because the course wants the student to FEEL the bug in chap04 before learning the fix in chap05<br/>
C) Because chap05 uses a different operating system<br/>
D) Because the fix is too complex for a single chapter

### Question 21
The chap01 → chap05 progression teaches MLflow by:

A) Dumping all features into chap01<br/>
B) Adding ONE new concept per chapter, building up complexity gradually<br/>
C) Starting with the most advanced topics first<br/>
D) Focusing only on the model registry

### Question 22
The progression chap01 → chap05 ends with:

A) A multi-service Docker setup where the trainer reads its tracking URI from an environment variable<br/>
B) A single container running MLflow on a developer laptop<br/>
C) A Kubernetes deployment with Helm charts<br/>
D) A serverless AWS Lambda function

### Question 23
Why does the course switch from `docker compose exec mlflow python train.py` (chap01-03) to `docker compose run --rm trainer ...` (chap04+)?

A) Because `docker compose exec` was deprecated<br/>
B) To SEPARATE the training environment (with sklearn/pandas) from the MLflow server environment (mlflow only)<br/>
C) Because `exec` requires root privileges<br/>
D) Because `run` is mathematically faster than `exec`

---

## Section 3 - Experiments, runs, params, metrics, artifacts, tags

### Question 24
In MLflow, an **EXPERIMENT** is:

A) A single execution of `train.py`<br/>
B) A named container that groups related runs together (e.g., "wine_quality_elasticnet")<br/>
C) The output file of a model<br/>
D) A configuration file

### Question 25
In MLflow, a **RUN** is:

A) The same thing as an experiment<br/>
B) A single execution of training code, with params, metrics, tags and artifacts attached<br/>
C) A folder on disk only, with no metadata<br/>
D) A REST API call

### Question 26
Which of the following should be logged with `mlflow.log_param()` ?

A) The training time in seconds<br/>
B) The value `alpha=0.5` used at training start<br/>
C) The trained model `.pkl` file<br/>
D) The validation RMSE

### Question 27
Which of the following should be logged with `mlflow.log_metric()` ?

A) The git commit hash<br/>
B) The training RMSE at the end of training<br/>
C) The value `alpha=0.5`<br/>
D) The version of scikit-learn

### Question 28
Which of the following should be logged with `mlflow.set_tag()` ?

A) The training RMSE<br/>
B) The trained ElasticNet object<br/>
C) A short string label like `version=v1` or `owner=alice`<br/>
D) The whole training dataset CSV

### Question 29
Which of the following should be logged with `mlflow.sklearn.log_model()` ?

A) The training RMSE value<br/>
B) The `alpha` hyperparameter value<br/>
C) The trained `ElasticNet` Python object (the model itself)<br/>
D) The dataset CSV file

### Question 30
A `run_id` in MLflow is:

A) An incremental integer like 1, 2, 3<br/>
B) A 128-bit UUID generated at run creation<br/>
C) The current Unix timestamp<br/>
D) The SHA-256 hash of the model weights

### Question 31
Why does MLflow use a UUID for `run_id` rather than an incremental integer?

A) UUIDs take less storage than integers<br/>
B) UUIDs allow many trackers (and many clients) to generate ids in parallel without collisions<br/>
C) UUIDs are faster to sort in SQL<br/>
D) MLflow actually uses incremental integers internally

### Question 32
Can a parameter logged with `log_param("alpha", 0.5)` be UPDATED later in the same run?

A) Yes, the new value silently overwrites the old one<br/>
B) No — params are IMMUTABLE once logged (a second call raises an exception)<br/>
C) Yes, but only by an admin user<br/>
D) Yes, but only with a paid license

### Question 33
Can a metric logged with `log_metric("rmse", x)` be logged MULTIPLE times in a single run?

A) No, only one value per metric per run<br/>
B) Yes — multiple logs produce a TIME SERIES (one value per step or per epoch), shown as a curve in the UI<br/>
C) Yes, but only if you change the metric name each time<br/>
D) No, MLflow raises an exception on the second call

### Question 34
The MLflow "Model" format is:

A) A `.h5` file (Keras only)<br/>
B) An `MLmodel` YAML file (alongside the model binary) that describes how to LOAD the model regardless of framework<br/>
C) A pure `torch.save` `.pt` file<br/>
D) An ONNX `.onnx` file only

### Question 35
Which of these is captured AUTOMATICALLY by MLflow for every run, with no explicit logging call?

A) The user's git diff at run time<br/>
B) The START TIME and END TIME of the run<br/>
C) The full RAM dump of the Python process<br/>
D) A screenshot of the developer's screen

### Question 36
MLflow Tracking separates METADATA storage (SQLite/Postgres) from ARTIFACT storage (filesystem/S3) because:

A) Metadata is much larger than artifacts<br/>
B) Artifacts are binary blobs that belong in object storage, while metadata is small structured data that belongs in a SQL DB<br/>
C) The MLflow license requires it<br/>
D) There is no separation; both go in the same place

---

## Section 4 - Comparing experiments (the heart of MLflow)

### Question 37
To find your BEST ElasticNet model among 3 chap03 runs in the MLflow UI, you would:

A) Open each run individually and write the metrics down on paper<br/>
B) Sort the run list by the `rmse` column in ASCENDING order and pick the top one<br/>
C) Use the Python API to query the SQLite file manually<br/>
D) There is no way to compare runs in the UI

### Question 38
For RMSE and MAE, the BETTER model is the one with:

A) The HIGHER value<br/>
B) The LOWER value (closer to 0 is better)<br/>
C) The value closest to 1.0<br/>
D) The value closest to 0.5

### Question 39
For R² (R-squared), the BETTER model is the one with:

A) The LOWER value<br/>
B) The value closest to 0<br/>
C) The HIGHER value (closer to 1.0 is best)<br/>
D) Any negative value

### Question 40
The MAXIMUM theoretical value of R² (perfect prediction) is:

A) 0<br/>
B) 0.5<br/>
C) 1.0<br/>
D) Infinity

### Question 41
The MLflow UI **"Compare"** view (selecting multiple runs and clicking Compare) shows:

A) Only the source code differences between runs<br/>
B) A side-by-side table of params and metrics, plus scatter and parallel coordinates plots<br/>
C) Only the trained model files<br/>
D) Only the stdout logs

### Question 42
A **parallel coordinates plot** is most useful when you have:

A) One run with one hyperparameter<br/>
B) MANY runs with MANY hyperparameters — to spot which combinations correlate with low loss<br/>
C) No runs at all<br/>
D) Only categorical features

### Question 43
In chap03 we deliberately run THREE experiments (not just one) because:

A) MLflow requires at least three runs<br/>
B) COMPARING runs is the only honest way to know which hyperparameters work best — one run alone tells you nothing<br/>
C) The dataset has three columns<br/>
D) Three is the minimum for statistical significance

### Question 44
To FILTER runs in the MLflow UI where `alpha < 0.3` AND `metrics.rmse < 0.7`, you would use:

A) The SEARCH BAR at the top of the experiment view, with a filter expression like `params.alpha < 0.3 and metrics.rmse < 0.7`<br/>
B) The Python REPL with manual loops<br/>
C) A direct SQL query on the SQLite database<br/>
D) A shell script

### Question 45
Can you compare runs that belong to DIFFERENT experiments in the MLflow UI?

A) No, comparison is limited to a single experiment<br/>
B) Yes, you can pick runs across multiple experiments and compare them together<br/>
C) Only with a paid Databricks license<br/>
D) Only if the experiments share the same name

### Question 46
Two runs with the SAME `alpha` and `l1_ratio` can still produce DIFFERENT `rmse` values because:

A) MLflow randomly perturbs the metrics it stores<br/>
B) The random seed (`np.random.seed` / `random_state`) was different, or the `train_test_split` produced a different split<br/>
C) MLflow does not record metrics deterministically<br/>
D) It is impossible for them to differ

### Question 47
Why is "comparing experiments" considered the SINGLE MOST IMPORTANT value of MLflow Tracking?

A) Because comparison is faster than training<br/>
B) Because WITHOUT comparison, hyperparameter optimisation becomes invisible guesswork — you cannot improve what you cannot measure side by side<br/>
C) Because MLflow can only do comparison and nothing else<br/>
D) Because comparison saves money on GPU bills

---

## Section 5 - Tracking URI, hosts and config precedence

### Question 48
The DEFAULT value of `mlflow.get_tracking_uri()` when NOTHING has been configured is:

A) `http://localhost:5000`<br/>
B) `file:///<current working directory>/mlruns`<br/>
C) `sqlite:///mlflow.db`<br/>
D) `databricks`

### Question 49
In chap05, the tracking URI is `http://mlflow:5000`. The hostname **`mlflow`** comes from:

A) The DNS server of the host machine<br/>
B) The `/etc/hosts` file edited manually<br/>
C) The service NAME `mlflow` declared in `docker-compose.yml`, resolved by Docker's internal DNS on the shared bridge network `recap-net`<br/>
D) MLflow's internal hardcoded hostname

### Question 50
The PRECEDENCE order between configuration sources for the tracking URI, from HIGHEST to LOWEST priority, is:

A) MLflow default → environment variable → `mlflow.set_tracking_uri()`<br/>
B) `mlflow.set_tracking_uri()` in code → environment variable `MLFLOW_TRACKING_URI` → MLflow default<br/>
C) Environment variable → MLflow default → `mlflow.set_tracking_uri()`<br/>
D) All three have equal priority and any of them can win randomly

---

# Appendix - Look-ahead questions (bonus, NOT counted in the 50)

### Question B1
A **model signature** in MLflow is:

A) A digital cryptographic signature for the model file<br/>
B) A schema describing the input/output column names and dtypes the model expects at inference<br/>
C) The author's name and email<br/>
D) The git commit hash that produced the model

### Question B2
The difference between `mlflow.sklearn.log_model()` and `mlflow.pyfunc.log_model()` is:

A) There is no difference; they are aliases<br/>
B) `sklearn.log_model` is framework-specific; `pyfunc.log_model` lets you wrap ANY Python callable as an MLflow model<br/>
C) `pyfunc` is only for TensorFlow<br/>
D) `sklearn.log_model` is deprecated

### Question B3
The MLflow **Model Registry** differs from the Tracking server because:

A) It stores models in a separate place, with versions, stages (Staging/Production) and approval workflows<br/>
B) It does the same thing as Tracking under a different name<br/>
C) It is a third-party paid product<br/>
D) It only works with cloud storage

---

**End of quiz - 50 multiple-choice questions, chapters 01 to 05.**

Pick your answers and discuss them in pairs.
