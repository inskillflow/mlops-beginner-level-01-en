# chap20a - Step-by-step recap: Hello MLflow basics

The full lesson lives at [`../20a-practical-work-15a-mlflow-step-by-step-recap-hello-mlflow-basics.md`](../20a-practical-work-15a-mlflow-step-by-step-recap-hello-mlflow-basics.md).

## Run it (100% Docker, no Python on the host)

```bash
# First, make sure you have read the Dockerfile and docker-compose.yml files
# to understand how the mlflow server is set up and how the mlflow tracking
# server is configured to use a local directory for storing artifacts and a
# SQLite database for tracking metadata.

# 1. Create a new directory `database/` and `mlruns/` in the current
#    directory to store the mlflow tracking data and artifacts.
#    --> THIS IS REQUIRED. If the host folders don't exist, the bind mounts
#        in docker-compose.yml will fail (or end up owned by root and the
#        SQLite file will not be writable). Create them first:
cd chap20a-mlflow-step-by-step-recap-hello-mlflow-basics
mkdir database mlruns

# 2. Then execute the following commands to start the mlflow server, run the
#    hello_mlflow.py script, and then stop the mlflow server:

docker compose up -d --build
# This command starts the mlflow server in detached mode, allowing you to
# run hello_mlflow.py while the server is running in the background.
# Check this URL: http://localhost:5000  (empty UI: only "Default" experiment)

# python hello_mlflow.py
# This command will NOT work because the mlflow server is running inside a
# Docker container, and hello_mlflow.py tries to connect to the server using
# the localhost address, which (from your host) doesn't always reach the
# container -- and you may not have mlflow installed on the host either.
# To fix this, run hello_mlflow.py INSIDE the mlflow container:

docker compose exec -d mlflow python hello_mlflow.py
# This executes hello_mlflow.py inside the mlflow container, allowing it to
# connect to the mlflow server (at http://localhost:5000 from inside the
# container, which IS the mlflow server itself) and log the experiment data
# successfully.

# Check again http://localhost:5000  --> you should see the logged experiment
# data in the mlflow UI. Click on the experiment name "hello_mlflow" and
# the run "my_first_run" to see the parameters and metrics that were logged.

# 3. Stop the mlflow server when you are done.
docker compose down
```

## What ends up on your host

After you run the script, the bind mounts populate two real folders next to
`docker-compose.yml`:

```
chap20a-mlflow-step-by-step-recap-hello-mlflow-basics/
├── database/
│   └── mlflow.db                    <- SQLite metadata
└── mlruns/
    └── 0/
        └── <run_id>/                <- artifacts
```

You can browse them with your file explorer.

## Why does `set_tracking_uri("http://localhost:5000")` work?

Because the script runs **inside** the `mlflow` container, so `localhost:5000`
is the MLflow server itself.

## Tear down

```bash
docker compose down
# Optional -- delete the host folders to wipe everything:
rm -rf database mlruns
```

## Recap

```bash
cd 01-mlflow-step-by-step-recap-hello-mlflow-basics
mkdir database mlruns
docker compose up -d --build
docker compose exec -d mlflow python hello_mlflow.py
docker-compose down 
```
