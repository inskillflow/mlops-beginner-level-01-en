# chap20b - Step-by-step recap: print the tracking URI

The full lesson lives at [`../20b-practical-work-15b-mlflow-step-by-step-recap-printing-the-tracking-uri.md`](../20b-practical-work-15b-mlflow-step-by-step-recap-printing-the-tracking-uri.md).

## What's new vs chap20a

A single line:

```python
print("Tracking URI:", mlflow.get_tracking_uri())
```

## Run it (100% Docker, no Python on the host)

```bash
# First, make sure you have read the Dockerfile and docker-compose.yml files
# to understand how the mlflow server is set up and how it persists metadata
# (SQLite) and artifacts on local folders.

# 1. Create `database/` and `mlruns/` in the current directory.
#    REQUIRED: bind mounts in docker-compose.yml expect them to exist.
cd chap20b-mlflow-step-by-step-recap-printing-the-tracking-uri
mkdir database mlruns

# 2. Start the mlflow server in detached mode.
docker compose up -d --build
# Check this URL: http://localhost:5000  (empty UI: only "Default" experiment)

# python hello_mlflow.py
# Will NOT work: mlflow may not be installed on the host, and the local
# `mlflow/` folder also shadows the `mlflow` Python package on the host.
# Run it INSIDE the mlflow container instead:

docker compose exec -d mlflow python hello_mlflow.py
# Expected stdout (visible via `docker compose logs mlflow`):
#   Tracking URI: http://localhost:5000
#   Done. Open http://localhost:5000 to see your run.

# Refresh http://localhost:5000  -> experiment "hello_mlflow" appears with
# a new run named "my_first_run".

# 3. Stop the mlflow server.
docker compose down
```

## What ends up on your host

```
chap20b-mlflow-step-by-step-recap-printing-the-tracking-uri/
├── database/mlflow.db
└── mlruns/0/<run_id>/...
```

## Tear down (full wipe)

```bash
docker compose down
rm -rf database mlruns        # delete persisted DB + artifacts
```



## Recap

> Suppose that you are still located in the first project 
> you need to execute docker compose down to stop the mlflow server 
> Then you can navigate to the second project directory and repeat the same steps to start the mlflow server, run the hello_mlflow.py script, and then stop the mlflow server:


```bash
docker compose down
cd ../chap20b-mlflow-step-by-step-recap-printing-the-tracking-uri
mkdir database mlruns
docker compose up -d --build  
docker compose exec -d mlflow python hello_mlflow.py # This command will not show you the print commands since it's running in the background (detached mode)
docker compose exec mlflow python hello_mlflow.py  # This command will show you the 2 prints in the beging and end of execution of your .py file
# Check your terminal logs
# Check the http://localhost:5000
docker-compose down 
```





