# chap20c - Step-by-step recap: first ElasticNet pipeline on red-wine-quality

The full lesson lives at [`../20c-practical-work-15c-mlflow-step-by-step-recap-elasticnet-on-red-wine-quality.md`](../20c-practical-work-15c-mlflow-step-by-step-recap-elasticnet-on-red-wine-quality.md).


This lab shows how to run several MLflow experiments with different hyperparameters.

You will train the same ElasticNet model three times, but each run will use different values of:

```text
alpha
l1_ratio
````

The goal is to compare the results in the MLflow UI.


## What's new vs chap20b - MLflow ElasticNet on Red Wine Quality with Docker

- A real ML pipeline: `pd.read_csv` -> `train_test_split` -> `ElasticNet.fit` -> compute `rmse / mae / r2`
- `mlflow.sklearn.log_model(lr, "mymodel")` to persist the trained model

---

# 1. Clone the project

```bash
git clone https://github.com/inskillflow/mlops-beginner-level-01-en.git
```

Then enter the project folders in order:

```bash
cd mlops-beginner-level-01-en/chap20a-mlflow-step-by-step-recap-hello-mlflow-basics
# done

cd ../chap20b-mlflow-step-by-step-recap-printing-the-tracking-uri
# done

cd ../chap20c-mlflow-step-by-step-recap-elasticnet-on-red-wine-quality
# start project #3
```

---

# 2. Stop any running containers first

Before starting this lab, make sure that no other MLflow container is already running on port `5000`.

## Method 1 — Stop containers from another project

Go to the other project folder:

```bash
cd other-project
docker compose down
```

This stops and removes the containers created by that project.

---

## Method 2 — Use Docker Desktop

You can also open **Docker Desktop** and manually:

```text
1. Go to Containers
2. Find the running container
3. Stop it
4. Delete it if necessary
```

This is useful if you do not remember which folder started the container.

---

# 3. Start project #3

You should now be inside this folder:

```bash
chap20c-mlflow-step-by-step-recap-elasticnet-on-red-wine-quality
```

Create the required folders:

```bash
mkdir -p database mlruns
```

Or On Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force database, mlruns
```

Or create the 2 folders manually (database and mlruns)



Then start the MLflow server:

```bash
docker compose up -d --build
```

Open the MLflow UI:

```text
http://localhost:5000
```

At this point, the UI may be empty or may only show the default experiment. This is normal.



⚠️ Important: create `database/` and `mlruns/` before starting Docker.

If you forget them, stop Docker, create the folders, and restart with `--build`:

```bash
docker compose down
mkdir -p database mlruns
docker compose up -d --build
````

The `--build` option forces Docker to rebuild instead of using the cache.


---

# 4. Run experiment 1

Run the model with:

```text
alpha = 0.1
l1_ratio = 0.1
```

Command:

```bash
docker compose exec mlflow python train_with_mlflow.py --alpha 0.1 --l1_ratio 0.1
```

Then check:

```text
http://localhost:5000
```

You should see a new MLflow run.

---

# 5. Run experiment 2

Run the model with:

```text
alpha = 0.5
l1_ratio = 0.5
```

Command:

```bash
docker compose exec mlflow python train.py --alpha 0.5 --l1_ratio 0.5
```

Then check again:

```text
http://localhost:5000
```

You should now see another run.

---

# 6. Run experiment 3

Run the model with:

```text
alpha = 0.9
l1_ratio = 0.9
```

Command:

```bash
docker compose exec mlflow python train.py --alpha 0.9 --l1_ratio 0.9
```

Then check again:

```text
http://localhost:5000
```

You should now see three different runs.

---

# 7. Compare the runs in MLflow

In the MLflow UI, compare the runs using:

```text
Parameters
Metrics
Artifacts
Model output
```

The important idea is this:

```text
Each run uses the same training script, but different hyperparameters.
MLflow records each run separately.
This allows you to compare which configuration gives the best results.
```

For example:

```text
Run 1: alpha = 0.1, l1_ratio = 0.1
Run 2: alpha = 0.5, l1_ratio = 0.5
Run 3: alpha = 0.9, l1_ratio = 0.9
```

---

# 8. Stop the containers

When you are finished:


```bash
docker compose down       # keep all runs
docker compose down -v    # wipe everything
```






## ⚠️ Important warning — create the folders first

⚠️ Be careful: if the two folders `database/` and `mlruns/` are not created before starting Docker, MLflow may not save the experiment data correctly.

You may open the MLflow UI at:

```text
http://localhost:5000
````

but you may not see your runs, metrics, parameters, or artifacts.

Before running Docker, create the two folders manually:

```bash
mkdir -p database mlruns
```

On Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force database, mlruns
```

If you already started Docker without creating these folders, do this:

```bash
docker compose down
mkdir -p database mlruns
docker compose up -d --build
```

On Windows PowerShell:

```powershell
docker compose down
New-Item -ItemType Directory -Force database, mlruns
docker compose up -d --build
```

⚠️ The `--build` option is important here.

It forces Docker to rebuild the image instead of reusing the previous cached version.

Without `--build`, Docker may reuse the old cached configuration, and your fix may not be applied correctly.

---

# Final command recap

```bash
git clone https://github.com/inskillflow/mlops-beginner-level-01-en.git

cd mlops-beginner-level-01-en/chap20a-mlflow-step-by-step-recap-hello-mlflow-basics
# done

cd ../chap20b-mlflow-step-by-step-recap-printing-the-tracking-uri
# done

cd ../chap20c-mlflow-step-by-step-recap-elasticnet-on-red-wine-quality
# start project #3

mkdir -p database mlruns

docker compose up -d --build

docker compose exec mlflow python train.py --alpha 0.1 --l1_ratio 0.1
docker compose exec mlflow python train.py --alpha 0.5 --l1_ratio 0.5
docker compose exec mlflow python train.py --alpha 0.9 --l1_ratio 0.9

docker compose down
```

PowerShell version:

```powershell
git clone https://github.com/inskillflow/mlops-beginner-level-01-en.git

cd mlops-beginner-level-01-en/chap20a-mlflow-step-by-step-recap-hello-mlflow-basics
# done

cd ../chap20b-mlflow-step-by-step-recap-printing-the-tracking-uri
# done

cd ../chap20c-mlflow-step-by-step-recap-elasticnet-on-red-wine-quality
# start project #3

New-Item -ItemType Directory -Force database, mlruns

docker compose up -d --build

docker compose exec mlflow python train.py --alpha 0.1 --l1_ratio 0.1
docker compose exec mlflow python train.py --alpha 0.5 --l1_ratio 0.5
docker compose exec mlflow python train.py --alpha 0.9 --l1_ratio 0.9

docker compose down
```

---

# To enter the container manually

First list the running containers:

```bash
docker ps
```

Then enter the MLflow container:

```bash
docker exec -it <container_id> bash
```

Inside the container:

```bash
ls
python train_with_mlflow.py --alpha 0.1 --l1_ratio 0.1
exit
```

The Docker Compose equivalent is simpler:

```bash
docker compose exec mlflow bash
```

---

# Why not use `-d` with `docker compose exec`?

You may see this command:

```bash
docker compose exec -d mlflow python train.py --alpha 0.1 --l1_ratio 0.1
```

It works, but it runs the script in detached mode.


## Recommended version ⚠️⚠️ :

```bash
docker compose exec mlflow python train.py --alpha 0.1 --l1_ratio 0.1
```

This way, if something goes wrong, the error appears immediately in the console !!!!

---

# What happens if I forgot to create `mlruns` and `database`?

If you forgot to create these folders before starting Docker, you may have problems with:

```text
SQLite database permissions
MLflow metadata storage
artifact storage
root-owned folders
bind mount errors
```

In simple words:

```text
MLflow needs a place to store experiment information and artifacts.
The database/ folder stores the SQLite metadata.
The mlruns/ folder stores the run artifacts.
If these folders are missing or created incorrectly, MLflow may not be able to write data properly.
```

---

# How to fix it

## Step 1 — Stop the containers

```bash
docker compose down
```

## Step 2 — Create the required folders

Linux, macOS, Git Bash:

```bash
mkdir -p database mlruns
```

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force database, mlruns
```

## Step 3 — Rebuild and restart the containers

```bash
docker compose up -d --build
```

The `--build` option forces Docker to rebuild the image.

This is useful when:

```text
the Dockerfile changed
dependencies changed
the environment needs to be refreshed
the previous container was created incorrectly
```

## Step 4 — Run the training script again

```bash
docker compose exec mlflow python train.py --alpha 0.1 --l1_ratio 0.1
```

Then open:

```text
http://localhost:5000
```

---

# Troubleshooting 1 : port 5000 already used ⚠️

On Windows CMD:

```bat
netstat -ano | findstr :5000
tasklist | findstr 12345
taskkill /PID 12345 /F
```

On PowerShell:

```powershell
Get-NetTCPConnection -LocalPort 5000
Stop-Process -Id 12345 -Force
```

Replace `12345` with the PID shown by the command.

Simple explanation:

Port `5000` is like a door. If another application is already using this door, MLflow cannot start on the same port. You must either stop the other application or change the port used by MLflow.





---

# Troubleshooting 2 : docker Desktop not starting ⚠️



Open **PowerShell as Administrator**, then run this:

```powershell
# 1. Stop Docker Desktop processes
Get-Process *docker* -ErrorAction SilentlyContinue | Stop-Process -Force

# 2. Stop Docker Desktop service
Stop-Service com.docker.service -Force -ErrorAction SilentlyContinue

# 3. Force-stop WSL backend used by Docker
wsl --shutdown
```

Then wait **10–15 seconds**.

To restart Docker Desktop:

```powershell
Start-Service com.docker.service
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
```

If it is still frozen, use the stronger version:

```powershell
taskkill /F /IM "Docker Desktop.exe"
taskkill /F /IM "com.docker.backend.exe"
taskkill /F /IM "com.docker.service.exe"
taskkill /F /IM "dockerd.exe"
wsl --shutdown
```

Then restart Docker Desktop manually from the Start menu.

Do **not** delete Docker folders yet. First try force stop + `wsl --shutdown`.






