# Quiz - chap01 to chap05 - MLflow concepts, objectives and comparing experiments (English version)

50 questions, no answers.

This quiz focuses on **MLflow concepts**, **the objective of each chapter file**, and **how to compare experiments**. It is NOT a troubleshooting quiz.

Chapters covered:

- chap01 - Hello MLflow basics
- chap02 - Print the tracking URI
- chap03 - First ElasticNet pipeline on red-wine-quality
- chap04 - Running the training in a second Docker service
- chap05 - Passing the tracking URI via an environment variable

> [!TIP]
> Write your answers in your own words. For multiple-choice questions, pick the right answer **and** explain in one sentence why the other options are wrong. For "TRUE / FALSE" questions, justify your answer in one sentence.

---

## Section 1 - What is MLflow and why does it exist?

### Question 1
In one sentence, what is **MLflow**? Mention the type of project it belongs to (library? framework? platform?) and the language(s) you can use it from.

### Question 2
MLflow is built around four major components:
- **Tracking**
- **Projects**
- **Models**
- **Model Registry**

Briefly define each one in 1 to 2 sentences.

### Question 3
Which of the four MLflow components is at the **heart** of chapters 01 to 05? Justify in one sentence.

### Question 4
List **three pains** that a data scientist experiences **without** MLflow when they train many models per week (for example: "I forgot the value of `alpha` that gave my best model two days ago").

### Question 5
What does it mean to say MLflow is "**framework-agnostic**"? Cite **two ML libraries** other than scikit-learn that MLflow supports natively.

### Question 6
**TRUE / FALSE**: MLflow stores both the **model itself** (a `.pkl` file or equivalent) **and** the **metrics, params and tags** that describe how it was trained.
Justify your answer in one sentence.

### Question 7
**Quick-fire**: in MLflow, what is the difference between an **experiment** and a **run**? Give a one-sentence definition for each.

### Question 8
Among the following statements about MLflow, which ones are **true**? (more than one may be correct)
- a) MLflow needs a paid license for commercial use.
- b) MLflow can run entirely offline on a laptop.
- c) MLflow requires a GPU.
- d) MLflow can serve a logged model as a REST API.
- e) MLflow only works with Python.

---

## Section 2 - The objective of each chapter file

### Question 9
**chap01 - Hello MLflow basics**.
In your own words, what is the **single concrete thing** the student must be able to do after finishing this chapter? Limit your answer to two sentences.

### Question 10
**chap02 - Print the tracking URI**.
The whole chapter adds **one line** to chap01:

```python
print("Tracking URI:", mlflow.get_tracking_uri())
```

Why does the course dedicate an entire chapter to a single print statement? What concept is the student supposed to internalise?

### Question 11
**chap03 - First ElasticNet pipeline on red-wine-quality**.
What new MLflow features does chap03 introduce that did NOT appear in chap01 or chap02? List at least **four** API calls or concepts.

### Question 12
**chap04 - Running the training in a second Docker service**.
This chapter is the only one in the series where the **first run does NOT appear in the MLflow UI**. Why is this **intentional**? What does the student learn from this "broken-by-design" experience?

### Question 13
**chap05 - Passing the tracking URI via an environment variable**.
Why is the **fix** for the chap04 bug introduced as a **separate chapter** rather than being included directly in chap04?

### Question 14
Looking at the progression chap01 → chap02 → chap03 → chap04 → chap05, fill in the blank:
"The course teaches MLflow by **adding one new concept per chapter**, starting from a minimal `start_run` and ending with `___________`."

### Question 15
**Pedagogical question**: why do chap01 and chap02 use `docker compose exec mlflow python train.py` (running the script **inside** the MLflow server container), while chap04 and chap05 switch to a separate `trainer` service running via `docker compose run --rm trainer ...`? What lesson is the course trying to teach by making this switch?

---

## Section 3 - Experiments, runs, params, metrics, artifacts, tags

### Question 16
What does `mlflow.start_run()` actually do **behind the scenes**? Mention at least **three** things that happen the moment this function is called.

### Question 17
Match each of the following data points with the right MLflow API:
- a) `alpha = 0.5`
- b) `rmse = 0.74`
- c) `git_commit = "abc123"`
- d) the trained `ElasticNet` object (pickled)
- e) the source `train.py` file

Choose from:
- `mlflow.log_param(...)`
- `mlflow.log_metric(...)`
- `mlflow.set_tag(...)`
- `mlflow.sklearn.log_model(...)`
- `mlflow.log_artifact(...)`

### Question 18
**TRUE / FALSE**: a parameter (`log_param`) can be **updated** during a run (for example, change `alpha` from `0.5` to `0.6` halfway through training).
Justify.

### Question 19
**TRUE / FALSE**: a metric (`log_metric`) can be **logged multiple times** during a single run, for example one `rmse` value per epoch.
Justify.

### Question 20
What is a `run_id`? Why is it a **128-bit UUID** instead of a simple incremental integer like in a SQL database?

### Question 21
What is the **purpose of tags** on a run, beyond just decoration? Give **two concrete tag examples** that would be useful in a real ML project.

### Question 22
When chap03 calls `mlflow.sklearn.log_model(lr, "mymodel")`, what files end up under `artifacts/mymodel/` in the MLflow UI? Name at least **three** of them.

### Question 23
What is the difference between an **artifact** and a **metric** in MLflow? Give one example of each that does NOT belong to the other category.

### Question 24
Where does MLflow store **metrics** (numerically) versus **artifacts** (binary blobs)? Why does the design separate the two?

### Question 25
Cite **three pieces of metadata** that MLflow captures **automatically** for each run, without the user having to log anything explicitly.

---

## Section 4 - Comparing experiments (the heart of MLflow)

### Question 26
After running the three experiments of chap03 with the following hyperparameters:

```text
Run 1: alpha=0.1, l1_ratio=0.1
Run 2: alpha=0.5, l1_ratio=0.5
Run 3: alpha=0.9, l1_ratio=0.1
```

How would you **identify the best model** in the MLflow UI? Describe the exact steps (which column to sort, which value indicates "best").

### Question 27
RMSE, MAE and R² are the three metrics logged in chap03.
- For **RMSE** and **MAE**, do you want to **minimise** or **maximise** the value?
- For **R²**, what is the **maximum possible value**, and what does a value of **0** mean?

### Question 28
The MLflow UI offers a "**Compare**" button that lets you select multiple runs and view them side by side. Name **three things** the comparison view shows.

### Question 29
What is a **parallel coordinates plot** in the MLflow UI? Why is it especially useful when you have **many runs** with **many hyperparameters**?

### Question 30
In chap03, why does the lesson recommend running **three experiments instead of just one**? What is the **pedagogical and scientific reason**?

### Question 31
**Scenario**: you ran 50 experiments today. You want to find all the runs where `alpha < 0.3` **AND** `rmse < 0.7`. How would you do this in the MLflow UI? (Describe the feature, no need to give exact syntax.)

### Question 32
**TRUE / FALSE**: you can compare runs that belong to **different experiments** in the MLflow UI.
Justify.

### Question 33
Imagine two runs with the same `alpha` and `l1_ratio` but different `rmse`. Cite **two possible reasons** for this difference, given what you know from the code in chap03.

### Question 34
Cite **three sortable columns** that the MLflow UI shows by default for a list of runs in an experiment.

### Question 35
What is the **role of the search bar** at the top of an experiment view? Give a concrete example of a search query you could write.

### Question 36
**Pedagogical question**: why is "comparing experiments" considered the **single most important value proposition** of MLflow Tracking? Try to answer in 2 to 3 sentences.

---

## Section 5 - Tracking URI and where the data lives

### Question 37
What is the **default value** of `mlflow.get_tracking_uri()` when **nothing** has been configured? Be precise about the format.

### Question 38
Name the **three main URI schemes** that `mlflow.set_tracking_uri(...)` accepts and describe in one sentence what each one does:
- `file://...`
- `http://...` / `https://...`
- `databricks` / `databricks://...`

### Question 39
In chap04, the trainer's first run lands in `file:///code/mlruns` inside the container. Explain in 3 to 4 sentences **why this run never shows up in the MLflow UI** at `http://localhost:5000`.

### Question 40
In chap05, the URI is `http://mlflow:5000`. Why is the hostname **`mlflow`** rather than **`localhost`** or **`127.0.0.1`**? Where does the name `mlflow` come from?

### Question 41
Cite the **precedence order** between these three configuration sources for the tracking URI, from highest priority to lowest:
- explicit call to `mlflow.set_tracking_uri(...)` in the code,
- environment variable `MLFLOW_TRACKING_URI`,
- MLflow built-in default value.

### Question 42
**TRUE / FALSE**: if you completely remove the `mlflow.set_tracking_uri(...)` line from `train.py` in chap05, the runs will still appear in the UI **because** the `MLFLOW_TRACKING_URI` environment variable is set in `docker-compose.yml`.
Justify.

---

## Section 6 - The progression of the course (synthesis)

### Question 43
Fill in the missing chapter for each progression step:
- chap01: run MLflow + a single `log_metric` call.
- chap02: add `____________________`.
- chap03: add `____________________`.
- chap04: add `____________________`.
- chap05: add `____________________`.

### Question 44
Looking back at chap01 to chap05, what is the **single MLflow API call** that was introduced in **every** chapter from chap03 onwards? Why is this call so central to MLflow?

### Question 45
Why does chap04 split the project into **two separate Docker images** (`mlflow` server image + `trainer` training image) instead of running everything in a single container? Cite at least **two reasons**.

### Question 46
What is the difference between a **named Docker volume** (`mlflow-db`) and a **bind mount** (`./database:/mlflow/database`) in terms of:
- where the data is stored,
- portability across machines,
- visibility from the host file explorer?

### Question 47
In chap05, the `docker-compose.yml` declares:

```yaml
trainer:
  depends_on:
    mlflow:
      condition: service_healthy
```

In one sentence, what does `service_healthy` guarantee that a plain `depends_on: [mlflow]` does **NOT** guarantee?

### Question 48
**Open question**: if you had to extend chap05 by adding a **fourth experiment** with `(alpha=0.3, l1_ratio=0.7)`, give the **exact command** you would type. (Do not run it; just write it.)

### Question 49
**Scenario**: a new colleague clones the repository on their laptop and types:

```bash
cd chap05-mlflow-step-by-step-recap-passing-tracking-uri-via-env-var
docker compose up -d --build mlflow
docker compose run --rm trainer --alpha 0.1 --l1_ratio 0.1
```

In **what order** do the following events happen?
1. The MLflow UI becomes reachable at `http://localhost:5000`.
2. The MLflow server's healthcheck reports `healthy`.
3. The trainer image is built.
4. The MLflow image is built.
5. The trainer container reads `MLFLOW_TRACKING_URI` from its environment.
6. A new run appears in the experiment `experiment_1`.

Order them from 1 to 6.

### Question 50
**Synthesis question** (long answer, 5 to 10 sentences).
You are interviewing for a junior MLOps role. The interviewer asks:

> "We have a small team of three data scientists. Each one trains models on their laptop and emails me the metrics. Should we adopt MLflow? Why or why not?"

Write the answer you would give. Mention at least:
- two **benefits** MLflow brings to this team,
- one **cost or downside** of adopting MLflow,
- whether you would also bring Docker into the picture (and why),
- one **alternative** that is sometimes used instead of MLflow.

---

# Appendix - Look-ahead questions (bonus, not in the 50)

- **B1.** What is a **model signature** in MLflow? Why is it useful at serving time? (covered in chap14+)
- **B2.** What is the difference between `mlflow.sklearn.log_model` and `mlflow.pyfunc.log_model`? When would you prefer one over the other? (covered in chap16+)
- **B3.** What is the **MLflow Model Registry** and how does it differ from the **Tracking** server? Why does it usually require a real database like Postgres rather than SQLite? (covered in chap21+)

---

**End of quiz - 50 questions, chapters 01 to 05.**

Happy revising!
