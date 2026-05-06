<a id="top"></a>

# Chapter 20c0 — From a classic Machine Learning script to a tracked MLflow experiment

## Table of Contents

| #  | Section                                              |
| -- | ---------------------------------------------------- |
| 1  | [Objective](#section-1)                              |
| 2  | [What is ElasticNet?](#section-2)                    |
| 3  | [What are `alpha` and `l1_ratio`?](#section-3)       |
| 4  | [Why start without MLflow?](#section-4)              |
| 5  | [Script 1 — Without MLflow](#section-5)              |
| 6  | [Problem with the script without MLflow](#section-6) |
| 7  | [Script 2 — With MLflow](#section-7)                 |
| 8  | [What exactly did we add with MLflow?](#section-8)   |
| 9  | [What problem does MLflow solve?](#section-9)        |
| 10 | [How to run both versions](#section-10)              |
| 11 | [Mini exercise](#section-11)                         |
| 12 | [Recap](#section-12)                                 |

---

<a id="section-1"></a>

## 1. Objective

The objective of this chapter is to understand the difference between:

1. a normal machine learning script without MLflow;
2. the same script with MLflow tracking added.

The goal is not only to run a model.

The real goal is to understand why MLflow becomes useful when we start running several experiments with different parameters.

In this example, we train an `ElasticNet` regression model on the `red-wine-quality.csv` dataset.

The model tries to predict the wine quality score using chemical characteristics such as acidity, sugar, alcohol, pH, sulphates, and other numerical features.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. What is ElasticNet?

`ElasticNet` is a regression model from scikit-learn.

It is used when we want to predict a numerical value.

In this lab, the numerical value to predict is:

```text
quality
```

The model receives several input columns:

```text
fixed acidity
volatile acidity
citric acid
residual sugar
chlorides
free sulfur dioxide
total sulfur dioxide
density
pH
sulphates
alcohol
```

Then it tries to learn a mathematical relationship between these input variables and the target column `quality`.

ElasticNet is based on linear regression, but it adds regularization.

Regularization means that the model is penalized when it becomes too complex.

The goal is to reduce overfitting.

In simple terms:

```text
ElasticNet = Linear Regression + regularization
```

It combines two types of regularization:

| Regularization    | Meaning                                                                   |
| ----------------- | ------------------------------------------------------------------------- |
| L1 regularization | Can reduce some coefficients to zero. This helps select useful variables. |
| L2 regularization | Reduces the size of coefficients. This helps make the model more stable.  |

ElasticNet is useful when we have several input variables and we want a model that is simple, stable, and less likely to overfit.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. What are `alpha` and `l1_ratio`?

The model has two important hyperparameters:

```python
alpha
l1_ratio
```

These are not learned automatically by the model.

They are chosen before training.

That is why they are called hyperparameters.

---

### 3.1 `alpha`

`alpha` controls the strength of the regularization.

If `alpha` is small, the model is less penalized.

If `alpha` is large, the model is more penalized.

Example:

```text
alpha = 0.1
```

The model has more freedom.

```text
alpha = 0.9
```

The model is more constrained.

A very large `alpha` can make the model too simple.

A very small `alpha` can make the model too flexible.

---

### 3.2 `l1_ratio`

`l1_ratio` controls the balance between L1 and L2 regularization.

It must be between 0 and 1.

| Value of `l1_ratio` | Meaning                  |
| ------------------- | ------------------------ |
| `0`                 | Mostly L2 regularization |
| `1`                 | Mostly L1 regularization |
| Between 0 and 1     | Mix of L1 and L2         |

Example:

```text
l1_ratio = 0.7
```

This means the model uses more L1 regularization than L2 regularization.

---

### 3.3 Why do we change these parameters?

We change `alpha` and `l1_ratio` to see which combination gives the best model.

For example:

```bash
python train.py --alpha 0.1 --l1_ratio 0.1
python train.py --alpha 0.5 --l1_ratio 0.5
python train.py --alpha 0.9 --l1_ratio 0.9
```

Each command trains a different version of the model.

This is where the problem begins.

If we do not use MLflow, we must manually remember:

* which parameters we used;
* which result we obtained;
* which model was produced;
* which run was the best.

MLflow solves this problem.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. Why start without MLflow?

Before adding MLflow, it is important to understand what a normal machine learning script looks like.

A classic script usually does the following steps:

1. imports the required libraries;
2. reads the dataset;
3. splits the dataset into train and test sets;
4. separates the input columns from the target column;
5. trains the model;
6. makes predictions;
7. calculates evaluation metrics;
8. prints the results in the terminal.

This works.

However, it does not keep a history of experiments.

If we run the script ten times with ten different parameters, the terminal only shows the latest output unless we manually save everything.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. Script 1 — Without MLflow

This first version does not use MLflow.

It only trains the model and prints the results.

File name:

```text
train_without_mlflow.py
```

```python
import argparse
import logging
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Command-line arguments
# ------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--alpha",
    type=float,
    required=False,
    default=0.7,
    help="Regularization strength for ElasticNet"
)

parser.add_argument(
    "--l1_ratio",
    type=float,
    required=False,
    default=0.7,
    help="Balance between L1 and L2 regularization"
)

args = parser.parse_args()


# ------------------------------------------------------------
# Evaluation function
# ------------------------------------------------------------

def eval_metrics(actual, pred):
    """
    Compute regression evaluation metrics.

    RMSE: Root Mean Squared Error
    MAE : Mean Absolute Error
    R2  : Coefficient of determination
    """

    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)

    return rmse, mae, r2


# ------------------------------------------------------------
# Main program
# ------------------------------------------------------------

if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    # Make the experiment reproducible
    np.random.seed(40)

    # Load the dataset
    data = pd.read_csv("data/red-wine-quality.csv")

    # Split the dataset into training and testing sets
    train, test = train_test_split(data)

    # Separate input variables from the target variable
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)

    train_y = train["quality"]
    test_y = test["quality"]

    # Get hyperparameters from the command line
    alpha = args.alpha
    l1_ratio = args.l1_ratio

    # Create the ElasticNet model
    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        random_state=42
    )

    # Train the model
    model.fit(train_x, train_y)

    # Make predictions
    predicted_qualities = model.predict(test_x)

    # Evaluate the model
    rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)

    # Display the results
    print("ElasticNet model without MLflow")
    print("--------------------------------")
    print(f"alpha:    {alpha}")
    print(f"l1_ratio: {l1_ratio}")
    print(f"RMSE:     {rmse}")
    print(f"MAE:      {mae}")
    print(f"R2:       {r2}")
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-6"></a>

## 6. Problem with the script without MLflow

The script works correctly.

It trains the model and prints the results.

However, there is a major problem.

The results are not tracked.

For example, if we run:

```bash
python train_without_mlflow.py --alpha 0.1 --l1_ratio 0.1
python train_without_mlflow.py --alpha 0.5 --l1_ratio 0.5
python train_without_mlflow.py --alpha 0.9 --l1_ratio 0.9
```

we get three different outputs in the terminal.

But the script does not automatically save:

* the parameters used for each run;
* the RMSE value;
* the MAE value;
* the R2 value;
* the trained model;
* the date and time of the experiment;
* the comparison between runs.

Without MLflow, the user must manually copy the results into a file or spreadsheet.

This becomes difficult when we have many experiments.

Example of the problem:

```text
Run 1:
alpha = 0.1
l1_ratio = 0.1
RMSE = ?

Run 2:
alpha = 0.5
l1_ratio = 0.5
RMSE = ?

Run 3:
alpha = 0.9
l1_ratio = 0.9
RMSE = ?
```

If we do not save these results manually, we lose the experiment history.

That is the problem MLflow solves.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-7"></a>

## 7. Script 2 — With MLflow

Now we add MLflow.

The machine learning logic is almost the same.

The difference is that we add experiment tracking.

File name:

```text
train_with_mlflow.py
```

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


# ------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Command-line arguments
# ------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--alpha",
    type=float,
    required=False,
    default=0.7,
    help="Regularization strength for ElasticNet"
)

parser.add_argument(
    "--l1_ratio",
    type=float,
    required=False,
    default=0.7,
    help="Balance between L1 and L2 regularization"
)

args = parser.parse_args()


# ------------------------------------------------------------
# Evaluation function
# ------------------------------------------------------------

def eval_metrics(actual, pred):
    """
    Compute regression evaluation metrics.

    RMSE: Root Mean Squared Error
    MAE : Mean Absolute Error
    R2  : Coefficient of determination
    """

    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)

    return rmse, mae, r2


# ------------------------------------------------------------
# Main program
# ------------------------------------------------------------

if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    # Make the experiment reproducible
    np.random.seed(40)

    # --------------------------------------------------------
    # MLflow insertion 1:
    # Tell Python where the MLflow Tracking Server is running.
    # --------------------------------------------------------

    mlflow.set_tracking_uri("http://localhost:5000")
    print("Tracking URI:", mlflow.get_tracking_uri())

    # Load the dataset
    data = pd.read_csv("data/red-wine-quality.csv")

    # Split the dataset into training and testing sets
    train, test = train_test_split(data)

    # Separate input variables from the target variable
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)

    train_y = train["quality"]
    test_y = test["quality"]

    # Get hyperparameters from the command line
    alpha = args.alpha
    l1_ratio = args.l1_ratio

    # --------------------------------------------------------
    # MLflow insertion 2:
    # Create or select an experiment.
    # --------------------------------------------------------

    experiment = mlflow.set_experiment(
        experiment_name="experiment_1"
    )

    # --------------------------------------------------------
    # MLflow insertion 3:
    # Start a run.
    # Everything inside this block belongs to this experiment run.
    # --------------------------------------------------------

    with mlflow.start_run(experiment_id=experiment.experiment_id):

        # Create the ElasticNet model
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            random_state=42
        )

        # Train the model
        model.fit(train_x, train_y)

        # Make predictions
        predicted_qualities = model.predict(test_x)

        # Evaluate the model
        rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)

        # Display the results
        print("ElasticNet model with MLflow")
        print("-----------------------------")
        print(f"alpha:    {alpha}")
        print(f"l1_ratio: {l1_ratio}")
        print(f"RMSE:     {rmse}")
        print(f"MAE:      {mae}")
        print(f"R2:       {r2}")

        # ----------------------------------------------------
        # MLflow insertion 4:
        # Log the hyperparameters.
        # ----------------------------------------------------

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        # ----------------------------------------------------
        # MLflow insertion 5:
        # Log the evaluation metrics.
        # ----------------------------------------------------

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # ----------------------------------------------------
        # MLflow insertion 6:
        # Log the trained model as an artifact.
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="mymodel"
        )
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-8"></a>

## 8. What exactly did we add with MLflow?

The machine learning part did not change much.

The model is still:

```python
model = ElasticNet(
    alpha=alpha,
    l1_ratio=l1_ratio,
    random_state=42
)
```

The training is still:

```python
model.fit(train_x, train_y)
```

The predictions are still:

```python
predicted_qualities = model.predict(test_x)
```

The metrics are still:

```python
rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)
```

What we added is experiment tracking.

---

### 8.1 Import MLflow

```python
import mlflow
import mlflow.sklearn
```

These imports allow the script to communicate with MLflow.

`mlflow` is used to log parameters and metrics.

`mlflow.sklearn` is used to save the scikit-learn model.

---

### 8.2 Set the tracking URI

```python
mlflow.set_tracking_uri("http://localhost:5000")
```

This line tells the script where MLflow is running.

In this example, MLflow is running locally on port `5000`.

Without this line, the results may be stored somewhere else, for example in a local `mlruns` folder.

That can confuse beginners because the run may not appear in the MLflow web interface.

---

### 8.3 Print the tracking URI

```python
print("Tracking URI:", mlflow.get_tracking_uri())
```

This line is a safety check.

It confirms that the script is connected to the correct MLflow server.

Expected output:

```text
Tracking URI: http://localhost:5000
```

---

### 8.4 Create or select an experiment

```python
experiment = mlflow.set_experiment(
    experiment_name="experiment_1"
)
```

An experiment is a container for multiple runs.

For example:

```text
experiment_1
├── Run 1: alpha=0.1, l1_ratio=0.1
├── Run 2: alpha=0.5, l1_ratio=0.5
└── Run 3: alpha=0.9, l1_ratio=0.9
```

This makes it easier to organize experiments.

---

### 8.5 Start a run

```python
with mlflow.start_run(experiment_id=experiment.experiment_id):
```

A run represents one execution of the script.

Each time we run the script with different parameters, MLflow creates a new run.

Example:

```bash
python train_with_mlflow.py --alpha 0.1 --l1_ratio 0.1
```

creates one run.

```bash
python train_with_mlflow.py --alpha 0.5 --l1_ratio 0.5
```

creates another run.

---

### 8.6 Log parameters

```python
mlflow.log_param("alpha", alpha)
mlflow.log_param("l1_ratio", l1_ratio)
```

These lines save the hyperparameters used for the run.

Instead of writing the parameters manually in a notebook or spreadsheet, MLflow records them automatically.

---

### 8.7 Log metrics

```python
mlflow.log_metric("rmse", rmse)
mlflow.log_metric("mae", mae)
mlflow.log_metric("r2", r2)
```

These lines save the evaluation results.

MLflow allows us to compare the metrics between several runs.

For example, we can sort the runs by the smallest RMSE.

---

### 8.8 Log the model

```python
mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="mymodel"
)
```

This line saves the trained model.

The model is stored as an artifact inside the MLflow run.

This is important because we do not only want to know the metrics.

We also want to keep the model that produced those metrics.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-9"></a>

## 9. What problem does MLflow solve?

MLflow solves the problem of experiment tracking.

Without MLflow, a machine learning workflow often looks like this:

```text
I run the script.
I see the result in the terminal.
I change the parameters.
I run the script again.
I forget the previous result.
I copy some numbers into a file.
I lose track of which model was best.
```

With MLflow, the workflow becomes:

```text
I run the script.
MLflow saves the parameters.
MLflow saves the metrics.
MLflow saves the model.
I open the MLflow UI.
I compare all runs in a table.
I identify the best run.
I can recover the model later.
```

---

### 9.1 Comparison table

| Without MLflow                           | With MLflow                         |
| ---------------------------------------- | ----------------------------------- |
| Results appear only in the terminal      | Results are saved in MLflow         |
| Parameters are not tracked automatically | Parameters are stored automatically |
| Metrics must be copied manually          | Metrics are logged automatically    |
| Difficult to compare many runs           | Easy to compare runs in the UI      |
| The model is not saved by default        | The model is saved as an artifact   |
| No experiment history                    | Full experiment history             |
| Hard to know which run was best          | Easy to sort by RMSE, MAE, or R2    |

---

### 9.2 Concrete example

Suppose we run:

```bash
python train_with_mlflow.py --alpha 0.1 --l1_ratio 0.1
python train_with_mlflow.py --alpha 0.5 --l1_ratio 0.5
python train_with_mlflow.py --alpha 0.9 --l1_ratio 0.9
```

MLflow will create three runs.

Each run will contain:

```text
Parameters:
- alpha
- l1_ratio

Metrics:
- rmse
- mae
- r2

Artifacts:
- trained model
```

This allows us to answer questions such as:

```text
Which alpha gave the best RMSE?
Which l1_ratio gave the best R2?
Which run should we keep?
Where is the trained model?
Can we reproduce the experiment?
```

This is the main value of MLflow.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-10"></a>

## 10. How to run both versions

### 10.1 Run the version without MLflow

```bash
python train_without_mlflow.py --alpha 0.7 --l1_ratio 0.7
```

Expected output:

```text
ElasticNet model without MLflow
--------------------------------
alpha:    0.7
l1_ratio: 0.7
RMSE:     ...
MAE:      ...
R2:       ...
```

This output appears in the terminal only.

No experiment is saved in MLflow.

---

### 10.2 Start MLflow server

Before running the MLflow version, start the MLflow server.

Example:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Then open:

```text
http://localhost:5000
```

---

### 10.3 Run the version with MLflow

In another terminal:

```bash
python train_with_mlflow.py --alpha 0.7 --l1_ratio 0.7
```

Expected output:

```text
Tracking URI: http://localhost:5000
ElasticNet model with MLflow
-----------------------------
alpha:    0.7
l1_ratio: 0.7
RMSE:     ...
MAE:      ...
R2:       ...
```

Now refresh the MLflow UI.

You should see:

```text
experiment_1
```

Inside the experiment, you should see a run containing:

```text
alpha
l1_ratio
rmse
mae
r2
mymodel
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-11"></a>

## 11. Mini exercise

Run the script without MLflow:

```bash
python train_without_mlflow.py --alpha 0.1 --l1_ratio 0.1
python train_without_mlflow.py --alpha 0.5 --l1_ratio 0.5
python train_without_mlflow.py --alpha 0.9 --l1_ratio 0.9
```

Then answer:

```text
1. Is it easy to compare the three runs?
2. Where are the results saved?
3. Can you recover the trained model?
4. Can you easily know which run gave the best RMSE?
```

Now run the version with MLflow:

```bash
python train_with_mlflow.py --alpha 0.1 --l1_ratio 0.1
python train_with_mlflow.py --alpha 0.5 --l1_ratio 0.5
python train_with_mlflow.py --alpha 0.9 --l1_ratio 0.9
```

Then open the MLflow UI and answer:

```text
1. How many runs were created?
2. Which run has the lowest RMSE?
3. Which run has the best R2?
4. Where can you find the saved model?
5. Why is MLflow useful when testing several hyperparameters?
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-12"></a>

## 12. Recap

In this chapter, we compared two versions of the same machine learning pipeline.

The first version trained an ElasticNet model without MLflow.

It worked, but it did not keep a structured history of experiments.

The second version added MLflow.

With MLflow, we were able to save:

* the hyperparameters;
* the metrics;
* the trained model;
* the experiment run;
* the comparison between several executions.

The important idea is this:

```text
MLflow does not replace the machine learning model.
MLflow tracks the experiment around the model.
```

A machine learning script answers:

```text
What result did I get this time?
```

An MLflow experiment answers:

```text
What did I run?
With which parameters?
What were the results?
Where is the model?
Which run was the best?
Can I reproduce it later?
```

That is why MLflow is important in real machine learning projects.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 20d — From a classic Machine Learning script to MLflow experiment tracking</strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>
