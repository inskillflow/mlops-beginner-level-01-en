import argparse
import os
import warnings

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha",    type=float, default=0.4)
    parser.add_argument("--l1_ratio", type=float, default=0.4)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    np.random.seed(40)

    data = pd.read_csv("data/red-wine-quality.csv")
    train, test = train_test_split(data)
    train_x = train.drop(["quality"], axis=1)
    test_x  = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y  = test[["quality"]]

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    )

    lr = ElasticNet(alpha=args.alpha, l1_ratio=args.l1_ratio, random_state=42)
    lr.fit(train_x, train_y)
    rmse, mae, r2 = eval_metrics(test_y, lr.predict(test_x))

    print(f"ElasticNet (alpha={args.alpha}, l1_ratio={args.l1_ratio})")
    print(f"  RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")

    mlflow.log_params({"alpha": args.alpha, "l1_ratio": args.l1_ratio})
    mlflow.log_metrics({"rmse": rmse, "r2": r2, "mae": mae})
    mlflow.sklearn.log_model(lr, "model")


if __name__ == "__main__":
    main()
