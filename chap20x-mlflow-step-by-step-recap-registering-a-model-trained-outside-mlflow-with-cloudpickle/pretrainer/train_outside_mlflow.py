"""Trains an ElasticNet and dumps it to /shared/elastic-net-regression.pkl.
Runs OUTSIDE any MLflow context. Could just as well be a SageMaker job,
a notebook a colleague sent you, or a vendor's binary blob."""

import argparse
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--alpha",    type=float, default=0.4)
parser.add_argument("--l1_ratio", type=float, default=0.4)
args = parser.parse_args()


def eval_metrics(actual, pred):
    return (
        np.sqrt(mean_squared_error(actual, pred)),
        mean_absolute_error(actual, pred),
        r2_score(actual, pred),
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    data = pd.read_csv("data/red-wine-quality.csv")
    train, test = train_test_split(data, test_size=0.25)

    train_x = train.drop(["quality"], axis=1)
    test_x  = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y  = test[["quality"]]

    lr = ElasticNet(alpha=args.alpha, l1_ratio=args.l1_ratio, random_state=42)
    lr.fit(train_x, train_y)

    rmse, mae, r2 = eval_metrics(test_y, lr.predict(test_x))
    print(f"[pretrainer] ElasticNet trained: RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")

    out_dir = "/shared"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "elastic-net-regression.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(lr, f)
    print(f"[pretrainer] Wrote {out_path}")
