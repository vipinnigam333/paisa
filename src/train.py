"""
train.py
--------
Trains an XGBoost classifier to predict next-day stock direction,
using WALK-FORWARD (expanding window) validation instead of a random
train/test split.

Why walk-forward validation matters here:
    A random split lets the model "see the future" during training
    (e.g. train on Tuesday, test on Monday of the same week) which
    leaks information and gives an unrealistically high accuracy.
    Walk-forward validation always trains on the past and tests on
    a later, unseen period -- the same constraint you'd face in
    live trading.

Logs every run (params, metrics, model artifact) to MLflow so you
can compare experiments over time.

Usage:
    python -m src.train --data data/sample_data/AAPL.csv --n-splits 5
"""
import argparse
import json
import os

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

from src.features import build_features, FEATURE_COLUMNS


def walk_forward_splits(n_rows: int, n_splits: int = 5, min_train_size: int = 500):
    """
    Yields (train_idx, test_idx) pairs using an expanding window.
    Each fold trains on everything up to a point in time, and tests
    on the next contiguous chunk -- never on data from before the
    training window.
    """
    fold_size = (n_rows - min_train_size) // n_splits
    if fold_size <= 0:
        raise ValueError("Not enough data for the requested number of splits.")

    for i in range(n_splits):
        train_end = min_train_size + i * fold_size
        test_end = train_end + fold_size
        if i == n_splits - 1:
            test_end = n_rows  # last fold takes the remainder
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, test_end)
        yield train_idx, test_idx


def train_and_evaluate(data_path: str, n_splits: int = 5, model_out: str = "models/model.joblib"):
    raw = pd.read_csv(data_path, index_col=0, parse_dates=True)
    model_df = build_features(raw)

    X = model_df[FEATURE_COLUMNS].values
    y = model_df["target"].values
    n = len(model_df)

    mlflow.set_experiment("stock-direction-prediction")

    fold_metrics = []
    with mlflow.start_run(run_name=f"xgb_walkforward_{os.path.basename(data_path)}"):
        params = dict(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
        )
        mlflow.log_params(params)
        mlflow.log_param("n_splits", n_splits)
        mlflow.log_param("n_rows", n)
        mlflow.log_param("data_path", data_path)

        last_model = None
        oos_records = []  # out-of-sample predictions, for an honest backtest later
        for fold_i, (train_idx, test_idx) in enumerate(walk_forward_splits(n, n_splits)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            model = XGBClassifier(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]

            for idx, p, pr in zip(model_df.index[test_idx], preds, probs):
                oos_records.append({"date": idx, "pred": int(p), "prob_up": float(pr)})

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)
            try:
                auc = roc_auc_score(y_test, probs)
            except ValueError:
                auc = float("nan")

            # naive baseline: always predict the majority class in the training set
            baseline_pred = int(round(y_train.mean()))
            baseline_acc = accuracy_score(y_test, [baseline_pred] * len(y_test))

            fold_metrics.append(dict(
                fold=fold_i, accuracy=acc, precision=prec, recall=rec,
                f1=f1, auc=auc, baseline_acc=baseline_acc,
                train_size=len(train_idx), test_size=len(test_idx),
            ))

            mlflow.log_metric("accuracy", acc, step=fold_i)
            mlflow.log_metric("precision", prec, step=fold_i)
            mlflow.log_metric("recall", rec, step=fold_i)
            mlflow.log_metric("f1", f1, step=fold_i)
            mlflow.log_metric("auc", auc, step=fold_i)
            mlflow.log_metric("baseline_acc", baseline_acc, step=fold_i)

            print(f"[fold {fold_i}] acc={acc:.4f} (baseline={baseline_acc:.4f})  "
                  f"prec={prec:.4f} rec={rec:.4f} f1={f1:.4f} auc={auc:.4f}")

            last_model = model  # keep most recent fold's model as the "production" candidate

        avg_acc = np.mean([m["accuracy"] for m in fold_metrics])
        avg_baseline = np.mean([m["baseline_acc"] for m in fold_metrics])
        avg_auc = np.nanmean([m["auc"] for m in fold_metrics])
        mlflow.log_metric("avg_accuracy", avg_acc)
        mlflow.log_metric("avg_baseline_acc", avg_baseline)
        mlflow.log_metric("avg_auc", avg_auc)

        print(f"\n=== Summary over {n_splits} walk-forward folds ===")
        print(f"Avg model accuracy:    {avg_acc:.4f}")
        print(f"Avg naive baseline:    {avg_baseline:.4f}")
        print(f"Avg AUC:               {avg_auc:.4f}")
        print(f"Edge over baseline:    {avg_acc - avg_baseline:+.4f}")

        # Final model: retrain on ALL data for deployment
        final_model = XGBClassifier(**params)
        final_model.fit(X, y)

        os.makedirs(os.path.dirname(model_out), exist_ok=True)
        joblib.dump({"model": final_model, "feature_columns": FEATURE_COLUMNS}, model_out)
        mlflow.xgboost.log_model(final_model, artifact_path="model")
        mlflow.log_artifact(model_out)

        metrics_path = model_out.replace(".joblib", "_fold_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(fold_metrics, f, indent=2)
        mlflow.log_artifact(metrics_path)

        # Save out-of-sample predictions separately -- this is what evaluate.py
        # should backtest on. Using the final in-sample model for backtesting
        # would leak future information into the "predictions" and produce
        # unrealistically good results.
        oos_path = model_out.replace(".joblib", "_oos_predictions.csv")
        oos_df = pd.DataFrame(oos_records).set_index("date")
        oos_df.to_csv(oos_path)
        mlflow.log_artifact(oos_path)

        print(f"\nSaved deployable model to {model_out}")
        print(f"Saved fold metrics to {metrics_path}")
        print(f"Saved out-of-sample predictions to {oos_path} (use this for backtesting)")

    return fold_metrics, model_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_data/AAPL.csv")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--model-out", default="models/model.joblib")
    args = parser.parse_args()
    train_and_evaluate(args.data, args.n_splits, args.model_out)


if __name__ == "__main__":
    main()
