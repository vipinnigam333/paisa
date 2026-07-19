"""
retrain_pipeline.py
--------------------
The automated retraining job: fetch latest data -> engineer features ->
walk-forward train & evaluate -> compare new model to the currently
deployed one -> only promote (overwrite models/model.joblib) if the
new model is at least as good. This "champion/challenger" gate is
what makes retraining safe to automate instead of blindly overwriting
a working model with a worse one.

Meant to be run on a schedule (cron, Airflow, GitHub Actions cron, etc.)
See dags/retrain_dag.py for an Airflow version of this same logic.

Usage:
    python -m src.retrain_pipeline --ticker AAPL
"""
import argparse
import json
import os
import shutil
from datetime import datetime, timedelta

from src.data_loader import load_data
from src.train import train_and_evaluate

CHAMPION_PATH = "models/model.joblib"
CHALLENGER_PATH = "models/challenger_model.joblib"
DATA_DIR = "data/sample_data"


def get_champion_avg_accuracy(champion_metrics_path: str) -> float:
    if not os.path.exists(champion_metrics_path):
        return 0.0
    with open(champion_metrics_path) as f:
        metrics = json.load(f)
    return sum(m["accuracy"] for m in metrics) / len(metrics)


def run(ticker: str, lookback_years: int = 8):
    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=365 * lookback_years)).strftime("%Y-%m-%d")

    print(f"[retrain] Fetching latest data for {ticker} ({start} to {end})...")
    df = load_data(ticker, start, end)
    os.makedirs(DATA_DIR, exist_ok=True)
    data_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    df.to_csv(data_path)

    print("[retrain] Training challenger model...")
    challenger_metrics, _ = train_and_evaluate(data_path, n_splits=5, model_out=CHALLENGER_PATH)
    challenger_avg_acc = sum(m["accuracy"] for m in challenger_metrics) / len(challenger_metrics)

    champion_metrics_path = CHAMPION_PATH.replace(".joblib", "_fold_metrics.json")
    champion_avg_acc = get_champion_avg_accuracy(champion_metrics_path)

    print(f"[retrain] Champion avg accuracy:   {champion_avg_acc:.4f}")
    print(f"[retrain] Challenger avg accuracy: {challenger_avg_acc:.4f}")

    if challenger_avg_acc >= champion_avg_acc:
        print("[retrain] Challenger is at least as good. PROMOTING to champion.")
        shutil.copy(CHALLENGER_PATH, CHAMPION_PATH)
        shutil.copy(
            CHALLENGER_PATH.replace(".joblib", "_fold_metrics.json"),
            champion_metrics_path,
        )
        shutil.copy(
            CHALLENGER_PATH.replace(".joblib", "_oos_predictions.csv"),
            CHAMPION_PATH.replace(".joblib", "_oos_predictions.csv"),
        )
        return True
    else:
        print("[retrain] Challenger did NOT beat champion. Keeping existing model.")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--lookback-years", type=int, default=8)
    args = parser.parse_args()
    run(args.ticker, args.lookback_years)


if __name__ == "__main__":
    main()
