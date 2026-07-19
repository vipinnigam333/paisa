"""
dashboard/app.py
-----------------
Streamlit dashboard to visualize:
  1. Historical price + the model's out-of-sample predictions
  2. Fold-by-fold model accuracy vs the naive baseline
  3. A live prediction for a ticker of choice

Run:
    streamlit run dashboard/app.py
"""
import json
import os
import sys

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import load_data
from src.features import build_features

st.set_page_config(page_title="Stock Direction Predictor", layout="wide")
st.title("📈 Stock Direction Prediction — MLOps Dashboard")
st.caption(
    "Educational project. Predicts next-day price direction (up/down), not exact prices. "
    "Not financial advice."
)

MODEL_PATH = "models/model.joblib"
METRICS_PATH = "models/model_fold_metrics.json"
OOS_PATH = "models/model_oos_predictions.csv"
DATA_PATH = "data/sample_data/AAPL.csv"

tab1, tab2, tab3 = st.tabs(["Live Prediction", "Model Performance", "Backtest Chart"])

with tab1:
    st.subheader("Get a live prediction")
    col1, col2 = st.columns([1, 3])
    with col1:
        ticker = st.text_input("Ticker", value="AAPL")
        run = st.button("Predict")

    if run:
        if not os.path.exists(MODEL_PATH):
            st.error("No trained model found. Run `python -m src.train` first.")
        else:
            bundle = joblib.load(MODEL_PATH)
            model, feature_cols = bundle["model"], bundle["feature_columns"]
            with st.spinner(f"Fetching data for {ticker}..."):
                raw = load_data(ticker, "2023-01-01", pd.Timestamp.today().strftime("%Y-%m-%d"))
            feat_df = build_features(raw)
            if feat_df.empty:
                st.error("Not enough data to build features for this ticker.")
            else:
                latest = feat_df.iloc[[-1]][feature_cols]
                prob_up = float(model.predict_proba(latest.values)[0, 1])
                direction = "UP 📈" if prob_up >= 0.5 else "DOWN 📉"
                if raw.attrs.get("synthetic"):
                    st.warning("Live data unavailable — showing result on SYNTHETIC data for demo purposes.")
                m1, m2 = st.columns(2)
                m1.metric("Predicted next-day direction", direction)
                m2.metric("Model probability of UP", f"{prob_up:.1%}")
                st.line_chart(raw["Close"].tail(120))

with tab2:
    st.subheader("Walk-forward validation results")
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            fold_metrics = pd.DataFrame(json.load(f))
        st.dataframe(fold_metrics, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Accuracy: model vs naive baseline, per fold**")
            st.bar_chart(fold_metrics.set_index("fold")[["accuracy", "baseline_acc"]])
        with c2:
            st.markdown("**AUC per fold**")
            st.bar_chart(fold_metrics.set_index("fold")[["auc"]])

        avg_acc = fold_metrics["accuracy"].mean()
        avg_base = fold_metrics["baseline_acc"].mean()
        st.info(
            f"Average model accuracy: **{avg_acc:.1%}** vs naive baseline **{avg_base:.1%}** "
            f"(edge: {avg_acc - avg_base:+.1%}). A small, consistent edge is realistic and expected "
            f"for daily direction prediction — be skeptical of any result claiming much more."
        )
    else:
        st.warning("No metrics found. Run `python -m src.train` first.")

with tab3:
    st.subheader("Strategy vs Buy & Hold (out-of-sample)")
    if os.path.exists(OOS_PATH) and os.path.exists(DATA_PATH):
        oos = pd.read_csv(OOS_PATH, index_col=0, parse_dates=True)
        raw = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
        next_day_return = raw["Close"].pct_change().shift(-1).reindex(oos.index)
        aligned = pd.DataFrame({"pred": oos["pred"], "ret": next_day_return}).dropna()
        strategy_returns = (aligned["pred"] == 1) * aligned["ret"]
        strategy_cum = (1 + strategy_returns).cumprod()
        buyhold_cum = (1 + aligned["ret"]).cumprod()
        chart_df = pd.DataFrame({"Model Strategy": strategy_cum, "Buy & Hold": buyhold_cum})
        st.line_chart(chart_df)
        st.caption("Cumulative growth of $1. No transaction costs modeled.")
    else:
        st.warning("No out-of-sample predictions found. Run `python -m src.train` first.")
