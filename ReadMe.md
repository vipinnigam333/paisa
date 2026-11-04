# 📈 Production-Grade Stock Direction Predictor & MLOps Pipeline

> An end-to-end machine learning system that predicts next-day stock price direction (up/down) and serves it through a production-style engineering pipeline.

⚠️ **Disclaimer:** *This project is built strictly for educational and engineering portfolio purposes. It is not financial advice, and it should not be used to make real trading decisions.*

---

## 📌 Project Overview

Most online stock prediction projects suffer from severe data leakage and unrealistic evaluations. This project deliberately avoids those pitfalls by prioritizing **engineering rigor and honest backtesting** over deceptive 90%+ accuracy metrics.

### Key Highlights
* **No Lookahead Bias:** Uses walk-forward (expanding window) validation so models are strictly trained on past data.
* **Honest Out-of-Sample Backtesting:** Backtests use strictly out-of-sample (OOS) fold predictions, avoiding in-sample data leakage.
* **Production Architecture:** Fully automated pipeline covering data ingestion, feature engineering, MLflow tracking, FastAPI serving, Docker containerization, Streamlit visualization, Airflow orchestration, and GitHub Actions CI.

---

## 🏗 System Architecture

```text
                ┌─────────────────┐
                │  Yahoo Finance  │  (or synthetic data fallback
                │   (yfinance)    │   if offline / rate-limited)
                └────────┬────────┘
                         │
                 src/data_loader.py
                         │
                 src/features.py        (technical indicators:
                         │                RSI, MACD, SMA, volatility...)
                         │
                  src/train.py          (XGBoost + walk-forward CV
                         │                + MLflow experiment tracking)
                         │
              ┌──────────┴──────────┐
              │                     │
      src/evaluate.py       models/model.joblib
   (Sharpe, drawdown,           │
    backtest vs buy&hold)       │
                                 │
                          src/api.py (FastAPI)
                                 │
                 ┌───────────────┼────────────────┐
                 │               │                │
          dashboard/app.py   Docker container   GitHub Actions CI
           (Streamlit)        (docker-compose)   (test + build)
                                                        │
                                            dags/retrain_dag.py
                                          (Airflow: weekly retrain,
                                        champion/challenger promotion)
```

---

## 📁 Project Structure

```text
stock-mlops/
├── src/
│   ├── data_loader.py       # Fetch real data (yfinance) or generate fallback
│   ├── features.py          # Technical indicator feature engineering
│   ├── train.py             # Walk-forward training + MLflow tracking
│   ├── evaluate.py          # Backtest evaluation: Sharpe ratio, max drawdown
│   ├── retrain_pipeline.py  # Champion/Challenger automated retraining
│   └── api.py                # FastAPI model serving layer
├── dashboard/
│   └── app.py                # Interactive Streamlit dashboard
├── dags/
│   └── retrain_dag.py        # Airflow DAG for scheduled weekly retraining
├── tests/
│   └── test_pipeline.py      # Unit tests (including no-lookahead assertions)
├── .github/workflows/ci.yml  # CI workflow (tests, smoke training, Docker build)
├── data/sample_data/          # Output directory for data CSVs
├── models/                    # Trained models & metrics output directory
├── Dockerfile                 # Single container setup
├── docker-compose.yml         # Multi-service setup
└── requirements.txt           # Project dependencies
```

---

## 🚀 Setup & Execution Guide

### 1. Installation & Environment
```bash
# Clone the repository
git clone https://github.com/your-username/stock-mlops.git
cd stock-mlops

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Fetch Data
```bash
python -m src.data_loader --ticker AAPL --start 2015-01-01 --end 2024-12-31
```
*Note: If Yahoo Finance is rate-limited or offline, the script automatically falls back to synthetic data generation and sets an `is_synthetic_data` flag in downstream outputs.*

### 3. Model Training & Experiment Tracking
```bash
python -m src.train --data data/sample_data/AAPL.csv --n-splits 5
```
To launch the **MLflow UI** and inspect experiments, metrics, and parameters:
```bash
mlflow ui
# Navigate to http://localhost:5000 in your browser
```

### 4. Run Backtest Evaluation
```bash
python -m src.evaluate --data data/sample_data/AAPL.csv --oos-preds models/model_oos_predictions.csv
```
Prints Sharpe ratio, cumulative returns, and maximum drawdown for model signals vs. Buy & Hold baselines.

### 5. Launch FastAPI Model Server
```bash
uvicorn src.api:app --reload --port 8000
```
Test endpoints:
```bash
curl http://localhost:8000/health
curl "http://localhost:8000/predict?ticker=AAPL"
```

### 6. Run Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```

### 7. Run via Docker / Docker Compose
```bash
# Build and run single container
docker build -t stock-mlops .
docker run -p 8000:8000 stock-mlops

# Or using Docker Compose
docker-compose up --build
```

### 8. Run Unit Tests
```bash
pytest tests/ -v
```

### 9. Automated Champion/Challenger Retraining
```bash
python -m src.retrain_pipeline --ticker AAPL
```
Evaluates a newly trained model (Challenger) against the active production model (Champion) and updates `models/model.joblib` only if performance criteria are met.

---

## 🗓 4-Week Implementation Plan

* **Week 1: Data & Features** — Integrate `yfinance`, implement technical indicators (RSI, MACD, SMAs), and perform sanity checks.
* **Week 2: Modeling & Honest Validation** — Implement walk-forward CV, train XGBoost vs. baseline models, and add no-lookahead unit tests.
* **Week 3: MLOps Infrastructure** — Set up MLflow tracking, build FastAPI REST endpoints, and containerize with Docker.
* **Week 4: Automation & Deployment** — Configure GitHub Actions CI, build Streamlit dashboard, write Airflow DAG, and assemble final documentation.

---

## 💡 Optional Extensions

* Incorporate news headline sentiment features using NLP (BERT/VADER).
* Compare XGBoost performance against sequential models (LSTM, Temporal Fusion Transformer).
* Implement continuous monitoring for live prediction drift detection.