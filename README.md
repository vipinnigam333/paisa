# Stock Direction Prediction & MLOps Pipeline

An end-to-end machine learning system that predicts next-day stock price
*direction* (up/down) and serves it through a production-style pipeline:
data ingestion → feature engineering → walk-forward validated training →
experiment tracking → automated champion/challenger retraining → API
serving → dashboard → containerization → CI/CD.

**This is an educational project. It is not financial advice, and it
should not be used to make real trading decisions.**

---

## Why this project, and why it's built this way

Most "stock prediction" tutorials online make two mistakes that this
project deliberately avoids:

1. **Predicting exact prices with a random train/test split.** This
   leaks future information and produces deceptively good-looking
   results (a naive "tomorrow = today" model already scores well on
   price regression). This project predicts **direction** instead, and
   uses **walk-forward (expanding window) validation**, so every
   prediction only ever comes from a model trained on data strictly
   before it — see `src/train.py::walk_forward_splits`.
2. **Backtesting with the wrong model.** It's tempting to train a
   final model on all your data and then "backtest" it on that same
   data — but the model has already seen those days, so the backtest
   is dishonest. This project saves genuine **out-of-sample**
   predictions from each validation fold and backtests on those (see
   `models/model_oos_predictions.csv`), which is why the reported
   results are modest and realistic rather than suspiciously good.

Realistic expectation: daily stock direction is close to a coin flip.
A well-built model here should show a **small, consistent edge over a
naive baseline** (a few percentage points of accuracy) — not 90%+.
Framing your project around *engineering rigor* rather than *beating
the market* is both more honest and more impressive to an interviewer.

---

## Architecture

```
                ┌─────────────────┐
                │  Yahoo Finance   │  (or synthetic data fallback
                │   (yfinance)     │   if offline / rate-limited)
                └────────┬─────────┘
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

## Project structure

```
stock-mlops/
├── src/
│   ├── data_loader.py       # fetch real data or generate synthetic fallback
│   ├── features.py          # technical indicator feature engineering
│   ├── train.py             # walk-forward training + MLflow tracking
│   ├── evaluate.py          # backtest: Sharpe ratio, max drawdown
│   ├── retrain_pipeline.py  # champion/challenger automated retraining
│   └── api.py                # FastAPI serving layer
├── dashboard/
│   └── app.py                # Streamlit dashboard
├── dags/
│   └── retrain_dag.py        # Airflow DAG for scheduled retraining
├── tests/
│   └── test_pipeline.py      # unit tests, incl. no-lookahead guarantee
├── .github/workflows/ci.yml  # CI: tests, training smoke test, docker build
├── data/sample_data/          # generated CSVs land here
├── models/                    # trained model + metrics land here
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Setup & how to run everything

### 1. Install
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Fetch data
```bash
python -m src.data_loader --ticker AAPL --start 2015-01-01 --end 2024-12-31
```
This tries Yahoo Finance first. If you have no internet access at that
moment (or Yahoo rate-limits you), it automatically falls back to
generating realistic synthetic data so you can keep developing —
clearly logged, and flagged in every downstream output (`is_synthetic_data`).

### 3. Train the model
```bash
python -m src.train --data data/sample_data/AAPL.csv --n-splits 5
```
This trains with 5-fold walk-forward validation, logs everything to
MLflow, and saves `models/model.joblib` plus out-of-sample predictions.

View experiment history:
```bash
mlflow ui
# open http://localhost:5000
```

### 4. Backtest
```bash
python -m src.evaluate --data data/sample_data/AAPL.csv --oos-preds models/model_oos_predictions.csv
```
Reports total return, Sharpe ratio, and max drawdown for the model's
signal vs. plain buy-and-hold, using only out-of-sample predictions.

### 5. Serve the model via API
```bash
uvicorn src.api:app --reload --port 8000
```
```bash
curl http://localhost:8000/health
curl "http://localhost:8000/predict?ticker=AAPL"
```

### 6. Run the dashboard
```bash
streamlit run dashboard/app.py
```

### 7. Run in Docker
```bash
docker build -t stock-mlops .
docker run -p 8000:8000 stock-mlops
```
or
```bash
docker compose up --build
```

### 8. Run tests
```bash
pytest tests/ -v
```

### 9. Automated retraining (champion/challenger)
```bash
python -m src.retrain_pipeline --ticker AAPL
```
Trains a new "challenger" model on fresh data and only promotes it to
production (`models/model.joblib`) if it performs at least as well as
the current "champion" — see `dags/retrain_dag.py` for how to run this
weekly via Airflow.

---

## Suggested week-by-week build plan

If you're building this yourself rather than just running it, here's
a realistic pace:

**Week 1 — Data & features**
- Get comfortable pulling data with `yfinance`.
- Implement and unit-test technical indicators (RSI, MACD, SMAs).
- Plot a few features against price to sanity-check them visually.

**Week 2 — Modeling & honest validation**
- Implement walk-forward validation from scratch (don't skip this —
  it's the single most interview-relevant part of this project).
- Train XGBoost, compare to a naive baseline and a simple logistic
  regression.
- Write the no-lookahead unit test — this is a great thing to point to
  in an interview as evidence of engineering rigor.

**Week 3 — MLOps layer**
- Add MLflow tracking.
- Build the FastAPI serving endpoint.
- Containerize with Docker.

**Week 4 — Automation & polish**
- Build the champion/challenger retraining script.
- Wire up GitHub Actions CI (tests + training smoke test + docker build).
- Add the Airflow DAG (even if you only run it locally once).
- Build the Streamlit dashboard.
- Write this README, and record a short demo GIF/video for your portfolio.
