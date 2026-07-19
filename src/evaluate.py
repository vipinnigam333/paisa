"""
evaluate.py
-----------
Turns the model's directional predictions into a simple paper-trading
backtest, and reports the metrics an interviewer (or a finance-minded
hiring manager) will actually ask about: Sharpe ratio, max drawdown,
cumulative return -- compared against a buy-and-hold baseline.

Strategy simulated: go long (hold 1 share worth $1) when the model
predicts UP, go flat (cash) when it predicts DOWN. No shorting, no
leverage, no transaction costs modeled (noted as a limitation).

IMPORTANT: this backtests using OUT-OF-SAMPLE predictions saved by
train.py (models/model_oos_predictions.csv) -- i.e. each prediction
only ever came from a model trained on data strictly before it. If
you backtest using the final deployed model's predictions on its own
training data instead, you'll get unrealistically good results
because the model has already seen those days. Run train.py first.

Usage:
    python -m src.evaluate --data data/sample_data/AAPL.csv --oos-preds models/model_oos_predictions.csv
"""
import argparse

import numpy as np
import pandas as pd


def max_drawdown(cum_returns: pd.Series) -> float:
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    return drawdown.min()


def sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    excess = daily_returns - risk_free_rate / periods_per_year
    if excess.std() == 0:
        return 0.0
    return np.sqrt(periods_per_year) * excess.mean() / excess.std()


def backtest(data_path: str, oos_preds_path: str):
    raw = pd.read_csv(data_path, index_col=0, parse_dates=True)
    oos = pd.read_csv(oos_preds_path, index_col=0, parse_dates=True)

    close = raw["Close"]
    next_day_return = close.pct_change().shift(-1).reindex(oos.index)

    aligned = pd.DataFrame({"pred": oos["pred"], "next_day_return": next_day_return}).dropna()

    strategy_returns = pd.Series(
        np.where(aligned["pred"] == 1, aligned["next_day_return"], 0.0), index=aligned.index
    )
    buyhold_returns = aligned["next_day_return"]

    strategy_cum = (1 + strategy_returns).cumprod()
    buyhold_cum = (1 + buyhold_returns).cumprod()

    results = {
        "strategy_total_return_pct": (strategy_cum.iloc[-1] - 1) * 100,
        "buyhold_total_return_pct": (buyhold_cum.iloc[-1] - 1) * 100,
        "strategy_sharpe": sharpe_ratio(strategy_returns),
        "buyhold_sharpe": sharpe_ratio(buyhold_returns),
        "strategy_max_drawdown_pct": max_drawdown(strategy_cum) * 100,
        "buyhold_max_drawdown_pct": max_drawdown(buyhold_cum) * 100,
        "n_days": len(strategy_returns),
    }

    print("=== Backtest: Model Signal Strategy vs Buy & Hold ===")
    print(f"Period:                  {aligned.index[0].date()} to {aligned.index[-1].date()} "
          f"({results['n_days']} trading days)")
    print(f"Strategy total return:   {results['strategy_total_return_pct']:+.2f}%")
    print(f"Buy & hold total return: {results['buyhold_total_return_pct']:+.2f}%")
    print(f"Strategy Sharpe ratio:   {results['strategy_sharpe']:.3f}")
    print(f"Buy & hold Sharpe ratio: {results['buyhold_sharpe']:.3f}")
    print(f"Strategy max drawdown:   {results['strategy_max_drawdown_pct']:.2f}%")
    print(f"Buy & hold max drawdown: {results['buyhold_max_drawdown_pct']:.2f}%")
    print("\nNote: no transaction costs, slippage, or shorting modeled. "
          "This is a directional-signal sanity check, not a production trading strategy.")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_data/AAPL.csv")
    parser.add_argument("--oos-preds", default="models/model_oos_predictions.csv")
    args = parser.parse_args()
    backtest(args.data, args.oos_preds)


if __name__ == "__main__":
    main()
