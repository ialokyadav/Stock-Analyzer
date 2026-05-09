# NIFTY 50 ML Direction Strategy + Intraday Web App

This project now includes:
- `nifty_ml_strategy.py`: long-horizon NIFTY model/backtest pipeline
- `app.py`: Flask web app for intraday stock and options signals

## 1) NIFTY 50 Pipeline (2000-2025)

`nifty_ml_strategy.py` does all of the following:
- Downloads historical data with `yfinance`
- Cleans and preprocesses data
- Engineers technical features (MA20/MA50/MA200, volatility, momentum, rolling returns)
- Creates binary target labels for next-day direction
- Uses a **time-based split** (train before 2020, test 2020 onward)
- Trains `XGBoostClassifier` (fallback: `RandomForestClassifier`)
- Evaluates with accuracy, confusion matrix, and classification report
- Generates model-based trading signals and backtests strategy
- Computes cumulative return, Sharpe ratio, max drawdown
- Saves plots and test-period CSV output in `outputs/`

Run:
```bash
python3 nifty_ml_strategy.py
```

## 2) Intraday Frontend + Backend

`app.py` serves a frontend where you input a stock symbol and backend returns:
- Intraday action: `BUY`, `HOLD`, `SELL`
- Confidence score
- Options action: `BUY_CALL`, `BUY_PUT`, `NO_TRADE`
- Quick ATM strike hint for options

### Start web app

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

Open:
- `http://127.0.0.1:5000`

### Input format examples
- `RELIANCE` (auto-maps to `RELIANCE.NS`)
- `TCS`
- `HDFCBANK.NS`

## Important
- Signals are for educational use and are **not financial advice**.
- Intraday model predicts the next 5-minute candle direction from recent 5-minute data.
