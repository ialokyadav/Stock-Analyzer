"""
NIFTY 50 Direction Prediction and Backtest (2005-2025)

This script downloads daily NIFTY 50 data, engineers features,
trains a classifier, evaluates performance, and backtests a
prediction-driven trading strategy against buy-and-hold.

Run:
    python nifty_ml_strategy.py
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Tuple
import pickle

import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Try XGBoost first; if unavailable, fallback to RandomForest.
try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


SEED = 42
TRADING_DAYS_PER_YEAR = 252
# Use fixed start and dynamic end date (today).
DATA_START_DATE = "2000-01-01"
DATA_END_DATE = date.today().isoformat()
SPLIT_DATE = "2020-01-01"
CONFIDENCE_THRESHOLD = 0.70
USE_SAVED_MODEL_IF_AVAILABLE = True
AUTO_RETRAIN_DAILY = True


@dataclass
class BacktestResults:
    cumulative_strategy_return: float
    cumulative_buy_hold_return: float
    sharpe_ratio: float
    max_drawdown: float


def load_data(symbol: str = "^NSEI", start: str = "2005-01-01", end: str = "2025-12-31") -> pd.DataFrame:
    """Download historical OHLCV data using yfinance."""
    data = yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=False, progress=False)

    if data.empty:
        raise ValueError("No data downloaded. Check ticker symbol, date range, or internet connection.")

    # yfinance can return MultiIndex columns (e.g., price field + ticker).
    # Normalize to single-level OHLCV columns so each field is a Series.
    if isinstance(data.columns, pd.MultiIndex):
        level0 = pd.Index(data.columns.get_level_values(0)).map(str).str.title()
        level1 = pd.Index(data.columns.get_level_values(1)).map(str).str.title()

        required_set = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        if required_set.issubset(set(level0)):
            data.columns = level0
        elif required_set.issubset(set(level1)):
            data.columns = level1
        else:
            # Last-resort flattening when yfinance changes column schema.
            data.columns = [
                "_".join([str(part) for part in col if str(part)]).title() for col in data.columns.to_flat_index()
            ]

    # Standardize columns and keep expected fields.
    data = data.rename(columns=str.title)
    required_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing_cols = [c for c in required_cols if c not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    data = data[required_cols].copy()
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    return data


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: remove duplicates, handle missing values, and enforce numeric dtypes."""
    df = df[~df.index.duplicated(keep="first")].copy()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Forward-fill short gaps in market data, then drop any remaining NaNs.
    df = df.ffill().dropna()
    return df


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Create technical indicators and predictive features."""
    feat = df.copy()

    # Daily return (close-to-close).
    feat["return_1d"] = feat["Close"].pct_change()

    # Moving averages.
    feat["ma20"] = feat["Close"].rolling(window=20).mean()
    feat["ma50"] = feat["Close"].rolling(window=50).mean()
    feat["ma200"] = feat["Close"].rolling(window=200).mean()
    feat["ema12"] = feat["Close"].ewm(span=12, adjust=False).mean()
    feat["ema26"] = feat["Close"].ewm(span=26, adjust=False).mean()

    # MACD indicators.
    feat["macd"] = feat["ema12"] - feat["ema26"]
    feat["macd_signal"] = feat["macd"].ewm(span=9, adjust=False).mean()
    feat["macd_hist"] = feat["macd"] - feat["macd_signal"]

    # Volatility (annualized rolling std of daily returns).
    feat["volatility_20d"] = feat["return_1d"].rolling(window=20).std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Momentum features.
    feat["momentum_5d"] = feat["Close"].pct_change(periods=5)
    feat["momentum_10d"] = feat["Close"].pct_change(periods=10)
    feat["momentum_20d"] = feat["Close"].pct_change(periods=20)

    # Rolling returns.
    feat["rolling_return_20d"] = feat["Close"].pct_change(periods=20)
    feat["rolling_return_60d"] = feat["Close"].pct_change(periods=60)

    # RSI(14).
    delta = feat["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    feat["rsi14"] = 100 - (100 / (1 + rs))

    # Bollinger Bands(20,2) and position inside band.
    bb_mid = feat["Close"].rolling(20).mean()
    bb_std = feat["Close"].rolling(20).std()
    feat["bb_upper"] = bb_mid + 2 * bb_std
    feat["bb_lower"] = bb_mid - 2 * bb_std
    feat["bb_width"] = (feat["bb_upper"] - feat["bb_lower"]) / bb_mid
    feat["bb_percent_b"] = (feat["Close"] - feat["bb_lower"]) / (feat["bb_upper"] - feat["bb_lower"])

    # ATR(14).
    prev_close = feat["Close"].shift(1)
    tr1 = feat["High"] - feat["Low"]
    tr2 = (feat["High"] - prev_close).abs()
    tr3 = (feat["Low"] - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    feat["atr14"] = true_range.rolling(14).mean()
    feat["atr14_pct"] = feat["atr14"] / feat["Close"]

    # Price relative to moving averages.
    feat["price_vs_ma20"] = feat["Close"] / feat["ma20"] - 1
    feat["price_vs_ma50"] = feat["Close"] / feat["ma50"] - 1
    feat["price_vs_ma200"] = feat["Close"] / feat["ma200"] - 1

    # Volume trend.
    feat["volume_change_5d"] = feat["Volume"].pct_change(periods=5)

    # Target: next day direction (1 if next close is higher, else 0).
    feat["target"] = (feat["Close"].shift(-1) > feat["Close"]).astype(int)

    # Guard against infinite values from percentage/division operations.
    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.dropna().copy()
    return feat


def time_based_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    split_date: str = "2020-01-01",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Time-series split without random shuffling."""
    split_ts = pd.Timestamp(split_date)

    train_df = df[df.index < split_ts]
    test_df = df[df.index >= split_ts]

    if train_df.empty or test_df.empty:
        raise ValueError("Train or test split is empty. Adjust split_date or date range.")

    x_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    y_train = train_df["target"]
    x_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    y_test = test_df["target"]

    # Drop rows with any missing/non-finite features and align targets.
    train_mask = x_train.notna().all(axis=1)
    test_mask = x_test.notna().all(axis=1)
    x_train = x_train.loc[train_mask]
    y_train = y_train.loc[train_mask]
    x_test = x_test.loc[test_mask]
    y_test = y_test.loc[test_mask]

    return x_train, x_test, y_train, y_test


def train_model(x_train: pd.DataFrame, y_train: pd.Series):
    """Train XGBoost classifier if available, otherwise RandomForest."""
    if HAS_XGBOOST:
        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            random_state=SEED,
            eval_metric="logloss",
            n_jobs=-1,
        )
    else:
        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=5,
            random_state=SEED,
            n_jobs=-1,
        )

    model.fit(x_train, y_train)
    return model


def save_model(model, feature_cols: list[str], output_dir: Path = Path("models")) -> Path:
    """Save trained model to disk and return saved path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"nifty_direction_model_{date.today().isoformat()}.pkl"
    payload = {
        "model": model,
        "feature_cols": feature_cols,
        "saved_on": date.today().isoformat(),
    }

    with model_path.open("wb") as f:
        pickle.dump(payload, f)

    # Keep a stable latest pointer for easy loading.
    latest_path = output_dir / "nifty_direction_model_latest.pkl"
    with latest_path.open("wb") as f:
        pickle.dump(payload, f)

    return model_path


def load_saved_model_payload(model_path: Path = Path("models/nifty_direction_model_latest.pkl")) -> Dict[str, object] | None:
    """Load saved model payload (model + metadata) if available."""
    if not model_path.exists():
        return None

    with model_path.open("rb") as f:
        payload = pickle.load(f)

    # Backward compatibility with older files that stored only the raw model object.
    if isinstance(payload, dict) and "model" in payload:
        return payload
    return {"model": payload, "saved_on": None}


def evaluate_model(model, x_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, object]:
    """Compute standard classification metrics."""
    y_pred = model.predict(x_test)
    y_pred = np.asarray(y_pred).astype(int)

    # Probability of class 1 (UP). Fallback keeps script robust across models.
    if hasattr(model, "predict_proba"):
        y_prob_up = model.predict_proba(x_test)[:, 1]
    elif hasattr(model, "decision_function"):
        raw_score = model.decision_function(x_test)
        y_prob_up = 1 / (1 + np.exp(-raw_score))
    else:
        y_prob_up = y_pred.astype(float)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    metrics = {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "classification_report": report,
        "y_pred": y_pred,
        "y_prob_up": y_prob_up,
    }
    return metrics


def create_signals(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    y_prob_up: np.ndarray,
    test_index: pd.DatetimeIndex,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> pd.DataFrame:
    """Create long/cash signals from model predictions for the test period."""
    out = df.loc[test_index].copy()

    out["predicted_target"] = np.asarray(y_pred).astype(int)
    out["prob_up"] = np.asarray(y_prob_up).astype(float)
    out["prob_down"] = 1 - out["prob_up"]
    out["prediction_confidence"] = np.where(
        out["predicted_target"] == 1, out["prob_up"], out["prob_down"]
    )

    # Signal: go long only when prediction is UP and confidence >= threshold, else stay in cash.
    out["signal"] = np.where(
        (out["predicted_target"] == 1) & (out["prediction_confidence"] >= confidence_threshold),
        1,
        0,
    )

    # Strategy return uses previous day's prediction to avoid look-ahead bias.
    out["market_return"] = out["Close"].pct_change()
    out["strategy_return"] = out["signal"].shift(1).fillna(0) * out["market_return"]

    return out


def backtest_strategy(signals_df: pd.DataFrame) -> BacktestResults:
    """Calculate cumulative returns, Sharpe ratio, and max drawdown."""
    bt = signals_df.copy()

    bt["cum_strategy"] = (1 + bt["strategy_return"].fillna(0)).cumprod()
    bt["cum_buy_hold"] = (1 + bt["market_return"].fillna(0)).cumprod()

    strategy_ret = bt["strategy_return"].dropna()
    if strategy_ret.std() > 0:
        sharpe = np.sqrt(TRADING_DAYS_PER_YEAR) * strategy_ret.mean() / strategy_ret.std()
    else:
        sharpe = 0.0

    running_max = bt["cum_strategy"].cummax()
    drawdown = bt["cum_strategy"] / running_max - 1
    max_dd = float(drawdown.min())

    return BacktestResults(
        cumulative_strategy_return=float(bt["cum_strategy"].iloc[-1] - 1),
        cumulative_buy_hold_return=float(bt["cum_buy_hold"].iloc[-1] - 1),
        sharpe_ratio=float(sharpe),
        max_drawdown=max_dd,
    )


def plot_results(full_df: pd.DataFrame, signal_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate and save key visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Price history + MA ribbon.
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(full_df.index, full_df["Close"], label="Close", linewidth=1.2)
    ax.plot(full_df.index, full_df["ma20"], label="MA20", alpha=0.9)
    ax.plot(full_df.index, full_df["ma50"], label="MA50", alpha=0.9)
    ax.plot(full_df.index, full_df["ma200"], label="MA200", alpha=0.9)
    ax.set_title("NIFTY 50 Price with Moving Average Ribbon")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "price_ma_ribbon.png", dpi=150)
    plt.close(fig)

    # 2) Prediction vs Actual direction.
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(signal_df.index, signal_df["actual_target"], label="Actual (Up=1, Down=0)", alpha=0.7)
    ax.plot(signal_df.index, signal_df["predicted_target"], label="Predicted", alpha=0.7)
    ax.set_title("Prediction vs Actual Direction (Test Period)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Direction")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "prediction_vs_actual.png", dpi=150)
    plt.close(fig)

    # 3) Cumulative strategy vs buy-and-hold.
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(signal_df.index, signal_df["cum_strategy"], label="ML Strategy")
    ax.plot(signal_df.index, signal_df["cum_buy_hold"], label="Buy & Hold")
    ax.set_title("Cumulative Returns: Strategy vs Buy-and-Hold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "cumulative_returns.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Run the full pipeline end-to-end."""
    print("Downloading data...")
    raw_df = load_data(symbol="^NSEI", start=DATA_START_DATE, end=DATA_END_DATE)

    print("Cleaning data...")
    clean_df = clean_data(raw_df)

    print("Engineering features...")
    feat_df = feature_engineer(clean_df)

    feature_cols = [
        "return_1d",
        "ma20",
        "ma50",
        "ma200",
        "ema12",
        "ema26",
        "macd",
        "macd_signal",
        "macd_hist",
        "volatility_20d",
        "momentum_5d",
        "momentum_10d",
        "momentum_20d",
        "rolling_return_20d",
        "rolling_return_60d",
        "rsi14",
        "bb_width",
        "bb_percent_b",
        "atr14_pct",
        "price_vs_ma20",
        "price_vs_ma50",
        "price_vs_ma200",
        "volume_change_5d",
    ]

    print("Splitting train/test with time-based split...")
    x_train, x_test, y_train, y_test = time_based_split(
        feat_df,
        feature_cols=feature_cols,
        split_date=SPLIT_DATE,
    )

    model = None
    today_str = date.today().isoformat()
    if USE_SAVED_MODEL_IF_AVAILABLE:
        print("Checking for saved model...")
        saved_payload = load_saved_model_payload()
        if saved_payload is not None:
            saved_on = saved_payload.get("saved_on")
            if AUTO_RETRAIN_DAILY and saved_on != today_str:
                if saved_on is None:
                    print("Saved model found but without date metadata. Retraining now...")
                else:
                    print(f"Saved model is from {saved_on}. Retraining for today ({today_str})...")
            else:
                model = saved_payload["model"]
                print("Loaded saved model: models/nifty_direction_model_latest.pkl")

    if model is None:
        print("Training model...")
        model = train_model(x_train, y_train)
        saved_model_path = save_model(model, feature_cols=feature_cols)
        print(f"Model saved: {saved_model_path}")

    print("Evaluating model...")
    eval_out = evaluate_model(model, x_test, y_test)
    print(f"\nAccuracy: {eval_out['accuracy']:.4f}")
    print("\nConfusion Matrix:")
    print(eval_out["confusion_matrix"])
    print("\nClassification Report:")
    print(eval_out["classification_report"])

    print("Generating signals and backtest...")
    signal_df = create_signals(
        feat_df,
        eval_out["y_pred"],
        eval_out["y_prob_up"],
        x_test.index,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )
    signal_df["actual_target"] = y_test.values

    bt_results = backtest_strategy(signal_df)

    # Recompute cumulative series for plotting.
    signal_df["cum_strategy"] = (1 + signal_df["strategy_return"].fillna(0)).cumprod()
    signal_df["cum_buy_hold"] = (1 + signal_df["market_return"].fillna(0)).cumprod()

    print("\nBacktest Results (Test Period):")
    print(f"Cumulative Strategy Return: {bt_results.cumulative_strategy_return:.2%}")
    print(f"Cumulative Buy&Hold Return: {bt_results.cumulative_buy_hold_return:.2%}")
    print(f"Sharpe Ratio: {bt_results.sharpe_ratio:.3f}")
    print(f"Max Drawdown: {bt_results.max_drawdown:.2%}")
    print(f"Confidence Threshold for Trading: {CONFIDENCE_THRESHOLD:.0%}")

    # Show highest-confidence prediction in the test period.
    top_row = signal_df["prediction_confidence"].idxmax()
    top = signal_df.loc[top_row]
    top_label = "UP (1)" if int(top["predicted_target"]) == 1 else "DOWN (0)"
    print("\nHighest-Probability Prediction (Test Period):")
    print(f"Date: {top_row.date()}")
    print(f"Predicted Direction: {top_label}")
    print(f"Confidence: {top['prediction_confidence']:.2%}")
    print(f"Prob(UP): {top['prob_up']:.2%}, Prob(DOWN): {top['prob_down']:.2%}")

    # Also show latest available prediction, useful for quick decision checks.
    last_idx = signal_df.index[-1]
    last = signal_df.iloc[-1]
    last_label = "UP (1)" if int(last["predicted_target"]) == 1 else "DOWN (0)"
    print("\nLatest Prediction:")
    print(f"Date: {last_idx.date()}")
    print(f"Predicted Direction: {last_label}")
    print(f"Confidence: {last['prediction_confidence']:.2%}")
    print(f"Prob(UP): {last['prob_up']:.2%}, Prob(DOWN): {last['prob_down']:.2%}")

    output_dir = Path("outputs")
    print("Saving plots...")
    plot_results(feat_df, signal_df, output_dir=output_dir)

    # Save a table of test-period predictions and returns.
    signal_df.to_csv(output_dir / "test_predictions_and_returns.csv", index=True)

    print("\nDone. Files created in ./outputs:")
    print("- price_ma_ribbon.png")
    print("- prediction_vs_actual.png")
    print("- cumulative_returns.png")
    print("- test_predictions_and_returns.csv")
    print("- models/nifty_direction_model_YYYY-MM-DD.pkl")


if __name__ == "__main__":
    main()
