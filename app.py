from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request, build_opener
from urllib.parse import quote

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, render_template, request
from sklearn.ensemble import RandomForestClassifier

APP_TITLE = "Intraday Stock & Options Signal"
INTRADAY_PERIOD = "1mo"
ANALYSIS_DAYS = 5
DEFAULT_HOLD_MINUTES = 60
MIN_HOLD_MINUTES = 5
MAX_HOLD_MINUTES = 240
INTERVAL_CANDIDATES = ["5m", "15m", "30m", "60m"]
DEFAULT_CANDLE_INTERVAL = "auto"
SEED = 42

# Signal thresholds for horizon-based UP probability.
BUY_THRESHOLD = 0.58
SELL_THRESHOLD = 0.42
OPTION_CONFIDENCE_MIN = 0.60
MIN_FEATURE_ROWS = 20
MIN_TRAIN_ROWS = 20
EXTRA_ROWS_BUFFER = 8
INDEX_ALIASES = {
    "NIFTY": "^NSEI",
    "NIFTY50": "^NSEI",
    "NIFTY 50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "NIFTYBANK": "^NSEBANK",
    "BANK NIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
}
INDEX_PROXY = {
    "^NSEI": "NIFTYBEES.NS",
    "^NSEBANK": "BANKBEES.NS",
}
NSE_BASE_URL = "https://www.nseindia.com"
NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
    "Connection": "keep-alive",
}
INDEX_NAME_BY_SYMBOL = {
    "^NSEI": "NIFTY 50",
    "^NSEBANK": "NIFTY BANK",
}
LIVE_CALL_CSV = Path("outputs/live_buy_call_signals.csv")
LIVE_PUT_CSV = Path("outputs/live_put_signals.csv")


@dataclass
class SignalResult:
    symbol: str
    spot_price: float
    probability_up: float
    action: str
    confidence: float
    option_action: str
    option_hint: str
    stock_suggestion: str
    option_suggestion: str
    winning_probability: float
    decision_factors: list[str]
    market_snapshot: dict
    analysis_text: str
    model_note: str
    analysis_interval: str
    hold_minutes: int


def normalize_symbol(raw_symbol: str) -> str:
    symbol = (raw_symbol or "").strip().upper()
    if not symbol:
        raise ValueError("Please enter a stock symbol.")

    if symbol in INDEX_ALIASES:
        return INDEX_ALIASES[symbol]

    # Allow direct index tickers like ^NSEI, ^NSEBANK, ^BSESN.
    if symbol.startswith("^"):
        return symbol

    # If user gives RELIANCE or TCS, map to NSE by default.
    if "." not in symbol and symbol.isalpha():
        return f"{symbol}.NS"
    return symbol


def interval_to_minutes(interval: str) -> int:
    if interval.endswith("m"):
        return int(interval[:-1])
    if interval.endswith("h"):
        return int(interval[:-1]) * 60
    raise ValueError(f"Unsupported interval: {interval}")


def parse_hold_minutes(raw_hold_minutes: object) -> int:
    if raw_hold_minutes is None or str(raw_hold_minutes).strip() == "":
        return DEFAULT_HOLD_MINUTES

    try:
        hold_minutes = int(raw_hold_minutes)
    except Exception as exc:
        raise ValueError("Hold time must be a whole number in minutes.") from exc

    if hold_minutes < MIN_HOLD_MINUTES or hold_minutes > MAX_HOLD_MINUTES:
        raise ValueError(
            f"Hold time must be between {MIN_HOLD_MINUTES} and {MAX_HOLD_MINUTES} minutes."
        )
    return hold_minutes


def parse_candle_interval(raw_interval: object) -> str:
    if raw_interval is None:
        return DEFAULT_CANDLE_INTERVAL
    interval = str(raw_interval).strip().lower()
    if not interval:
        return DEFAULT_CANDLE_INTERVAL
    if interval == "auto":
        return "auto"
    allowed = set(INTERVAL_CANDIDATES)
    if interval not in allowed:
        raise ValueError(f"Candle interval must be one of: auto, {', '.join(INTERVAL_CANDIDATES)}.")
    return interval


def load_intraday_data(symbol: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        tickers=symbol,
        period=INTRADAY_PERIOD,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise ValueError("No intraday data found for this symbol.")

    if isinstance(df.columns, pd.MultiIndex):
        cols0 = pd.Index(df.columns.get_level_values(0)).map(str).str.title()
        df.columns = cols0

    df = df.rename(columns=str.title)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[needed].copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().replace([np.inf, -np.inf], np.nan).ffill().dropna()

    # Keep only the last N trading days for model analysis.
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    trade_days = df.index.normalize()
    last_days = sorted(trade_days.unique())[-ANALYSIS_DAYS:]
    df = df[trade_days.isin(last_days)].copy()

    return df


def _nse_get_json(path: str) -> dict | list:
    opener = build_opener()

    # Warm-up request to set cookies required by NSE APIs.
    warm_req = Request(f"{NSE_BASE_URL}/", headers=NSE_HEADERS)
    opener.open(warm_req, timeout=8).read()

    req = Request(f"{NSE_BASE_URL}{path}", headers=NSE_HEADERS)
    with opener.open(req, timeout=8) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def fetch_nse_live_price(symbol: str) -> float | None:
    try:
        # NSE indices from allIndices payload.
        if symbol in INDEX_NAME_BY_SYMBOL:
            target_name = INDEX_NAME_BY_SYMBOL[symbol]
            payload = _nse_get_json("/api/allIndices")
            if isinstance(payload, dict):
                for row in payload.get("data", []) or []:
                    if str(row.get("index", "")).strip().upper() == target_name.upper():
                        value = row.get("last")
                        if value is not None:
                            return float(value)
            return None

        # NSE equities via quote-equity.
        eq_symbol = symbol[:-3] if symbol.endswith(".NS") else symbol
        eq_symbol = eq_symbol.strip().upper()
        if not eq_symbol or not eq_symbol.isalnum():
            return None
        path = f"/api/quote-equity?symbol={quote(eq_symbol)}"
        payload = _nse_get_json(path)
        if isinstance(payload, dict):
            price_info = payload.get("priceInfo", {}) or {}
            last = price_info.get("lastPrice")
            if last is not None:
                return float(str(last).replace(",", ""))
    except (URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError):
        return None
    except Exception:
        return None
    return None


def build_intraday_features(df: pd.DataFrame, horizon_candles: int) -> pd.DataFrame:
    feat = df.copy()

    feat["ret_1"] = feat["Close"].pct_change(1)
    feat["ret_3"] = feat["Close"].pct_change(3)
    feat["ret_6"] = feat["Close"].pct_change(6)

    feat["sma_9"] = feat["Close"].rolling(9).mean()
    feat["sma_21"] = feat["Close"].rolling(21).mean()
    feat["vol_20"] = feat["ret_1"].rolling(20).std()

    feat["price_vs_sma9"] = feat["Close"] / feat["sma_9"] - 1
    feat["price_vs_sma21"] = feat["Close"] / feat["sma_21"] - 1
    feat["vol_chg_5"] = feat["Volume"].pct_change(5)

    # Popular technical indicators.
    feat["ema12"] = feat["Close"].ewm(span=12, adjust=False).mean()
    feat["ema26"] = feat["Close"].ewm(span=26, adjust=False).mean()
    feat["macd"] = feat["ema12"] - feat["ema26"]
    feat["macd_signal"] = feat["macd"].ewm(span=9, adjust=False).mean()
    feat["macd_hist"] = feat["macd"] - feat["macd_signal"]

    delta = feat["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    feat["rsi14"] = 100 - (100 / (1 + rs))

    bb_mid = feat["Close"].rolling(20).mean()
    bb_std = feat["Close"].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    feat["bb_width"] = (bb_upper - bb_lower) / bb_mid
    feat["bb_percent_b"] = (feat["Close"] - bb_lower) / (bb_upper - bb_lower)

    prev_close = feat["Close"].shift(1)
    tr1 = feat["High"] - feat["Low"]
    tr2 = (feat["High"] - prev_close).abs()
    tr3 = (feat["Low"] - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    feat["atr14"] = true_range.rolling(14).mean()
    feat["atr14_pct"] = feat["atr14"] / feat["Close"]

    # Target for next 1 hour direction based on selected interval.
    feat["target"] = (feat["Close"].shift(-horizon_candles) > feat["Close"]).astype(int)

    feat = feat.replace([np.inf, -np.inf], np.nan).dropna().copy()
    min_rows_needed = max(MIN_FEATURE_ROWS, horizon_candles + EXTRA_ROWS_BUFFER)
    if len(feat) < min_rows_needed:
        raise ValueError(
            f"Insufficient intraday candles for this symbol right now (need at least {min_rows_needed})."
        )
    return feat


def train_and_predict_latest(feat_df: pd.DataFrame) -> Dict[str, float | int]:
    features = [
        "ret_1",
        "ret_3",
        "ret_6",
        "sma_9",
        "sma_21",
        "vol_20",
        "price_vs_sma9",
        "price_vs_sma21",
        "vol_chg_5",
        "ema12",
        "ema26",
        "macd",
        "macd_signal",
        "macd_hist",
        "rsi14",
        "bb_width",
        "bb_percent_b",
        "atr14_pct",
    ]

    # Time-based split on intraday candles (no random shuffle).
    split_idx = int(len(feat_df) * 0.8)
    split_idx = max(split_idx, MIN_TRAIN_ROWS)
    split_idx = min(split_idx, len(feat_df) - 1)
    train_df = feat_df.iloc[:split_idx]
    if len(train_df) < MIN_TRAIN_ROWS:
        raise ValueError(
            f"Not enough training candles after split (need at least {MIN_TRAIN_ROWS}). "
            "Try another symbol or wait for more market data."
        )

    x_train = train_df[features]
    y_train = train_df["target"]

    x_latest = feat_df[features].iloc[[-1]]

    model = RandomForestClassifier(
        n_estimators=600,
        max_depth=10,
        min_samples_leaf=4,
        random_state=SEED,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(x_train, y_train)

    prob_up = float(model.predict_proba(x_latest)[0, 1])
    pred = int(prob_up >= 0.5)
    return {"prob_up": prob_up, "pred": pred}


def classify_action(prob_up: float) -> tuple[str, float]:
    if prob_up >= BUY_THRESHOLD:
        return "BUY", prob_up
    if prob_up <= SELL_THRESHOLD:
        return "SELL", 1 - prob_up
    return "HOLD", max(prob_up, 1 - prob_up)


def nearest_strike(spot: float, symbol: str) -> int:
    # Typical strike spacing approximation for hints.
    if symbol == "^NSEI":
        step = 50
    elif symbol == "^NSEBANK":
        step = 100
    elif symbol == "^BSESN":
        step = 100
    else:
        step = 10
    return int(round(spot / step) * step)


def option_signal(action: str, confidence: float, spot: float, symbol: str) -> tuple[str, str]:
    strike = nearest_strike(spot, symbol)
    instrument_type = "index option" if symbol.startswith("^") else "stock option"

    if confidence < OPTION_CONFIDENCE_MIN or action == "HOLD":
        return "NO_TRADE", f"Confidence below {OPTION_CONFIDENCE_MIN:.0%}. Consider no options entry."

    if action == "BUY":
        return "BUY_CALL", f"Consider near ATM CE strike around {strike} ({instrument_type})."

    return "BUY_PUT", f"Consider near ATM PE strike around {strike} ({instrument_type})."


def stock_suggestion_text(action: str, confidence: float, hold_minutes: int) -> str:
    if action == "BUY":
        return f"Stock Intraday: BUY for ~{hold_minutes} min with ~{confidence:.1%} win probability."
    if action == "SELL":
        return f"Stock Intraday: SELL for ~{hold_minutes} min with ~{confidence:.1%} win probability."
    return (
        f"Stock Intraday: HOLD for now. Win probability ~{confidence:.1%} is not strong enough "
        f"for a {hold_minutes}-minute hold."
    )


def option_suggestion_text(option_action: str, option_hint: str, confidence: float, hold_minutes: int) -> str:
    if option_action == "NO_TRADE":
        return f"Options: NO TRADE for {hold_minutes} min. Win probability ~{confidence:.1%}. {option_hint}"
    return (
        f"Options: {option_action} for ~{hold_minutes} min with ~{confidence:.1%} win probability. "
        f"{option_hint}"
    )


def build_decision_explanation(
    latest_row: pd.Series,
    prob_up: float,
    action: str,
    hold_minutes: int,
    interval: str,
) -> tuple[list[str], dict]:
    sma9 = float(latest_row["sma_9"])
    sma21 = float(latest_row["sma_21"])
    ret_1 = float(latest_row["ret_1"])
    ret_3 = float(latest_row["ret_3"])
    vol_chg_5 = float(latest_row["vol_chg_5"])
    vol_20 = float(latest_row["vol_20"])
    price_vs_sma9 = float(latest_row["price_vs_sma9"])
    price_vs_sma21 = float(latest_row["price_vs_sma21"])
    ema12 = float(latest_row["ema12"])
    ema26 = float(latest_row["ema26"])
    macd = float(latest_row["macd"])
    macd_signal = float(latest_row["macd_signal"])
    macd_hist = float(latest_row["macd_hist"])
    rsi14 = float(latest_row["rsi14"])
    bb_width = float(latest_row["bb_width"])
    bb_percent_b = float(latest_row["bb_percent_b"])
    atr14_pct = float(latest_row["atr14_pct"])

    direction_text = "UP" if prob_up >= 0.5 else "DOWN"
    factors = [
        f"Model probability for {direction_text} move over next {hold_minutes} min: {max(prob_up, 1 - prob_up):.2%}.",
        f"Trend check: SMA9 ({sma9:.2f}) vs SMA21 ({sma21:.2f}) and price-vs-SMA9 ({price_vs_sma9:.2%}).",
        f"Momentum: 1-candle return {ret_1:.2%}, 3-candle return {ret_3:.2%}.",
        f"Volume context: last 5-candle volume change {vol_chg_5:.2%}.",
        f"Volatility context: rolling 20-candle volatility {vol_20:.4f} on {interval} candles.",
        f"MACD set: macd {macd:.4f}, macd_signal {macd_signal:.4f}, macd_hist {macd_hist:.4f}.",
        f"EMA set: ema12 {ema12:.2f}, ema26 {ema26:.2f}.",
        f"RSI: rsi14 {rsi14:.2f}. Bollinger: bb_width {bb_width:.2%}, bb_percent_b {bb_percent_b:.2f}.",
        f"ATR: atr14_pct {atr14_pct:.2%}.",
    ]

    if action == "BUY":
        factors.append("Signal bias: bullish setup supports stock BUY / option CALL side.")
    elif action == "SELL":
        factors.append("Signal bias: bearish setup supports stock SELL / option PUT side.")
    else:
        factors.append("Signal bias: mixed setup, so HOLD/NO_TRADE preferred.")

    snapshot = {
        "ret_1": round(ret_1, 4),
        "ret_3": round(ret_3, 4),
        "price_vs_sma9": round(price_vs_sma9, 4),
        "price_vs_sma21": round(price_vs_sma21, 4),
        "vol_chg_5": round(vol_chg_5, 4),
        "vol_20": round(vol_20, 4),
        "sma_9": round(sma9, 2),
        "sma_21": round(sma21, 2),
        "ema12": round(ema12, 2),
        "ema26": round(ema26, 2),
        "macd": round(macd, 4),
        "macd_signal": round(macd_signal, 4),
        "macd_hist": round(macd_hist, 4),
        "rsi14": round(rsi14, 2),
        "bb_width": round(bb_width, 4),
        "bb_percent_b": round(bb_percent_b, 4),
        "atr14_pct": round(atr14_pct, 4),
    }
    return factors, snapshot


def build_analysis_text(snapshot: dict, factors: list[str]) -> str:
    lines = [
        "Why This Signal",
        f"ret_1: {snapshot['ret_1'] * 100:.2f}%",
        f"ret_3: {snapshot['ret_3'] * 100:.2f}%",
        f"price_vs_sma9: {snapshot['price_vs_sma9'] * 100:.2f}%",
        f"vol_chg_5: {snapshot['vol_chg_5'] * 100:.2f}%",
        "",
        "MACD set:",
        f"macd: {snapshot['macd']}",
        f"macd_signal: {snapshot['macd_signal']}",
        f"macd_hist: {snapshot['macd_hist']}",
        "EMA:",
        f"ema12: {snapshot['ema12']}",
        f"ema26: {snapshot['ema26']}",
        "RSI:",
        f"rsi14: {snapshot['rsi14']}",
        "Bollinger:",
        f"bb_width: {snapshot['bb_width']}",
        f"bb_percent_b: {snapshot['bb_percent_b']}",
        "ATR:",
        f"atr14_pct: {snapshot['atr14_pct']}",
    ]
    lines.extend(factors)
    return "\\n".join(lines)


def generate_signal(symbol_input: str, hold_minutes: int, preferred_interval: str = "auto") -> SignalResult:
    symbol = normalize_symbol(symbol_input)
    candidate_symbols = [symbol]
    if symbol in INDEX_PROXY:
        candidate_symbols.append(INDEX_PROXY[symbol])

    feat_df = None
    used_symbol = None
    used_interval = None
    last_error = "No intraday data."
    if preferred_interval == "auto":
        interval_order = INTERVAL_CANDIDATES.copy()
    else:
        interval_order = [preferred_interval] + [i for i in INTERVAL_CANDIDATES if i != preferred_interval]

    for candidate in candidate_symbols:
        for interval in interval_order:
            try:
                interval_mins = interval_to_minutes(interval)
                horizon_candles = max(1, int(round(hold_minutes / interval_mins)))
                price_df = load_intraday_data(candidate, interval=interval)
                feat_df = build_intraday_features(price_df, horizon_candles=horizon_candles)
                used_symbol = candidate
                used_interval = interval
                break
            except Exception as exc:
                last_error = str(exc)
        if feat_df is not None:
            break

    if feat_df is None or used_symbol is None or used_interval is None:
        raise ValueError(f"Could not build intraday model input. Last error: {last_error}")

    latest_price_model = float(feat_df["Close"].iloc[-1])
    latest_price_nse = fetch_nse_live_price(symbol)
    latest_price = latest_price_nse if latest_price_nse is not None else latest_price_model
    pred_info = train_and_predict_latest(feat_df)
    latest_row = feat_df.iloc[-1]

    action, confidence = classify_action(pred_info["prob_up"])
    opt_action, opt_hint = option_signal(action, confidence, latest_price, symbol)
    stock_suggestion = stock_suggestion_text(action, confidence, hold_minutes)
    option_suggestion = option_suggestion_text(opt_action, opt_hint, confidence, hold_minutes)
    decision_factors, market_snapshot = build_decision_explanation(
        latest_row=latest_row,
        prob_up=float(pred_info["prob_up"]),
        action=action,
        hold_minutes=hold_minutes,
        interval=used_interval,
    )
    analysis_text = build_analysis_text(market_snapshot, decision_factors)
    source_note = f"Data source: {used_symbol}, interval: {used_interval}."
    if latest_price_nse is not None:
        source_note += " Spot price source: NSE live quote."
    else:
        source_note += " Spot price source: model candle close (NSE quote unavailable)."
    if used_symbol != symbol:
        source_note += f" Used proxy for {symbol} due to intraday feed availability."
    if preferred_interval != "auto" and used_interval != preferred_interval:
        source_note += f" Requested interval {preferred_interval} not available; fallback used."

    return SignalResult(
        symbol=symbol,
        spot_price=latest_price,
        probability_up=float(pred_info["prob_up"]),
        action=action,
        confidence=float(confidence),
        option_action=opt_action,
        option_hint=opt_hint,
        stock_suggestion=stock_suggestion,
        option_suggestion=option_suggestion,
        winning_probability=float(confidence),
        decision_factors=decision_factors,
        market_snapshot=market_snapshot,
        analysis_text=analysis_text,
        model_note=(
            f"Model uses last {ANALYSIS_DAYS} trading days and predicts next {hold_minutes} minutes direction. "
            f"{source_note}"
        ),
        analysis_interval=used_interval,
        hold_minutes=hold_minutes,
    )


def quote_url(symbol: str) -> str:
    return f"https://finance.yahoo.com/quote/{quote(str(symbol), safe='')}"


def load_live_rows(csv_path: Path, limit: int, option_action: str) -> list[dict]:
    if not csv_path.exists():
        return []

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return []

    if df.empty:
        return []

    if "option_action" in df.columns:
        df = df[df["option_action"] == option_action]

    if "scan_time" in df.columns:
        df = df.sort_values("scan_time", ascending=False)
    df = df.head(limit).copy()

    rows = []
    for _, row in df.iterrows():
        symbol = str(row.get("normalized_symbol") or row.get("symbol") or "")
        rows.append(
            {
                "scan_time": str(row.get("scan_time", "")),
                "symbol": str(row.get("symbol", "")),
                "normalized_symbol": symbol,
                "action": str(row.get("action", "")),
                "option_action": str(row.get("option_action", "")),
                "confidence": float(row.get("confidence", 0) or 0),
                "probability_up": float(row.get("probability_up", 0) or 0),
                "spot_price": float(row.get("spot_price", 0) or 0),
                "analysis_interval": str(row.get("analysis_interval", "")),
                "hold_minutes": int(float(row.get("hold_minutes", 0) or 0)),
                "option_hint": str(row.get("option_hint", "")),
                "quote_url": quote_url(symbol),
            }
        )
    return rows


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html", app_title=APP_TITLE)


@app.route("/api/signal", methods=["POST"])
def api_signal():
    payload = request.get_json(silent=True) or {}
    symbol = payload.get("symbol", "")

    try:
        hold_minutes = parse_hold_minutes(payload.get("hold_minutes"))
        candle_interval = parse_candle_interval(payload.get("candle_interval"))
        result = generate_signal(symbol, hold_minutes=hold_minutes, preferred_interval=candle_interval)
        return jsonify(
            {
                "ok": True,
                "symbol": result.symbol,
                "spot_price": round(result.spot_price, 2),
                "probability_up": round(result.probability_up, 4),
                "probability_down": round(1 - result.probability_up, 4),
                "action": result.action,
                "confidence": round(result.confidence, 4),
                "winning_probability": round(result.winning_probability, 4),
                "option_action": result.option_action,
                "option_hint": result.option_hint,
                "stock_suggestion": result.stock_suggestion,
                "option_suggestion": result.option_suggestion,
                "decision_factors": result.decision_factors,
                "market_snapshot": result.market_snapshot,
                "analysis_text": result.analysis_text,
                "model_note": result.model_note,
                "analysis_interval": result.analysis_interval,
                "requested_interval": candle_interval,
                "hold_minutes": result.hold_minutes,
                "horizon": f"next_{result.hold_minutes}_minutes",
                "warning": "Educational signal only, not financial advice.",
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route("/api/live-signals", methods=["GET"])
def api_live_signals():
    try:
        limit = int(request.args.get("limit", 20))
        limit = max(1, min(limit, 200))
    except Exception:
        limit = 20

    calls = load_live_rows(LIVE_CALL_CSV, limit=limit, option_action="BUY_CALL")
    puts = load_live_rows(LIVE_PUT_CSV, limit=limit, option_action="BUY_PUT")

    return jsonify(
        {
            "ok": True,
            "buy_call": calls,
            "buy_put": puts,
            "sources": {
                "buy_call_csv": str(LIVE_CALL_CSV),
                "buy_put_csv": str(LIVE_PUT_CSV),
            },
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
