from __future__ import annotations

import argparse
import csv
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

from app import generate_signal

DEFAULT_SYMBOLS = [
    "NIFTY",
    "BANKNIFTY",
    "SENSEX",
    "RELIANCE",
    "TCS",
    "INFY",
    "HDFCBANK",
    "ICICIBANK",
    "SBIN",
    "LT",
    "AXISBANK",
    "KOTAKBANK",
    "ITC",
]

CSV_COLUMNS = [
    "scan_time",
    "symbol",
    "normalized_symbol",
    "option_action",
    "action",
    "confidence",
    "probability_up",
    "spot_price",
    "analysis_interval",
    "hold_minutes",
    "option_hint",
]

RUNNING = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live intraday option signal scans every N minutes.")
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=300,
        help="Seconds between scans (default: 300).",
    )
    parser.add_argument(
        "--hold-minutes",
        type=int,
        default=60,
        help="Holding horizon passed to the model (default: 60).",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=",".join(DEFAULT_SYMBOLS),
        help="Comma-separated symbols to scan.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output CSV path (default auto: outputs/live_<action>_signals.csv).",
    )
    parser.add_argument(
        "--option-action",
        type=str,
        default="BUY_PUT",
        help="Option action to track: BUY_PUT or BUY_CALL (default: BUY_PUT).",
    )
    parser.add_argument(
        "--max-scans",
        type=int,
        default=0,
        help="Optional max scan count for testing (0 means run forever).",
    )
    return parser.parse_args()


def on_signal(_sig: int, _frame: object) -> None:
    global RUNNING
    RUNNING = False


def ensure_csv(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        return
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()


def append_rows(csv_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerows(rows)


def scan_once(symbols: list[str], hold_minutes: int, option_action: str) -> tuple[list[dict], list[str]]:
    scan_time = datetime.now().isoformat(timespec="seconds")
    put_rows: list[dict] = []
    errors: list[str] = []

    for symbol in symbols:
        try:
            result = generate_signal(symbol, hold_minutes=hold_minutes, preferred_interval="auto")
            if result.option_action != option_action:
                continue

            put_rows.append(
                {
                    "scan_time": scan_time,
                    "symbol": symbol,
                    "normalized_symbol": result.symbol,
                    "option_action": result.option_action,
                    "action": result.action,
                    "confidence": round(float(result.confidence), 4),
                    "probability_up": round(float(result.probability_up), 4),
                    "spot_price": round(float(result.spot_price), 2),
                    "analysis_interval": result.analysis_interval,
                    "hold_minutes": int(result.hold_minutes),
                    "option_hint": result.option_hint,
                }
            )
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")

    return put_rows, errors


def main() -> int:
    args = parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    option_action = args.option_action.strip().upper()
    if option_action not in {"BUY_PUT", "BUY_CALL"}:
        print("option-action must be BUY_PUT or BUY_CALL", file=sys.stderr)
        return 1

    if args.output.strip():
        output_path = Path(args.output.strip())
    else:
        output_path = Path(f"outputs/live_{option_action.lower()}_signals.csv")

    if args.interval_seconds < 10:
        print("interval-seconds must be >= 10", file=sys.stderr)
        return 1
    if args.hold_minutes <= 0:
        print("hold-minutes must be > 0", file=sys.stderr)
        return 1
    if not symbols:
        print("No symbols provided.", file=sys.stderr)
        return 1

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    ensure_csv(output_path)

    print(f"Starting live {option_action} monitor. Output: {output_path}", flush=True)
    print(f"Symbols: {', '.join(symbols)}", flush=True)
    print(f"Scan every {args.interval_seconds}s, hold_minutes={args.hold_minutes}", flush=True)

    scan_count = 0
    while RUNNING:
        scan_count += 1
        rows, errors = scan_once(
            symbols=symbols,
            hold_minutes=args.hold_minutes,
            option_action=option_action,
        )
        append_rows(output_path, rows)

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"[{ts}] scan #{scan_count}: {option_action} rows={len(rows)} errors={len(errors)}",
            flush=True,
        )
        for err in errors[:5]:
            print(f"  error: {err}", flush=True)

        if args.max_scans > 0 and scan_count >= args.max_scans:
            break

        sleep_left = args.interval_seconds
        while RUNNING and sleep_left > 0:
            step = min(1, sleep_left)
            time.sleep(step)
            sleep_left -= step

    print(f"Stopped live {option_action} monitor.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
