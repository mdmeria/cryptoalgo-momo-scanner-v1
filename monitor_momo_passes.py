"""Run strict momentum scan on a schedule and log passing symbols with timestamps."""

from __future__ import annotations

import argparse
import csv
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from momentum_quality import evaluate_momentum_setup

BASE_URL = "https://data-api.binance.vision/api/v3"


def now_est_iso() -> str:
    # Use fixed EST offset (-05:00) per user preference.
    est_now = datetime.now(timezone.utc) - timedelta(hours=5)
    return est_now.strftime("%Y-%m-%dT%H:%M:%S-05:00")


def infer_direction(df: pd.DataFrame) -> str:
    sm30 = df["close"].ewm(alpha=1 / 30, adjust=False).mean().iloc[-1]
    sm120 = df["close"].ewm(alpha=1 / 120, adjust=False).mean().iloc[-1]
    close = float(df["close"].iloc[-1])

    if close > sm30 > sm120:
        return "long"
    if close < sm30 < sm120:
        return "short"
    return "long"


def fetch_usdt_pairs() -> list[str]:
    exchange_info = requests.get(f"{BASE_URL}/exchangeInfo", timeout=20).json()
    return [
        s["symbol"]
        for s in exchange_info.get("symbols", [])
        if s.get("quoteAsset") == "USDT"
        and s.get("status") == "TRADING"
        and s.get("isSpotTradingAllowed")
    ]


def fetch_klines(symbol: str, limit: int = 500) -> pd.DataFrame | None:
    response = requests.get(
        f"{BASE_URL}/klines",
        params={"symbol": symbol, "interval": "1m", "limit": limit},
        timeout=8,
    )
    if response.status_code != 200:
        return None

    raw = response.json()
    if not isinstance(raw, list) or len(raw) < 120:
        return None

    df = pd.DataFrame(
        [
            {
                "timestamp": pd.to_datetime(k[0], unit="ms"),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            }
            for k in raw
        ]
    )
    return df.set_index("timestamp")


def evaluate_symbol(symbol: str, min_score: float):
    try:
        df = fetch_klines(symbol)
        if df is None:
            return None

        direction = infer_direction(df)
        quality = evaluate_momentum_setup(
            df=df,
            direction=direction,
            min_quality_score=min_score,
            symbol=symbol,
            enforce_extended_rules=True,
        )

        return symbol, direction, quality
    except Exception:
        return None


def ensure_header(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run_timestamp_est",
                "symbol",
                "direction",
                "quality_tier",
                "score",
                "day_change_pct",
                "entry_cross_count_6h",
                "retracements_found",
                "staircase_bars_120",
                "trend_stack_bars_120",
                "checks",
            ]
        )


def ensure_summary_header(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run_timestamp_est",
                "symbols_scanned",
                "symbols_evaluated",
                "symbols_passed",
                "top_failure_checks",
                "passes_log_path",
            ]
        )


def append_passes(path: Path, run_ts: str, passes: list[tuple[str, str, object]]) -> None:
    if not passes:
        return

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for symbol, direction, quality in passes:
            writer.writerow(
                [
                    run_ts,
                    symbol,
                    direction,
                    quality.quality_tier,
                    f"{quality.score:.2f}",
                    f"{quality.metrics.get('day_change_pct', float('nan')):.2f}",
                    int(quality.metrics.get("entry_cross_count_6h", -1)),
                    int(quality.metrics.get("retracements_found", 0)),
                    int(quality.metrics.get("staircase_bars_120", 0)),
                    int(quality.metrics.get("trend_stack_bars_120", 0)),
                    (
                        f"approach={quality.checks.get('slow_grind_approach')}|"
                        f"staircase={quality.checks.get('left_side_staircase')}|"
                        f"volume={quality.checks.get('volume_not_decreasing')}|"
                        f"not_choppy={quality.checks.get('not_choppy')}|"
                        f"balanced_2h={quality.checks.get('balanced_momo_2h')}|"
                        f"parallel_30smma_2h={quality.checks.get('parallel_to_smma30_2h')}|"
                        f"spread_30_120_increasing_2h={quality.checks.get('smma_spread_increasing_2h')}|"
                        f"day_change={quality.checks.get('day_change_ok')}|"
                        f"vwap_side={quality.checks.get('vwap_side_ok')}|"
                        f"first_2h_prev_vwap={quality.checks.get('first_2h_prev_day_vwap_ok')}|"
                        f"entry_fresh_6h={quality.checks.get('entry_not_crossed_6h')}"
                    ),
                ]
            )


def append_summary(
    path: Path,
    run_ts: str,
    scanned: int,
    evaluated: int,
    passed: int,
    top_failures: list[tuple[str, int]],
    pass_log_path: Path,
) -> None:
    top_failure_str = "|".join([f"{name}={count}" for name, count in top_failures])
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([run_ts, scanned, evaluated, passed, top_failure_str, str(pass_log_path.resolve())])


def run_scan(min_score: float, max_workers: int):
    symbols = fetch_usdt_pairs()
    passes: list[tuple[str, str, object]] = []
    fail_counter: Counter = Counter()
    evaluated = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(evaluate_symbol, symbol, min_score) for symbol in symbols]
        for future in as_completed(futures):
            out = future.result()
            if out is None:
                continue

            evaluated += 1
            symbol, direction, quality = out
            if quality.passed:
                passes.append((symbol, direction, quality))
            else:
                for check_name, check_ok in quality.checks.items():
                    if check_name == "enough_data":
                        continue
                    if not check_ok:
                        fail_counter[check_name] += 1

    passes.sort(key=lambda x: x[0])
    return symbols, passes, evaluated, fail_counter.most_common(5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor strict momentum passes every N minutes")
    parser.add_argument("--interval-min", type=int, default=15, help="Scan interval in minutes")
    parser.add_argument("--min-score", type=float, default=0.60, help="Minimum core quality score")
    parser.add_argument("--max-workers", type=int, default=20, help="Thread pool size")
    parser.add_argument("--output", default="momo_passes_log.csv", help="CSV output path")
    parser.add_argument(
        "--summary-output",
        default="momo_scan_summary_est.csv",
        help="Per-run summary CSV path (logs scanned/passed every run)",
    )
    parser.add_argument("--once", action="store_true", help="Run one scan and exit")
    args = parser.parse_args()

    out_path = Path(args.output)
    summary_path = Path(args.summary_output)
    ensure_header(out_path)
    ensure_summary_header(summary_path)

    while True:
        started = time.time()
        run_ts = now_est_iso()

        symbols, passes, evaluated, top_failures = run_scan(
            min_score=args.min_score,
            max_workers=args.max_workers,
        )
        append_passes(out_path, run_ts, passes)
        append_summary(
            summary_path,
            run_ts,
            len(symbols),
            evaluated,
            len(passes),
            top_failures,
            out_path,
        )

        print(
            f"[{run_ts}] scanned={len(symbols)} evaluated={evaluated} passed={len(passes)} "
            f"logged_to={out_path.resolve()}"
        )
        if top_failures:
            top_failures_line = ", ".join([f"{name}={count}" for name, count in top_failures])
            print(f"  top_failures: {top_failures_line}")
        if passes:
            symbols_line = ", ".join([f"{s}:{d}[{q.quality_tier}]" for s, d, q in passes])
            print(f"  passes: {symbols_line}")

        if args.once:
            break

        elapsed = time.time() - started
        sleep_seconds = max(1, int(args.interval_min * 60 - elapsed))
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
