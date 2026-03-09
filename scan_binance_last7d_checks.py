#!/usr/bin/env python3
"""Generate per-check historical samples from Binance symbols over last 7 days.

Outputs:
- Full row-level review CSV (chart-friendly)
- Per-check sample CSV (PASS/FAIL examples for manual TradingView validation)
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from typing import Any

import pandas as pd
import requests
import numpy as np

from momentum_quality import MomentumCheckConfig, evaluate_momentum_setup

BINANCE_API = "https://data-api.binance.vision/api/v3"


def fetch_usdt_symbols() -> list[str]:
    r = requests.get(f"{BINANCE_API}/exchangeInfo", timeout=20)
    r.raise_for_status()
    data = r.json()
    symbols = []
    for s in data.get("symbols", []):
        if (
            s.get("status") == "TRADING"
            and s.get("quoteAsset") == "USDT"
            and s.get("isSpotTradingAllowed", False)
        ):
            symbols.append(s["symbol"])
    return sorted(set(symbols))


def fetch_1m_range(symbol: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame | None:
    rows: list[dict[str, Any]] = []
    end_ms = int(end_ts.timestamp() * 1000)
    start_ms = int(start_ts.timestamp() * 1000)

    # Paginate backwards with Binance max limit=1000.
    while True:
        params = {
            "symbol": symbol,
            "interval": "1m",
            "limit": 1000,
            "endTime": end_ms,
        }
        r = requests.get(f"{BINANCE_API}/klines", params=params, timeout=20)
        if r.status_code != 200:
            return None
        raw = r.json()
        if not isinstance(raw, list) or len(raw) == 0:
            break

        batch = [
            {
                "timestamp": pd.to_datetime(k[0], unit="ms", utc=True),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            }
            for k in raw
        ]
        rows.extend(batch)

        oldest_ms = int(raw[0][0])
        if oldest_ms <= start_ms:
            break

        end_ms = oldest_ms - 1

        # Safety stop: enough for >7 days + headroom.
        if len(rows) > 13000:
            break

    if not rows:
        return None

    df = pd.DataFrame(rows).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]
    if len(df) < 500:
        return None
    return df.set_index("timestamp")


def check_day_change_ok(df: pd.DataFrame, idx: int, direction: str) -> tuple[bool, float]:
    if idx < 1440:
        return False, float("nan")
    now_close = float(df["close"].iloc[idx])
    prev_close = float(df["close"].iloc[idx - 1440])
    day_change = ((now_close - prev_close) / max(abs(prev_close), 1e-9)) * 100.0
    if direction == "long":
        return day_change >= 5.0, day_change
    return day_change <= -5.0, day_change


def session_vwap(df: pd.DataFrame) -> float:
    if df.empty:
        return float("nan")
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"]
    denom = float(vol.sum())
    if denom <= 0:
        return float("nan")
    return float((tp * vol).sum() / denom)


def check_vwap_side(df: pd.DataFrame, ts: pd.Timestamp, direction: str) -> tuple[bool, bool]:
    close_now = float(df.loc[:ts]["close"].iloc[-1])
    day = ts.date()
    today_df = df[(df.index.date == day) & (df.index <= ts)]
    today_vwap = session_vwap(today_df)

    if direction == "long":
        vwap_ok = pd.notna(today_vwap) and close_now > today_vwap
    else:
        vwap_ok = pd.notna(today_vwap) and close_now < today_vwap

    if ts.hour < 2:
        prev_day = (ts - pd.Timedelta(days=1)).date()
        prev_df = df[df.index.date == prev_day]
        prev_vwap = session_vwap(prev_df)
        day_change_ok, _ = check_day_change_ok(df, df.index.get_loc(ts), direction)
        if direction == "long":
            first2h_ok = pd.notna(prev_vwap) and close_now > prev_vwap and day_change_ok
        else:
            first2h_ok = pd.notna(prev_vwap) and close_now < prev_vwap and day_change_ok
    else:
        first2h_ok = True

    return bool(vwap_ok), bool(first2h_ok)


def check_entry_not_crossed_6h(df: pd.DataFrame, idx: int, direction: str) -> tuple[bool, int]:
    if idx < 370:
        return False, -1
    win = df.iloc[: idx + 1]
    if direction == "long":
        entry_price = float(win["high"].iloc[-10:].max())
    else:
        entry_price = float(win["low"].iloc[-10:].min())

    prior = win.iloc[-370:-10]
    crossed = (prior["low"] <= entry_price) & (prior["high"] >= entry_price)
    n_cross = int(crossed.sum())
    return n_cross == 0, n_cross


def check_volume_threshold(df: pd.DataFrame, ts: pd.Timestamp, min_avg_volume: float = 50000.0) -> tuple[bool, float]:
    """Check if average 1m volume over last 5 minutes meets threshold."""
    try:
        # Get last 5 bars (5 minutes) up to and including the target timestamp
        last_5 = df.loc[:ts].tail(5)
        if len(last_5) < 5:
            return False, 0.0
        avg_vol = float(last_5["volume"].mean())
        return avg_vol >= min_avg_volume, avg_vol
    except:
        return False, 0.0


def evaluate_symbol_at_all_timestamps(
    symbol: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    eval_timestamps: list[pd.Timestamp],
    min_avg_volume: float = 50000.0,
) -> list[dict[str, Any]]:
    """Fetch symbol data once, then evaluate at all specified timestamps."""
    # Fetch all data for the symbol once
    df = fetch_1m_range(symbol, start_ts, end_ts + pd.Timedelta(minutes=1))
    if df is None or len(df) < 500:
        return []

    results: list[dict[str, Any]] = []

    for eval_ts in eval_timestamps:
        # Check if we have data up to this timestamp
        if eval_ts not in df.index:
            # Find closest timestamp before eval_ts
            prior = df.loc[:eval_ts]
            if prior.empty:
                continue
            eval_ts = prior.index[-1]

        # Check volume threshold
        vol_ok, avg_vol = check_volume_threshold(df, eval_ts, min_avg_volume)
        if not vol_ok:
            continue

        # Get the index position for the evaluation timestamp
        try:
            idx = df.index.get_loc(eval_ts)
        except:
            continue

        if idx < 370:  # Need at least 370 bars for all checks
            continue

        window = df.iloc[max(0, idx - 500) : idx + 1].copy()

        for direction in ("long", "short"):
            res_on = evaluate_momentum_setup(
                df=window,
                direction=direction,
                symbol=symbol,
                enforce_extended_rules=False,
                eval_time=eval_ts,
                check_config=MomentumCheckConfig(require_grind_subchecks_in_balanced_2h=True),
            )
            res_off = evaluate_momentum_setup(
                df=window,
                direction=direction,
                symbol=symbol,
                enforce_extended_rules=False,
                eval_time=eval_ts,
                check_config=MomentumCheckConfig(require_grind_subchecks_in_balanced_2h=False),
            )

            day_change_ok, day_change_pct = check_day_change_ok(df, idx, direction)
            vwap_side_ok, first2h_ok = check_vwap_side(df, eval_ts, direction)
            entry_ok, entry_cross = check_entry_not_crossed_6h(df, idx, direction)

            row = {
                "timestamp_utc": eval_ts.isoformat(),
                "timestamp_est": eval_ts.tz_convert("America/New_York").isoformat(),
                "symbol": symbol,
                "tv_symbol": f"BINANCE:{symbol}",
                "direction": direction,
                "avg_volume_5m": float(avg_vol),
                "slow_grind_approach": bool(res_on.checks.get("slow_grind_approach", False)),
                "left_side_staircase": bool(res_on.checks.get("left_side_staircase", False)),
                "volume_not_decreasing": bool(res_on.checks.get("volume_not_decreasing", False)),
                "not_choppy": bool(res_on.checks.get("not_choppy", False)),
                "balanced_momo_2h_grind_on": bool(res_on.checks.get("balanced_momo_2h", False)),
                "balanced_momo_2h_grind_off": bool(res_off.checks.get("balanced_momo_2h", False)),
                "day_change_ok": bool(day_change_ok),
                "vwap_side_ok": bool(vwap_side_ok),
                "first_2h_prev_day_vwap_ok": bool(first2h_ok),
                "entry_not_crossed_6h": bool(entry_ok),
                "day_change_pct": float(day_change_pct),
                "entry_cross_count_6h": int(entry_cross),
                "dir_move_2h_pct": float(res_on.metrics.get("dir_move_2h_pct", float("nan"))),
                "efficiency_2h": float(res_on.metrics.get("efficiency_2h", float("nan"))),
                "smma30_crosses_2h": float(res_on.metrics.get("smma30_crosses_2h", float("nan"))),
                "smma30_direction_2h": float(res_on.metrics.get("smma30_direction_2h", float("nan"))),
                "noise_class_momentum": float(res_on.metrics.get("noise_class_momentum", float("nan"))),
                "pre_entry_move_30m_pct": float(res_on.metrics.get("pre_entry_move_30m_pct", float("nan"))),
                "pre_entry_efficiency_30m": float(res_on.metrics.get("pre_entry_efficiency_30m", float("nan"))),
                "pre_entry_move_10m_pct": float(res_on.metrics.get("pre_entry_move_10m_pct", float("nan"))),
                "pre_entry_efficiency_10m": float(res_on.metrics.get("pre_entry_efficiency_10m", float("nan"))),
                "pre_entry_dir_bar_ratio_10m": float(res_on.metrics.get("pre_entry_dir_bar_ratio_10m", float("nan"))),
                "pre_entry_opp_candles_10m": float(res_on.metrics.get("pre_entry_opp_candles_10m", float("nan"))),
                "pre_entry_grind_10m_ok": bool(res_on.metrics.get("pre_entry_grind_10m_ok", 0.0) > 0.5),
            }
            # NOTE: balanced_momo_2h_grind_on removed from gating checks
            # Still calculated and exported for diagnostic purposes
            row["overall_pass_grind_on"] = all(
                [
                    row["slow_grind_approach"],
                    row["left_side_staircase"],
                    row["volume_not_decreasing"],
                    row["not_choppy"],
                    # row["balanced_momo_2h_grind_on"],  # removed - diagnostic only
                    row["day_change_ok"],
                    row["vwap_side_ok"],
                    row["first_2h_prev_day_vwap_ok"],
                    row["entry_not_crossed_6h"],
                ]
            )
            row["overall_pass_grind_off"] = all(
                [
                    row["slow_grind_approach"],
                    row["left_side_staircase"],
                    row["volume_not_decreasing"],
                    row["not_choppy"],
                    row["balanced_momo_2h_grind_off"],
                    row["day_change_ok"],
                    row["vwap_side_ok"],
                    row["first_2h_prev_day_vwap_ok"],
                    row["entry_not_crossed_6h"],
                ]
            )

            results.append(row)

    return results


def write_outputs(rows: list[dict[str, Any]], out_prefix: str, samples_per_check: int) -> tuple[str, str]:
    full_path = f"{out_prefix}_full.csv"
    samples_path = f"{out_prefix}_samples.csv"

    if not rows:
        with open(full_path, "w", newline="") as f:
            f.write("")
        with open(samples_path, "w", newline="") as f:
            f.write("")
        return full_path, samples_path

    fields = list(rows[0].keys())
    with open(full_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    check_cols = [
        "slow_grind_approach",
        "left_side_staircase",
        "volume_not_decreasing",
        "not_choppy",
        "balanced_momo_2h_grind_on",
        "balanced_momo_2h_grind_off",
        "day_change_ok",
        "vwap_side_ok",
        "first_2h_prev_day_vwap_ok",
        "entry_not_crossed_6h",
    ]

    sample_rows: list[dict[str, Any]] = []
    n = max(1, samples_per_check)
    for c in check_cols:
        fail_rows = [r for r in rows if not bool(r.get(c, False))][:n]
        pass_rows = [r for r in rows if bool(r.get(c, False))][:n]

        for r in fail_rows:
            sample_rows.append(
                {
                    "check_name": c,
                    "check_result": "FAIL",
                    "timestamp_est": r["timestamp_est"],
                    "symbol": r["symbol"],
                    "tv_symbol": r["tv_symbol"],
                    "direction": r["direction"],
                    "overall_pass_grind_on": r["overall_pass_grind_on"],
                    "overall_pass_grind_off": r["overall_pass_grind_off"],
                }
            )

        for r in pass_rows:
            sample_rows.append(
                {
                    "check_name": c,
                    "check_result": "PASS",
                    "timestamp_est": r["timestamp_est"],
                    "symbol": r["symbol"],
                    "tv_symbol": r["tv_symbol"],
                    "direction": r["direction"],
                    "overall_pass_grind_on": r["overall_pass_grind_on"],
                    "overall_pass_grind_off": r["overall_pass_grind_off"],
                }
            )

    with open(samples_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "check_name",
                "check_result",
                "timestamp_est",
                "symbol",
                "tv_symbol",
                "direction",
                "overall_pass_grind_on",
                "overall_pass_grind_off",
            ],
        )
        w.writeheader()
        w.writerows(sample_rows)

    return full_path, samples_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan across all Binance USDT symbols over N days")
    parser.add_argument("--days", type=int, default=7, help="Lookback days (default 7)")
    parser.add_argument("--interval-hours", type=float, default=1.0, help="Scan interval in hours (default 1, supports decimals like 0.5)")
    parser.add_argument("--min-volume", type=float, default=50000.0, help="Min avg 1m volume over last 5 minutes")
    parser.add_argument("--samples-per-check", type=int, default=8, help="PASS/FAIL examples exported per check")
    parser.add_argument("--workers", type=int, default=8, help="Parallel symbol workers per timestamp")
    parser.add_argument("--max-symbols", type=int, default=0, help="Optional symbol cap for quick tests (0=all)")
    parser.add_argument(
        "--out-prefix",
        default="C:/Projects/CryptoAlgo/binance_hourly_scan",
        help="Output file prefix path",
    )
    args = parser.parse_args()

    end_ts = pd.Timestamp.now(tz="UTC").floor("1h")  # Round to hour
    start_ts = end_ts - timedelta(days=args.days)
    
    # Convert interval hours to minutes for flexibility and generate timestamps
    interval_minutes = int(args.interval_hours * 60)
    timestamps = pd.date_range(start=start_ts, end=end_ts, freq=f"{interval_minutes}min", tz="UTC").tolist()
    
    symbols = fetch_usdt_symbols()
    if args.max_symbols > 0:
        symbols = symbols[: args.max_symbols]

    print(f"Symbols: {len(symbols)} | Timestamps: {len(timestamps)} ({args.interval_hours}h interval, {interval_minutes}m) | Min volume: {args.min_volume:,.0f}/1m")
    print(f"Date range: {start_ts.isoformat()} to {end_ts.isoformat()}")

    all_rows: list[dict[str, Any]] = []
    
    # Process symbols in parallel, each evaluating all timestamps
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(evaluate_symbol_at_all_timestamps, s, start_ts, end_ts, timestamps, args.min_volume): s
            for s in symbols
        }
        for i, fut in enumerate(as_completed(futures), start=1):
            symbol = futures[fut]
            try:
                rows = fut.result()
                all_rows.extend(rows)
            except Exception as e:
                pass
            if i % 25 == 0 or i == len(symbols):
                print(f"Processed {i}/{len(symbols)} symbols | total rows={len(all_rows)}")

    full_path, samples_path = write_outputs(all_rows, args.out_prefix, args.samples_per_check)
    print(f"Done. Total rows={len(all_rows)}")
    print(f"Full output: {full_path}")
    print(f"Samples output: {samples_path}")


if __name__ == "__main__":
    main()
