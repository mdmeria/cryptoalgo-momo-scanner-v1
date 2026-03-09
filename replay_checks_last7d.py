#!/usr/bin/env python3
"""Replay and summarize per-check outcomes on historical rows from scanner CSV.

This script uses only the last N days (default 7) to match chart lookback limits.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from datetime import datetime, timedelta

import pandas as pd
import requests

from momentum_quality import MomentumCheckConfig, evaluate_momentum_setup


def parse_ts(ts: str) -> pd.Timestamp:
    dt = datetime.fromisoformat(ts)
    t = pd.Timestamp(dt)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def fetch_1m_window(symbol: str, end_ts: pd.Timestamp, limit: int = 420) -> pd.DataFrame | None:
    url = "https://data-api.binance.vision/api/v3/klines"
    params = {
        "symbol": symbol.upper(),
        "interval": "1m",
        "limit": limit,
        "endTime": int(end_ts.timestamp() * 1000),
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        raw = r.json()
        if not isinstance(raw, list) or len(raw) < 120:
            return None
        df = pd.DataFrame(
            [
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
        )
        return df.set_index("timestamp")
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay checks on historical CSV rows")
    parser.add_argument("--csv", required=True, help="Path to scanner CSV log")
    parser.add_argument("--days", type=int, default=7, help="Only evaluate last N days (default 7)")
    parser.add_argument("--limit", type=int, default=0, help="Max number of rows to evaluate (0 = all)")
    parser.add_argument(
        "--out-prefix",
        default="replay_last7d",
        help="Prefix for output files (saved near --csv path)",
    )
    parser.add_argument(
        "--samples-per-check",
        type=int,
        default=5,
        help="How many PASS and FAIL examples per check to export",
    )
    parser.add_argument(
        "--disable-grind-subchecks",
        action="store_true",
        help="Disable grind-quality subchecks inside balanced_momo_2h",
    )
    args = parser.parse_args()

    with open(args.csv, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    now_utc = pd.Timestamp.now(tz="UTC")
    cutoff = now_utc - timedelta(days=args.days)
    rows = [r for r in rows if parse_ts(r["run_timestamp_est"]) >= cutoff]
    if args.limit > 0:
        rows = rows[-args.limit :]

    if not rows:
        print("No rows found in requested lookback.")
        return

    cfg = MomentumCheckConfig(
        require_grind_subchecks_in_balanced_2h=not args.disable_grind_subchecks
    )

    check_keys = [
        "slow_grind_approach",
        "left_side_staircase",
        "volume_not_decreasing",
        "not_choppy",
        "balanced_momo_2h",
        "day_change_ok",
        "vwap_side_ok",
        "first_2h_prev_day_vwap_ok",
        "entry_not_crossed_6h",
    ]

    pass_counts = Counter()
    fail_counts = Counter()
    skipped = 0
    total = 0
    review_rows: list[dict] = []

    print(
        f"Evaluating {len(rows)} rows from last {args.days} days | "
        f"grind_subchecks={'OFF' if args.disable_grind_subchecks else 'ON'}"
    )

    for r in rows:
        symbol = r["symbol"]
        direction = r["direction"]
        ts = parse_ts(r["run_timestamp_est"])

        df = fetch_1m_window(symbol, ts, limit=420)
        if df is None:
            skipped += 1
            continue

        res = evaluate_momentum_setup(
            df=df,
            direction=direction,
            symbol=symbol,
            enforce_extended_rules=True,
            eval_time=ts,
            check_config=cfg,
        )

        total += 1
        failed_checks = [k for k in check_keys if not res.checks.get(k, False)]

        row_out = {
            "run_timestamp_est": r["run_timestamp_est"],
            "eval_timestamp_utc": ts.isoformat(),
            "symbol": symbol,
            "tv_symbol": f"BINANCE:{symbol}",
            "direction": direction,
            "overall_pass": bool(res.passed),
            "quality_tier": res.quality_tier,
            "failed_checks": "|".join(failed_checks),
            "day_change_pct": float(res.metrics.get("day_change_pct", float("nan"))),
            "dir_move_2h_pct": float(res.metrics.get("dir_move_2h_pct", float("nan"))),
            "efficiency_2h": float(res.metrics.get("efficiency_2h", float("nan"))),
            "pre_entry_move_30m_pct": float(res.metrics.get("pre_entry_move_30m_pct", float("nan"))),
            "pre_entry_efficiency_30m": float(res.metrics.get("pre_entry_efficiency_30m", float("nan"))),
            "pre_entry_dir_bar_ratio_30m": float(res.metrics.get("pre_entry_dir_bar_ratio_30m", float("nan"))),
            "entry_cross_count_6h": float(res.metrics.get("entry_cross_count_6h", float("nan"))),
        }

        for k in check_keys:
            row_out[k] = bool(res.checks.get(k, False))

        review_rows.append(row_out)

        for k in check_keys:
            if res.checks.get(k, False):
                pass_counts[k] += 1
            else:
                fail_counts[k] += 1

    if total == 0:
        print(f"No evaluable rows. Skipped={skipped}")
        return

    print("\nPer-check pass/fail summary")
    for k in check_keys:
        p = pass_counts[k]
        f = fail_counts[k]
        rate = (p / total) * 100.0
        print(f"{k:28} pass={p:4d} fail={f:4d} pass_rate={rate:6.2f}%")

    print(f"\nRows evaluated={total}, skipped={skipped}")

    # Save full review sheet and per-check sample sheet for manual TradingView validation.
    out_dir = os.path.dirname(os.path.abspath(args.csv))
    suffix = "nogrind" if args.disable_grind_subchecks else "grind"
    review_path = os.path.join(out_dir, f"{args.out_prefix}_{suffix}_full.csv")
    samples_path = os.path.join(out_dir, f"{args.out_prefix}_{suffix}_samples.csv")

    review_fields = [
        "run_timestamp_est",
        "eval_timestamp_utc",
        "symbol",
        "tv_symbol",
        "direction",
        "overall_pass",
        "quality_tier",
        "failed_checks",
        "day_change_pct",
        "dir_move_2h_pct",
        "efficiency_2h",
        "pre_entry_move_30m_pct",
        "pre_entry_efficiency_30m",
        "pre_entry_dir_bar_ratio_30m",
        "entry_cross_count_6h",
    ] + check_keys

    with open(review_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=review_fields)
        w.writeheader()
        w.writerows(review_rows)

    # Build per-check review samples (PASS and FAIL examples for quick chart validation).
    sample_rows: list[dict] = []
    n = max(1, int(args.samples_per_check))
    for k in check_keys:
        fails = [r for r in review_rows if not r.get(k, False)][:n]
        passes = [r for r in review_rows if r.get(k, False)][:n]

        for r in fails:
            sample_rows.append(
                {
                    "check_name": k,
                    "check_result": "FAIL",
                    "run_timestamp_est": r["run_timestamp_est"],
                    "symbol": r["symbol"],
                    "tv_symbol": r["tv_symbol"],
                    "direction": r["direction"],
                    "overall_pass": r["overall_pass"],
                    "failed_checks": r["failed_checks"],
                }
            )

        for r in passes:
            sample_rows.append(
                {
                    "check_name": k,
                    "check_result": "PASS",
                    "run_timestamp_est": r["run_timestamp_est"],
                    "symbol": r["symbol"],
                    "tv_symbol": r["tv_symbol"],
                    "direction": r["direction"],
                    "overall_pass": r["overall_pass"],
                    "failed_checks": r["failed_checks"],
                }
            )

    with open(samples_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "check_name",
                "check_result",
                "run_timestamp_est",
                "symbol",
                "tv_symbol",
                "direction",
                "overall_pass",
                "failed_checks",
            ],
        )
        w.writeheader()
        w.writerows(sample_rows)

    print(f"Saved full review CSV: {review_path}")
    print(f"Saved sample review CSV: {samples_path}")


if __name__ == "__main__":
    main()
