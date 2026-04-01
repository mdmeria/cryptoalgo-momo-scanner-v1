#!/usr/bin/env python3
"""
Measure proposed quality gate metrics on the exact PDF example trades.
Goal: find empirical thresholds from known-good setups.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("datasets/binance_futures_1m")

# PDF trades: (symbol, date, entry_time_utc, direction, description)
PDF_TRADES = [
    ("HIGHUSDT",  "2024-04-30", "06:09", "long",  "Trade 1: Long HIGH"),
    ("ETHFIUSDT", "2024-07-23", "08:37", "long",  "Trade 2: Long ETHFI"),
    # BNX not available
    ("CRVUSDT",   "2024-06-13", "03:01", "short", "Trade 4: Short CRV"),
    ("ORDIUSDT",  "2024-05-17", "16:10", "long",  "Trade 5: Long ORDI"),
    ("WIFUSDT",   "2024-04-19", "01:12", "short", "Trade 7: Short WIF"),
    # Trade 8: unknown ticker
    # Trade 9: Short ETHFI — different date
    ("ETHFIUSDT", "2024-07-25", "01:26", "short", "Trade 9: Short ETHFI (UTC+8->UTC)"),
    # Trade 10: ZK not available
    ("WIFUSDT",   "2024-07-19", "02:06", "short", "Trade 11: Short WIF (UTC-6->UTC)"),
]


def load_around(symbol, date_str, time_str, hours_before=5, hours_after=2):
    fpath = DATA_DIR / f"{symbol}_1m.csv"
    if not fpath.exists():
        return None, None
    df = pd.read_csv(str(fpath), parse_dates=["timestamp"])
    entry_dt = pd.Timestamp(f"{date_str} {time_str}:00", tz="UTC")
    start = entry_dt - pd.Timedelta(hours=hours_before)
    end = entry_dt + pd.Timedelta(hours=hours_after)
    mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
    df_slice = df[mask].copy().reset_index(drop=True)
    if len(df_slice) < 60:
        return None, None
    diffs = (df_slice["timestamp"] - entry_dt).abs()
    entry_idx = diffs.idxmin()
    return df_slice, entry_idx


def smma(arr, period):
    result = np.zeros(len(arr))
    result[0] = arr[0]
    alpha = 1.0 / period
    for i in range(1, len(arr)):
        result[i] = result[i-1] * (1 - alpha) + arr[i] * alpha
    return result


def compute_metrics(df, entry_idx, direction, lookback_bars=120):
    """Compute all proposed quality metrics for the staircase window."""
    start_idx = max(0, entry_idx - lookback_bars)
    sl = slice(start_idx, entry_idx + 1)

    c = df["close"].values[sl].astype(float)
    h = df["high"].values[sl].astype(float)
    l = df["low"].values[sl].astype(float)
    o = df["open"].values[sl].astype(float)
    v = df["volume"].values[sl].astype(float)
    n = len(c)

    if n < 30:
        return None

    results = {"n_bars": n}

    # ── 1. STAIRCASE R² (linear regression of closes) ──
    x = np.arange(n)
    slope, intercept = np.polyfit(x, c, 1)
    predicted = slope * x + intercept
    ss_res = np.sum((c - predicted) ** 2)
    ss_tot = np.sum((c - np.mean(c)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    results["r2"] = round(r2, 4)
    results["slope_pct_per_bar"] = round(slope / np.mean(c) * 100, 5)

    # Check direction consistency
    if direction == "long":
        results["slope_correct"] = slope > 0
    else:
        results["slope_correct"] = slope < 0

    # ── 2. CANDLE UNIFORMITY (max body / median body) ──
    bodies = np.abs(c - o)
    # Filter out doji (near-zero bodies)
    nonzero_bodies = bodies[bodies > 0]
    if len(nonzero_bodies) > 5:
        median_body = np.median(nonzero_bodies)
        max_body = np.max(nonzero_bodies)
        results["body_max_median_ratio"] = round(max_body / median_body, 2) if median_body > 0 else 999
        results["body_cv"] = round(np.std(nonzero_bodies) / np.mean(nonzero_bodies), 3) if np.mean(nonzero_bodies) > 0 else 0
    else:
        results["body_max_median_ratio"] = 0
        results["body_cv"] = 0

    # Also: max close-to-close move / median close-to-close move
    c2c = np.abs(np.diff(c))
    if len(c2c) > 5:
        results["c2c_max_median"] = round(np.max(c2c) / np.median(c2c), 2) if np.median(c2c) > 0 else 999
    else:
        results["c2c_max_median"] = 0

    # ── 3. MAX PULLBACK DEPTH ──
    # For longs: max drawdown from any running high
    # For shorts: max rally from any running low
    total_range = abs(c[-1] - c[0])

    if direction == "long":
        running_high = np.maximum.accumulate(c)
        drawdowns = (running_high - c) / running_high * 100
        max_pullback_pct = np.max(drawdowns)
        # As fraction of total move
        if total_range > 0:
            max_pullback_frac = np.max(running_high - c) / total_range
        else:
            max_pullback_frac = 0
    else:
        running_low = np.minimum.accumulate(c)
        rallies = (c - running_low) / running_low * 100
        max_pullback_pct = np.max(rallies)
        if total_range > 0:
            max_pullback_frac = np.max(c - running_low) / total_range
        else:
            max_pullback_frac = 0

    results["max_pullback_pct"] = round(max_pullback_pct, 3)
    results["max_pullback_frac"] = round(max_pullback_frac, 3)
    results["total_move_pct"] = round(total_range / c[0] * 100, 3) if c[0] > 0 else 0

    # ── 4. HIGHER LOWS / LOWER HIGHS CONSISTENCY ──
    # Rolling 5-bar windows: what fraction make HL (long) or LH (short)
    window = 5
    if n > window * 3:
        rolling_lows = np.array([np.min(l[i:i+window]) for i in range(0, n - window + 1)])
        rolling_highs = np.array([np.max(h[i:i+window]) for i in range(0, n - window + 1)])

        if direction == "long":
            hl_count = np.sum(np.diff(rolling_lows) > 0)
            results["hl_pct"] = round(hl_count / (len(rolling_lows) - 1) * 100, 1) if len(rolling_lows) > 1 else 0
        else:
            lh_count = np.sum(np.diff(rolling_highs) < 0)
            results["lh_pct"] = round(lh_count / (len(rolling_highs) - 1) * 100, 1) if len(rolling_highs) > 1 else 0

    # Also: bar-by-bar higher lows / lower highs
    if direction == "long":
        bar_hl = np.sum(np.diff(l) > 0) / (n - 1) * 100
        results["bar_hl_pct"] = round(bar_hl, 1)
    else:
        bar_lh = np.sum(np.diff(h) < 0) / (n - 1) * 100
        results["bar_lh_pct"] = round(bar_lh, 1)

    # ── 5. WICK RATIO ──
    rng = h - l
    body_abs = np.abs(c - o)
    wick_ratio = np.where(rng > 1e-12, (rng - body_abs) / rng, 0)
    results["avg_wick"] = round(np.mean(wick_ratio), 3)
    results["median_wick"] = round(np.median(wick_ratio), 3)

    # ── 6. SMMA30 METRICS ──
    sm30 = smma(c, 30)
    # Crosses
    above = c > sm30
    cross_count = 0
    last_side = above[0]
    run = 0
    for i in range(1, n):
        if above[i] != last_side:
            run += 1
            if run >= 5:
                cross_count += 1
                last_side = above[i]
                run = 0
        else:
            run = 0
    results["smma30_crosses"] = cross_count

    # SMMA30 slope
    sm30_change = abs(sm30[-1] - sm30[10]) / sm30[10] * 100 if sm30[10] > 0 else 0
    results["smma30_change_pct"] = round(sm30_change, 3)

    # % of bars on correct side of SMMA30
    if direction == "long":
        correct_side = np.sum(c > sm30) / n * 100
    else:
        correct_side = np.sum(c < sm30) / n * 100
    results["pct_correct_side_smma"] = round(correct_side, 1)

    # ── 7. VOLUME METRICS ──
    vol_usd = v * c
    # nama-style EMA of volume
    vol_ema60 = pd.Series(vol_usd).ewm(span=60, adjust=False).mean().values
    if len(vol_ema60) > 30:
        vol_start = np.mean(vol_ema60[20:30])
        vol_end = np.mean(vol_ema60[-10:])
        results["vol_ratio"] = round(vol_end / vol_start, 3) if vol_start > 0 else 0

    # ── 8. SPIKE RATIO (current gate metric) ──
    last10 = c[-10:]
    bar_moves = np.abs(np.diff(last10))
    if len(bar_moves) > 0 and np.mean(bar_moves) > 0:
        results["spike_ratio"] = round(np.max(bar_moves) / np.mean(bar_moves), 2)
        results["max_bar_pct"] = round(np.max(bar_moves) / np.mean(last10) * 100, 4)

    # ── 9. EFFICIENCY (net move / total distance) ──
    all_moves = np.abs(np.diff(c))
    net = abs(c[-1] - c[0])
    total_dist = np.sum(all_moves)
    results["efficiency"] = round(net / total_dist, 3) if total_dist > 0 else 0

    return results


# ── MAIN ──
print("=" * 120)
print("QUALITY METRICS FOR PDF EXAMPLE TRADES (known-good setups)")
print("=" * 120)

all_results = []

for sym, date_str, time_str, direction, desc in PDF_TRADES:
    print(f"\n{'─' * 100}")
    print(f"  {desc} | {sym} {date_str} {time_str} UTC | {direction}")
    print(f"{'─' * 100}")

    df, entry_idx = load_around(sym, date_str, time_str)
    if df is None:
        print("  !! NO DATA")
        continue

    print(f"  Data: {len(df)} bars, entry at index {entry_idx} ({df['timestamp'].iloc[entry_idx]})")

    metrics = compute_metrics(df, entry_idx, direction)
    if metrics is None:
        print("  !! Not enough data")
        continue

    all_results.append({"trade": desc, "direction": direction, **metrics})

    # Print in groups
    print(f"\n  STAIRCASE QUALITY:")
    print(f"    R²:                    {metrics['r2']:.4f}  (>0.70 = clean staircase)")
    print(f"    Slope correct:         {metrics['slope_correct']}")
    print(f"    Slope %/bar:           {metrics['slope_pct_per_bar']:.5f}%")
    print(f"    Efficiency (net/total): {metrics['efficiency']:.3f}")

    print(f"\n  CANDLE UNIFORMITY:")
    print(f"    Body max/median:       {metrics['body_max_median_ratio']:.2f}x  (<2.5x = uniform)")
    print(f"    Body CV:               {metrics['body_cv']:.3f}")
    print(f"    C2C max/median:        {metrics['c2c_max_median']:.2f}x")

    print(f"\n  PULLBACK DEPTH:")
    print(f"    Max pullback:          {metrics['max_pullback_pct']:.3f}%")
    print(f"    Pullback/move fraction: {metrics['max_pullback_frac']:.3f}  (<0.40 = shallow)")
    print(f"    Total move:            {metrics['total_move_pct']:.3f}%")

    print(f"\n  DIRECTIONAL STRUCTURE:")
    hl_key = "hl_pct" if direction == "long" else "lh_pct"
    bar_key = "bar_hl_pct" if direction == "long" else "bar_lh_pct"
    print(f"    Rolling 5-bar HL/LH:   {metrics.get(hl_key, 'N/A')}%  (>50% = consistent)")
    print(f"    Bar-by-bar HL/LH:      {metrics.get(bar_key, 'N/A')}%")

    print(f"\n  NOISE / WICKS:")
    print(f"    Avg wick ratio:        {metrics['avg_wick']:.3f}  (<0.50 = clean)")
    print(f"    Median wick ratio:     {metrics['median_wick']:.3f}")
    print(f"    SMMA30 crosses:        {metrics['smma30_crosses']}")
    print(f"    SMMA30 change:         {metrics['smma30_change_pct']:.3f}%")
    print(f"    % correct side SMMA:   {metrics['pct_correct_side_smma']:.1f}%")

    print(f"\n  VOLUME:")
    print(f"    Vol EMA60 ratio:       {metrics.get('vol_ratio', 'N/A')}")

    print(f"\n  SPIKE (current gate):")
    print(f"    Spike ratio (last 10): {metrics.get('spike_ratio', 'N/A')}")
    print(f"    Max bar %:             {metrics.get('max_bar_pct', 'N/A')}%")


# ── SUMMARY TABLE ──
print(f"\n\n{'=' * 120}")
print("SUMMARY — THRESHOLD CALIBRATION")
print(f"{'=' * 120}")

if all_results:
    rdf = pd.DataFrame(all_results)

    key_metrics = ["r2", "body_max_median_ratio", "c2c_max_median", "body_cv",
                   "max_pullback_pct", "max_pullback_frac", "efficiency",
                   "avg_wick", "smma30_crosses", "smma30_change_pct",
                   "pct_correct_side_smma", "spike_ratio"]

    print(f"\n{'Metric':<28} {'Min':>8} {'Max':>8} {'Mean':>8} {'Median':>8} | Suggested Threshold")
    print("─" * 90)

    for m in key_metrics:
        if m in rdf.columns:
            vals = rdf[m].dropna()
            if len(vals) == 0:
                continue
            mn, mx, avg, med = vals.min(), vals.max(), vals.mean(), vals.median()

            # Suggest threshold based on worst example (all are "good" trades)
            if m == "r2":
                thresh = f"> {mn - 0.05:.2f}  (worst example minus margin)"
            elif m == "body_max_median_ratio":
                thresh = f"< {mx + 1:.1f}  (worst example plus margin)"
            elif m == "c2c_max_median":
                thresh = f"< {mx + 1:.1f}"
            elif m == "max_pullback_frac":
                thresh = f"< {mx + 0.10:.2f}"
            elif m == "efficiency":
                thresh = f"> {mn - 0.03:.2f}"
            elif m == "avg_wick":
                thresh = f"< {mx + 0.02:.2f}"
            elif m == "spike_ratio":
                thresh = f"< {mx + 0.5:.1f}"
            else:
                thresh = ""

            print(f"  {m:<26} {mn:>8.3f} {mx:>8.3f} {avg:>8.3f} {med:>8.3f} | {thresh}")

    # HL/LH separately (different column names for long/short)
    for key in ["hl_pct", "lh_pct", "bar_hl_pct", "bar_lh_pct"]:
        if key in rdf.columns:
            vals = rdf[key].dropna()
            if len(vals) > 0:
                print(f"  {key:<26} {vals.min():>8.1f} {vals.max():>8.1f} {vals.mean():>8.1f} {vals.median():>8.1f} |")
