#!/usr/bin/env python3
"""
Analyze "grind into level" quality from ZCT 100 B+ trades PDF.
The grind = last 10 bars before signal. Characteristics:
  - Small uniform candles (low CV of body sizes)
  - No single large candle (low max/median ratio)
  - Steady slope (not accelerating)
  - Bodies getting smaller or staying same (not growing)
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd, numpy as np
from pathlib import Path

DATA_DIR = Path("datasets/binance_futures_1m")

trades = pd.read_csv("zct_momo_results.csv")
mkt = pd.read_csv("datasets/binance_futures_1m/enhanced_market_cache.csv",
                   parse_dates=["timestamp"])
trades["ts_dt"] = pd.to_datetime(trades["ts"])
trades["hour"] = trades["ts_dt"].dt.hour
mkt_ts = mkt["timestamp"].values.astype("datetime64[ns]").astype(np.int64)
trade_ts = trades["ts_dt"].values.astype("datetime64[ns]").astype(np.int64)
idxs = np.searchsorted(mkt_ts, trade_ts, side="right") - 1
idxs = np.clip(idxs, 0, len(mkt) - 1)
for col in ["direction_score", "strategy_score", "activity_mod"]:
    trades[col] = mkt[col].values[idxs]

filled = trades[trades["outcome"].isin(["TP", "SL", "TRAIL_SL", "OPEN"])].copy()
closed = filled[filled["outcome"].isin(["TP", "SL"])].copy()

no_close = ~(((closed["hour"] == 23) & (closed["ts_dt"].dt.minute >= 30)) | (closed["hour"] == 0))
avoid_hrs = ~closed["hour"].isin([8, 21, 22, 23])
short_base = (no_close & avoid_hrs & (closed["side"] == "short") &
              (closed["direction_score"] <= -1) & (closed["strategy_score"] >= 1) &
              (closed["activity_mod"] >= 0))
long_base = (no_close & avoid_hrs & (closed["side"] == "long") &
             (closed["direction_score"] >= 2) & (closed["strategy_score"] >= 2) &
             (closed["activity_mod"] >= 0))

_cache = {}
def load_sym(sym):
    if sym not in _cache:
        p = DATA_DIR / f"{sym}_1m.csv"
        if p.exists():
            _cache[sym] = pd.read_csv(str(p), parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        else:
            _cache[sym] = None
    return _cache[sym]


def compute_grind_metrics(row):
    """Compute grind quality metrics for last 10 bars before signal."""
    df = load_sym(row["symbol"])
    if df is None:
        return None
    ts = pd.Timestamp(row["ts"])
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    diffs = (df["timestamp"] - ts).abs()
    idx = diffs.idxmin()
    if idx < 15:
        return None

    c = df["close"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    o = df["open"].values.astype(float)

    # Last 10 bars before signal
    last10_c = c[idx-9:idx+1]
    last10_h = h[idx-9:idx+1]
    last10_l = l[idx-9:idx+1]
    last10_o = o[idx-9:idx+1]

    if len(last10_c) < 10:
        return None

    # 1. Body sizes (absolute)
    bodies = np.abs(last10_c - last10_o)
    ranges = last10_h - last10_l
    nonzero_bodies = bodies[bodies > 0]
    nonzero_ranges = ranges[ranges > 0]

    # 2. Body uniformity: CV of body sizes (lower = more uniform)
    if len(nonzero_bodies) > 3:
        body_cv = np.std(nonzero_bodies) / np.mean(nonzero_bodies)
    else:
        body_cv = 0

    # 3. Max/median body ratio (lower = no outlier large candle)
    if len(nonzero_bodies) > 3:
        body_max_med = np.max(nonzero_bodies) / np.median(nonzero_bodies)
    else:
        body_max_med = 0

    # 4. Max range as % of price (smaller = calmer approach)
    max_range_pct = np.max(ranges) / np.mean(last10_c) * 100

    # 5. Avg range as % of price
    avg_range_pct = np.mean(ranges) / np.mean(last10_c) * 100

    # 6. Are bodies getting smaller in last 5 vs first 5? (good for grind)
    first5_avg = np.mean(bodies[:5]) if np.mean(bodies[:5]) > 0 else 1e-12
    last5_avg = np.mean(bodies[5:])
    body_decay = last5_avg / first5_avg  # <1 means getting smaller (good)

    # 7. Acceleration: slope of last 5 vs first 5
    first5_move = abs(last10_c[4] - last10_c[0]) / last10_c[0] * 100
    last5_move = abs(last10_c[9] - last10_c[5]) / last10_c[5] * 100
    if first5_move > 0:
        accel_ratio = last5_move / first5_move
    else:
        accel_ratio = 1.0

    # 8. Wick ratio in last 10 (clean bodies = grind, long wicks = choppy)
    wick_ratio = np.mean(np.where(ranges > 1e-12, (ranges - bodies) / ranges, 0))

    # 9. Net efficiency in last 10 (how much net move vs total distance)
    c2c_moves = np.abs(np.diff(last10_c))
    net_move = abs(last10_c[-1] - last10_c[0])
    total_dist = np.sum(c2c_moves)
    efficiency_10 = net_move / total_dist if total_dist > 0 else 0

    # 10. Consecutive direction: how many bars moved in the correct direction
    side = row["side"]
    if side == "long":
        correct_bars = np.sum(np.diff(last10_c) > 0) / 9 * 100
    else:
        correct_bars = np.sum(np.diff(last10_c) < 0) / 9 * 100

    return {
        "body_cv": round(body_cv, 3),
        "body_max_med": round(body_max_med, 2),
        "max_range_pct": round(max_range_pct, 4),
        "avg_range_pct": round(avg_range_pct, 4),
        "body_decay": round(body_decay, 3),
        "accel_ratio": round(accel_ratio, 3),
        "wick_ratio_10": round(wick_ratio, 3),
        "efficiency_10": round(efficiency_10, 3),
        "correct_bars_pct": round(correct_bars, 1),
    }


# Analyze TP and SL trades from both sides
np.random.seed(42)
for label, base in [("SHORTS", short_base), ("LONGS", long_base)]:
    tp_rows = closed[base & (closed["outcome"] == "TP")]
    sl_rows = closed[base & (closed["outcome"] == "SL")]
    tp_sample = tp_rows.sample(n=min(200, len(tp_rows)))
    sl_sample = sl_rows.sample(n=min(200, len(sl_rows)))

    print(f"\nAnalyzing {label} grind quality...")
    tp_m = [m for _, r in tp_sample.iterrows() if (m := compute_grind_metrics(r)) is not None]
    sl_m = [m for _, r in sl_sample.iterrows() if (m := compute_grind_metrics(r)) is not None]
    tp_df = pd.DataFrame(tp_m)
    sl_df = pd.DataFrame(sl_m)
    print(f"  {len(tp_df)} TP, {len(sl_df)} SL")

    print(f"\n{'='*95}")
    print(f"{label} — GRIND QUALITY (last 10 bars): TP vs SL")
    print(f"{'='*95}")

    metrics = list(tp_df.columns)
    print(f"\n{'Metric':<22} | {'TP Mean':>10} {'TP Med':>10} | {'SL Mean':>10} {'SL Med':>10} | {'Delta%':>8}")
    print("-" * 85)

    for m in metrics:
        tv = tp_df[m]; sv = sl_df[m]
        tm, tmed = tv.mean(), tv.median()
        sm, smed = sv.mean(), sv.median()
        delta = abs(tm - sm) / max(abs(sm), 0.001) * 100
        sig = "***" if delta > 15 else ("**" if delta > 8 else ("*" if delta > 5 else ""))
        print(f"  {m:<20} | {tm:>10.4f} {tmed:>10.4f} | {sm:>10.4f} {smed:>10.4f} | {delta:>7.1f}% {sig}")

    # Test thresholds for most promising
    print(f"\n  THRESHOLD SEARCH:")
    for m in metrics:
        tv = tp_df[m]; sv = sl_df[m]
        all_vals = np.concatenate([tv.values, sv.values])
        found = False
        for p in [25, 50, 75]:
            thresh = np.percentile(all_vals, p)
            for op_fn, op_name in [(lambda x, t: x >= t, ">="), (lambda x, t: x <= t, "<=")]:
                tp_pass = op_fn(tv, thresh).mean() * 100
                sl_pass = op_fn(sv, thresh).mean() * 100
                if tp_pass > 25 and sl_pass > 25:
                    lift = tp_pass - sl_pass
                    if abs(lift) > 5:
                        if not found:
                            print(f"\n    {m}:")
                            found = True
                        print(f"      {op_name} {thresh:.4f} (p{p}): TP={tp_pass:.0f}% SL={sl_pass:.0f}% lift={lift:+.1f}pp")
