#!/usr/bin/env python3
"""
Compare proposed quality metrics between:
  - PDF example trades (known good)
  - Backtest TP trades (algo-detected winners)
  - Backtest SL trades (algo-detected losers)

Goal: find metrics that SEPARATE good from bad.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("datasets/binance_futures_1m")
RESULTS_CSV = "zct_momo_results.csv"

_DF_CACHE = {}

def load_sym_df(sym):
    if sym not in _DF_CACHE:
        fpath = DATA_DIR / f"{sym}_1m.csv"
        if not fpath.exists():
            _DF_CACHE[sym] = None
        else:
            df = pd.read_csv(str(fpath), parse_dates=["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)
            _DF_CACHE[sym] = df
    return _DF_CACHE[sym]


def smma(arr, period):
    result = np.zeros(len(arr))
    result[0] = arr[0]
    alpha = 1.0 / period
    for i in range(1, len(arr)):
        result[i] = result[i-1] * (1 - alpha) + arr[i] * alpha
    return result


def compute_metrics(c, h, l, o, v, direction, lookback=120):
    """Compute quality metrics on arrays ending at entry bar."""
    n = min(len(c), lookback)
    if n < 40:
        return None
    c = c[-n:]
    h = h[-n:]
    l = l[-n:]
    o = o[-n:]
    v = v[-n:]

    results = {"n_bars": n}

    # 1. R-squared (staircase linearity)
    x = np.arange(n)
    slope, intercept = np.polyfit(x, c, 1)
    predicted = slope * x + intercept
    ss_res = np.sum((c - predicted) ** 2)
    ss_tot = np.sum((c - np.mean(c)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    results["r2"] = round(r2, 4)

    # 2. Efficiency
    net = abs(c[-1] - c[0])
    total_dist = np.sum(np.abs(np.diff(c)))
    results["efficiency"] = round(net / total_dist, 4) if total_dist > 0 else 0

    # 3. Max pullback %
    if direction == "long":
        running_high = np.maximum.accumulate(c)
        drawdowns = (running_high - c) / running_high * 100
        results["max_pullback_pct"] = round(np.max(drawdowns), 3)
    else:
        running_low = np.minimum.accumulate(c)
        rallies = (c - running_low) / running_low * 100
        results["max_pullback_pct"] = round(np.max(rallies), 3)

    # 4. Body max/median (excluding zero-body bars)
    bodies = np.abs(c - o)
    nonzero = bodies[bodies > 0]
    if len(nonzero) > 10:
        results["body_max_med"] = round(np.max(nonzero) / np.median(nonzero), 2)
    else:
        results["body_max_med"] = 0

    # 5. C2C max/median
    c2c = np.abs(np.diff(c))
    nonzero_c2c = c2c[c2c > 0]
    if len(nonzero_c2c) > 10:
        results["c2c_max_med"] = round(np.max(nonzero_c2c) / np.median(nonzero_c2c), 2)
    else:
        results["c2c_max_med"] = 0

    # 6. Avg wick
    rng = h - l
    body_abs = np.abs(c - o)
    wick_ratio = np.where(rng > 1e-12, (rng - body_abs) / rng, 0)
    results["avg_wick"] = round(np.mean(wick_ratio), 3)

    # 7. SMMA30 metrics
    sm30 = smma(c, 30)
    if direction == "long":
        pct_above = np.sum(c > sm30) / n * 100
    else:
        pct_above = np.sum(c < sm30) / n * 100
    results["pct_correct_side"] = round(pct_above, 1)

    # 8. HL/LH bar-by-bar
    if direction == "long":
        bar_hl = np.sum(np.diff(l) > 0) / (n - 1) * 100
        results["bar_hl_lh_pct"] = round(bar_hl, 1)
    else:
        bar_lh = np.sum(np.diff(h) < 0) / (n - 1) * 100
        results["bar_hl_lh_pct"] = round(bar_lh, 1)

    # 9. Pullback count (how many times does price retrace > 0.3%?)
    if direction == "long":
        running_high = np.maximum.accumulate(c)
        dd_pct = (running_high - c) / running_high * 100
    else:
        running_low = np.minimum.accumulate(c)
        dd_pct = (c - running_low) / running_low * 100
    results["pullback_03_count"] = int(np.sum(dd_pct > 0.3))
    results["pullback_05_count"] = int(np.sum(dd_pct > 0.5))

    # 10. Candle body CV
    if len(nonzero) > 10:
        results["body_cv"] = round(np.std(nonzero) / np.mean(nonzero), 3)
    else:
        results["body_cv"] = 0

    return results


def get_trade_data(sym, ts_str, direction, lookback=120):
    """Get arrays for a backtest trade."""
    df = load_sym_df(sym)
    if df is None:
        return None

    trade_ts = pd.Timestamp(ts_str)
    if trade_ts.tzinfo is None:
        trade_ts = trade_ts.tz_localize("UTC")

    diffs = (df["timestamp"] - trade_ts).abs()
    idx = diffs.idxmin()

    start = max(0, idx - lookback)
    sl = slice(start, idx + 1)

    c = df["close"].values[sl].astype(float)
    h = df["high"].values[sl].astype(float)
    l = df["low"].values[sl].astype(float)
    o = df["open"].values[sl].astype(float)
    v = df["volume"].values[sl].astype(float)

    return compute_metrics(c, h, l, o, v, direction, lookback)


# ── LOAD BACKTEST RESULTS ──
results = pd.read_csv(RESULTS_CSV)
filled = results[results["outcome"].isin(["TP", "SL"])].copy()

# Sample TP and SL trades (stratified by DPS)
np.random.seed(42)
tp_trades = filled[filled["outcome"] == "TP"].sample(n=min(200, len(filled[filled["outcome"]=="TP"])))
sl_trades = filled[filled["outcome"] == "SL"].sample(n=min(200, len(filled[filled["outcome"]=="SL"])))

print("=" * 100)
print("COMPARING QUALITY METRICS: TP WINNERS vs SL LOSERS")
print(f"  TP sample: {len(tp_trades)} | SL sample: {len(sl_trades)}")
print("=" * 100)

groups = {"TP": tp_trades, "SL": sl_trades}
group_metrics = {}

for label, trades_df in groups.items():
    metrics_list = []
    done = 0
    for _, row in trades_df.iterrows():
        m = get_trade_data(row["symbol"], row["ts"], row["side"])
        if m is not None:
            m["dps"] = row["dps"]
            m["dur_hrs"] = row["dur_hrs"]
            m["pnl"] = row["pnl"]
            metrics_list.append(m)
            done += 1
        if done >= 200:
            break

    group_metrics[label] = pd.DataFrame(metrics_list)
    print(f"\n  {label}: computed metrics for {len(metrics_list)} trades")

# ── COMPARE ──
key_metrics = ["r2", "efficiency", "max_pullback_pct", "body_max_med", "c2c_max_med",
               "avg_wick", "pct_correct_side", "bar_hl_lh_pct",
               "pullback_03_count", "pullback_05_count", "body_cv"]

print(f"\n{'=' * 100}")
print(f"{'Metric':<22} | {'TP Mean':>10} {'TP Med':>10} | {'SL Mean':>10} {'SL Med':>10} | {'Delta':>8} | Direction")
print("-" * 100)

for m in key_metrics:
    tp_vals = group_metrics["TP"][m].dropna()
    sl_vals = group_metrics["SL"][m].dropna()
    if len(tp_vals) == 0 or len(sl_vals) == 0:
        continue

    tp_mean = tp_vals.mean()
    tp_med = tp_vals.median()
    sl_mean = sl_vals.mean()
    sl_med = sl_vals.median()
    delta = tp_mean - sl_mean
    pct_diff = abs(delta) / max(abs(sl_mean), 0.001) * 100

    if pct_diff > 5:
        direction = "<<< TP better" if delta > 0 else ">>> SL higher"
        if m in ["max_pullback_pct", "body_max_med", "c2c_max_med", "avg_wick",
                 "pullback_03_count", "pullback_05_count", "body_cv"]:
            direction = "<<< TP better (lower)" if delta < 0 else ">>> SL lower (?)"
    else:
        direction = "~same"

    star = " ***" if pct_diff > 10 else ""
    print(f"  {m:<20} | {tp_mean:>10.3f} {tp_med:>10.3f} | {sl_mean:>10.3f} {sl_med:>10.3f} | {delta:>+8.3f} | {direction}{star}")

# ── DISTRIBUTION ANALYSIS for most promising metrics ──
print(f"\n\n{'=' * 100}")
print("DISTRIBUTION ANALYSIS — best separating metrics")
print(f"{'=' * 100}")

for m in ["r2", "efficiency", "max_pullback_pct", "pct_correct_side", "avg_wick",
          "pullback_03_count", "body_max_med"]:
    tp_vals = group_metrics["TP"][m].dropna()
    sl_vals = group_metrics["SL"][m].dropna()

    print(f"\n  {m}:")
    print(f"    TP:  p10={tp_vals.quantile(0.1):.3f}  p25={tp_vals.quantile(0.25):.3f}  "
          f"p50={tp_vals.median():.3f}  p75={tp_vals.quantile(0.75):.3f}  p90={tp_vals.quantile(0.9):.3f}")
    print(f"    SL:  p10={sl_vals.quantile(0.1):.3f}  p25={sl_vals.quantile(0.25):.3f}  "
          f"p50={sl_vals.median():.3f}  p75={sl_vals.quantile(0.75):.3f}  p90={sl_vals.quantile(0.9):.3f}")

    # Test different thresholds
    if m in ["r2", "efficiency", "pct_correct_side"]:
        # Higher is better
        for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            tp_pass = (tp_vals >= thresh).mean() * 100
            sl_pass = (sl_vals >= thresh).mean() * 100
            if tp_pass > 5 and sl_pass > 5:
                tp_wr = tp_pass / (tp_pass + sl_pass) * 100 if (tp_pass + sl_pass) > 0 else 0
                # Adjust: account for equal sample sizes
                kept_pct = (tp_pass + sl_pass) / 2
                print(f"    >= {thresh}: TP pass={tp_pass:.0f}% SL pass={sl_pass:.0f}% "
                      f"| implied WR boost={tp_pass-sl_pass:+.1f}pp | kept ~{kept_pct:.0f}%")
    elif m in ["max_pullback_pct", "avg_wick", "pullback_03_count", "body_max_med"]:
        # Lower is better
        thresholds = {
            "max_pullback_pct": [0.5, 0.8, 1.0, 1.5, 2.0],
            "avg_wick": [0.40, 0.45, 0.50, 0.55],
            "pullback_03_count": [5, 10, 15, 20, 30],
            "body_max_med": [3.0, 4.0, 5.0, 6.0, 8.0],
        }
        for thresh in thresholds.get(m, []):
            tp_pass = (tp_vals <= thresh).mean() * 100
            sl_pass = (sl_vals <= thresh).mean() * 100
            if tp_pass > 5 and sl_pass > 5:
                kept_pct = (tp_pass + sl_pass) / 2
                print(f"    <= {thresh}: TP pass={tp_pass:.0f}% SL pass={sl_pass:.0f}% "
                      f"| implied WR boost={tp_pass-sl_pass:+.1f}pp | kept ~{kept_pct:.0f}%")

# ── CHECK: DPS + new metrics combined ──
print(f"\n\n{'=' * 100}")
print("COMBO: DPS >= 4 + new metric filters")
print(f"{'=' * 100}")

for label in ["TP", "SL"]:
    group_metrics[label]["dps_ge4"] = group_metrics[label]["dps"] >= 4

tp_df = group_metrics["TP"]
sl_df = group_metrics["SL"]

combos = [
    ("baseline (no filter)", lambda df: pd.Series([True]*len(df), index=df.index)),
    ("DPS >= 4", lambda df: df["dps_ge4"]),
    ("R2 >= 0.5", lambda df: df["r2"] >= 0.5),
    ("R2 >= 0.6", lambda df: df["r2"] >= 0.6),
    ("R2 >= 0.7", lambda df: df["r2"] >= 0.7),
    ("max_pb <= 1.5%", lambda df: df["max_pullback_pct"] <= 1.5),
    ("max_pb <= 1.0%", lambda df: df["max_pullback_pct"] <= 1.0),
    ("efficiency >= 0.15", lambda df: df["efficiency"] >= 0.15),
    ("efficiency >= 0.20", lambda df: df["efficiency"] >= 0.20),
    ("pct_correct >= 80%", lambda df: df["pct_correct_side"] >= 80),
    ("pct_correct >= 90%", lambda df: df["pct_correct_side"] >= 90),
    ("pb_03_count <= 15", lambda df: df["pullback_03_count"] <= 15),
    ("pb_05_count <= 5", lambda df: df["pullback_05_count"] <= 5),
    ("DPS>=4 + R2>=0.5", lambda df: (df["dps_ge4"]) & (df["r2"] >= 0.5)),
    ("DPS>=4 + R2>=0.6", lambda df: (df["dps_ge4"]) & (df["r2"] >= 0.6)),
    ("DPS>=4 + R2>=0.5 + pb<=1.5", lambda df: (df["dps_ge4"]) & (df["r2"] >= 0.5) & (df["max_pullback_pct"] <= 1.5)),
    ("DPS>=4 + R2>=0.5 + eff>=0.15", lambda df: (df["dps_ge4"]) & (df["r2"] >= 0.5) & (df["efficiency"] >= 0.15)),
    ("R2>=0.5 + pb<=1.5", lambda df: (df["r2"] >= 0.5) & (df["max_pullback_pct"] <= 1.5)),
    ("R2>=0.5 + eff>=0.15", lambda df: (df["r2"] >= 0.5) & (df["efficiency"] >= 0.15)),
    ("R2>=0.5 + correct>=80", lambda df: (df["r2"] >= 0.5) & (df["pct_correct_side"] >= 80)),
    ("R2>=0.5 + pb<=1.5 + eff>=0.15", lambda df: (df["r2"] >= 0.5) & (df["max_pullback_pct"] <= 1.5) & (df["efficiency"] >= 0.15)),
]

print(f"\n{'Filter':<40} | {'TP pass':>8} {'SL pass':>8} | {'WR':>6} | {'WR lift':>8}")
print("-" * 90)

baseline_tp = len(tp_df)
baseline_sl = len(sl_df)
baseline_wr = baseline_tp / (baseline_tp + baseline_sl) * 100

for name, fn in combos:
    tp_mask = fn(tp_df)
    sl_mask = fn(sl_df)
    tp_pass = tp_mask.sum()
    sl_pass = sl_mask.sum()
    total = tp_pass + sl_pass
    wr = tp_pass / total * 100 if total > 0 else 0
    lift = wr - baseline_wr
    kept = total / (baseline_tp + baseline_sl) * 100
    print(f"  {name:<38} | {tp_pass:>8d} {sl_pass:>8d} | {wr:>5.1f}% | {lift:>+7.1f}pp  (kept {kept:.0f}%)")
