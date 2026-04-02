#!/usr/bin/env python3
"""Analyze price action differences between TP and SL trades."""
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

np.random.seed(42)
tp_trades = closed[short_base & (closed["outcome"] == "TP")].sample(
    n=min(200, (closed[short_base]["outcome"] == "TP").sum()))
sl_trades = closed[short_base & (closed["outcome"] == "SL")].sample(
    n=min(200, (closed[short_base]["outcome"] == "SL").sum()))

_cache = {}
def load_sym(sym):
    if sym not in _cache:
        p = DATA_DIR / f"{sym}_1m.csv"
        if p.exists():
            _cache[sym] = pd.read_csv(str(p), parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        else:
            _cache[sym] = None
    return _cache[sym]


def analyze_trade(row):
    df = load_sym(row["symbol"])
    if df is None:
        return None
    ts = pd.Timestamp(row["ts"])
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    diffs = (df["timestamp"] - ts).abs()
    idx = diffs.idxmin()
    c = df["close"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    o = df["open"].values.astype(float)
    v = df["volume"].values.astype(float)
    level = row["level"]
    if idx < 120 or idx + 30 >= len(c):
        return None

    # 1. Distance from level at signal
    dist_from_level = abs(c[idx] - level) / level * 100

    # 2. Slope of last 30 bars
    last30 = c[idx - 29:idx + 1]
    slope30 = np.polyfit(np.arange(30), last30, 1)[0]
    slope30_pct = slope30 / np.mean(last30) * 100

    # 3. R2 of last 60 bars
    last60 = c[idx - 59:idx + 1]
    x60 = np.arange(60)
    s60, i60 = np.polyfit(x60, last60, 1)
    pred = s60 * x60 + i60
    ss_res = np.sum((last60 - pred) ** 2)
    ss_tot = np.sum((last60 - np.mean(last60)) ** 2)
    r2_60 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # 4. R2 of last 30 bars
    s30, i30 = np.polyfit(np.arange(30), last30, 1)
    pred30 = s30 * np.arange(30) + i30
    ss_r30 = np.sum((last30 - pred30) ** 2)
    ss_t30 = np.sum((last30 - np.mean(last30)) ** 2)
    r2_30 = 1 - ss_r30 / ss_t30 if ss_t30 > 0 else 0

    # 5. Volume acceleration
    vol_usd = v[idx - 119:idx + 1] * c[idx - 119:idx + 1]
    recent_vol = np.mean(vol_usd[-30:])
    prior_vol = np.mean(vol_usd[:30])
    vol_accel = recent_vol / prior_vol if prior_vol > 0 else 1

    # 6. Red/green candle count last 20
    c20 = c[idx - 19:idx + 1]; o20 = o[idx - 19:idx + 1]
    red_pct = np.sum(c20 < o20) / 20 * 100

    # 7. Wick ratio last 10
    rng10 = h[idx - 9:idx + 1] - l[idx - 9:idx + 1]
    body10 = np.abs(c[idx - 9:idx + 1] - o[idx - 9:idx + 1])
    wick10 = np.mean(np.where(rng10 > 1e-12, (rng10 - body10) / rng10, 0))

    # 8. Consecutive bars below level
    bars_below = 0
    for k in range(idx, max(idx - 120, 0), -1):
        if c[k] < level:
            bars_below += 1
        else:
            break

    # 9. Max bar in last 5
    rng5 = h[idx - 4:idx + 1] - l[idx - 4:idx + 1]
    max_bar_5 = np.max(rng5 / c[idx - 4:idx + 1] * 100)

    # 10. Volatility: std of last 30 closes / mean
    close_vol = np.std(last30) / np.mean(last30) * 100

    # 11. How far from SMMA30?
    smma30 = pd.Series(c[:idx + 1]).ewm(alpha=1 / 30, adjust=False).mean().values
    dist_smma = (smma30[idx] - c[idx]) / c[idx] * 100  # positive = price below SMMA (good for shorts)

    # 12. Number of higher lows in last 20 (counter-trend for shorts = bad)
    lows20 = l[idx - 19:idx + 1]
    higher_lows = np.sum(np.diff(lows20) > 0) / 19 * 100

    # 13. Total move in staircase (how much has it already moved?)
    stair_move = abs(c[idx] - c[idx - 119]) / c[idx - 119] * 100

    # 14. Grind quality: how many 5-bar windows make new lows (for shorts)
    new_lows = 0
    for k in range(idx - 24, idx + 1, 5):
        if k >= 5:
            if np.min(l[k - 4:k + 1]) < np.min(l[k - 9:k - 4]):
                new_lows += 1
    new_low_pct = new_lows / 5 * 100 if True else 0

    return {
        "dist_from_level": round(dist_from_level, 4),
        "slope30_pct": round(slope30_pct, 5),
        "r2_60": round(r2_60, 4),
        "r2_30": round(r2_30, 4),
        "vol_accel": round(vol_accel, 3),
        "red_pct_20": round(red_pct, 1),
        "wick10": round(wick10, 3),
        "bars_below": bars_below,
        "max_bar_5": round(max_bar_5, 4),
        "close_vol": round(close_vol, 4),
        "dist_smma": round(dist_smma, 4),
        "higher_lows_pct": round(higher_lows, 1),
        "stair_move_pct": round(stair_move, 3),
        "new_low_pct": round(new_low_pct, 1),
    }


print("Analyzing TP vs SL trades...")
tp_m = [m for _, r in tp_trades.iterrows() if (m := analyze_trade(r)) is not None]
sl_m = [m for _, r in sl_trades.iterrows() if (m := analyze_trade(r)) is not None]
tp_df = pd.DataFrame(tp_m)
sl_df = pd.DataFrame(sl_m)
print(f"  {len(tp_df)} TP, {len(sl_df)} SL analyzed")

print(f"\n{'='*100}")
print("TP vs SL — PRICE ACTION BEFORE ENTRY (shorts, best config)")
print(f"{'='*100}")

metrics = list(tp_df.columns)
print(f"\n{'Metric':<22} | {'TP Mean':>10} {'TP Med':>10} | {'SL Mean':>10} {'SL Med':>10} | {'Delta%':>8} | Sig")
print("-" * 95)

for m in metrics:
    tv = tp_df[m]; sv = sl_df[m]
    tm, tmed = tv.mean(), tv.median()
    sm, smed = sv.mean(), sv.median()
    delta = abs(tm - sm) / max(abs(sm), 0.001) * 100
    sig = "***" if delta > 15 else ("**" if delta > 8 else ("*" if delta > 5 else ""))
    print(f"  {m:<20} | {tm:>10.4f} {tmed:>10.4f} | {sm:>10.4f} {smed:>10.4f} | {delta:>7.1f}% | {sig}")

# Threshold search for most promising
print(f"\n{'='*100}")
print("THRESHOLD SEARCH — can we filter SL trades?")
print(f"{'='*100}")

for m in metrics:
    tv = tp_df[m]; sv = sl_df[m]
    if abs(tv.mean() - sv.mean()) / max(abs(sv.mean()), 0.001) < 0.05:
        continue
    print(f"\n  {m}:")
    # Try different thresholds
    all_vals = np.concatenate([tv.values, sv.values])
    for p in [25, 50, 75]:
        thresh = np.percentile(all_vals, p)
        # Which direction filters better?
        for op, op_name in [(lambda x, t: x >= t, ">="), (lambda x, t: x <= t, "<=")]:
            tp_pass = op(tv, thresh).mean() * 100
            sl_pass = op(sv, thresh).mean() * 100
            if tp_pass > 30 and sl_pass > 30:  # meaningful filter
                lift = tp_pass - sl_pass
                if abs(lift) > 3:
                    print(f"    {op_name} {thresh:.4f} (p{p}): TP pass={tp_pass:.0f}% SL pass={sl_pass:.0f}% | lift={lift:+.1f}pp")
