#!/usr/bin/env python3
"""Analyze price action differences between TP and SL — separately for longs and shorts."""
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


def smma_arr(arr, period=30):
    out = np.zeros(len(arr))
    out[0] = arr[0]
    alpha = 1.0 / period
    for i in range(1, len(arr)):
        out[i] = out[i-1] * (1 - alpha) + arr[i] * alpha
    return out


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
    side = row["side"]
    is_long = side == "long"
    if idx < 120 or idx + 30 >= len(c):
        return None

    # 1. Distance from level at signal
    dist_from_level = abs(c[idx] - level) / level * 100

    # 2. Slope of last 30 bars (positive for longs, negative for shorts when trending)
    last30 = c[idx - 29:idx + 1]
    slope30 = np.polyfit(np.arange(30), last30, 1)[0]
    slope30_pct = slope30 / np.mean(last30) * 100
    # Normalize: positive = correct direction
    slope_correct = slope30_pct if is_long else -slope30_pct

    # 3. R2 of last 60 bars and 30 bars
    def calc_r2(arr):
        x = np.arange(len(arr))
        s, intercept = np.polyfit(x, arr, 1)
        pred = s * x + intercept
        ss_res = np.sum((arr - pred) ** 2)
        ss_tot = np.sum((arr - np.mean(arr)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0
    r2_60 = calc_r2(c[idx - 59:idx + 1])
    r2_30 = calc_r2(c[idx - 29:idx + 1])

    # 4. Volume acceleration
    vol_usd = v[idx - 119:idx + 1] * c[idx - 119:idx + 1]
    recent_vol = np.mean(vol_usd[-30:])
    prior_vol = np.mean(vol_usd[:30])
    vol_accel = recent_vol / prior_vol if prior_vol > 0 else 1

    # 5. Candle color dominance (% of candles in correct direction)
    c20 = c[idx - 19:idx + 1]; o20 = o[idx - 19:idx + 1]
    if is_long:
        correct_candles = np.sum(c20 > o20) / 20 * 100
    else:
        correct_candles = np.sum(c20 < o20) / 20 * 100

    # 6. Wick ratio last 10
    rng10 = h[idx - 9:idx + 1] - l[idx - 9:idx + 1]
    body10 = np.abs(c[idx - 9:idx + 1] - o[idx - 9:idx + 1])
    wick10 = np.mean(np.where(rng10 > 1e-12, (rng10 - body10) / rng10, 0))

    # 7. Max bar in last 5 and 10
    rng5 = h[idx - 4:idx + 1] - l[idx - 4:idx + 1]
    max_bar_5 = np.max(rng5 / c[idx - 4:idx + 1] * 100)
    rng10_full = h[idx - 9:idx + 1] - l[idx - 9:idx + 1]
    max_bar_10 = np.max(rng10_full / c[idx - 9:idx + 1] * 100)

    # 8. Volatility: std of last 30 closes / mean
    close_vol = np.std(last30) / np.mean(last30) * 100

    # 9. SMMA30 distance
    sm30 = smma_arr(c[:idx + 1], 30)
    if is_long:
        dist_smma = (c[idx] - sm30[idx]) / c[idx] * 100  # positive = above SMMA (good)
    else:
        dist_smma = (sm30[idx] - c[idx]) / c[idx] * 100  # positive = below SMMA (good)

    # 10. Total staircase move
    stair_move = abs(c[idx] - c[idx - 119]) / c[idx - 119] * 100

    # 11. Higher lows / lower highs consistency
    if is_long:
        hl_pct = np.sum(np.diff(l[idx - 19:idx + 1]) > 0) / 19 * 100
    else:
        hl_pct = np.sum(np.diff(h[idx - 19:idx + 1]) < 0) / 19 * 100

    # 12. Pullback depth in last 60 bars
    last60_c = c[idx - 59:idx + 1]
    if is_long:
        peak = np.maximum.accumulate(last60_c)
        max_pb_60 = np.max((peak - l[idx - 59:idx + 1]) / peak * 100)
    else:
        trough = np.minimum.accumulate(last60_c)
        max_pb_60 = np.max((h[idx - 59:idx + 1] - trough) / trough * 100)

    # 13. Number of SMMA30 crosses in 2h
    above = c[idx - 119:idx + 1] > sm30[idx - 119:idx + 1]
    crosses = np.sum(np.diff(above.astype(int)) != 0)

    # 14. EMA7 position relative to close (momentum confirmation)
    ema7 = pd.Series(c[:idx + 1]).ewm(span=7, adjust=False).mean().values
    if is_long:
        ema7_above = c[idx] > ema7[idx]
    else:
        ema7_above = c[idx] < ema7[idx]

    # 15. Body/range ratio of signal candle
    sig_body = abs(c[idx] - o[idx])
    sig_range = h[idx] - l[idx]
    sig_body_ratio = sig_body / sig_range if sig_range > 0 else 0

    # 16. Was the signal candle in the correct direction?
    sig_correct = (c[idx] > o[idx]) if is_long else (c[idx] < o[idx])

    return {
        "dist_from_level": round(dist_from_level, 4),
        "slope_correct": round(slope_correct, 5),
        "r2_60": round(r2_60, 4),
        "r2_30": round(r2_30, 4),
        "vol_accel": round(vol_accel, 3),
        "correct_candles_pct": round(correct_candles, 1),
        "wick10": round(wick10, 3),
        "max_bar_5": round(max_bar_5, 4),
        "max_bar_10": round(max_bar_10, 4),
        "close_vol": round(close_vol, 4),
        "dist_smma": round(dist_smma, 4),
        "stair_move_pct": round(stair_move, 3),
        "hl_lh_pct": round(hl_pct, 1),
        "max_pb_60": round(max_pb_60, 4),
        "smma_crosses": crosses,
        "ema7_correct": int(ema7_above),
        "sig_body_ratio": round(sig_body_ratio, 3),
        "sig_correct": int(sig_correct),
    }


def run_analysis(label, base_mask):
    np.random.seed(42)
    tp_rows = closed[base_mask & (closed["outcome"] == "TP")]
    sl_rows = closed[base_mask & (closed["outcome"] == "SL")]
    tp_sample = tp_rows.sample(n=min(200, len(tp_rows)))
    sl_sample = sl_rows.sample(n=min(200, len(sl_rows)))

    print(f"\nAnalyzing {label}...")
    tp_m = [m for _, r in tp_sample.iterrows() if (m := analyze_trade(r)) is not None]
    sl_m = [m for _, r in sl_sample.iterrows() if (m := analyze_trade(r)) is not None]
    tp_df = pd.DataFrame(tp_m)
    sl_df = pd.DataFrame(sl_m)
    print(f"  {len(tp_df)} TP, {len(sl_df)} SL analyzed")

    print(f"\n{'='*100}")
    print(f"{label} — TP vs SL")
    print(f"{'='*100}")

    metrics = list(tp_df.columns)
    print(f"\n{'Metric':<22} | {'TP Mean':>10} {'TP Med':>10} | {'SL Mean':>10} {'SL Med':>10} | {'Delta%':>8} | Sig")
    print("-" * 95)

    promising = []
    for m in metrics:
        tv = tp_df[m]; sv = sl_df[m]
        tm, tmed = tv.mean(), tv.median()
        sm, smed = sv.mean(), sv.median()
        delta = abs(tm - sm) / max(abs(sm), 0.001) * 100
        sig = "***" if delta > 15 else ("**" if delta > 8 else ("*" if delta > 5 else ""))
        print(f"  {m:<20} | {tm:>10.4f} {tmed:>10.4f} | {sm:>10.4f} {smed:>10.4f} | {delta:>7.1f}% | {sig}")
        if delta > 5:
            promising.append(m)

    # Threshold search for promising metrics
    if promising:
        print(f"\n  THRESHOLD SEARCH:")
        for m in promising:
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
                        if abs(lift) > 4:
                            if not found:
                                print(f"\n    {m}:")
                                found = True
                            print(f"      {op_name} {thresh:.4f} (p{p}): TP={tp_pass:.0f}% SL={sl_pass:.0f}% lift={lift:+.1f}pp")


# Run for both sides
run_analysis("SHORTS (enh+act, avoid bad hrs)", short_base)
run_analysis("LONGS (dir>=2, strat>=2, act>=0, avoid bad hrs)", long_base)
