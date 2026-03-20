#!/usr/bin/env python3
"""Analyze depth backtest trades with market condition scores."""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

# --- 1. Load BTC and compute SMMA30 + VWAP ---
btc = pd.read_csv("datasets/live/candles_1m/BTCUSDT_1m.csv", parse_dates=["timestamp"])
btc = btc.sort_values("timestamp").reset_index(drop=True)
btc["smma30"] = btc["close"].ewm(alpha=1/30, adjust=False).mean()
btc["smma30_slope"] = (btc["smma30"] - btc["smma30"].shift(120)) / btc["smma30"].shift(120) * 100

btc["date"] = btc["timestamp"].dt.normalize()
btc["tp_col"] = (btc["high"] + btc["low"] + btc["close"]) / 3
btc["tp_vol"] = btc["tp_col"] * btc["volume"]
btc["cum_tp_vol"] = btc.groupby("date")["tp_vol"].cumsum()
btc["cum_vol"] = btc.groupby("date")["volume"].cumsum()
btc["vwap"] = btc["cum_tp_vol"] / btc["cum_vol"].replace(0, np.nan)

# --- 2. Load symbols for breadth ---
candle_dir = Path("datasets/live/candles_1m")
all_files = sorted(candle_dir.glob("*_1m.csv"))

# Build per-timestamp arrays for breadth
# Use BTC timestamps as reference
btc_ts_list = btc["timestamp"].tolist()
btc_ts_set = set(btc_ts_list)

sym_above_count = np.zeros(len(btc), dtype=np.float32)
sym_total_count = np.zeros(len(btc), dtype=np.float32)

btc_ts_to_idx = {ts: i for i, ts in enumerate(btc_ts_list)}

loaded = 0
for f in all_files:
    sym = f.stem.replace("_1m", "")
    if sym == "BTCUSDT":
        continue
    try:
        sdf = pd.read_csv(str(f), parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        if len(sdf) < 100:
            continue
        sdf["smma30"] = sdf["close"].ewm(alpha=1/30, adjust=False).mean()
        for _, row in sdf.iterrows():
            ts = row["timestamp"]
            if ts in btc_ts_to_idx:
                idx = btc_ts_to_idx[ts]
                sym_total_count[idx] += 1
                if row["close"] > row["smma30"]:
                    sym_above_count[idx] += 1
        loaded += 1
    except:
        continue

print(f"Loaded {loaded} symbols for breadth")

# Breadth %
breadth_pct = np.where(sym_total_count > 10,
                       sym_above_count / sym_total_count * 100, 50.0)

# --- 3. Compute market score per BTC bar ---
scores = np.zeros(len(btc), dtype=np.int32)
for i in range(len(btc)):
    s = 0
    # Signal 1: SMMA slope
    slope = btc.iloc[i]["smma30_slope"]
    if not pd.isna(slope):
        if slope > 0.01:
            s += 1
        elif slope < -0.01:
            s -= 1
    # Signal 2: vs VWAP
    vwap = btc.iloc[i]["vwap"]
    close = btc.iloc[i]["close"]
    if not pd.isna(vwap) and vwap > 0:
        if close > vwap:
            s += 1
        else:
            s -= 1
    # Signal 3: Breadth
    bp = breadth_pct[i]
    if bp > 60:
        s += 1
    elif bp < 40:
        s -= 1
    scores[i] = s

btc["market_score"] = scores

# Distribution
dist = Counter(scores)
print("Score distribution:")
for s in sorted(dist):
    print(f"  Score {s:+d}: {dist[s]} bars ({dist[s]/len(btc)*100:.1f}%)")

# --- 4. Match depth trades to market scores ---
depth_df = pd.read_csv("backtest_depth_trades.csv")
depth_df["market_score"] = 0

for idx, row in depth_df.iterrows():
    ts = pd.Timestamp(row["timestamp"])
    # Find nearest BTC bar
    diffs = (btc["timestamp"] - ts).abs()
    nearest = diffs.idxmin()
    depth_df.at[idx, "market_score"] = int(btc.iloc[nearest]["market_score"])

# --- 5. Filters ---
def is_allowed(row):
    score = row["market_score"]
    side = row["side"]
    if score >= 2 and side == "short":
        return False
    if score <= -2 and side == "long":
        return False
    return True

depth_df["allowed"] = depth_df.apply(is_allowed, axis=1)
depth_df["abs_imb"] = (abs(depth_df["imbalance_1pct"]) + abs(depth_df["imbalance_2pct"])) / 2
depth_df["strong_imb"] = depth_df["abs_imb"] >= 0.15
depth_df["strong_walls"] = depth_df["sl_wall_strength"] >= 5

# --- 6. Report ---
combos = []
def calc(sub, label):
    if len(sub) < 3:
        return
    tp = (sub["outcome"] == "TP").sum()
    sl = (sub["outcome"] == "SL").sum()
    to = (sub["outcome"] == "TIMEOUT").sum()
    closed = tp + sl
    if closed < 2:
        return
    wr = tp / closed * 100
    avg = sub["pnl_pct"].mean()
    combos.append({"filter": label, "trades": len(sub), "tp": tp, "sl": sl, "to": to,
                   "wr": round(wr,1), "avg_pnl": round(avg,3), "total_pnl": round(sub["pnl_pct"].sum(),2)})

calc(depth_df, "Depth (all)")
calc(depth_df[depth_df["side"]=="short"], "Depth short")
calc(depth_df[depth_df["side"]=="long"], "Depth long")
calc(depth_df[depth_df["allowed"]==True], "Depth + mkt filter")
calc(depth_df[(depth_df["allowed"]==True) & (depth_df["side"]=="short")], "Depth + mkt short")
calc(depth_df[(depth_df["allowed"]==True) & (depth_df["side"]=="long")], "Depth + mkt long")
calc(depth_df[depth_df["strong_imb"]==True], "Depth strong imbalance")
calc(depth_df[(depth_df["strong_imb"]==True) & (depth_df["allowed"]==True)], "Depth strong imb + mkt")
calc(depth_df[(depth_df["strong_imb"]==True) & (depth_df["side"]=="short")], "Depth strong imb short")
calc(depth_df[depth_df["strong_walls"]==True], "Depth strong wall (>=5x)")
calc(depth_df[(depth_df["strong_walls"]==True) & (depth_df["allowed"]==True)], "Depth strong wall + mkt")
calc(depth_df[(depth_df["strong_walls"]==True) & (depth_df["side"]=="short")], "Depth strong wall short")
calc(depth_df[depth_df["rr"]>=1.2], "Depth RR>=1.2")
calc(depth_df[(depth_df["rr"]>=1.2) & (depth_df["allowed"]==True)], "Depth RR>=1.2 + mkt")
calc(depth_df[(depth_df["rr"]>=1.2) & (depth_df["side"]=="short")], "Depth RR>=1.2 short")
calc(depth_df[(depth_df["strong_imb"]==True) & (depth_df["rr"]>=1.2)], "Depth imb + RR>=1.2")
calc(depth_df[(depth_df["strong_imb"]==True) & (depth_df["rr"]>=1.2) & (depth_df["allowed"]==True)], "Depth imb+RR+mkt")
calc(depth_df[(depth_df["strong_walls"]==True) & (depth_df["rr"]>=1.2)], "Depth wall + RR>=1.2")
calc(depth_df[(depth_df["strong_walls"]==True) & (depth_df["rr"]>=1.2) & (depth_df["allowed"]==True)], "Depth wall+RR+mkt")
calc(depth_df[(depth_df["strong_walls"]==True) & (depth_df["strong_imb"]==True)], "Depth wall + imb")
calc(depth_df[(depth_df["strong_walls"]==True) & (depth_df["strong_imb"]==True) & (depth_df["allowed"]==True)], "Depth wall+imb+mkt")
calc(depth_df[(depth_df["strong_walls"]==True) & (depth_df["strong_imb"]==True) & (depth_df["side"]=="short")], "Depth wall+imb short")

# Market score only short
calc(depth_df[(depth_df["market_score"]<=-1) & (depth_df["side"]=="short")], "Depth score<=-1 short")
calc(depth_df[(depth_df["market_score"]>=1) & (depth_df["side"]=="long")], "Depth score>=1 long")
calc(depth_df[(depth_df["market_score"]<=-2) & (depth_df["side"]=="short")], "Depth score<=-2 short")

# By market score
print(f"\nBy Market Score:")
print(f"{'Score':>6s} {'Trades':>6s} {'TP':>4s} {'SL':>4s} {'TO':>4s} {'WR':>6s} {'PnL':>8s}")
print("-" * 42)
for score in sorted(depth_df["market_score"].unique()):
    s = depth_df[depth_df["market_score"] == score]
    tp = (s["outcome"] == "TP").sum()
    sl = (s["outcome"] == "SL").sum()
    to = (s["outcome"] == "TIMEOUT").sum()
    closed = tp + sl
    wr = tp / closed * 100 if closed > 0 else 0
    print(f"{int(score):+6d} {len(s):6d} {tp:4d} {sl:4d} {to:4d} {wr:5.1f}% {s['pnl_pct'].sum():+8.2f}")

# Sort by WR
cdf = pd.DataFrame(combos).sort_values("wr", ascending=False)
print(f"\n{'='*90}")
print(f"DEPTH FILTER RANKING — Sorted by Win Rate")
print(f"{'='*90}")
print(f"{'Filter':>30s} {'Trades':>6s} {'TP':>4s} {'SL':>4s} {'WR':>6s} {'Avg PnL':>8s} {'Total':>8s}")
print(f"{'-'*75}")
for _, r in cdf.iterrows():
    marker = " <<<" if r["wr"] >= 55 else ""
    print(f"{r['filter']:>30s} {r['trades']:6d} {r['tp']:4d} {r['sl']:4d} {r['wr']:5.1f}% {r['avg_pnl']:+8.3f} {r['total_pnl']:+8.2f}{marker}")
