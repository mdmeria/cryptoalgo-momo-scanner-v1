#!/usr/bin/env python3
"""Analyze winner vs loser traits from MR backtest."""
import pandas as pd
import numpy as np
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

rdf = pd.read_csv("mr_choppy_backtest.csv")

# Compute pnl
pnl = []
for _, r in rdf.iterrows():
    if r["outcome"] == "TP":
        pnl.append(r["tp_pct"])
    elif r["outcome"] == "SL":
        pnl.append(-r["sl_pct"])
    else:
        if r["side"] == "long":
            pnl.append((r["exit_price"] - r["entry"]) / r["entry"] * 100)
        else:
            pnl.append((r["entry"] - r["exit_price"]) / r["entry"] * 100)
rdf["pnl_pct"] = pnl

print("=== CURRENT DISTRIBUTION ===")
print(f"TP% < 1: {(rdf['tp_pct'] < 1).sum()} trades ({(rdf['tp_pct'] < 1).sum()/len(rdf)*100:.0f}%)")
print(f"RR < 1:  {(rdf['rr'] < 1).sum()} trades")
print()

# Filter: TP >= 1% and RR > 1
clean = rdf[(rdf["tp_pct"] >= 1.0) & (rdf["rr"] > 1.0)].copy()
print(f"After TP>=1% + RR>1 filter: {len(clean)} trades")
tp_n = (clean["outcome"] == "TP").sum()
sl_n = (clean["outcome"] == "SL").sum()
op_n = (clean["outcome"] == "OPEN").sum()
wr = tp_n / (tp_n + sl_n) * 100 if (tp_n + sl_n) > 0 else 0
print(f"  {tp_n} TP / {sl_n} SL / {op_n} OPEN, WR={wr:.1f}%, PnL={clean['pnl_pct'].sum():.2f}%")
print(f"  Avg TP%={clean['tp_pct'].mean():.2f}, Avg SL%={clean['sl_pct'].mean():.2f}, Avg RR={clean['rr'].mean():.2f}")
print()

winners = clean[clean["outcome"] == "TP"]
losers = clean[clean["outcome"] == "SL"]

print("=== WINNERS vs LOSERS TRAITS ===")
for col in ["range_width_pct", "range_duration_hrs", "touches", "break_pct", "rr", "tp_pct", "sl_pct"]:
    w_mean = winners[col].mean() if len(winners) > 0 else 0
    l_mean = losers[col].mean() if len(losers) > 0 else 0
    print(f"  {col:22s}  W={w_mean:7.3f}  L={l_mean:7.3f}  diff={w_mean - l_mean:+.3f}")

print("\n--- Side ---")
for side in ["long", "short"]:
    w = (winners["side"] == side).sum()
    l = (losers["side"] == side).sum()
    wr_s = w / (w + l) * 100 if (w + l) > 0 else 0
    pnl_s = clean[clean["side"] == side]["pnl_pct"].sum()
    print(f"  {side:5s}: {w}W / {l}L  WR={wr_s:.1f}%  PnL={pnl_s:+.2f}%")

print("\n--- Noise Level ---")
for noise in ["low", "medium", "high"]:
    w = (winners["noise_level"] == noise).sum()
    l = (losers["noise_level"] == noise).sum()
    if w + l == 0:
        continue
    wr_s = w / (w + l) * 100
    pnl_s = clean[clean["noise_level"] == noise]["pnl_pct"].sum()
    print(f"  {noise:6s}: {w}W / {l}L  WR={wr_s:.1f}%  PnL={pnl_s:+.2f}%")

print("\n--- Pre-Chop Trend ---")
for trend in ["up", "down", "unclear"]:
    w = (winners["pre_chop_trend"] == trend).sum()
    l = (losers["pre_chop_trend"] == trend).sum()
    if w + l == 0:
        continue
    wr_s = w / (w + l) * 100
    pnl_s = clean[clean["pre_chop_trend"] == trend]["pnl_pct"].sum()
    print(f"  {trend:7s}: {w}W / {l}L  WR={wr_s:.1f}%  PnL={pnl_s:+.2f}%")

print("\n--- Approach ---")
for app in ["spike", "unclear", "grind"]:
    w = (winners["dps_v2_label"] == app).sum()
    l = (losers["dps_v2_label"] == app).sum()
    if w + l == 0:
        continue
    wr_s = w / (w + l) * 100
    pnl_s = clean[clean["dps_v2_label"] == app]["pnl_pct"].sum()
    print(f"  {app:8s}: {w}W / {l}L  WR={wr_s:.1f}%  PnL={pnl_s:+.2f}%")

print("\n--- Volume Trend ---")
for vol in ["flat", "decreasing", "increasing"]:
    w = (winners["dps_v3_vol_trend"] == vol).sum()
    l = (losers["dps_v3_vol_trend"] == vol).sum()
    if w + l == 0:
        continue
    wr_s = w / (w + l) * 100
    pnl_s = clean[clean["dps_v3_vol_trend"] == vol]["pnl_pct"].sum()
    print(f"  {vol:11s}: {w}W / {l}L  WR={wr_s:.1f}%  PnL={pnl_s:+.2f}%")

print("\n--- DPS Score ---")
for dps in sorted(clean["dps_total"].unique()):
    sub = clean[clean["dps_total"] == dps]
    w = (sub["outcome"] == "TP").sum()
    l = (sub["outcome"] == "SL").sum()
    if w + l == 0:
        continue
    wr_s = w / (w + l) * 100
    print(f"  DPS {dps}: {w}W / {l}L  WR={wr_s:.1f}%  PnL={sub['pnl_pct'].sum():+.2f}%")

print("\n--- Range Width Buckets ---")
for lo, hi, label in [(1.0, 1.5, "1.0-1.5%"), (1.5, 2.0, "1.5-2.0%"), (2.0, 3.0, "2.0-3.0%"), (3.0, 5.0, "3.0-5.0%")]:
    sub = clean[(clean["range_width_pct"] >= lo) & (clean["range_width_pct"] < hi)]
    if len(sub) == 0:
        continue
    w = (sub["outcome"] == "TP").sum()
    l = (sub["outcome"] == "SL").sum()
    wr_s = w / (w + l) * 100 if (w + l) > 0 else 0
    print(f"  {label:8s}: {w}W / {l}L  WR={wr_s:.1f}%  PnL={sub['pnl_pct'].sum():+.2f}%  avg_rr={sub['rr'].mean():.2f}")

print("\n--- Touches Buckets ---")
for lo, hi, label in [(4, 5, "4"), (5, 7, "5-6"), (7, 10, "7-9"), (10, 30, "10+")]:
    sub = clean[(clean["touches"] >= lo) & (clean["touches"] < hi)]
    if len(sub) == 0:
        continue
    w = (sub["outcome"] == "TP").sum()
    l = (sub["outcome"] == "SL").sum()
    wr_s = w / (w + l) * 100 if (w + l) > 0 else 0
    print(f"  {label:4s}: {w}W / {l}L  WR={wr_s:.1f}%  PnL={sub['pnl_pct'].sum():+.2f}%")

print("\n--- Break % Buckets ---")
for lo, hi, label in [(0, 1, "0-1%"), (1, 3, "1-3%"), (3, 5, "3-5%")]:
    sub = clean[(clean["break_pct"] >= lo) & (clean["break_pct"] < hi)]
    if len(sub) == 0:
        continue
    w = (sub["outcome"] == "TP").sum()
    l = (sub["outcome"] == "SL").sum()
    wr_s = w / (w + l) * 100 if (w + l) > 0 else 0
    print(f"  {label:5s}: {w}W / {l}L  WR={wr_s:.1f}%  PnL={sub['pnl_pct'].sum():+.2f}%")

print("\n--- Duration Buckets ---")
for lo, hi, label in [(2, 3, "2-3h"), (3, 5, "3-5h"), (5, 9, "5-8h")]:
    sub = clean[(clean["range_duration_hrs"] >= lo) & (clean["range_duration_hrs"] < hi)]
    if len(sub) == 0:
        continue
    w = (sub["outcome"] == "TP").sum()
    l = (sub["outcome"] == "SL").sum()
    wr_s = w / (w + l) * 100 if (w + l) > 0 else 0
    print(f"  {label:4s}: {w}W / {l}L  WR={wr_s:.1f}%  PnL={sub['pnl_pct'].sum():+.2f}%")

print("\n--- Bars Held ---")
print(f"  Winners avg bars: {winners['bars_held'].mean():.0f}")
print(f"  Losers avg bars:  {losers['bars_held'].mean():.0f}")

print("\n=== BEST FILTER COMBOS ===")
combos = [
    ("break>=3", clean["break_pct"] >= 3),
    ("break>=3 + low noise", (clean["break_pct"] >= 3) & (clean["noise_level"] == "low")),
    ("width>=2%", clean["range_width_pct"] >= 2.0),
    ("width>=2% + low noise", (clean["range_width_pct"] >= 2.0) & (clean["noise_level"] == "low")),
    ("down+short", (clean["pre_chop_trend"] == "down") & (clean["side"] == "short")),
    ("up+long", (clean["pre_chop_trend"] == "up") & (clean["side"] == "long")),
    ("touches>=5", clean["touches"] >= 5),
    ("touches>=5 + low noise", (clean["touches"] >= 5) & (clean["noise_level"] == "low")),
    ("rr>=1.5", clean["rr"] >= 1.5),
    ("rr>=1.5 + low noise", (clean["rr"] >= 1.5) & (clean["noise_level"] == "low")),
    ("rr>=1.5 + touches>=5", (clean["rr"] >= 1.5) & (clean["touches"] >= 5)),
    ("rr>=2.0", clean["rr"] >= 2.0),
    ("width>=2 + dur>=4h", (clean["range_width_pct"] >= 2.0) & (clean["range_duration_hrs"] >= 4)),
    ("width>=2 + touches>=5", (clean["range_width_pct"] >= 2.0) & (clean["touches"] >= 5)),
    ("width>=2 + touches>=5 + low noise", (clean["range_width_pct"] >= 2.0) & (clean["touches"] >= 5) & (clean["noise_level"] == "low")),
    ("down+short + rr>=1.5", (clean["pre_chop_trend"] == "down") & (clean["side"] == "short") & (clean["rr"] >= 1.5)),
    ("unclear trend + touches>=5", (clean["pre_chop_trend"] == "unclear") & (clean["touches"] >= 5)),
    ("flat vol + width>=2", (clean["dps_v3_vol_trend"] == "flat") & (clean["range_width_pct"] >= 2.0)),
    ("dur>=4h + touches>=5 + rr>=1.5", (clean["range_duration_hrs"] >= 4) & (clean["touches"] >= 5) & (clean["rr"] >= 1.5)),
]
for label, mask in combos:
    sub = clean[mask]
    if len(sub) < 3:
        continue
    w = (sub["outcome"] == "TP").sum()
    l = (sub["outcome"] == "SL").sum()
    if w + l == 0:
        continue
    wr_s = w / (w + l) * 100
    pnl_s = sub["pnl_pct"].sum()
    avg_pnl = sub["pnl_pct"].mean()
    print(f"  {label:42s}: {len(sub):3d} trades, {w}W/{l}L, WR={wr_s:.1f}%, PnL={pnl_s:+.2f}%, avg={avg_pnl:+.3f}%")
