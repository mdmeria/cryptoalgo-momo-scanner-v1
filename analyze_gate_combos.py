#!/usr/bin/env python3
"""
Analyze all gate combinations against enhanced market conditions.
Run after backtest with gate metrics logged (not filtered).
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd, numpy as np

# Load trades + enhanced market cache
trades = pd.read_csv("zct_momo_results.csv")
mkt = pd.read_csv("datasets/binance_futures_1m/enhanced_market_cache.csv",
                   parse_dates=["timestamp"])

# Join enhanced market data
trades["ts_dt"] = pd.to_datetime(trades["ts"])
mkt_ts = mkt["timestamp"].values.astype("datetime64[ns]").astype(np.int64)
trade_ts = trades["ts_dt"].values.astype("datetime64[ns]").astype(np.int64)
idxs = np.searchsorted(mkt_ts, trade_ts, side="right") - 1
idxs = np.clip(idxs, 0, len(mkt) - 1)
for col in ["direction_score", "strategy_score", "activity_mod", "quadrant"]:
    trades[col] = mkt[col].values[idxs]

filled = trades[trades["outcome"].isin(["TP", "SL", "TRAIL_SL", "OPEN"])].copy()
closed = filled[filled["outcome"].isin(["TP", "SL"])].copy()

def stats(mask_closed, mask_filled=None):
    sc = closed[mask_closed]
    sa = filled[mask_filled] if mask_filled is not None else filled[mask_closed.reindex(filled.index, fill_value=False)]
    tp = (sc["outcome"] == "TP").sum()
    sl = (sc["outcome"] == "SL").sum()
    wr = tp / (tp + sl) * 100 if (tp + sl) > 0 else 0
    pnl = sa["pnl"].sum()
    avg = sa["pnl"].mean() if len(sa) > 0 else 0
    return len(sc), wr, pnl, avg

# Gate filters
def candle_gate(thresh=0.5):
    return closed["max_candle_15m"] < thresh

def dd_gate(thresh=1.0):
    return closed["max_dd_2h"] <= thresh

def short_only():
    return closed["side"] == "short"

def long_only():
    return closed["side"] == "long"

def enhanced_mkt():
    return (closed["direction_score"] <= -1) & (closed["strategy_score"] >= 1)

def enhanced_mkt_act():
    return enhanced_mkt() & (closed["activity_mod"] >= 0)

# ── ALL COMBOS ──
print("=" * 100)
print("GATE COMBINATION ANALYSIS (all trades, post-hoc filtering)")
print("=" * 100)

combos = [
    ("Baseline (all)", pd.Series(True, index=closed.index)),
    ("Short only", short_only()),
    ("Long only", long_only()),

    # Candle gate only
    ("candle<0.5%", candle_gate(0.5)),
    ("candle<0.5% short", candle_gate(0.5) & short_only()),

    # Drawdown gate only
    ("dd<=1%", dd_gate(1.0)),
    ("dd<=1% short", dd_gate(1.0) & short_only()),
    ("dd<=1.5%", dd_gate(1.5)),
    ("dd<=1.5% short", dd_gate(1.5) & short_only()),

    # Both gates
    ("candle<0.5% + dd<=1%", candle_gate(0.5) & dd_gate(1.0)),
    ("candle<0.5% + dd<=1% short", candle_gate(0.5) & dd_gate(1.0) & short_only()),
    ("candle<0.5% + dd<=1.5%", candle_gate(0.5) & dd_gate(1.5)),
    ("candle<0.5% + dd<=1.5% short", candle_gate(0.5) & dd_gate(1.5) & short_only()),

    # Enhanced market only
    ("enhanced_mkt short", enhanced_mkt() & short_only()),
    ("enhanced_mkt+act short", enhanced_mkt_act() & short_only()),

    # Enhanced market + candle
    ("enh+candle<0.5% short", enhanced_mkt() & candle_gate(0.5) & short_only()),
    ("enh+act+candle<0.5% short", enhanced_mkt_act() & candle_gate(0.5) & short_only()),

    # Enhanced market + drawdown
    ("enh+dd<=1% short", enhanced_mkt() & dd_gate(1.0) & short_only()),
    ("enh+act+dd<=1% short", enhanced_mkt_act() & dd_gate(1.0) & short_only()),
    ("enh+dd<=1.5% short", enhanced_mkt() & dd_gate(1.5) & short_only()),
    ("enh+act+dd<=1.5% short", enhanced_mkt_act() & dd_gate(1.5) & short_only()),

    # Enhanced market + both gates
    ("enh+candle+dd<=1% short", enhanced_mkt() & candle_gate(0.5) & dd_gate(1.0) & short_only()),
    ("enh+act+candle+dd<=1% short", enhanced_mkt_act() & candle_gate(0.5) & dd_gate(1.0) & short_only()),
    ("enh+candle+dd<=1.5% short", enhanced_mkt() & candle_gate(0.5) & dd_gate(1.5) & short_only()),
    ("enh+act+candle+dd<=1.5% short", enhanced_mkt_act() & candle_gate(0.5) & dd_gate(1.5) & short_only()),

    # Try different candle thresholds
    ("enh+act+candle<0.4% short", enhanced_mkt_act() & candle_gate(0.4) & short_only()),
    ("enh+act+candle<0.3% short", enhanced_mkt_act() & candle_gate(0.3) & short_only()),

    # Try different drawdown thresholds
    ("enh+act+dd<=0.8% short", enhanced_mkt_act() & dd_gate(0.8) & short_only()),
    ("enh+act+dd<=0.5% short", enhanced_mkt_act() & dd_gate(0.5) & short_only()),
]

print(f"\n{'Filter':<42} | {'Trades':>6} | {'WR':>6} | {'PnL':>8} | {'Avg':>8}")
print("-" * 85)

for label, mask in combos:
    n, wr, pnl, avg = stats(mask)
    if n < 10:
        continue
    print(f"  {label:<40} | {n:>6} | {wr:>5.1f}% | {pnl:>+7.1f}% | {avg:>+7.3f}%")
