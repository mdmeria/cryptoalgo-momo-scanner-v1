#!/usr/bin/env python3
"""Test daily key level impact on the 18 backtest trades."""

import pandas as pd
import numpy as np
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def find_daily_key_levels(df, min_continuation_days=2, cluster_pct=1.5):
    """Find levels where price changed direction on daily and continued 1+ days."""
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    levels = []

    for i in range(1, len(df) - min_continuation_days):
        # Swing high -> resistance
        if df.loc[i, "high"] >= df.loc[i - 1, "high"]:
            future_closes = df.loc[i + 1 : i + min_continuation_days, "close"].values
            if len(future_closes) >= min_continuation_days:
                days_lower = sum(1 for c in future_closes if c < df.loc[i, "high"])
                if days_lower >= min_continuation_days:
                    future_highs = df.loc[i + 1 : i + min_continuation_days, "high"].values
                    if all(h < df.loc[i, "high"] for h in future_highs):
                        levels.append({
                            "date": df.loc[i, "timestamp"],
                            "price": df.loc[i, "high"],
                            "type": "resistance",
                            "continuation": int(days_lower),
                        })

        # Swing low -> support
        if df.loc[i, "low"] <= df.loc[i - 1, "low"]:
            future_closes = df.loc[i + 1 : i + min_continuation_days, "close"].values
            if len(future_closes) >= min_continuation_days:
                days_higher = sum(1 for c in future_closes if c > df.loc[i, "low"])
                if days_higher >= min_continuation_days:
                    future_lows = df.loc[i + 1 : i + min_continuation_days, "low"].values
                    if all(l > df.loc[i, "low"] for l in future_lows):
                        levels.append({
                            "date": df.loc[i, "timestamp"],
                            "price": df.loc[i, "low"],
                            "type": "support",
                            "continuation": int(days_higher),
                        })

    # Deduplicate nearby same-type levels
    levels.sort(key=lambda x: x["price"])
    deduped = []
    for l in levels:
        if deduped and deduped[-1]["type"] == l["type"]:
            diff_pct = abs(l["price"] - deduped[-1]["price"]) / deduped[-1]["price"] * 100
            if diff_pct < 0.5:
                if l["continuation"] > deduped[-1]["continuation"]:
                    deduped[-1] = l
                continue
        deduped.append(l)

    # Cluster detection
    if len(deduped) >= 2:
        prices = [l["price"] for l in deduped]
        for l in deduped:
            nearby = sum(
                1 for p in prices
                if p != l["price"] and abs(p - l["price"]) / l["price"] * 100 < cluster_pct
            )
            l["in_cluster"] = nearby >= 1
    else:
        for l in deduped:
            l["in_cluster"] = False

    return deduped


def main():
    trades = pd.read_csv("momo_run_trade_list.csv")

    print(f"{'Symbol':20s} {'Side':5s} {'Out':4s}  {'Entry':>10s}  {'OrigTP':>10s}  "
          f"{'OrigSL':>10s}  Block  {'AdjTP':>10s}  {'AdjSL':>10s}  NewOut")
    print("-" * 120)

    results = []
    for _, t in trades.iterrows():
        sym = t["symbol"]
        daily_file = f"datasets/daily_candles/{sym}_1d.csv"
        if not os.path.exists(daily_file):
            print(f"{sym:20s} -- no daily data")
            continue

        daily = pd.read_csv(daily_file, parse_dates=["timestamp"])
        entry_ts = pd.Timestamp(t["timestamp"])

        # Only use levels BEFORE the trade date
        daily_before = daily[daily["timestamp"] < entry_ts.normalize()].copy()
        if len(daily_before) < 5:
            print(f"{sym:20s} -- not enough daily history")
            continue

        levels = find_daily_key_levels(daily_before)

        # Add prev day and prev-prev day high/low as levels
        prev_day = daily_before.iloc[-1]
        levels.append({"price": prev_day["high"], "type": "resistance",
                       "date": prev_day["timestamp"], "continuation": 0, "in_cluster": False})
        levels.append({"price": prev_day["low"], "type": "support",
                       "date": prev_day["timestamp"], "continuation": 0, "in_cluster": False})
        if len(daily_before) >= 2:
            prev2 = daily_before.iloc[-2]
            levels.append({"price": prev2["high"], "type": "resistance",
                           "date": prev2["timestamp"], "continuation": 0, "in_cluster": False})
            levels.append({"price": prev2["low"], "type": "support",
                           "date": prev2["timestamp"], "continuation": 0, "in_cluster": False})

        entry = t["entry"]
        orig_tp = t["tp"]
        orig_sl = t["sl"]
        side = t["side"]
        outcome = t["outcome"]

        if side == "long":
            # Blocking resistance between entry and TP
            blocking = [l for l in levels if l["type"] == "resistance"
                        and entry < l["price"] < orig_tp]
            support_below = [l for l in levels if l["type"] == "support"
                             and l["price"] < entry and l["price"] > orig_sl]

            if blocking:
                nearest = min(blocking, key=lambda l: l["price"])
                adj_tp = nearest["price"] * 0.998
            else:
                adj_tp = orig_tp

            if support_below:
                nearest_sup = max(support_below, key=lambda l: l["price"])
                adj_sl = nearest_sup["price"] * 0.998
            else:
                adj_sl = orig_sl

            block_str = f"{len(blocking)}R"
        else:
            blocking = [l for l in levels if l["type"] == "support"
                        and orig_tp < l["price"] < entry]
            res_above = [l for l in levels if l["type"] == "resistance"
                         and l["price"] > entry and l["price"] < orig_sl]

            if blocking:
                nearest = max(blocking, key=lambda l: l["price"])
                adj_tp = nearest["price"] * 1.002
            else:
                adj_tp = orig_tp

            if res_above:
                nearest_res = min(res_above, key=lambda l: l["price"])
                adj_sl = nearest_res["price"] * 1.002
            else:
                adj_sl = orig_sl

            block_str = f"{len(blocking)}S"

        # Enforce SL bounds: 1-2%
        sl_pct = abs(entry - adj_sl) / entry * 100
        tp_pct = abs(adj_tp - entry) / entry * 100

        if sl_pct < 1.0:
            adj_sl = entry * (1 - 0.01) if side == "long" else entry * (1 + 0.01)
            sl_pct = 1.0
        if sl_pct > 2.0:
            adj_sl = entry * (1 - 0.02) if side == "long" else entry * (1 + 0.02)
            sl_pct = 2.0

        # If adjusted TP is too close (< 0.5%), revert to original
        if tp_pct < 0.5:
            adj_tp = orig_tp
            tp_pct = abs(adj_tp - entry) / entry * 100

        # Simulate new outcome using 1m data
        data_file = f"datasets/perp_mar2_mar15/{sym}_1m.csv"
        if not os.path.exists(data_file):
            data_file = f"datasets/momo_1m_7d_top100_midcap_30d/{sym}_1m.csv"

        new_outcome = outcome
        if os.path.exists(data_file):
            m1 = pd.read_csv(data_file, parse_dates=["timestamp"])
            m1 = m1.sort_values("timestamp").reset_index(drop=True)
            trade_bars = m1[m1["timestamp"] >= entry_ts].head(60)

            if len(trade_bars) > 0:
                new_outcome = "OPEN"
                for _, bar in trade_bars.iterrows():
                    if side == "long":
                        if bar["low"] <= adj_sl:
                            new_outcome = "SL"
                            break
                        if bar["high"] >= adj_tp:
                            new_outcome = "TP"
                            break
                    else:
                        if bar["high"] >= adj_sl:
                            new_outcome = "SL"
                            break
                        if bar["low"] <= adj_tp:
                            new_outcome = "TP"
                            break

        changed = " ***" if new_outcome != outcome else ""
        print(f"{sym:20s} {side:5s} {outcome:4s}  {entry:10.6f}  {orig_tp:10.6f}  "
              f"{orig_sl:10.6f}  {block_str:5s}  {adj_tp:10.6f}  {adj_sl:10.6f}  "
              f"{new_outcome:4s}{changed}")

        results.append({
            "symbol": sym, "side": side,
            "orig_outcome": outcome, "new_outcome": new_outcome,
            "orig_tp_pct": t["tp_pct"], "orig_sl_pct": t["sl_pct"],
            "adj_tp_pct": tp_pct, "adj_sl_pct": sl_pct,
        })

    rdf = pd.DataFrame(results)
    print()
    print("=== COMPARISON ===")
    for label, col in [("Original", "orig_outcome"), ("Key-Level Adj", "new_outcome")]:
        tp_n = sum(rdf[col] == "TP")
        sl_n = sum(rdf[col] == "SL")
        op_n = sum(rdf[col] == "OPEN")
        wr = tp_n / (tp_n + sl_n) * 100 if (tp_n + sl_n) > 0 else 0
        print(f"  {label:15s}  {tp_n} TP / {sl_n} SL / {op_n} OPEN  WR={wr:.1f}%")

    changes = rdf[rdf["orig_outcome"] != rdf["new_outcome"]]
    if len(changes) > 0:
        print(f"\nChanged trades ({len(changes)}):")
        for _, c in changes.iterrows():
            print(f"  {c['symbol']:20s}  {c['orig_outcome']} -> {c['new_outcome']}")


if __name__ == "__main__":
    main()
