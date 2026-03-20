#!/usr/bin/env python3
"""
Backtest: Market Condition Filter for MR and Momo strategies.

Computes a Combined Direction Score (-3 to +3) from BTC SMMA30 slope,
BTC session VWAP position, and market breadth (% of symbols above SMMA30).
Compares strategy results WITH and WITHOUT the market condition filter.
"""
from __future__ import annotations

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scan_mean_reversion import (
    MRSettings,
    check_mr_gates_at_bar,
)
from backtest_momo_vwap_grind15_full import (
    GateSettings as MomoGateSettings,
    check_momo_gates_at_bar,
    prepare_features,
)

DATASET_DIR = Path("datasets/spot_jan17_mar17")
MAX_BARS = 120       # timeout after 120 bars
COOLDOWN_BARS = 60   # cooldown between trades per symbol per strategy
WARMUP = 720         # need enough bars for indicators

# ── helpers ──────────────────────────────────────────────────────────────

def smma(series: np.ndarray, period: int) -> np.ndarray:
    """SMMA = EWM with alpha = 1/period."""
    s = pd.Series(series)
    return s.ewm(alpha=1.0 / period, adjust=False).mean().values


def compute_session_vwap(df: pd.DataFrame) -> np.ndarray:
    """Per-bar session VWAP, resetting each UTC day."""
    tp = (df["high"].values + df["low"].values + df["close"].values) / 3.0
    vol = df["volume"].values
    tp_vol = tp * vol

    dates = df["timestamp"].dt.normalize().values
    cum_tp_vol = np.zeros(len(df))
    cum_vol = np.zeros(len(df))
    vwap = np.full(len(df), np.nan)

    prev_date = None
    running_tp_vol = 0.0
    running_vol = 0.0

    for i in range(len(df)):
        if dates[i] != prev_date:
            running_tp_vol = 0.0
            running_vol = 0.0
            prev_date = dates[i]
        running_tp_vol += tp_vol[i]
        running_vol += vol[i]
        if running_vol > 0:
            vwap[i] = running_tp_vol / running_vol
        else:
            vwap[i] = tp[i]

    return vwap


def simulate_trade(df_1m: pd.DataFrame, entry_idx: int,
                   side: str, entry_price: float,
                   sl: float, tp: float,
                   max_bars: int = MAX_BARS) -> dict:
    """Simulate a trade starting from entry_idx."""
    bars = df_1m.iloc[entry_idx:entry_idx + max_bars]
    if len(bars) == 0:
        return {"outcome": "NO_DATA", "bars_held": 0, "exit_price": entry_price}

    for offset, (idx, bar) in enumerate(bars.iterrows()):
        if side == "long":
            if bar["low"] <= sl:
                return {"outcome": "SL", "bars_held": offset + 1,
                        "exit_price": sl, "exit_ts": str(bar["timestamp"])}
            if bar["high"] >= tp:
                return {"outcome": "TP", "bars_held": offset + 1,
                        "exit_price": tp, "exit_ts": str(bar["timestamp"])}
        else:
            if bar["high"] >= sl:
                return {"outcome": "SL", "bars_held": offset + 1,
                        "exit_price": sl, "exit_ts": str(bar["timestamp"])}
            if bar["low"] <= tp:
                return {"outcome": "TP", "bars_held": offset + 1,
                        "exit_price": tp, "exit_ts": str(bar["timestamp"])}

    last_close = float(bars.iloc[-1]["close"])
    return {"outcome": "OPEN", "bars_held": len(bars), "exit_price": last_close,
            "exit_ts": str(bars.iloc[-1]["timestamp"])}


def calc_pnl_pct(outcome, side, entry, exit_price, sl_pct, tp_pct):
    if outcome == "TP":
        return tp_pct
    elif outcome == "SL":
        return -sl_pct
    else:
        if side == "long":
            return (exit_price - entry) / entry * 100
        else:
            return (entry - exit_price) / entry * 100


def direction_allowed(score: int, side: str, strategy: str) -> bool:
    """Check if a trade direction is allowed by the market condition score."""
    if score >= 2:
        # Strong long: skip ALL shorts
        if side == "short":
            return False
        return True
    elif score == 1:
        # Lean long: both sides OK, both strategies OK
        return True
    elif score == 0:
        # Neutral: MR both sides OK, skip Momo
        if strategy == "momo":
            return False
        return True
    elif score == -1:
        # Lean short: both sides OK, both strategies OK
        return True
    else:
        # score <= -2: Strong short: skip ALL longs
        if side == "long":
            return False
        return True


# ── main ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("MARKET CONDITION FILTER BACKTEST")
    print(f"Dataset: {DATASET_DIR}")
    print("=" * 70)

    # ── Step 1: Load BTC data ──
    btc_path = DATASET_DIR / "BTCUSDT_1m.csv"
    if not btc_path.exists():
        print(f"ERROR: BTC data not found at {btc_path}")
        return
    btc_df = pd.read_csv(str(btc_path), parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    print(f"BTC data: {len(btc_df)} bars, {btc_df['timestamp'].min()} to {btc_df['timestamp'].max()}")

    # ── Step 2: Pre-compute BTC SMMA30 and session VWAP ──
    btc_closes = btc_df["close"].values
    btc_smma30 = smma(btc_closes, 30)
    btc_vwap = compute_session_vwap(btc_df)

    # Build BTC timestamp -> index mapping
    btc_ts_to_idx = {}
    for i, ts in enumerate(btc_df["timestamp"].values):
        btc_ts_to_idx[ts] = i

    # ── Step 3: Load ALL symbols ──
    all_files = sorted(DATASET_DIR.glob("*USDT_1m.csv"))
    symbol_data = {}  # sym -> df
    symbol_smma30 = {}  # sym -> np.ndarray of smma30 values

    print(f"\nLoading {len(all_files)} symbol files...")
    for fi, fpath in enumerate(all_files):
        sym = fpath.stem.replace("_1m", "")  # e.g. "BTCUSDT"
        df = pd.read_csv(str(fpath), parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        if len(df) < WARMUP + 100:
            continue
        symbol_data[sym] = df
        symbol_smma30[sym] = smma(df["close"].values, 30)
        if (fi + 1) % 50 == 0:
            print(f"  Loaded {fi + 1}/{len(all_files)} files")

    print(f"  Loaded {len(symbol_data)} symbols with sufficient data")

    # ── Step 4: Pre-compute market breadth for each BTC bar ──
    print("\nPre-computing market breadth...")

    # For each symbol, build a timestamp->smma30 mapping aligned to BTC timeline
    # We need: at each BTC timestamp, what % of symbols have close > smma30

    # Build aligned arrays: for each symbol, map its data to BTC bar indices
    # This avoids per-bar dict lookups
    n_btc = len(btc_df)
    btc_timestamps = btc_df["timestamp"].values

    # For each symbol, create arrays aligned to BTC index
    sym_close_aligned = {}   # sym -> np.ndarray(n_btc) with NaN where no data
    sym_smma30_aligned = {}

    for sym, df in symbol_data.items():
        if sym == "BTCUSDT":
            continue  # skip BTC itself from breadth
        ts_to_idx_sym = {}
        for j in range(len(df)):
            ts_to_idx_sym[df["timestamp"].values[j]] = j

        close_arr = np.full(n_btc, np.nan)
        smma_arr = np.full(n_btc, np.nan)
        sym_closes = df["close"].values
        sym_smma = symbol_smma30[sym]

        for bi in range(n_btc):
            ts = btc_timestamps[bi]
            if ts in ts_to_idx_sym:
                si = ts_to_idx_sym[ts]
                close_arr[bi] = sym_closes[si]
                smma_arr[bi] = sym_smma[si]

        sym_close_aligned[sym] = close_arr
        sym_smma30_aligned[sym] = smma_arr

    # Now compute breadth: at each BTC bar, % of symbols where close > smma30
    syms_list = list(sym_close_aligned.keys())
    n_syms = len(syms_list)

    # Stack into 2D arrays for vectorized computation
    close_matrix = np.array([sym_close_aligned[s] for s in syms_list])  # (n_syms, n_btc)
    smma_matrix = np.array([sym_smma30_aligned[s] for s in syms_list])  # (n_syms, n_btc)

    # Compute: for each bar, count symbols where close > smma30 (ignoring NaN)
    above = close_matrix > smma_matrix  # (n_syms, n_btc) bool
    valid = ~(np.isnan(close_matrix) | np.isnan(smma_matrix))  # (n_syms, n_btc) bool

    above_count = np.sum(above & valid, axis=0)  # (n_btc,)
    valid_count = np.sum(valid, axis=0)           # (n_btc,)
    breadth_pct = np.where(valid_count > 0, above_count / valid_count * 100.0, 50.0)

    print(f"  Breadth computed for {n_btc} BTC bars, {n_syms} symbols")

    # ── Step 5: Compute Combined Direction Score for each BTC bar ──
    market_scores = np.zeros(n_btc, dtype=int)

    for i in range(n_btc):
        score = 0

        # Signal 1: BTC SMMA30 slope (last 120 bars)
        if i >= 120:
            slope = (btc_smma30[i] - btc_smma30[i - 120]) / btc_smma30[i - 120] * 100
            if slope > 0.01:
                score += 1
            elif slope < -0.01:
                score -= 1

        # Signal 2: BTC close vs session VWAP
        if not np.isnan(btc_vwap[i]):
            if btc_closes[i] > btc_vwap[i]:
                score += 1
            elif btc_closes[i] < btc_vwap[i]:
                score -= 1

        # Signal 3: Market breadth
        if breadth_pct[i] > 60:
            score += 1
        elif breadth_pct[i] < 40:
            score -= 1

        market_scores[i] = score

    # Score distribution
    print("\n  Market Condition Score Distribution:")
    for s in range(-3, 4):
        count = np.sum(market_scores == s)
        pct = count / n_btc * 100
        print(f"    Score {s:+d}: {count:>7d} bars ({pct:.1f}%)")

    # ── Step 6: Load strategy configs ──
    mr_cfg = MRSettings()
    momo_cfg = MomoGateSettings.from_json("momo_gate_settings.json")
    print(f"\nMR settings: cooldown={COOLDOWN_BARS}, timeout={MAX_BARS}")
    print(f"Momo settings: loaded from momo_gate_settings.json")

    # ── Step 7: Scan symbols ──
    # Determine scan window: last 7 days
    max_ts = btc_df["timestamp"].max()
    cutoff = max_ts - pd.Timedelta(days=7)
    print(f"\nScan window: {cutoff} to {max_ts} (last 7 days)")

    all_trades = []  # each trade has: strategy, symbol, side, score, filtered, + trade details

    symbols_to_scan = sorted(symbol_data.keys())
    print(f"Scanning {len(symbols_to_scan)} symbols...\n")

    for si, sym in enumerate(symbols_to_scan):
        df = symbol_data[sym]

        # Find cutoff index
        cutoff_mask = df["timestamp"] >= cutoff
        if not cutoff_mask.any():
            continue
        cutoff_idx = cutoff_mask.idxmax()
        scan_start = max(WARMUP, cutoff_idx)
        scan_end = len(df) - 1

        if scan_start >= scan_end:
            continue

        # Build timestamp -> BTC index mapping for this symbol
        sym_ts_to_btc_idx = {}
        for j in range(len(df)):
            ts = df["timestamp"].values[j]
            if ts in btc_ts_to_idx:
                sym_ts_to_btc_idx[j] = btc_ts_to_idx[ts]

        # Prepare momo features (needs DatetimeIndex)
        df_momo = df.copy()
        df_momo = df_momo.set_index("timestamp")
        df_momo.index = pd.DatetimeIndex(df_momo.index)
        try:
            df_momo = prepare_features(df_momo)
        except Exception:
            df_momo = None

        cooldown_mr = -1
        cooldown_momo = -1

        for i in range(scan_start, scan_end):
            # Get market condition score
            btc_idx = sym_ts_to_btc_idx.get(i)
            if btc_idx is None:
                continue
            score = int(market_scores[btc_idx])
            ts_str = str(df.iloc[i]["timestamp"])

            # ── MR scan ──
            if i > cooldown_mr:
                result_mr = check_mr_gates_at_bar(df, i, mr_cfg)
                if result_mr["passed"]:
                    side = result_mr["side"]
                    allowed = direction_allowed(score, side, "mr")

                    trade = simulate_trade(df, i, side,
                                           result_mr["entry"],
                                           result_mr["sl"], result_mr["tp"])
                    pnl = calc_pnl_pct(trade["outcome"], side,
                                       result_mr["entry"], trade["exit_price"],
                                       result_mr["sl_pct"], result_mr["tp_pct"])

                    all_trades.append({
                        "strategy": "MR",
                        "symbol": sym,
                        "side": side,
                        "timestamp": ts_str,
                        "entry": result_mr["entry"],
                        "sl": result_mr["sl"],
                        "tp": result_mr["tp"],
                        "sl_pct": result_mr["sl_pct"],
                        "tp_pct": result_mr["tp_pct"],
                        "rr": result_mr.get("rr", 1.0),
                        "dps_total": result_mr["dps_total"],
                        "dps_confidence": result_mr["dps_confidence"],
                        "outcome": trade["outcome"],
                        "bars_held": trade["bars_held"],
                        "exit_price": trade["exit_price"],
                        "pnl_pct": pnl,
                        "market_score": score,
                        "filtered": not allowed,
                    })
                    cooldown_mr = i + COOLDOWN_BARS

            # ── Momo scan ──
            if i > cooldown_momo and df_momo is not None:
                for momo_side in ("long", "short"):
                    # df_slice: all bars up to and including i
                    # df_momo uses iloc index; the original df index maps 1:1
                    if i + 1 > len(df_momo):
                        continue
                    df_slice = df_momo.iloc[:i + 1]
                    result_momo = check_momo_gates_at_bar(df_slice, momo_side, momo_cfg)
                    if result_momo["passed"]:
                        side = result_momo["side"]
                        allowed = direction_allowed(score, side, "momo")

                        trade = simulate_trade(df, i, side,
                                               result_momo["entry"],
                                               result_momo["sl"], result_momo["tp"])
                        pnl = calc_pnl_pct(trade["outcome"], side,
                                           result_momo["entry"], trade["exit_price"],
                                           result_momo["sl_pct"], result_momo["tp_pct"])

                        all_trades.append({
                            "strategy": "Momo",
                            "symbol": sym,
                            "side": side,
                            "timestamp": ts_str,
                            "entry": result_momo["entry"],
                            "sl": result_momo["sl"],
                            "tp": result_momo["tp"],
                            "sl_pct": result_momo["sl_pct"],
                            "tp_pct": result_momo["tp_pct"],
                            "rr": result_momo.get("rr", 1.0),
                            "dps_total": result_momo["dps_total"],
                            "dps_confidence": result_momo["dps_confidence"],
                            "outcome": trade["outcome"],
                            "bars_held": trade["bars_held"],
                            "exit_price": trade["exit_price"],
                            "pnl_pct": pnl,
                            "market_score": score,
                            "filtered": not allowed,
                        })
                        cooldown_momo = i + COOLDOWN_BARS
                        break  # one momo trade per bar

        if (si + 1) % 20 == 0 or si == len(symbols_to_scan) - 1:
            print(f"  [{si+1}/{len(symbols_to_scan)}] {sym:20s}  total_trades={len(all_trades)}")

    # ── Step 8: Build results DataFrame ──
    trades_df = pd.DataFrame(all_trades)
    if len(trades_df) == 0:
        print("\nNo trades found!")
        return

    # Save all trades
    trades_df.to_csv("backtest_market_condition_trades.csv", index=False)
    print(f"\nSaved {len(trades_df)} trades to backtest_market_condition_trades.csv")

    # ── Step 9: Report ──
    unfiltered_df = trades_df  # all trades (without filter)
    filtered_df = trades_df[~trades_df["filtered"]]  # only allowed trades (with filter)
    removed_df = trades_df[trades_df["filtered"]]  # trades that would be removed

    print(f"\n{'='*70}")
    print("RESULTS COMPARISON")
    print(f"{'='*70}")

    for label, tdf in [("A) WITHOUT market condition filter (all trades)", unfiltered_df),
                        ("B) WITH market condition filter", filtered_df)]:
        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"{'='*70}")

        if len(tdf) == 0:
            print("  No trades.")
            continue

        for strat in ["MR", "Momo"]:
            sdf = tdf[tdf["strategy"] == strat]
            if len(sdf) == 0:
                print(f"\n  --- {strat} --- No trades")
                continue

            tp_n = (sdf["outcome"] == "TP").sum()
            sl_n = (sdf["outcome"] == "SL").sum()
            op_n = (sdf["outcome"] == "OPEN").sum()
            closed = tp_n + sl_n
            wr = tp_n / closed * 100 if closed > 0 else 0
            total_pnl = sdf["pnl_pct"].sum()
            avg_pnl = sdf["pnl_pct"].mean()

            print(f"\n  --- {strat} ---")
            print(f"    Total trades:   {len(sdf)}")
            print(f"    TP / SL / TO:   {tp_n} / {sl_n} / {op_n}")
            print(f"    Win Rate:       {wr:.1f}%")
            print(f"    Total PnL:      {total_pnl:.2f}%")
            print(f"    Avg PnL/trade:  {avg_pnl:.4f}%")

            for side in ["long", "short"]:
                sub = sdf[sdf["side"] == side]
                if len(sub) == 0:
                    continue
                tp = (sub["outcome"] == "TP").sum()
                sl = (sub["outcome"] == "SL").sum()
                wr_s = tp / (tp + sl) * 100 if (tp + sl) > 0 else 0
                print(f"      {side:5s}: {len(sub)} trades, {tp} TP / {sl} SL, "
                      f"WR={wr_s:.1f}%, PnL={sub['pnl_pct'].sum():.2f}%")

    # ── Filtered trades analysis ──
    print(f"\n{'='*70}")
    print("FILTERED TRADES ANALYSIS (trades removed by market condition filter)")
    print(f"{'='*70}")

    if len(removed_df) > 0:
        print(f"\n  Total filtered out: {len(removed_df)}")

        for strat in ["MR", "Momo"]:
            sdf = removed_df[removed_df["strategy"] == strat]
            if len(sdf) == 0:
                continue
            tp_n = (sdf["outcome"] == "TP").sum()
            sl_n = (sdf["outcome"] == "SL").sum()
            op_n = (sdf["outcome"] == "OPEN").sum()
            closed = tp_n + sl_n
            wr = tp_n / closed * 100 if closed > 0 else 0
            total_pnl = sdf["pnl_pct"].sum()
            print(f"\n  {strat} filtered: {len(sdf)} trades")
            print(f"    TP / SL / TO:   {tp_n} / {sl_n} / {op_n}")
            print(f"    Win Rate:       {wr:.1f}%")
            print(f"    Total PnL:      {total_pnl:.2f}%")
            print(f"    (Negative PnL = filter correctly removed losers)")

            for side in ["long", "short"]:
                sub = sdf[sdf["side"] == side]
                if len(sub) == 0:
                    continue
                tp = (sub["outcome"] == "TP").sum()
                sl = (sub["outcome"] == "SL").sum()
                wr_s = tp / (tp + sl) * 100 if (tp + sl) > 0 else 0
                print(f"      {side:5s}: {len(sub)} trades, {tp} TP / {sl} SL, "
                      f"WR={wr_s:.1f}%, PnL={sub['pnl_pct'].sum():.2f}%")
    else:
        print("  No trades were filtered.")

    # ── PnL by score bucket ──
    print(f"\n{'='*70}")
    print("PnL BY MARKET CONDITION SCORE BUCKET")
    print(f"{'='*70}")

    for s in range(-3, 4):
        bucket = trades_df[trades_df["market_score"] == s]
        if len(bucket) == 0:
            print(f"  Score {s:+d}: no trades")
            continue
        tp_n = (bucket["outcome"] == "TP").sum()
        sl_n = (bucket["outcome"] == "SL").sum()
        closed = tp_n + sl_n
        wr = tp_n / closed * 100 if closed > 0 else 0
        pnl = bucket["pnl_pct"].sum()
        # Break down by strategy
        mr_sub = bucket[bucket["strategy"] == "MR"]
        mo_sub = bucket[bucket["strategy"] == "Momo"]
        mr_pnl = mr_sub["pnl_pct"].sum() if len(mr_sub) > 0 else 0
        mo_pnl = mo_sub["pnl_pct"].sum() if len(mo_sub) > 0 else 0
        print(f"  Score {s:+d}: {len(bucket):>4d} trades, WR={wr:.1f}%, "
              f"PnL={pnl:+.2f}% (MR={mr_pnl:+.2f}%, Momo={mo_pnl:+.2f}%)")

    # ── Summary verdict ──
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    all_pnl = trades_df["pnl_pct"].sum()
    filt_pnl = filtered_df["pnl_pct"].sum() if len(filtered_df) > 0 else 0
    all_n = len(trades_df)
    filt_n = len(filtered_df)
    rem_n = len(removed_df) if len(removed_df) > 0 else 0

    all_closed = ((trades_df["outcome"] == "TP") | (trades_df["outcome"] == "SL")).sum()
    all_tp = (trades_df["outcome"] == "TP").sum()
    all_wr = all_tp / all_closed * 100 if all_closed > 0 else 0

    filt_closed = ((filtered_df["outcome"] == "TP") | (filtered_df["outcome"] == "SL")).sum() if len(filtered_df) > 0 else 0
    filt_tp = (filtered_df["outcome"] == "TP").sum() if len(filtered_df) > 0 else 0
    filt_wr = filt_tp / filt_closed * 100 if filt_closed > 0 else 0

    print(f"  Without filter: {all_n} trades, WR={all_wr:.1f}%, PnL={all_pnl:+.2f}%")
    print(f"  With filter:    {filt_n} trades, WR={filt_wr:.1f}%, PnL={filt_pnl:+.2f}%")
    print(f"  Trades removed: {rem_n}")
    improvement = filt_pnl - all_pnl
    print(f"  PnL difference: {improvement:+.2f}% "
          f"({'Filter helps' if improvement > 0 else 'Filter hurts'})")
    if all_n > 0 and filt_n > 0:
        avg_all = all_pnl / all_n
        avg_filt = filt_pnl / filt_n
        print(f"  Avg PnL/trade:  {avg_all:.4f}% (no filter) vs {avg_filt:.4f}% (with filter)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
