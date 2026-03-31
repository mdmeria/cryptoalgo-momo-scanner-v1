#!/usr/bin/env python3
"""Combined parallel backtest: MR v2 + Momo v1 (no quality filter) + Momo v2 (with quality filter)."""
import sys, os, time, pandas as pd, numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATASET_DIR = Path("datasets/binance_futures_1m")
MAX_BARS = 120
LIMIT_EXPIRY = 3
MIN_VOL_5M_USD = 500_000  # minimum 5-min volume in USD to enter
MARKET_EVAL_INTERVAL = 120  # evaluate market conditions every 120 bars
OUTPUT_CSV = "baseline_binance_results.csv"
LOG_FILE = "baseline_binance_log.txt"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def precompute_market_conditions(dataset_dir, syms):
    """
    Precompute market condition scores every 120 bars.
    Returns dict: {bar_index: score} based on BTC timestamps.
    Also returns a function to look up score at any timestamp.
    """
    # Check for cached market conditions
    cache_path = dataset_dir / "market_conditions_cache.csv"
    if cache_path.exists():
        cache_df = pd.read_csv(str(cache_path), parse_dates=["timestamp"])
        score_timestamps = cache_df["timestamp"].values
        score_values = cache_df["score"].tolist()
        scores = {i: s for i, s in enumerate(score_values)}
        score_data = (score_timestamps, score_values)
        return scores, score_data

    # Load BTC data
    btc_path = dataset_dir / "BTCUSDT_1m.csv"
    if not btc_path.exists():
        return None, None

    btc_df = pd.read_csv(str(btc_path), parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    btc_closes = btc_df["close"].values.astype(float)
    btc_highs = btc_df["high"].values.astype(float)
    btc_lows = btc_df["low"].values.astype(float)
    # Binance futures volume is in quote (USD) — derive base volume for VWAP
    btc_volumes_usd = btc_df["volume"].values.astype(float)
    btc_volumes = btc_volumes_usd / btc_closes  # approximate base volume
    btc_timestamps = btc_df["timestamp"].values

    # Precompute BTC SMMA30
    btc_smma30 = pd.Series(btc_closes).ewm(alpha=1.0/30, adjust=False).mean().values

    # Precompute SMMA30 for all symbols (for breadth) — numpy arrays for speed
    sym_above_smma = {}  # {sym: (timestamps_array, above_array)}
    for sym in syms:
        fpath = dataset_dir / f"{sym}_1m.csv"
        if not fpath.exists() or sym == "BTCUSDT":
            continue
        try:
            df = pd.read_csv(str(fpath), parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            closes = df["close"].values.astype(float)
            smma30 = pd.Series(closes).ewm(alpha=1.0/30, adjust=False).mean().values
            above = closes > smma30
            sym_above_smma[sym] = (df["timestamp"].values, above)
        except Exception:
            continue

    # Compute score every 120 bars (using BTC timestamps as reference)
    scores = {}  # {btc_bar_idx: score}
    n = len(btc_closes)

    for i in range(150, n, MARKET_EVAL_INTERVAL):
        # BTC SMMA30 slope (120-bar)
        if i >= 121:
            slope_pct = (btc_smma30[i] - btc_smma30[i - 120]) / btc_smma30[i - 120] * 100
            btc_smma_signal = 1 if slope_pct > 0.01 else (-1 if slope_pct < -0.01 else 0)
        else:
            btc_smma_signal = 0

        # BTC vs session VWAP
        ts = pd.Timestamp(btc_timestamps[i])
        day_str = str(ts)[:10]
        # Use numpy searchsorted on raw timestamp array to avoid tz issues
        day_start_np = np.datetime64(day_str)
        day_start_idx = int(np.searchsorted(btc_timestamps, day_start_np))
        if i - day_start_idx > 5:
            tp_arr = (btc_highs[day_start_idx:i+1] + btc_lows[day_start_idx:i+1] + btc_closes[day_start_idx:i+1]) / 3
            vol_arr = btc_volumes[day_start_idx:i+1]
            cum_tp_vol = np.cumsum(tp_arr * vol_arr)
            cum_vol = np.cumsum(vol_arr)
            cum_vol[cum_vol == 0] = np.nan
            vwap = cum_tp_vol[-1] / cum_vol[-1] if cum_vol[-1] > 0 else btc_closes[i]
            btc_vwap_signal = 1 if btc_closes[i] > vwap else -1
        else:
            btc_vwap_signal = 0

        # Breadth: % of coins above SMMA30 at this BTC timestamp
        current_ts = btc_timestamps[i]
        above = 0
        total = 0
        for sym, (sym_ts, sym_above) in sym_above_smma.items():
            # Find nearest timestamp <= current_ts using numpy
            idx = int(np.searchsorted(sym_ts, current_ts, side="right")) - 1
            if 0 <= idx < len(sym_above):
                if sym_above[idx]:
                    above += 1
                total += 1
        if total > 10:
            breadth_pct = above / total * 100
            breadth_signal = 1 if breadth_pct > 60 else (-1 if breadth_pct < 40 else 0)
        else:
            breadth_signal = 0

        score = btc_smma_signal + btc_vwap_signal + breadth_signal
        scores[i] = score

    # Build serializable lookup data
    score_indices = sorted(scores.keys())
    score_values = [scores[k] for k in score_indices]
    score_timestamps = [btc_timestamps[k] for k in score_indices]

    # Save cache
    cache_df = pd.DataFrame({"timestamp": score_timestamps, "score": score_values})
    cache_df.to_csv(str(cache_path), index=False)

    # Return as tuple of lists (pickle-safe for multiprocessing)
    score_data = (score_timestamps, score_values)
    return scores, score_data


def is_market_allowed(score, strategy, side):
    """Check if trade is allowed given market condition score."""
    if strategy == "mr_v2":
        # MR: no market filter (SMA alignment handles direction)
        # mkt_score logged for post-analysis
        return True
    elif strategy in ("momo_v1", "momo_v2"):
        # Momo: only trade with strong market conviction
        # Longs only at score >= +2, shorts only at score <= -2
        if side == "long" and score >= 2:
            return True
        if side == "short" and score <= -2:
            return True
        return False
    return True


def simulate_momo_live_entry(df, signal_idx, side, entry, tp, sl, tp_pct, sl_pct, max_bars=480):
    """
    Simulate momo trade with live dummy entry criteria:
    1. 2-bar confirmation (next 2 bars close on correct side)
    2. 0.75R cancel (price moved too far)
    3. Limit order at entry price, 10-min expiry
    4. Trail SL: move to breakeven+0.1R at 0.9R or after 60 bars onside
    Returns (outcome, pnl, reason)
    """
    n = len(df)

    # Phase 1: Wait for price to cross entry level, then 2-bar confirmation
    # Entry = high of signal bar (long) or low of signal bar (short)
    # Next bars must cross that level and close beyond it for 2 consecutive bars
    confirm_count = 0
    confirm_bar = None
    for cb in range(signal_idx + 1, min(signal_idx + 15, n)):  # look up to 15 bars
        close = float(df.iloc[cb]["close"])
        if side == "long" and close > entry:
            confirm_count += 1
        elif side == "short" and close < entry:
            confirm_count += 1
        else:
            confirm_count = 0  # reset on failure

        if confirm_count >= 2:
            confirm_bar = cb + 1
            break

    if confirm_bar is None or confirm_bar >= n:
        return "MISSED", 0, "confirm_failed"
    if confirm_bar >= n:
        return "MISSED", 0, "no_data"

    cancel_075r = entry + (tp - entry) * 0.75 if side == "long" else entry - (entry - tp) * 0.75

    fill_bar = None
    for fb in range(confirm_bar, min(confirm_bar + 10, n)):
        h = float(df.iloc[fb]["high"])
        l = float(df.iloc[fb]["low"])
        if side == "long" and h >= cancel_075r:
            return "MISSED", 0, "075r_cancel"
        if side == "short" and l <= cancel_075r:
            return "MISSED", 0, "075r_cancel"
        if side == "long" and l <= entry:
            fill_bar = fb; break
        if side == "short" and h >= entry:
            fill_bar = fb; break

    if fill_bar is None:
        return "MISSED", 0, "expired_10m"

    # Phase 4: trade filled — simulate with trail SL
    trail_trigger = entry + (tp - entry) * 0.9 if side == "long" else entry - (entry - tp) * 0.9
    trail_sl = entry * 1.001 if side == "long" else entry * 0.999
    current_sl = sl
    trailed = False
    bars_onside = 0

    for tb in range(fill_bar, min(fill_bar + max_bars, n)):
        h = float(df.iloc[tb]["high"])
        l = float(df.iloc[tb]["low"])
        c = float(df.iloc[tb]["close"])

        if side == "long" and c > entry:
            bars_onside += 1
        elif side == "short" and c < entry:
            bars_onside += 1

        if not trailed:
            if side == "long" and (h >= trail_trigger or bars_onside >= 60):
                current_sl = trail_sl; trailed = True
            elif side == "short" and (l <= trail_trigger or bars_onside >= 60):
                current_sl = trail_sl; trailed = True

        if side == "long":
            if l <= current_sl:
                outcome = "TRAIL_SL" if trailed else "SL"
                pnl = 0.1 * sl_pct if trailed else -sl_pct
                return outcome, round(pnl, 3), "filled"
            if h >= tp:
                return "TP", tp_pct, "filled"
        else:
            if h >= current_sl:
                outcome = "TRAIL_SL" if trailed else "SL"
                pnl = 0.1 * sl_pct if trailed else -sl_pct
                return outcome, round(pnl, 3), "filled"
            if l <= tp:
                return "TP", tp_pct, "filled"

    # Timeout
    lc = float(df.iloc[min(fill_bar + max_bars, n) - 1]["close"])
    pnl = (lc - entry) / entry * 100 if side == "long" else (entry - lc) / entry * 100
    return "OPEN", round(pnl, 3), "filled"


def simulate_trade(df, fill_bar, side, entry, tp, sl, max_bars=MAX_BARS):
    """Simulate a trade from fill_bar. Returns outcome and pnl."""
    bars = df.iloc[fill_bar:fill_bar + max_bars]
    outcome = "OPEN"
    for _, bar in bars.iterrows():
        if side == "long":
            if bar["low"] <= sl: outcome = "SL"; break
            if bar["high"] >= tp: outcome = "TP"; break
        else:
            if bar["high"] >= sl: outcome = "SL"; break
            if bar["low"] <= tp: outcome = "TP"; break
    return outcome


def process_symbol(sym, dataset_dir, market_score_data=None):
    """Process one symbol for all 3 strategies."""
    # Reconstruct market score lookup from serializable data
    market_score_fn = None
    if market_score_data is not None:
        score_timestamps, score_values = market_score_data
        def market_score_fn(timestamp):
            idx = np.searchsorted(score_timestamps, np.datetime64(timestamp), side="right") - 1
            return score_values[idx] if idx >= 0 else 0
    from scan_mean_reversion import MRSettings, check_mr_gates_at_bar
    from backtest_momo_vwap_grind15_full import (
        GateSettings as MomoGateSettings,
        check_momo_gates_at_bar,
        prepare_features as prepare_momo_features,
    )
    from strategies import _momo_quality_filter

    fpath = dataset_dir / f"{sym}_1m.csv"
    df = pd.read_csv(str(fpath), parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    mr_cfg = MRSettings()
    momo_cfg = MomoGateSettings.from_json("momo_gate_settings.json") if Path("momo_gate_settings.json").exists() else MomoGateSettings()
    mr_warmup = max(mr_cfg.range_max_bars, mr_cfg.noise_lookback_bars, 720)
    momo_warmup = 150

    if len(df) < max(mr_warmup, momo_warmup) + MAX_BARS + 10:
        return []

    scan_start = max(mr_warmup, momo_warmup)
    scan_end = len(df) - MAX_BARS - LIMIT_EXPIRY

    # Prepare momo features once
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    df_indexed = df_sorted.set_index("timestamp").copy()
    df_indexed.index = pd.to_datetime(df_indexed.index, utc=True)
    try:
        df_prepped = prepare_momo_features(df_indexed)
    except Exception:
        df_prepped = None

    trades = []
    cooldown_mr = -1
    cooldown_momo_v1 = -1
    cooldown_momo_v2 = -1

    # Precompute volume arrays for fast 5-min lookback
    # Note: Binance futures volume is already in quote currency (USD)
    vol_usd = df["volume"].values.astype(float)

    for i in range(scan_start, scan_end):
        # --- Volume pre-filter: skip if last 5 min < $500k ---
        if MIN_VOL_5M_USD > 0 and i >= 5:
            vol_5m = vol_usd[i-5:i].sum()
            if vol_5m < MIN_VOL_5M_USD:
                continue

        # --- Get market condition score ---
        mkt_score = 0
        if market_score_fn is not None:
            mkt_score = market_score_fn(df.iloc[i]["timestamp"])

        # --- MR v2 ---
        if i > cooldown_mr:
            r = check_mr_gates_at_bar(df, i, mr_cfg)
            if r["passed"] and is_market_allowed(mkt_score, "mr_v2", r["side"]):
                side, level, tp, sl = r["side"], r["entry"], r["tp"], r["sl"]
                mr_zct = {
                    "dur_score": r.get("dps_v1_duration", ""),
                    "dur_label": r.get("dps_v1_label", ""),
                    "approach_score": r.get("dps_v2_approach", ""),
                    "approach_label": r.get("dps_v2_label", ""),
                    "vol_score": r.get("dps_v3_volume", ""),
                    "vol_label": r.get("dps_v3_vol_trend", ""),
                    "range_width": r.get("range_width_pct", ""),
                    "range_hrs": r.get("range_duration_hrs", ""),
                }
                fill_bar = None
                for fb in range(i + 1, min(i + 1 + LIMIT_EXPIRY, len(df))):
                    if side == "long" and df.iloc[fb]["low"] <= level:
                        fill_bar = fb; break
                    elif side == "short" and df.iloc[fb]["high"] >= level:
                        fill_bar = fb; break

                if fill_bar is None:
                    trades.append({
                        "symbol": sym, "ts": str(df.iloc[i + 1]["timestamp"]),
                        "strategy": "mr_v2", "side": side,
                        "entry": round(level, 8), "tp": round(tp, 8), "sl": round(sl, 8),
                        "tp_pct": round(r["tp_pct"], 3), "sl_pct": round(r["sl_pct"], 3),
                        "dps": r["dps_total"], "conf": r["dps_confidence"],
                        "touches": r["touches"], "outcome": "MISSED", "pnl": 0,
                        "mkt_score": mkt_score, **mr_zct,
                    })
                else:
                    outcome = simulate_trade(df, fill_bar, side, level, tp, sl)
                    if outcome == "TP": pnl = r["tp_pct"]
                    elif outcome == "SL": pnl = -r["sl_pct"]
                    else:
                        lc = float(df.iloc[min(fill_bar + MAX_BARS, len(df)) - 1]["close"])
                        pnl = (lc - level) / level * 100 if side == "long" else (level - lc) / level * 100
                    trades.append({
                        "symbol": sym, "ts": str(df.iloc[fill_bar]["timestamp"]),
                        "strategy": "mr_v2", "side": side,
                        "entry": round(level, 8), "tp": round(tp, 8), "sl": round(sl, 8),
                        "tp_pct": round(r["tp_pct"], 3), "sl_pct": round(r["sl_pct"], 3),
                        "dps": r["dps_total"], "conf": r["dps_confidence"],
                        "touches": r["touches"], "outcome": outcome, "pnl": round(pnl, 3),
                        "mkt_score": mkt_score, **mr_zct,
                    })
                cooldown_mr = i + mr_cfg.cooldown_bars

        # --- Momo v1 + Momo v2 (both with live entry criteria) ---
        if df_prepped is not None and (i > cooldown_momo_v1 or i > cooldown_momo_v2):
            for direction in ["long", "short"]:
                # Slice up to current bar
                ts_at_i = df.iloc[i]["timestamp"]
                slice_end = df_prepped.index.searchsorted(ts_at_i, side="right")
                if slice_end < 150:
                    continue
                df_slice = df_prepped.iloc[max(0, slice_end - 500):slice_end]
                if len(df_slice) < 150:
                    continue

                result = check_momo_gates_at_bar(df_slice, direction, momo_cfg)
                if not result["passed"]:
                    continue

                # Last 15 bars grind filter
                last15 = df.iloc[max(0, i-14):i+1]
                if len(last15) >= 15:
                    l15_c = last15["close"].values.astype(float)
                    l15_o = last15["open"].values.astype(float)
                    l15_h = last15["high"].values.astype(float)
                    l15_l = last15["low"].values.astype(float)

                    # 1. Max bar < 2x median of last 15 bars (uniform size, no spikes)
                    bar_pcts = (l15_h - l15_l) / l15_l * 100
                    median_15 = np.median(bar_pcts)
                    if median_15 > 0 and bar_pcts.max() > median_15 * 2:
                        continue

                    # 2. Each 5-bar segment must be net positive in trade direction
                    grind_ok = True
                    for seg_start in [0, 5, 10]:
                        seg_c = l15_c[seg_start:seg_start+5]
                        if direction == "long":
                            net_positive = seg_c[-1] > seg_c[0]
                        else:
                            net_positive = seg_c[-1] < seg_c[0]
                        if not net_positive:
                            grind_ok = False
                            break
                    if not grind_ok:
                        continue

                # 2h volatility proxy: avg bar range as % of price
                vol_2h_bars = df.iloc[max(0, i-119):i+1]
                avg_bar_range_pct = ((vol_2h_bars["high"] - vol_2h_bars["low"]) / vol_2h_bars["low"] * 100).mean() if len(vol_2h_bars) > 0 else 0

                # Market condition filter for momo
                if not is_market_allowed(mkt_score, "momo_v1", direction):
                    continue

                # Entry level: high of signal bar for longs, low for shorts
                signal_bar = df.iloc[i]
                entry_level = float(signal_bar["high"]) if direction == "long" else float(signal_bar["low"])
                entry = round(entry_level, 8)
                # Recompute SL/TP from the correct entry level
                if direction == "long":
                    sl_val = round(entry_level * (1 - result["sl_pct"] / 100), 8)
                    tp_val = round(entry_level * (1 + result["tp_pct"] / 100), 8)
                else:
                    sl_val = round(entry_level * (1 + result["sl_pct"] / 100), 8)
                    tp_val = round(entry_level * (1 - result["tp_pct"] / 100), 8)
                tp_pct = result["tp_pct"]
                sl_pct = result["sl_pct"]

                # ZCT component fields from gate result
                momo_zct = {
                    "dur_score": result.get("dps_v1", ""),
                    "approach_score": result.get("dps_v2", ""),
                    "approach_label": result.get("approach", ""),
                    "vol_score": result.get("dps_v3", ""),
                    "vol_label": result.get("vol_trend", ""),
                }

                # Momo v1: gates only + live entry
                if i > cooldown_momo_v1:
                    outcome, pnl, reason = simulate_momo_live_entry(
                        df, i, direction, entry_level, tp_val, sl_val,
                        tp_pct, sl_pct)
                    trades.append({
                        "symbol": sym, "ts": str(df.iloc[i]["timestamp"]),
                        "strategy": "momo_v1", "side": direction,
                        "entry": entry, "tp": tp_val, "sl": sl_val,
                        "tp_pct": round(tp_pct, 3), "sl_pct": round(sl_pct, 3),
                        "dps": result["dps_total"], "conf": result["dps_confidence"],
                        "touches": 0, "outcome": outcome, "pnl": round(pnl, 3),
                        "mkt_score": mkt_score, "volatility_2h": round(avg_bar_range_pct, 4), **momo_zct,
                    })
                    cooldown_momo_v1 = i + 30

                # Momo v2: gates + quality filter + live entry
                if i > cooldown_momo_v2:
                    qf = _momo_quality_filter(df_sorted.iloc[:i + 1], direction)
                    if qf is not None:
                        dps_t = qf["dps_total_quality"]
                        conf = "max" if dps_t >= 6 else ("high" if dps_t >= 4 else ("low" if dps_t >= 3 else "avoid"))
                        if dps_t >= 3:
                            outcome, pnl, reason = simulate_momo_live_entry(
                                df, i, direction, entry_level, tp_val, sl_val,
                                tp_pct, sl_pct)
                            trades.append({
                                "symbol": sym, "ts": str(df.iloc[i]["timestamp"]),
                                "strategy": "momo_v2", "side": direction,
                                "entry": entry, "tp": tp_val, "sl": sl_val,
                                "tp_pct": round(tp_pct, 3), "sl_pct": round(sl_pct, 3),
                                "dps": dps_t, "conf": conf,
                                "touches": 0, "outcome": outcome, "pnl": round(pnl, 3),
                                "mkt_score": mkt_score, **momo_zct,
                            })
                            cooldown_momo_v2 = i + 30

    # --- ZCT Momo v3 (separate signal detection + gates) ---
    try:
        from run_zct_momo_backtest import process_symbol as zct_momo_process
        v3_trades = zct_momo_process(sym, dataset_dir, market_score_data)
        if v3_trades:
            trades.extend(v3_trades)
    except Exception:
        pass

    return trades


def main():
    with open(LOG_FILE, "w") as f:
        f.write("")

    syms = sorted([f.replace("_1m.csv", "") for f in os.listdir(DATASET_DIR) if f.endswith("_1m.csv")])
    log(f"Starting: {len(syms)} symbols, dataset={DATASET_DIR}")

    # Precompute market conditions
    log("Precomputing market conditions (BTC + breadth every 120 bars)...")
    market_scores, market_score_data = precompute_market_conditions(DATASET_DIR, syms)
    if market_scores:
        log(f"  Market conditions: {len(market_scores)} checkpoints computed")
    else:
        log("  WARNING: No BTC data found — market conditions disabled")
        market_score_data = None

    log(f"Volume filter: ${MIN_VOL_5M_USD:,} 5-min minimum")

    workers = 10
    log(f"Using {workers} parallel workers")
    log(f"Strategies: MR v2, Momo v1 (no quality filter), Momo v2 (with quality filter)")

    all_trades = []
    done = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_symbol, sym, DATASET_DIR, market_score_data): sym for sym in syms}
        for future in as_completed(futures):
            sym = futures[future]
            done += 1
            try:
                trades = future.result()
                if trades:
                    all_trades.extend(trades)
                    tdf = pd.DataFrame(all_trades)
                    tdf.to_csv(OUTPUT_CSV, index=False)
                    for t in trades:
                        log(f"  {t['strategy']:8s} {sym} {t['ts']} {t['side']} entry={t['entry']} outcome={t['outcome']} pnl={t['pnl']}")
            except Exception as e:
                log(f"  ERROR {sym}: {e}")

            if done % 25 == 0 or done == len(syms):
                elapsed = time.time() - t0
                log(f"Progress: {done}/{len(syms)} symbols, {len(all_trades)} trades, {elapsed:.0f}s elapsed")

    tdf = pd.DataFrame(all_trades)
    tdf.to_csv(OUTPUT_CSV, index=False)

    elapsed = time.time() - t0
    log(f"\nDone in {elapsed:.0f}s")

    if len(tdf) == 0:
        log("No trades found!")
        return

    # Summary per strategy
    for strat in ["mr_v2", "momo_v1", "momo_v2", "zct_momo"]:
        sdf = tdf[tdf["strategy"] == strat]
        filled = sdf[sdf["outcome"] != "MISSED"]
        if len(filled) == 0:
            log(f"\n--- {strat} ---\n  No filled trades.")
            continue
        tp_n = (filled["outcome"] == "TP").sum()
        sl_n = (filled["outcome"] == "SL").sum()
        op_n = (filled["outcome"] == "OPEN").sum()
        wr = tp_n / (tp_n + sl_n) * 100 if (tp_n + sl_n) > 0 else 0
        log(f"\n--- {strat} ---")
        log(f"  Signals: {len(sdf)} | Filled: {len(filled)} | Missed: {len(sdf) - len(filled)}")
        log(f"  Filled: {len(filled)} | TP={tp_n} SL={sl_n} OPEN={op_n} | WR={wr:.1f}% | PnL={filled['pnl'].sum():+.2f}%")
        for side in ["long", "short"]:
            sub = filled[filled["side"] == side]
            if len(sub) == 0: continue
            tp = (sub["outcome"] == "TP").sum()
            sl = (sub["outcome"] == "SL").sum()
            wr_s = tp / (tp + sl) * 100 if (tp + sl) > 0 else 0
            log(f"    {side:5s}: {len(sub)} | WR={wr_s:.1f}% | PnL={sub['pnl'].sum():+.2f}%")

    # Combined
    filled_all = tdf[tdf["outcome"] != "MISSED"]
    tp_n = (filled_all["outcome"] == "TP").sum()
    sl_n = (filled_all["outcome"] == "SL").sum()
    wr = tp_n / (tp_n + sl_n) * 100 if (tp_n + sl_n) > 0 else 0
    log(f"\n--- COMBINED ---")
    log(f"  Filled: {len(filled_all)} | TP={tp_n} SL={sl_n} | WR={wr:.1f}% | PnL={filled_all['pnl'].sum():+.2f}%")

    log(f"\nResults saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
