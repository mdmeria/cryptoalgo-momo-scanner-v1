#!/usr/bin/env python3
"""
ZCT Momentum Strategy Backtest — v3 (correct flow)

ZCT Momo flow:
  1. Staircase regime builds: 2h+ of price grinding in one direction
  2. Price BREAKS a key level (6h high for longs, 6h low for shorts)
  3. 2 candle closes beyond the level = confirmation
  4. Limit order at the broken level = retest entry

Signal = the breakout bar (close beyond 6h extreme)
Gates check the quality of price action BEFORE the breakout:
  - Duration: 2h+ on correct side of SMMA30
  - SMMA30 trending (not flat/sideways)
  - Low noise (SMMA30 crosses <= 6)
  - Grind approach (no spike into the level)
  - Volume: nama30 not falling
"""
import sys, os, time, math
import pandas as pd, numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATASET_DIR = Path("datasets/binance_futures_1m")
MAX_BARS = 120
MIN_VOL_5M_USD = 500_000
OUTPUT_CSV = "zct_momo_results_v5.csv"
LOG_FILE = "zct_momo_log_v5.txt"
SCAN_FROM = None  # None = full dataset

# Symbols too slow/large for momo — exclude from scanning
EXCLUDED_SYMBOLS = {
    "BTCUSDT", "BNBUSDT", "ETHUSDT", "XRPUSDT", "LTCUSDT",
    "PAXGUSDT", "XAUUSDT", "XAGUSDT", "XPTUSDT", "BCHUSDT", "YFIUSDT",
}

MAX_PRICE = 100.0      # skip coins priced above $100
MIN_2H_MOVE_PCT = 0.0  # disabled for now


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ═══════════════════════════════════════════════════════════════════════════
#  INDICATORS
# ═══════════════════════════════════════════════════════════════════════════

def _nama(arr, seed_len=30):
    """Noise-adjusted moving average."""
    out = np.full(len(arr), np.nan)
    if len(arr) < seed_len:
        return out
    seed = np.nanmean(arr[:seed_len])
    if np.isnan(seed) or seed <= 0:
        seed = 1.0
    out[seed_len - 1] = seed
    for i in range(seed_len, len(arr)):
        prev = out[i - 1]
        diff = abs(arr[i] - prev)
        noise = diff / prev if prev > 0 else 1.0
        alpha = noise / (noise + 1.0)
        out[i] = prev + alpha * (arr[i] - prev)
    return out


def _count_smma_crosses(closes, smma, min_bars_away=5):
    """Count debounced SMMA30 crosses."""
    above = closes > smma
    crossovers = 0
    confirmed_side = above[0]
    tentative_side = None
    tentative_run = 0
    for i in range(1, len(above)):
        if tentative_side is None:
            if above[i] != confirmed_side:
                tentative_side = above[i]
                tentative_run = 1
        else:
            if above[i] == tentative_side:
                tentative_run += 1
                if tentative_run >= min_bars_away:
                    crossovers += 1
                    confirmed_side = tentative_side
                    tentative_side = None
            else:
                tentative_side = None
                tentative_run = 0
                if above[i] != confirmed_side:
                    tentative_side = above[i]
                    tentative_run = 1
    return crossovers


# ═══════════════════════════════════════════════════════════════════════════
#  SL / TP (ZCT: 2nd swing low + buffer)
# ═══════════════════════════════════════════════════════════════════════════

def find_second_swing(arr, entry, is_long, lookback=20, buffer_pct=0.15):
    tail = arr[-lookback:]
    n = len(tail)
    swings = []
    for i in range(1, n - 1):
        if is_long:
            if tail[i] < tail[i - 1] and tail[i] < tail[i + 1] and tail[i] < entry:
                swings.append(tail[i])
        else:
            if tail[i] > tail[i - 1] and tail[i] > tail[i + 1] and tail[i] > entry:
                swings.append(tail[i])
    if not swings:
        return None
    if is_long:
        swings.sort(reverse=True)
    else:
        swings.sort()
    ref = swings[1] if len(swings) >= 2 else swings[0]
    if is_long:
        return ref * (1.0 - buffer_pct / 100.0)
    else:
        return ref * (1.0 + buffer_pct / 100.0)


def compute_sl_tp(h_arr, l_arr, direction, entry_price,
                  min_sl=1.0, max_sl=3.0, rr=1.1, max_tp=4.0):
    is_long = direction == "long"
    swing_ref = find_second_swing(
        l_arr if is_long else h_arr, entry_price, is_long)
    if swing_ref is not None:
        if is_long:
            struct_pct = (entry_price - swing_ref) / max(entry_price, 1e-9) * 100
        else:
            struct_pct = (swing_ref - entry_price) / max(entry_price, 1e-9) * 100
    else:
        struct_pct = min_sl
    sl_pct = max(struct_pct, min_sl)
    sl_pct = min(sl_pct, max_sl)
    tp_pct = min(sl_pct * rr, max_tp)
    if is_long:
        sl = entry_price * (1.0 - sl_pct / 100.0)
        tp = entry_price * (1.0 + tp_pct / 100.0)
    else:
        sl = entry_price * (1.0 + sl_pct / 100.0)
        tp = entry_price * (1.0 - tp_pct / 100.0)
    return sl, tp, sl_pct, tp_pct


# ═══════════════════════════════════════════════════════════════════════════
#  LIVE ENTRY SIMULATION
# ═══════════════════════════════════════════════════════════════════════════

def simulate_momo_live_entry(df_c, df_h, df_l, signal_idx, side,
                             level, tp, sl, tp_pct, sl_pct, max_bars=480):
    """
    ZCT momo entry:
    1. Signal = breakout bar (close beyond level)
    2. Next 2 bars must also close beyond level = confirmation
    3. Then limit order AT the level, wait for price to retest it
    4. 0.75R cancel if price runs too far before retest
    5. Trail SL to BE+0.1R at 0.9R or after 60 bars onside
    """
    n = len(df_c)

    # Phase 1: 2-bar confirmation (next 2 bars close beyond the level)
    # If after 1 close beyond, the next candle closes back through → recycled level
    confirm_count = 0
    confirm_bar = None
    for cb in range(signal_idx + 1, min(signal_idx + 15, n)):
        if side == "long" and df_c[cb] > level:
            confirm_count += 1
        elif side == "short" and df_c[cb] < level:
            confirm_count += 1
        else:
            if confirm_count >= 1:
                # Had 1 close beyond but next candle closed back through
                # Level is weak/recycled — abandon entirely
                return "MISSED", 0, "recycled_level"
            confirm_count = 0
        if confirm_count >= 2:
            confirm_bar = cb + 1
            break

    if confirm_bar is None or confirm_bar >= n:
        return "MISSED", 0, "confirm_failed"

    # Phase 2: limit order at the level, 10-bar expiry
    cancel_075r = level + (tp - level) * 0.75 if side == "long" else level - (level - tp) * 0.75
    fill_bar = None
    for fb in range(confirm_bar, min(confirm_bar + 15, n)):
        if side == "long" and df_h[fb] >= cancel_075r:
            return "MISSED", 0, "075r_cancel"
        if side == "short" and df_l[fb] <= cancel_075r:
            return "MISSED", 0, "075r_cancel"
        if side == "long" and df_l[fb] <= level:
            fill_bar = fb; break
        if side == "short" and df_h[fb] >= level:
            fill_bar = fb; break

    if fill_bar is None:
        return "MISSED", 0, "expired_10m"

    # Phase 3: trade resolution with trail SL
    trail_trigger = level + (tp - level) * 0.9 if side == "long" else level - (level - tp) * 0.9
    trail_sl = level * 1.001 if side == "long" else level * 0.999
    current_sl = sl
    trailed = False
    bars_onside = 0

    for tb in range(fill_bar, min(fill_bar + max_bars, n)):
        bh, bl, bc = df_h[tb], df_l[tb], df_c[tb]
        if side == "long" and bc > level:
            bars_onside += 1
        elif side == "short" and bc < level:
            bars_onside += 1
        if not trailed:
            if side == "long" and (bh >= trail_trigger or bars_onside >= 60):
                current_sl = trail_sl; trailed = True
            elif side == "short" and (bl <= trail_trigger or bars_onside >= 60):
                current_sl = trail_sl; trailed = True
        if side == "long":
            if bl <= current_sl:
                pnl = 0.1 * sl_pct if trailed else -sl_pct
                return ("TRAIL_SL" if trailed else "SL"), round(pnl, 3), "filled"
            if bh >= tp:
                return "TP", tp_pct, "filled"
        else:
            if bh >= current_sl:
                pnl = 0.1 * sl_pct if trailed else -sl_pct
                return ("TRAIL_SL" if trailed else "SL"), round(pnl, 3), "filled"
            if bl <= tp:
                return "TP", tp_pct, "filled"

    lc = df_c[min(fill_bar + max_bars, n) - 1]
    pnl = (lc - level) / level * 100 if side == "long" else (level - lc) / level * 100
    return "OPEN", round(pnl, 3), "filled"


# ═══════════════════════════════════════════════════════════════════════════
#  MARKET CONDITIONS
# ═══════════════════════════════════════════════════════════════════════════

def load_market_scores(dataset_dir):
    cache_path = dataset_dir / "market_conditions_cache.csv"
    if cache_path.exists():
        cache_df = pd.read_csv(str(cache_path), parse_dates=["timestamp"])
        return cache_df["timestamp"].values, np.array(cache_df["score"].tolist())
    return None, None


# ═══════════════════════════════════════════════════════════════════════════
#  PROCESS ONE SYMBOL
# ═══════════════════════════════════════════════════════════════════════════

def process_symbol(sym, dataset_dir, market_score_data=None):
    if sym in EXCLUDED_SYMBOLS:
        return []
    fpath = dataset_dir / f"{sym}_1m.csv"
    if not fpath.exists():
        return []
    df = pd.read_csv(str(fpath), parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    if n < 500:
        return []

    # Extract arrays once
    c = df["close"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    o = df["open"].values.astype(float)
    vol = df["volume"].values.astype(float)
    ts = df["timestamp"].values

    # Compute indicators
    smma30 = pd.Series(c).ewm(alpha=1.0 / 30, adjust=False).mean().values
    ema7 = pd.Series(c).ewm(span=7, adjust=False).mean().values
    vol_usd = vol * c
    nama30 = _nama(vol_usd, seed_len=30)

    # Wick ratio
    rng = h - l
    body = np.abs(c - o)
    wick = np.where(rng > 1e-12, (rng - body) / rng, 0.0)

    # Rolling 6h high/low (the KEY LEVEL for breakout detection)
    # Use bars [-361:-1] to exclude current bar — level is from PRIOR 6h
    hi6h = np.full(n, np.nan)
    lo6h = np.full(n, np.nan)
    for i in range(361, n):
        hi6h[i] = np.max(h[i - 361:i])  # prior 6h high (excluding current bar)
        lo6h[i] = np.min(l[i - 361:i])  # prior 6h low

    # Precompute SMMA30 streaks
    above_smma = c > smma30
    streak_above = np.zeros(n, dtype=int)
    streak_below = np.zeros(n, dtype=int)
    for j in range(1, n):
        if above_smma[j]:
            streak_above[j] = streak_above[j - 1] + 1
        else:
            streak_below[j] = streak_below[j - 1] + 1

    # Rolling 5-min volume
    vol_5m = np.zeros(n)
    for j in range(5, n):
        vol_5m[j] = vol_usd[j - 5:j].sum()

    # Session VWAP
    tp_arr = (h + l + c) / 3.0
    # Simple approximation: daily reset VWAP
    dates = pd.to_datetime(ts).normalize()
    vwap = np.full(n, np.nan)
    cum_tp_vol = 0.0
    cum_vol = 0.0
    current_date = None
    for j in range(n):
        d = dates[j]
        if d != current_date:
            cum_tp_vol = 0.0
            cum_vol = 0.0
            current_date = d
        cum_tp_vol += tp_arr[j] * vol[j]
        cum_vol += vol[j]
        vwap[j] = cum_tp_vol / cum_vol if cum_vol > 0 else c[j]

    # Market scores
    if market_score_data is not None:
        score_ts, score_vals = market_score_data
        if not isinstance(score_vals, np.ndarray):
            score_vals = np.array(score_vals)
        bar_mkt = np.zeros(n, dtype=int)
        idxs = np.searchsorted(score_ts, ts.astype("datetime64[ns]"), side="right") - 1
        valid = idxs >= 0
        bar_mkt[valid] = score_vals[idxs[valid]]
    else:
        bar_mkt = np.zeros(n, dtype=int)

    # ── SCAN ──────────────────────────────────────────────────────────────
    trades = []
    cooldown = -1
    scan_start = 400
    scan_end = n - MAX_BARS - 15
    min_gap = 60
    min_dur_bars = 120  # 2h

    if SCAN_FROM is not None:
        scan_from_ts = pd.Timestamp(SCAN_FROM, tz="UTC")
        from_idx = int(np.searchsorted(ts.astype("datetime64[ns]"),
                                        np.datetime64(scan_from_ts), side="left"))
        scan_start = max(scan_start, from_idx)

    for i in range(scan_start, scan_end):
        if i <= cooldown:
            continue

        # ── CHEAPEST: 5-min volume ──
        if vol_5m[i] < MIN_VOL_5M_USD:
            continue

        # ── CHEAPEST: market condition (skip only at neutral) ──
        mkt = bar_mkt[i]
        can_long = mkt > 0
        can_short = mkt < 0
        if not can_long and not can_short:
            continue

        # ── SIGNAL DETECTION: price above 6h high / below 6h low ──
        # No "first time" requirement — cooldown handles dedup
        if np.isnan(hi6h[i]) or np.isnan(lo6h[i]):
            continue

        # ── CHEAP: Price cap ──
        if c[i] > MAX_PRICE:
            continue

        # ── CHEAP: 2h price move minimum ──
        if i >= 120:
            move_2h = abs(c[i] - c[i - 120]) / c[i - 120] * 100
            if move_2h < MIN_2H_MOVE_PCT:
                continue
        else:
            continue

        signals = []
        if can_long and c[i] > hi6h[i]:
            signals.append(("long", hi6h[i]))
        if can_short and c[i] < lo6h[i]:
            signals.append(("short", lo6h[i]))

        if not signals:
            continue

        for direction, level in signals:
            is_long = direction == "long"

            # ── GATE: Duration — 2h with ≤2 debounced SMMA30 crosses ──
            # Price must be on correct side at entry
            if is_long and c[i] < smma30[i]:
                continue
            if not is_long and c[i] > smma30[i]:
                continue
            # Count debounced crosses in last 2h
            dur_c = c[max(0, i - min_dur_bars + 1):i + 1]
            dur_sm = smma30[max(0, i - min_dur_bars + 1):i + 1]
            if len(dur_c) < min_dur_bars:
                continue
            dur_crosses = _count_smma_crosses(dur_c, dur_sm)
            if dur_crosses > 3:
                continue

            # ── GATE: SMMA30 trending (not flat/sideways) ──
            lb = min(i + 1, 120)
            sm_start = np.mean(smma30[i - lb + 10:i - lb + 20])
            sm_end = np.mean(smma30[i - 9:i + 1])
            sm_change = abs(sm_end - sm_start) / sm_start * 100 if sm_start > 0 else 0
            if sm_change < 0.15:  # SMMA30 barely moved = sideways
                continue
            if is_long and sm_end <= sm_start:
                continue
            if not is_long and sm_end >= sm_start:
                continue

            # ── GATE: Low noise (SMMA30 crosses ≤ 6 in 2h) ──
            stair_c = c[max(0, i - 119):i + 1]
            stair_h = h[max(0, i - 119):i + 1]
            stair_l = l[max(0, i - 119):i + 1]
            stair_sm = smma30[max(0, i - 119):i + 1]
            crosses = _count_smma_crosses(stair_c, stair_sm)
            if crosses > 6:
                continue

            # ── GATE: R² > 0.85 and channel < 0.25% over last 2h ──
            if len(stair_c) >= 60:
                n_s = len(stair_c)
                x_s = np.arange(n_s, dtype=float)
                xm, ym = np.mean(x_s), np.mean(stair_c)
                sxy = np.sum((x_s - xm) * (stair_c - ym))
                sxx = np.sum((x_s - xm) ** 2)
                syy = np.sum((stair_c - ym) ** 2)
                if sxx > 0 and syy > 0:
                    r2 = (sxy ** 2) / (sxx * syy)
                    slope = sxy / sxx
                    pred = slope * x_s + (ym - slope * xm)
                    std_res = np.std(stair_c - pred)
                    channel = std_res / ym * 100 * 2 if ym > 0 else 99
                    if r2 < 0.85:
                        continue
                    if channel > 0.25:
                        continue
                else:
                    continue

            # ── GATE: Grind approach (no spike into the level) ──
            if i >= 10:
                tail_c = c[i - 9:i + 1]
                bar_moves = np.abs(np.diff(tail_c))
                avg_m = np.mean(bar_moves)
                max_m = np.max(bar_moves)
                spike_ratio = max_m / avg_m if avg_m > 0 else 0
                max_bar_pct = max_m / np.mean(tail_c) * 100 if np.mean(tail_c) > 0 else 0
                if spike_ratio > 4.5 and max_bar_pct > 0.5:
                    continue
                # Acceleration check
                if len(bar_moves) >= 7:
                    accel = np.mean(bar_moves[-3:]) / np.mean(bar_moves[:-3]) if np.mean(bar_moves[:-3]) > 0 else 0
                    if accel > 3.0 and max_bar_pct > 0.3:
                        continue

            # ── GATE: Volume — longs: nama30 not falling. Shorts: allow decreasing, block flat ──
            if i >= 11 and not np.isnan(nama30[i]) and not np.isnan(nama30[i - 10]):
                if is_long and nama30[i] < nama30[i - 10] * 0.95:
                    continue
                # Shorts: block flat volume (ratio ~1.0), allow decreasing or increasing
                if not is_long:
                    vol_ratio = nama30[i] / nama30[i - 10] if nama30[i - 10] > 0 else 1.0
                    if 0.95 <= vol_ratio <= 1.05:  # flat = bad for shorts
                        continue

            # ── GATE: VWAP side ──
            if is_long and c[i] < vwap[i]:
                continue
            if not is_long and c[i] > vwap[i]:
                continue

            # ── GATE: 30-bar wick noise ──
            avg_wick = np.mean(wick[max(0, i - 29):i + 1])
            if avg_wick > 0.55:
                continue

            # ── LONG-ONLY GATE: Volume acceleration >= 1.3 ──
            if is_long and i >= 120:
                vol_usd_stair = vol_usd[max(0, i - 119):i + 1]
                recent_v = np.mean(vol_usd_stair[-30:])
                prior_v = np.mean(vol_usd_stair[:30])
                vol_accel = recent_v / prior_v if prior_v > 0 else 1.0
                if vol_accel < 1.3:
                    continue
            else:
                vol_accel = 0.0

            # ── LONG-ONLY GATE: R² of last 30 bars >= 0.5 ──
            if is_long and i >= 30:
                last30_c = c[i - 29:i + 1]
                x30 = np.arange(30)
                s30, i30 = np.polyfit(x30, last30_c, 1)
                pred30 = s30 * x30 + i30
                ss_res30 = np.sum((last30_c - pred30) ** 2)
                ss_tot30 = np.sum((last30_c - np.mean(last30_c)) ** 2)
                r2_30 = 1 - ss_res30 / ss_tot30 if ss_tot30 > 0 else 0
                if r2_30 < 0.5:
                    continue
            else:
                r2_30 = 0.0

            # ── MEASURE (not gate): Candle size metrics in last 15 bars ──
            if i >= 15:
                last15_range = h[i - 14:i + 1] - l[i - 14:i + 1]
                last15_pct = last15_range / c[i - 14:i + 1] * 100
                max_candle_15m = float(np.max(last15_pct))
                pct_small_15m = float(np.sum(last15_pct < 0.2) / len(last15_pct) * 100)
            else:
                max_candle_15m = 0.0
                pct_small_15m = 0.0

            # ── MEASURE (not gate): Max drawdown in last 2h ──
            stair_start = max(0, i - 119)
            stair_cl = c[stair_start:i + 1]
            stair_lo = l[stair_start:i + 1]
            stair_hi = h[stair_start:i + 1]
            if is_long:
                running_peak = np.maximum.accumulate(stair_cl)
                dd_pct = float(np.max((running_peak - stair_lo) / running_peak * 100))
            else:
                running_trough = np.minimum.accumulate(stair_cl)
                dd_pct = float(np.max((stair_hi - running_trough) / running_trough * 100))

            # ── ALL GATES PASSED — compute SL/TP ──
            entry_level = level

            sl_val, tp_val, sl_pct, tp_pct = compute_sl_tp(
                h[:i + 1], l[:i + 1], direction, entry_level)

            # RR guard
            if is_long:
                eff_rr = (tp_val - entry_level) / (entry_level - sl_val) if entry_level > sl_val else 0
            else:
                eff_rr = (entry_level - tp_val) / (sl_val - entry_level) if sl_val > entry_level else 0
            if eff_rr < 0.95:
                continue

            # ── DPS scoring ──
            # Duration: score=2 if 0 crosses in 2h, score=1 if 1-3 crosses (passed gate)
            dur_hrs = min_dur_bars / 60.0  # 2h window
            dps_dur = 2 if dur_crosses == 0 else 1

            if i >= 10:
                sr = spike_ratio
                dps_app = 2 if sr < 2.5 else (1 if sr < 3.5 else 0)
            else:
                dps_app = 1

            # Volume DPS — ZCT scoring:
            #   Longs:  increasing(2) > flat(1) > decreasing(0)
            #   Shorts: increasing(2) > decreasing(1) > flat(0)
            if i >= 11 and not np.isnan(nama30[i]) and not np.isnan(nama30[i - 10]):
                vol_change = (nama30[i] - nama30[i - 10]) / nama30[i - 10] if nama30[i - 10] > 0 else 0
                if vol_change > 0.05:
                    dps_vol = 2; vol_label = "increasing"
                elif vol_change > -0.05:
                    # Flat
                    if is_long:
                        dps_vol = 1; vol_label = "flat"
                    else:
                        dps_vol = 0; vol_label = "flat"
                else:
                    # Decreasing
                    if is_long:
                        dps_vol = 0; vol_label = "decreasing"
                    else:
                        dps_vol = 1; vol_label = "decreasing"
            else:
                dps_vol = 1; vol_label = "flat"

            dps_total = dps_dur + dps_app + dps_vol
            dps_conf = "max" if dps_total >= 6 else ("high" if dps_total >= 4 else ("low" if dps_total >= 3 else "avoid"))

            # ── SIMULATE ENTRY ──
            outcome, pnl, reason = simulate_momo_live_entry(
                c, h, l, i, direction,
                entry_level, tp_val, sl_val, tp_pct, sl_pct, MAX_BARS)

            trades.append({
                "symbol": sym,
                "ts": str(ts[i]),
                "strategy": "zct_momo",
                "side": direction,
                "level": round(level, 8),
                "entry": round(entry_level, 8),
                "tp": round(tp_val, 8),
                "sl": round(sl_val, 8),
                "tp_pct": round(tp_pct, 3),
                "sl_pct": round(sl_pct, 3),
                "eff_rr": round(eff_rr, 2),
                "dps": dps_total,
                "conf": dps_conf,
                "dps_dur": dps_dur,
                "dps_app": dps_app,
                "dps_vol": dps_vol,
                "dur_hrs": round(dur_hrs, 1),
                "vol_trend": vol_label,
                "smma_crosses": crosses,
                "vol_accel": round(vol_accel, 3),
                "r2_30": round(r2_30, 4),
                "max_candle_15m": round(max_candle_15m, 3),
                "pct_small_15m": round(pct_small_15m, 1),
                "max_dd_2h": round(dd_pct, 3),
                "outcome": outcome,
                "pnl": round(pnl, 3),
                "mkt_score": int(mkt),
            })
            # Only cooldown on filled trades, not MISSED
            if outcome != "MISSED":
                cooldown = i + min_gap
            break

    return trades


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    with open(LOG_FILE, "w") as f:
        f.write("")
    # Clear output CSV for fresh run
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)

    syms = sorted([f.replace("_1m.csv", "") for f in os.listdir(DATASET_DIR) if f.endswith("_1m.csv")])
    log(f"ZCT Momo v3: {len(syms)} symbols")
    if SCAN_FROM:
        log(f"  From: {SCAN_FROM}")

    score_ts, score_vals = load_market_scores(DATASET_DIR)
    market_data = (score_ts, score_vals) if score_ts is not None else None
    if market_data:
        log(f"  Market conditions: {len(score_vals)} checkpoints")

    log(f"  Signal: price above 6h high / below 6h low (no first-time req)")
    log(f"  Gates: duration(2h+) + SMMA30 trending + noise(xr<=6) + grind + volume(nama30) + VWAP side")
    log(f"  Entry: 2-bar confirm + limit at broken level + 0.75R cancel")
    log(f"  SL: 2nd swing + 0.15% buffer [1-2%] | TP: SL*1.1R [max 4%]")

    workers = min(10, len(syms))
    log(f"  Workers: {workers}")

    all_trades = []
    done = 0
    t0 = time.time()
    header_written = False

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_symbol, sym, DATASET_DIR, market_data): sym for sym in syms}
        for future in as_completed(futures):
            sym = futures[future]
            done += 1
            try:
                sym_trades = future.result()
                if sym_trades:
                    all_trades.extend(sym_trades)
                    # Write incrementally to disk
                    batch_df = pd.DataFrame(sym_trades)
                    batch_df.to_csv(OUTPUT_CSV, mode="a", index=False,
                                   header=not header_written)
                    header_written = True
                    for t in sym_trades:
                        log(f"  {sym:>15} {t['ts'][:19]} {t['side']:>5} lvl={t['level']:.4f} "
                            f"dur={t['dur_hrs']}h xr={t['smma_crosses']} dps={t['dps']} "
                            f"=> {t['outcome']:>8} pnl={t['pnl']:+.2f}%")
            except Exception as e:
                log(f"  ERROR {sym}: {e}")
                import traceback
                log(traceback.format_exc())

            if done % 50 == 0 or done == len(syms):
                elapsed = time.time() - t0
                log(f"Progress: {done}/{len(syms)}, {len(all_trades)} trades, {elapsed:.0f}s")

    if all_trades:
        tdf = pd.DataFrame(all_trades)
        # Rewrite clean sorted version
        tdf.to_csv(OUTPUT_CSV, index=False)
    else:
        tdf = pd.DataFrame()

    elapsed = time.time() - t0
    log(f"\nDone in {elapsed:.0f}s -- {len(all_trades)} trades")

    if len(tdf) == 0:
        log("No trades found!")
        return

    filled = tdf[tdf["outcome"] != "MISSED"]
    tp_n = (filled["outcome"] == "TP").sum()
    sl_n = (filled["outcome"] == "SL").sum()
    trail_n = (filled["outcome"] == "TRAIL_SL").sum()
    open_n = (filled["outcome"] == "OPEN").sum()
    wr = tp_n / (tp_n + sl_n) * 100 if (tp_n + sl_n) > 0 else 0

    log(f"\n{'='*70}")
    log(f"ZCT MOMO v3 RESULTS")
    log(f"{'='*70}")
    log(f"  Signals: {len(tdf)} | Filled: {len(filled)} | Missed: {len(tdf)-len(filled)}")
    log(f"  TP={tp_n} SL={sl_n} TRAIL_SL={trail_n} OPEN={open_n}")
    log(f"  Win Rate: {wr:.1f}%")
    log(f"  Total PnL: {filled['pnl'].sum():+.2f}%")
    if len(filled) > 0:
        log(f"  Avg PnL: {filled['pnl'].mean():+.3f}%")

    for side in ["long", "short"]:
        sub = filled[filled["side"] == side]
        if len(sub) == 0: continue
        tp_s = (sub["outcome"] == "TP").sum()
        sl_s = (sub["outcome"] == "SL").sum()
        wr_s = tp_s / (tp_s + sl_s) * 100 if (tp_s + sl_s) > 0 else 0
        log(f"  {side.upper()}: {len(sub)} filled | WR={wr_s:.1f}% | PnL={sub['pnl'].sum():+.2f}%")

    log(f"\n  By DPS:")
    for conf in ["max", "high", "low", "avoid"]:
        sub = filled[filled["conf"] == conf]
        if len(sub) == 0: continue
        tp_c = (sub["outcome"] == "TP").sum()
        sl_c = (sub["outcome"] == "SL").sum()
        wr_c = tp_c / (tp_c + sl_c) * 100 if (tp_c + sl_c) > 0 else 0
        log(f"    {conf:6s}: {len(sub):4d} | WR={wr_c:.1f}% | PnL={sub['pnl'].sum():+.2f}%")

    log(f"\nResults: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
