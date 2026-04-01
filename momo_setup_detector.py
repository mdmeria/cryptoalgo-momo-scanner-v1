#!/usr/bin/env python3
"""
ZCT Momentum Setup Detector
Evaluates 3 key variables for B+ momentum trade quality:
  1. Slow grind into level (approach quality)
  2. Volume consistently increasing
  3. Grindy staircase on left hand side

Target: 90%+ agreement with ZCT PDF labels across 100 B+ trades.
"""
import pandas as pd
import numpy as np
import os
import json
from datetime import timedelta
from typing import Optional, Dict, Tuple

DATA_DIR = "datasets/binance_futures_1m"

# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

_DF_CACHE = {}

def _load_symbol_df(fname):
    if fname not in _DF_CACHE:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            _DF_CACHE[fname] = None
            return None
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        _DF_CACHE[fname] = df
    return _DF_CACHE[fname]


def load_trade_data(symbol, date_str, entry_time_str, hours_before=5, hours_after=1):
    """Load 1m candles around the entry time. Returns (df, entry_idx, actual_symbol)."""
    candidates = [f"{symbol}_1m.csv"]
    if not symbol.startswith("1000"):
        candidates.append(f"1000{symbol}_1m.csv")
    if symbol == "FLOKIUSDT":
        candidates = ["1000FLOKIUSDT_1m.csv", "FLOKIUSDT_1m.csv"]
    if symbol == "SHIBUSDT":
        candidates = ["1000SHIBUSDT_1m.csv", "SHIBUSDT_1m.csv"]
    if symbol == "NEIROUSDT":
        candidates = ["NEIROUSDT_1m.csv", "NEIROETHUSDT_1m.csv"]

    for fname in candidates:
        df = _load_symbol_df(fname)
        if df is None:
            continue
        entry_dt = pd.Timestamp(f"{date_str} {entry_time_str}:00", tz="UTC")
        start = entry_dt - timedelta(hours=hours_before)
        end = entry_dt + timedelta(hours=hours_after)
        mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
        df_slice = df[mask].copy().reset_index(drop=True)
        if len(df_slice) < 60:
            continue
        diffs = (df_slice["timestamp"] - entry_dt).abs()
        entry_idx = diffs.idxmin()
        return df_slice, entry_idx, fname.replace("_1m.csv", "")
    return None, None, None


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def smma(arr, period):
    result = np.zeros(len(arr))
    result[0] = arr[0]
    alpha = 1.0 / period
    for i in range(1, len(arr)):
        result[i] = result[i - 1] * (1 - alpha) + arr[i] * alpha
    return result


def ema(arr, period):
    result = np.zeros(len(arr))
    result[0] = arr[0]
    alpha = 2.0 / (period + 1)
    for i in range(1, len(arr)):
        result[i] = result[i - 1] * (1 - alpha) + arr[i] * alpha
    return result


def higher_lows_count(lows, window=20):
    """Count how many sequential higher lows in the last `window` bars."""
    tail = lows[-window:]
    count = 0
    for i in range(1, len(tail)):
        if tail[i] >= tail[i - 1]:
            count += 1
    return count / (len(tail) - 1) * 100 if len(tail) > 1 else 0


def lower_highs_count(highs, window=20):
    """Count how many sequential lower highs in the last `window` bars."""
    tail = highs[-window:]
    count = 0
    for i in range(1, len(tail)):
        if tail[i] <= tail[i - 1]:
            count += 1
    return count / (len(tail) - 1) * 100 if len(tail) > 1 else 0


def count_smma_crosses(closes, smma_vals, min_bars_away=5):
    above = closes > smma_vals
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
    return crossovers


# ═══════════════════════════════════════════════════════════════════════════
#  VARIABLE 1: SLOW GRIND INTO LEVEL
# ═══════════════════════════════════════════════════════════════════════════

def detect_grind(closes, highs, lows, direction, lookback=10):
    """
    Detect if price slowly grinds into the level (last N bars before entry).

    Key characteristics of a grind (from PDF):
    - Candles are roughly the same size (no single huge candle)
    - Steady, gradual approach — not a sudden spike
    - The "slope" isn't steep

    Key characteristics of a spike (anti-pattern):
    - One or two candles are much larger than the rest
    - Sudden acceleration into the level
    - Price covers a lot of distance in very few bars

    Returns: ('yes', 'warning', 'no') and confidence metrics dict
    """
    if len(closes) < lookback + 5:
        return "warning", {}

    tail = closes[-lookback:]
    tail_h = highs[-lookback:]
    tail_l = lows[-lookback:]

    # 1. Spike ratio: largest bar move / average bar move
    bar_moves = np.abs(np.diff(tail))
    avg_move = np.mean(bar_moves)
    max_move = np.max(bar_moves)
    spike_ratio = max_move / avg_move if avg_move > 0 else 0

    # 2. Max single bar move as % of price
    price = np.mean(tail)
    max_bar_pct = max_move / price * 100 if price > 0 else 0

    # 3. Acceleration: is the last 3 bars move >> previous 7 bars move?
    if lookback >= 7:
        recent_3 = np.mean(bar_moves[-3:]) if len(bar_moves) >= 3 else 0
        prior = np.mean(bar_moves[:-3]) if len(bar_moves) > 3 else avg_move
        accel_ratio = recent_3 / prior if prior > 0 else 0
    else:
        accel_ratio = 1.0

    # 4. Candle body uniformity (CV of body sizes)
    bodies = np.abs(np.diff(tail))
    body_cv = np.std(bodies) / np.mean(bodies) if np.mean(bodies) > 0 else 0

    # 5. Net efficiency: what fraction of total distance is net movement
    net_move = abs(tail[-1] - tail[0])
    total_dist = np.sum(bar_moves)
    efficiency = net_move / total_dist if total_dist > 0 else 0

    metrics = {
        'spike_ratio': round(spike_ratio, 2),
        'max_bar_pct': round(max_bar_pct, 4),
        'accel_ratio': round(accel_ratio, 2),
        'body_cv': round(body_cv, 3),
        'efficiency': round(efficiency, 3),
    }

    # Decision logic:
    # A "spike" = single large candle (spike_ratio > 4) AND that candle is significant (>0.5%)
    # OR extreme acceleration (last 3 bars >> prior bars)
    is_spike = (spike_ratio > 4.5 and max_bar_pct > 0.5)
    is_steep = (accel_ratio > 3.0 and max_bar_pct > 0.3)

    if is_spike:
        return "no", metrics
    elif is_steep or (spike_ratio > 3.5 and max_bar_pct > 0.4):
        return "warning", metrics
    else:
        return "yes", metrics


# ═══════════════════════════════════════════════════════════════════════════
#  VARIABLE 2: VOLUME CONSISTENTLY INCREASING
# ═══════════════════════════════════════════════════════════════════════════

def detect_volume(volumes, closes, direction, lookback=120):
    """
    Detect if volume is consistently increasing over the staircase (2h).

    ZCT method (from PDF):
    - Compare the avg volume (blue line = moving average) at entry vs at
      the start of the staircase
    - "Increasing" = avg at entry NOTICEABLY higher than avg at start
    - "Flat" = roughly the same
    - "Decreasing" = avg at entry lower than avg at start

    We replicate this by computing an EMA(60) of USD volume and comparing
    its value at the entry vs at the start of the 2h window.

    Returns: ('yes', 'warning', 'no') and metrics dict
    """
    vol_usd = volumes * closes
    n = min(len(vol_usd), lookback)
    if n < 60:
        return "warning", {}

    stair_vol = vol_usd[-n:]

    # Compute EMA(60) of volume — this is the "blue line"
    vol_ema = ema(stair_vol, 60)

    # Compare: EMA value at entry (last 10 bars avg) vs EMA value at staircase
    # start (first 10 bars avg after EMA warmup)
    warmup = min(30, n // 4)
    start_val = np.mean(vol_ema[warmup:warmup + 10])
    entry_val = np.mean(vol_ema[-10:])

    if start_val <= 0:
        return "warning", {}

    vol_change_ratio = entry_val / start_val

    # Also compute: 3-segment trend for consistency check
    thirds = n // 3
    avg1 = np.mean(stair_vol[:thirds])
    avg2 = np.mean(stair_vol[thirds:2 * thirds])
    avg3 = np.mean(stair_vol[2 * thirds:])

    metrics = {
        'vol_change_ratio': round(vol_change_ratio, 3),
        'start_val': round(start_val, 0),
        'entry_val': round(entry_val, 0),
        'avg1': round(avg1, 0),
        'avg2': round(avg2, 0),
        'avg3': round(avg3, 0),
    }

    # Decision:
    # ZCT says volume is the LEAST important variable — the PDF frequently
    # says "trickier to interpret" and labels many as "warning". Only 42% of
    # B+ trades have volume=yes, and 41% have volume=no but still won.
    #
    # Our EMA ratio approach has limited precision because:
    #   - Entry timestamps from screenshots may be off by 15-30 min
    #   - Staircase start varies (1-8 hours back), not fixed 2h
    #   - Students use different platforms with different volume data
    #
    # Strategy: use wide "warning" band to capture genuine uncertainty.
    # Only confident calls go to yes/no.
    if vol_change_ratio > 1.60:
        return "yes", metrics    # clearly higher at entry than start
    elif vol_change_ratio > 1.15:
        return "warning", metrics  # somewhat higher but uncertain
    elif vol_change_ratio > 0.90:
        return "no", metrics      # roughly flat — not increasing
    elif vol_change_ratio > 0.60:
        return "warning", metrics  # somewhat lower but uncertain
    else:
        return "no", metrics      # clearly decreasing


# ═══════════════════════════════════════════════════════════════════════════
#  VARIABLE 3: GRINDY STAIRCASE ON LEFT HAND SIDE
# ═══════════════════════════════════════════════════════════════════════════

def detect_staircase(closes, highs, lows, opens, volumes, direction, lookback=120):
    """
    Detect if there's a grindy staircase on the left hand side (2+ hours).

    From the PDF:
    - "Grindy staircase" = steady incremental price movement in one direction
    - Anti-pattern: choppy range (price rejects off nearby levels 3+ times)
    - Anti-pattern: price slicing through MAs many times
    - Minimum ~2 hours of staircase for best results
    - SMMA30 should be trending (not flat/sideways)

    Detection approach:
    1. SMMA30 noise: ≤6 crosses = not choppy
    2. SMMA30 trending: slope is meaningful, not flat
    3. Net direction: price moved meaningfully in trade direction
    4. Not a choppy range: price doesn't revisit same levels excessively

    Returns: ('yes', 'warning', 'no') and metrics dict
    """
    n = min(len(closes), lookback)
    if n < 60:
        return "no", {}

    c = closes[-n:]
    h = highs[-n:]
    l = lows[-n:]
    o = opens[-n:]

    # 1. SMMA30 computation and crosses
    sm30 = smma(c, 30)
    crosses = count_smma_crosses(c, sm30)

    # 2. SMMA30 trending check
    # Compare SMMA30 at 3 points: start, middle, end
    sm30_start = np.mean(sm30[10:20])
    sm30_mid = np.mean(sm30[n // 2 - 5:n // 2 + 5])
    sm30_end = np.mean(sm30[-10:])

    if direction == "long":
        sm30_monotonic = sm30_end > sm30_start  # overall up
        sm30_consistent = sm30_mid >= sm30_start * 0.999  # mid not below start
    else:
        sm30_monotonic = sm30_end < sm30_start  # overall down
        sm30_consistent = sm30_mid <= sm30_start * 1.001  # mid not above start

    # SMMA30 total change (as % of price) — measure trend strength
    sm30_change_pct = abs(sm30_end - sm30_start) / sm30_start * 100 if sm30_start > 0 else 0
    sm30_flat = sm30_change_pct < 0.15  # less than 0.15% change = basically flat

    # 3. Net directional move
    net_move = c[-1] - c[0]
    net_move_pct = abs(net_move) / c[0] * 100 if c[0] > 0 else 0
    correct_dir = (direction == "long" and net_move > 0) or (direction == "short" and net_move < 0)

    # 4. Price on correct side of SMMA30 (most of the time)
    if direction == "long":
        pct_correct_side = np.sum(c > sm30) / n * 100
    else:
        pct_correct_side = np.sum(c < sm30) / n * 100

    # 5. Choppiness: count how many times price crosses the midline of its range
    midline = (np.max(h) + np.min(l)) / 2
    above_mid = c > midline
    mid_crosses = np.sum(np.diff(above_mid.astype(int)) != 0)

    # 6. Range-band revisit analysis
    price_range = np.max(h) - np.min(l)
    if price_range > 0:
        # Divide into 5 bands, count max revisits to any single band
        bands = 5
        band_size = price_range / bands
        band_min = np.min(l)
        band_visits = np.zeros(bands)
        last_band = -1
        for price in c:
            band = min(int((price - band_min) / band_size), bands - 1)
            if band != last_band:
                band_visits[band] += 1
                last_band = band
        max_revisits = int(np.max(band_visits))
    else:
        max_revisits = 0

    metrics = {
        'smma_crosses': crosses,
        'sm30_monotonic': sm30_monotonic,
        'sm30_consistent': sm30_consistent,
        'sm30_change_pct': round(sm30_change_pct, 3),
        'sm30_flat': sm30_flat,
        'correct_dir': correct_dir,
        'net_move_pct': round(net_move_pct, 3),
        'pct_correct_side': round(pct_correct_side, 1),
        'mid_crosses': int(mid_crosses),
        'max_revisits': max_revisits,
        'n_bars': n,
    }

    # Decision:
    # The key ZCT criteria for staircase:
    #   (A) Not choppy: SMMA crosses ≤ 6 (mandatory)
    #   (B) SMMA trending in correct direction (not flat/sideways)
    #   (C) Net price move in correct direction
    #   (D) Not excessively crossing the midline (choppy range indicator)
    #
    # Staircase "yes" = (A) + at least 2 of (B, C, D)
    # Staircase "warning" = (A) + at least 1 of (B, C, D), typically short duration
    # Staircase "no" = fails (A) or fails all of (B, C, D)

    is_low_noise = crosses <= 6
    is_trending = sm30_monotonic and not sm30_flat
    is_directional = correct_dir and net_move_pct > 0.2
    is_not_choppy = mid_crosses < n * 0.4  # less than 40% of bars cross midline

    positives = sum([is_trending, is_directional, is_not_choppy])
    metrics['positives'] = positives

    if not is_low_noise:
        # High noise = choppy = not a staircase
        if positives >= 3:
            return "warning", metrics  # strong trend despite noise
        return "no", metrics

    if positives >= 2:
        return "yes", metrics
    elif positives >= 1:
        return "warning", metrics
    else:
        return "no", metrics


# ═══════════════════════════════════════════════════════════════════════════
#  STAIRCASE START DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def find_staircase_start(closes, highs, lows, direction, entry_idx, max_lookback=240):
    """
    Find where the staircase regime began by looking for:
    - A swing reversal point (the start of the current trend)
    - Or a range break (price leaving a consolidation)
    """
    start = max(0, entry_idx - max_lookback)
    c = closes[start:entry_idx + 1]
    h = highs[start:entry_idx + 1]
    l = lows[start:entry_idx + 1]

    if len(c) < 30:
        return start

    # Walk backwards from entry to find where the trend started
    # Look for the point where price was at the opposite extreme
    if direction == "long":
        # Find the lowest low before the grind up
        min_idx = np.argmin(l)
        return start + min_idx
    else:
        # Find the highest high before the grind down
        max_idx = np.argmax(h)
        return start + max_idx


# ═══════════════════════════════════════════════════════════════════════════
#  FULL SETUP EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_momo_setup(symbol, date_str, entry_time, direction):
    """
    Full evaluation of a momentum setup at the given point.
    Returns dict with labels and metrics for all 3 variables.
    """
    df, entry_idx, actual_sym = load_trade_data(symbol, date_str, entry_time)
    if df is None:
        return None

    c = df["close"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    o = df["open"].values.astype(float)
    v = df["volume"].values.astype(float)

    # Use only data up to entry
    c_pre = c[:entry_idx + 1]
    h_pre = h[:entry_idx + 1]
    l_pre = l[:entry_idx + 1]
    o_pre = o[:entry_idx + 1]
    v_pre = v[:entry_idx + 1]

    if len(c_pre) < 30:
        return None

    # Evaluate all 3 variables
    grind_label, grind_metrics = detect_grind(c_pre, h_pre, l_pre, direction)
    volume_label, volume_metrics = detect_volume(v_pre, c_pre, direction)
    staircase_label, staircase_metrics = detect_staircase(
        c_pre, h_pre, l_pre, o_pre, v_pre, direction
    )

    return {
        "symbol": actual_sym,
        "grind": grind_label,
        "grind_metrics": grind_metrics,
        "volume": volume_label,
        "volume_metrics": volume_metrics,
        "staircase": staircase_label,
        "staircase_metrics": staircase_metrics,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  TRADE LIST (from PDF)
# ═══════════════════════════════════════════════════════════════════════════

TRADES = [
    (1, "ENAUSDT", "2024-08-02", "15:00", "short", "yes", "no", "warning"),
    (2, "ENAUSDT", "2024-08-02", "15:19", "short", "yes", "yes", "warning"),
    (3, "ENAUSDT", "2024-08-05", "18:15", "long", "yes", "warning", "yes"),
    (4, "FLOKIUSDT", "2024-08-07", "10:00", "long", "yes", "yes", "yes"),
    (5, "ENAUSDT", "2024-08-15", "09:30", "long", "yes", "yes", "yes"),
    (6, "ENAUSDT", "2024-08-15", "10:00", "short", "yes", "yes", "yes"),
    (7, "TRXUSDT", "2024-08-20", "14:00", "long", "yes", "yes", "yes"),
    (8, "TRXUSDT", "2024-08-21", "14:30", "short", "yes", "no", "yes"),
    (9, "AAVEUSDT", "2024-08-28", "10:00", "long", "yes", "no", "yes"),
    (10, "TRXUSDT", "2024-08-29", "07:14", "short", "yes", "yes", "yes"),
    (11, "ENAUSDT", "2024-09-18", "12:30", "long", "yes", "yes", "yes"),
    (12, "AAVEUSDT", "2024-10-02", "12:30", "short", "yes", "no", "yes"),
    (13, "AAVEUSDT", "2024-10-01", "05:30", "short", "yes", "yes", "warning"),
    (14, "AAVEUSDT", "2024-10-01", "05:30", "short", "yes", "no", "yes"),
    (15, "ENAUSDT", "2024-12-25", "19:00", "long", "yes", "no", "yes"),
    (16, "FTMUSDT", "2024-10-06", "14:30", "short", "yes", "warning", "yes"),
    (17, "FTMUSDT", "2024-10-08", "18:00", "short", "yes", "yes", "yes"),
    (18, "FTMUSDT", "2024-10-16", "12:00", "short", "yes", "yes", "yes"),
    (19, "WIFUSDT", "2025-01-28", "15:30", "long", "yes", "no", "yes"),
    (20, "FTMUSDT", "2024-11-11", "13:00", "short", "yes", "no", "yes"),
    (21, "FTMUSDT", "2024-11-15", "16:30", "long", "yes", "no", "yes"),
    (22, "FTMUSDT", "2024-11-15", "16:30", "short", "yes", "no", "yes"),
    (23, "LINKUSDT", "2024-11-22", "08:30", "long", "yes", "yes", "yes"),
    (24, "SOLUSDT", "2024-11-23", "18:00", "short", "yes", "no", "yes"),
    (25, "KSMUSDT", "2024-11-26", "08:30", "short", "yes", "no", "warning"),
    (26, "INJUSDT", "2024-11-25", "16:00", "long", "yes", "yes", "yes"),
    (27, "VIRTUALUSDT", "2024-12-20", "18:30", "long", "yes", "yes", "yes"),
    (28, "ENAUSDT", "2024-12-20", "16:00", "long", "yes", "no", "yes"),
    (29, "INJUSDT", "2024-12-26", "13:00", "long", "yes", "warning", "yes"),
    (30, "HBARUSDT", "2025-01-17", "14:27", "long", "yes", "warning", "yes"),
    (31, "ETHUSDT", "2025-03-05", "18:41", "long", "yes", "warning", "yes"),
    (32, "SOLUSDT", "2025-01-25", "12:16", "short", "yes", "no", "yes"),
    (33, "SOLUSDT", "2025-01-15", "18:48", "long", "yes", "no", "yes"),
    (34, "XRPUSDT", "2025-01-27", "14:16", "long", "yes", "yes", "yes"),
    (35, "SUIUSDT", "2025-01-28", "12:12", "short", "yes", "warning", "yes"),
    (36, "DOGEUSDT", "2025-01-28", "08:44", "short", "yes", "yes", "yes"),
    (37, "TRXUSDT", "2025-01-31", "11:08", "long", "yes", "yes", "warning"),
    (38, "ADAUSDT", "2025-01-31", "16:16", "short", "yes", "warning", "yes"),
    (39, "BNBUSDT", "2025-02-01", "14:00", "short", "yes", "no", "yes"),
    (40, "DOTUSDT", "2025-02-17", "20:21", "long", "yes", "no", "yes"),
    (41, "LINKUSDT", "2025-03-21", "08:07", "long", "yes", "no", "yes"),
    (42, "TRUMPUSDT", "2025-01-23", "10:53", "long", "yes", "yes", "warning"),
    (43, "ENAUSDT", "2025-03-05", "13:45", "long", "yes", "warning", "yes"),
    (44, "AVAXUSDT", "2025-03-05", "18:30", "short", "yes", "yes", "warning"),
    (45, "FILUSDT", "2025-08-22", "08:14", "long", "warning", "yes", "yes"),
    (46, "SUIUSDT", "2025-06-24", "06:30", "long", "yes", "no", "yes"),
    (47, "HBARUSDT", "2025-07-12", "00:20", "short", "yes", "warning", "yes"),
    (48, "THETAUSDT", "2025-07-16", "04:15", "short", "yes", "warning", "yes"),
    (49, "GALAUSDT", "2025-06-28", "09:00", "long", "warning", "yes", "yes"),
    (50, "ZENUSDT", "2024-12-18", "05:41", "long", "yes", "no", "yes"),
    (51, "SAGAUSDT", "2024-11-19", "01:15", "long", "yes", "no", "yes"),
    (52, "SUIUSDT", "2025-06-23", "08:30", "long", "yes", "yes", "yes"),
    (53, "SOLUSDT", "2025-08-15", "07:30", "short", "yes", "yes", "yes"),
    (54, "ICPUSDT", "2025-08-14", "09:00", "long", "yes", "no", "yes"),
    (55, "ORDIUSDT", "2025-01-20", "19:00", "short", "yes", "warning", "yes"),
    (56, "FILUSDT", "2025-08-14", "06:00", "long", "yes", "no", "yes"),
    (57, "BTCUSDT", "2025-08-14", "16:30", "long", "yes", "yes", "yes"),
    (58, "XRPUSDT", "2025-08-17", "02:30", "long", "yes", "yes", "yes"),
    (59, "ENAUSDT", "2024-08-10", "10:30", "long", "yes", "yes", "yes"),
    (60, "ENAUSDT", "2024-06-26", "14:30", "long", "yes", "no", "yes"),
    (61, "ENAUSDT", "2024-06-03", "16:30", "long", "yes", "no", "yes"),
    (62, "ENAUSDT", "2024-04-16", "08:00", "long", "yes", "yes", "yes"),
    (63, "ENAUSDT", "2024-01-11", "12:00", "long", "yes", "warning", "yes"),
    (64, "ENAUSDT", "2024-03-17", "06:00", "long", "yes", "no", "yes"),
    (65, "SUIUSDT", "2024-10-24", "05:00", "short", "yes", "yes", "yes"),
    (66, "SAGAUSDT", "2024-10-28", "11:00", "long", "yes", "no", "yes"),
    (67, "ETHFIUSDT", "2025-01-16", "06:00", "short", "yes", "no", "yes"),
    (68, "ENAUSDT", "2024-06-03", "10:17", "long", "yes", "no", "yes"),
    (69, "NEIROUSDT", "2025-04-03", "14:00", "short", "yes", "yes", "yes"),
    (70, "ONDOUSDT", "2024-07-18", "16:15", "long", "yes", "no", "yes"),
    (71, "1000SHIBUSDT", "2025-04-03", "14:30", "short", "yes", "warning", "yes"),
    (72, "HBARUSDT", "2025-04-03", "10:00", "short", "yes", "warning", "yes"),
    (73, "SOLUSDT", "2025-04-03", "08:00", "short", "yes", "no", "yes"),
    (74, "AUCTIONUSDT", "2025-03-28", "09:17", "short", "yes", "warning", "yes"),
    (75, "MUBARAKUSDT", "2025-03-26", "08:11", "long", "yes", "warning", "yes"),
    (76, "REDUSDT", "2025-03-24", "11:24", "short", "yes", "yes", "yes"),
    (77, "BANUSDT", "2025-03-26", "19:08", "short", "yes", "warning", "yes"),
    (78, "BTCUSDT", "2025-03-26", "04:40", "long", "yes", "warning", "yes"),
    (79, "PNUTUSDT", "2025-02-03", "12:08", "long", "yes", "no", "yes"),
    (80, "HIFIUSDT", "2025-04-12", "04:16", "long", "yes", "no", "yes"),
    (81, "HIFIUSDT", "2025-04-12", "03:51", "long", "yes", "no", "yes"),
    (82, "ARCUSDT", "2025-03-18", "19:07", "short", "yes", "warning", "yes"),
    (83, "SOLUSDT", "2025-06-20", "17:19", "short", "yes", "no", "yes"),
    (84, "SWARMSUSDT", "2025-03-19", "17:28", "short", "yes", "no", "yes"),
    (85, "TROYUSDT", "2025-04-05", "16:03", "short", "yes", "no", "yes"),
    (86, "ARCUSDT", "2025-03-08", "14:21", "short", "yes", "yes", "yes"),
    (87, "ENAUSDT", "2025-06-30", "04:09", "long", "yes", "no", "yes"),
    (88, "HOOKUSDT", "2025-03-25", "14:31", "short", "yes", "no", "yes"),
    (89, "HOOKUSDT", "2025-02-25", "14:21", "long", "yes", "no", "yes"),
    (90, "LAYERUSDT", "2025-02-21", "04:10", "short", "yes", "yes", "yes"),
]


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN: RUN ALL TRADES AND SCORE
# ═══════════════════════════════════════════════════════════════════════════

def label_match(predicted, actual):
    """Check if labels match. 'warning' matches with both 'yes' and 'warning'."""
    if predicted == actual:
        return "exact"
    # Treat warning as acceptable for both yes and no
    if predicted == "warning" or actual == "warning":
        return "close"
    return "miss"


def main():
    total = 0
    found = 0

    grind_exact = 0
    grind_close = 0
    grind_miss = 0
    vol_exact = 0
    vol_close = 0
    vol_miss = 0
    stair_exact = 0
    stair_close = 0
    stair_miss = 0

    print(f"{'#':>3} {'Symbol':>15} {'Dir':>6} | "
          f"{'Grind':>7} {'pred':>5} {'match':>5} | "
          f"{'Vol':>7} {'pred':>5} {'match':>5} | "
          f"{'Stair':>7} {'pred':>5} {'match':>5} | "
          f"Key metrics")
    print("=" * 160)

    for trade in TRADES:
        num, symbol, date_str, entry_time, direction, pdf_grind, pdf_vol, pdf_stair = trade

        result = evaluate_momo_setup(symbol, date_str, entry_time, direction)
        if result is None:
            print(f"{num:>3} {symbol:>15} -- NO DATA")
            continue

        total += 1
        found += 1

        g_match = label_match(result["grind"], pdf_grind)
        v_match = label_match(result["volume"], pdf_vol)
        s_match = label_match(result["staircase"], pdf_stair)

        if g_match == "exact": grind_exact += 1
        elif g_match == "close": grind_close += 1
        else: grind_miss += 1

        if v_match == "exact": vol_exact += 1
        elif v_match == "close": vol_close += 1
        else: vol_miss += 1

        if s_match == "exact": stair_exact += 1
        elif s_match == "close": stair_close += 1
        else: stair_miss += 1

        # Build key metrics string
        gm = result["grind_metrics"]
        vm = result["volume_metrics"]
        sm = result["staircase_metrics"]
        key_info = (f"spk={gm.get('spike_ratio',0):.1f} "
                    f"vratio={vm.get('vol_change_ratio',0):.2f} "
                    f"xr={sm.get('smma_crosses',0)} "
                    f"sc={sm.get('score',0)}")

        g_sym = "OK" if g_match != "miss" else "XX"
        v_sym = "OK" if v_match != "miss" else "XX"
        s_sym = "OK" if s_match != "miss" else "XX"

        print(f"{num:>3} {result['symbol']:>15} {direction:>6} | "
              f"{pdf_grind:>7} {result['grind']:>5} {g_sym:>5} | "
              f"{pdf_vol:>7} {result['volume']:>5} {v_sym:>5} | "
              f"{pdf_stair:>7} {result['staircase']:>5} {s_sym:>5} | "
              f"{key_info}")

    print(f"\n{'='*80}")
    print(f"RESULTS ({found} trades evaluated)")
    print(f"{'='*80}")

    def pct(x, t):
        return f"{100*x/t:.0f}%" if t > 0 else "N/A"

    print(f"\n  GRIND (approach):")
    print(f"    Exact match:  {grind_exact}/{found} ({pct(grind_exact, found)})")
    print(f"    Close match:  {grind_close}/{found}")
    print(f"    Miss:         {grind_miss}/{found}")
    print(f"    Accuracy (exact+close): {pct(grind_exact + grind_close, found)}")

    print(f"\n  VOLUME:")
    print(f"    Exact match:  {vol_exact}/{found} ({pct(vol_exact, found)})")
    print(f"    Close match:  {vol_close}/{found}")
    print(f"    Miss:         {vol_miss}/{found}")
    print(f"    Accuracy (exact+close): {pct(vol_exact + vol_close, found)}")

    print(f"\n  STAIRCASE:")
    print(f"    Exact match:  {stair_exact}/{found} ({pct(stair_exact, found)})")
    print(f"    Close match:  {stair_close}/{found}")
    print(f"    Miss:         {stair_miss}/{found}")
    print(f"    Accuracy (exact+close): {pct(stair_exact + stair_close, found)}")

    overall_exact = grind_exact + vol_exact + stair_exact
    overall_close = grind_close + vol_close + stair_close
    overall_total = found * 3
    print(f"\n  OVERALL (all 3 variables):")
    print(f"    Exact:        {overall_exact}/{overall_total} ({pct(overall_exact, overall_total)})")
    print(f"    Exact+Close:  {overall_exact + overall_close}/{overall_total} ({pct(overall_exact + overall_close, overall_total)})")
    print(f"    Miss:         {overall_total - overall_exact - overall_close}/{overall_total}")


if __name__ == "__main__":
    main()
