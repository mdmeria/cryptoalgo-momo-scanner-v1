#!/usr/bin/env python3
"""
Analyze 100 B+ ZCT momentum trade setups against our dataset.
Computes: R², channel width, staircase quality, grind quality, volume trend,
SMMA noise, approach speed — the key metrics for detecting high-quality momo setups.
"""
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta

DATA_DIR = "datasets/binance_futures_1m"

# ── All 99 trades extracted from the PDF screenshots ──────────────────────
# Format: (trade_num, symbol, date, entry_time_utc, direction, grind, volume, staircase)
# grind/volume/staircase: 'yes', 'no', 'warning'
TRADES = [
    # Batch 1 (pages 7-20)
    (1, "ENAUSDT", "2024-08-02", "15:00", "short", "yes", "no", "warning"),
    (2, "ENAUSDT", "2024-08-02", "15:19", "short", "yes", "yes", "warning"),
    (3, "ENAUSDT", "2024-08-05", "18:15", "long", "yes", "warning", "yes"),
    (4, "FLOKIUSDT", "2024-08-07", "10:00", "long", "yes", "yes", "yes"),  # 1000FLOKIUSDT on binance
    (5, "ENAUSDT", "2024-08-15", "09:30", "long", "yes", "yes", "yes"),
    (6, "ENAUSDT", "2024-08-15", "10:00", "short", "yes", "yes", "yes"),
    (7, "TRXUSDT", "2024-08-20", "14:00", "long", "yes", "yes", "yes"),
    (8, "TRXUSDT", "2024-08-21", "14:30", "short", "yes", "no", "yes"),
    (9, "AAVEUSDT", "2024-08-28", "10:00", "long", "yes", "no", "yes"),
    (10, "TRXUSDT", "2024-08-29", "07:14", "short", "yes", "yes", "yes"),
    (11, "ENAUSDT", "2024-09-18", "12:30", "long", "yes", "yes", "yes"),
    (12, "AAVEUSDT", "2024-10-02", "12:30", "short", "yes", "no", "yes"),
    (13, "AAVEUSDT", "2024-10-01", "05:30", "short", "yes", "yes", "warning"),
    (14, "AAVEUSDT", "2024-10-01", "05:30", "short", "yes", "no", "yes"),  # page20

    # Batch 2 (pages 21-36)
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

    # Batch 3 (pages 37-52)
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

    # Batch 4 (pages 53-69)
    (45, "FILUSDT", "2025-08-22", "08:14", "long", "warning", "yes", "yes"),  # future date - may not have data
    (46, "SUIUSDT", "2025-06-24", "06:30", "long", "yes", "no", "yes"),  # future
    (47, "HBARUSDT", "2025-07-12", "00:20", "short", "yes", "warning", "yes"),  # future
    (48, "THETAUSDT", "2025-07-16", "04:15", "short", "yes", "warning", "yes"),  # future
    (49, "GALAUSDT", "2025-06-28", "09:00", "long", "warning", "yes", "yes"),  # future
    (50, "ZENUSDT", "2024-12-18", "05:41", "long", "yes", "no", "yes"),
    (51, "SAGAUSDT", "2024-11-19", "01:15", "long", "yes", "no", "yes"),
    (52, "SUIUSDT", "2025-06-23", "08:30", "long", "yes", "yes", "yes"),  # future
    (53, "SOLUSDT", "2025-08-15", "07:30", "short", "yes", "yes", "yes"),  # future
    (54, "ICPUSDT", "2025-08-14", "09:00", "long", "yes", "no", "yes"),  # future
    (55, "ORDIUSDT", "2025-01-20", "19:00", "short", "yes", "warning", "yes"),
    (56, "FILUSDT", "2025-08-14", "06:00", "long", "yes", "no", "yes"),  # future
    (57, "BTCUSDT", "2025-08-14", "16:30", "long", "yes", "yes", "yes"),  # future
    (58, "XRPUSDT", "2025-08-17", "02:30", "long", "yes", "yes", "yes"),  # future

    # Batch 5 (pages 70-87)
    (59, "ENAUSDT", "2024-08-10", "10:30", "long", "yes", "yes", "yes"),
    (60, "ENAUSDT", "2024-06-26", "14:30", "long", "yes", "no", "yes"),
    (61, "ENAUSDT", "2024-06-03", "16:30", "long", "yes", "no", "yes"),
    (62, "ENAUSDT", "2024-04-16", "08:00", "long", "yes", "yes", "yes"),
    (63, "ENAUSDT", "2024-01-11", "12:00", "long", "yes", "warning", "yes"),  # ENA may not exist Jan 2024
    (64, "ENAUSDT", "2024-03-17", "06:00", "long", "yes", "no", "yes"),  # same
    (65, "SUIUSDT", "2024-10-24", "05:00", "short", "yes", "yes", "yes"),
    (66, "SAGAUSDT", "2024-10-28", "11:00", "long", "yes", "no", "yes"),
    (67, "ETHFIUSDT", "2025-01-16", "06:00", "short", "yes", "no", "yes"),
    (68, "ENAUSDT", "2024-06-03", "10:17", "long", "yes", "no", "yes"),
    (69, "NEIROUSDT", "2025-04-03", "14:00", "short", "yes", "yes", "yes"),  # may not exist
    (70, "ONDOUSDT", "2024-07-18", "16:15", "long", "yes", "no", "yes"),
    (71, "1000SHIBUSDT", "2025-04-03", "14:30", "short", "yes", "warning", "yes"),
    (72, "HBARUSDT", "2025-04-03", "10:00", "short", "yes", "warning", "yes"),
    (73, "SOLUSDT", "2025-04-03", "08:00", "short", "yes", "no", "yes"),

    # Batch 6 (pages 88-106)
    (74, "AUCTIONUSDT", "2025-03-28", "09:17", "short", "yes", "warning", "yes"),
    (75, "MUBARAKUSDT", "2025-03-26", "08:11", "long", "yes", "warning", "yes"),
    (76, "REDUSDT", "2025-03-24", "11:24", "short", "yes", "yes", "yes"),
    (77, "BANUSDT", "2025-03-26", "19:08", "short", "yes", "warning", "yes"),
    (78, "BTCUSDT", "2025-03-26", "04:40", "long", "yes", "warning", "yes"),
    (79, "PNUTUSDT", "2025-02-03", "12:08", "long", "yes", "no", "yes"),
    (80, "HIFIUSDT", "2025-04-12", "04:16", "long", "yes", "no", "yes"),  # future-ish
    (81, "HIFIUSDT", "2025-04-12", "03:51", "long", "yes", "no", "yes"),  # same
    (82, "ARCUSDT", "2025-03-18", "19:07", "short", "yes", "warning", "yes"),
    (83, "SOLUSDT", "2025-06-20", "17:19", "short", "yes", "no", "yes"),  # future
    (84, "SWARMSUSDT", "2025-03-19", "17:28", "short", "yes", "no", "yes"),
    (85, "TROYUSDT", "2025-04-05", "16:03", "short", "yes", "no", "yes"),  # future-ish
    (86, "ARCUSDT", "2025-03-08", "14:21", "short", "yes", "yes", "yes"),
    (87, "ENAUSDT", "2025-06-30", "04:09", "long", "yes", "no", "yes"),  # future
    (88, "HOOKUSDT", "2025-03-25", "14:31", "short", "yes", "no", "yes"),
    (89, "HOOKUSDT", "2025-02-25", "14:21", "long", "yes", "no", "yes"),
    (90, "LAYERUSDT", "2025-02-21", "04:10", "short", "yes", "yes", "yes"),
]


def smma(arr, period):
    result = np.zeros(len(arr))
    result[0] = arr[0]
    alpha = 1.0 / period
    for i in range(1, len(arr)):
        result[i] = result[i-1] * (1 - alpha) + arr[i] * alpha
    return result


def calc_regression(closes, highs, lows):
    n = len(closes)
    if n < 10:
        return None
    x = np.arange(n, dtype=float)
    xm, ym = np.mean(x), np.mean(closes)
    sxy = np.sum((x - xm) * (closes - ym))
    sxx = np.sum((x - xm) ** 2)
    syy = np.sum((closes - ym) ** 2)
    if sxx == 0 or syy == 0:
        return None
    slope = sxy / sxx
    r2 = (sxy ** 2) / (sxx * syy)
    pred = slope * x + (ym - slope * xm)
    std = np.std(closes - pred)
    ch = std / ym * 100 * 2
    going_up = slope > 0
    if going_up:
        rh = np.maximum.accumulate(highs)
        max_dd = np.max((rh - lows) / rh * 100)
    else:
        rl = np.minimum.accumulate(lows)
        max_dd = np.max((highs - rl) / rl * 100)
    return {
        'r2': r2, 'channel': ch, 'max_dd': max_dd,
        'slope': slope / ym * 100, 'going_up': going_up,
    }


def count_smma_crosses(closes, smma_vals, min_bars_away=5):
    """Count debounced SMMA30 crosses."""
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
                    tentative_run = 0
            else:
                tentative_side = None
                tentative_run = 0
    return crossovers


def analyze_approach(closes, highs, lows, direction, lookback=10):
    """
    Analyze the last N bars approaching the entry level.
    Returns metrics about grind quality vs spike quality.
    """
    if len(closes) < lookback:
        return None

    tail_c = closes[-lookback:]
    tail_h = highs[-lookback:]
    tail_l = lows[-lookback:]

    # 1. Max single-bar move (spike detection)
    bar_moves = np.abs(np.diff(tail_c))
    avg_bar_move = np.mean(bar_moves) if len(bar_moves) > 0 else 0
    max_bar_move = np.max(bar_moves) if len(bar_moves) > 0 else 0
    price_level = np.mean(tail_c)
    max_bar_move_pct = (max_bar_move / price_level * 100) if price_level > 0 else 0
    avg_bar_move_pct = (avg_bar_move / price_level * 100) if price_level > 0 else 0

    # 2. Spike ratio: max bar move vs average bar move
    spike_ratio = (max_bar_move / avg_bar_move) if avg_bar_move > 0 else 0

    # 3. Consistency: how many bars move in the right direction
    if direction == "long":
        dir_bars = np.sum(np.diff(tail_c) > 0)
    else:
        dir_bars = np.sum(np.diff(tail_c) < 0)
    dir_consistency = dir_bars / (lookback - 1) * 100

    # 4. Candle body uniformity (std of body sizes / mean body size)
    bodies = np.abs(tail_c - closes[-lookback - 1:-1]) if len(closes) > lookback else np.abs(np.diff(tail_c))
    bodies = np.abs(tail_c[1:] - tail_c[:-1])
    body_mean = np.mean(bodies)
    body_std = np.std(bodies)
    body_cv = (body_std / body_mean) if body_mean > 0 else 0

    # 5. Net move vs total distance (efficiency)
    net_move = abs(tail_c[-1] - tail_c[0])
    total_dist = np.sum(np.abs(np.diff(tail_c)))
    efficiency = (net_move / total_dist) if total_dist > 0 else 0

    # 6. Linearity (R² of last N bars)
    reg = calc_regression(tail_c, tail_h, tail_l)
    r2_approach = reg['r2'] if reg else 0

    return {
        'max_bar_move_pct': round(max_bar_move_pct, 4),
        'avg_bar_move_pct': round(avg_bar_move_pct, 4),
        'spike_ratio': round(spike_ratio, 2),
        'dir_consistency_pct': round(dir_consistency, 1),
        'body_cv': round(body_cv, 3),
        'efficiency': round(efficiency, 3),
        'r2_approach': round(r2_approach, 4),
    }


def analyze_staircase(closes, highs, lows, opens, smma30, direction, lookback=120):
    """
    Analyze 2h staircase quality.
    """
    n = min(len(closes), lookback)
    c = closes[-n:]
    h = highs[-n:]
    l = lows[-n:]
    o = opens[-n:]
    sm = smma30[-n:]

    # 1. R² over 2h
    reg = calc_regression(c, h, l)
    if reg is None:
        return None

    # 2. SMMA30 crosses
    valid = ~np.isnan(sm)
    if valid.sum() < 30:
        smma_crosses = -1
    else:
        smma_crosses = count_smma_crosses(c[valid], sm[valid])

    # 3. Directional consistency
    if direction == "long":
        dir_candles = np.sum(c > o)
    else:
        dir_candles = np.sum(c < o)
    dir_pct = dir_candles / n * 100

    # 4. Wick ratio
    ranges = h - l
    bodies = np.abs(c - o)
    safe_ranges = np.where(ranges < 1e-12, np.nan, ranges)
    wicks = (ranges - bodies) / safe_ranges
    avg_wick = float(np.nanmean(wicks))

    # 5. Pullback analysis (how many times does price pull back > X%)
    price_mean = np.mean(c)
    if direction == "long":
        running_high = np.maximum.accumulate(h)
        pullbacks = (running_high - l) / running_high * 100
    else:
        running_low = np.minimum.accumulate(l)
        pullbacks = (h - running_low) / running_low * 100
    max_pullback = np.max(pullbacks)
    pullback_count_05 = np.sum(pullbacks > 0.5)  # pullbacks > 0.5%

    # 6. Step detection: count "steps" (pauses in the staircase)
    # A step = a cluster of 5+ bars within a tight range before moving further
    bar_returns = np.diff(c) / c[:-1] * 100
    step_count = 0
    in_step = False
    step_len = 0
    for r in bar_returns:
        if abs(r) < 0.02:  # very small move
            step_len += 1
            if step_len >= 5 and not in_step:
                step_count += 1
                in_step = True
        else:
            in_step = False
            step_len = 0

    return {
        'r2_2h': round(reg['r2'], 4),
        'channel_2h': round(reg['channel'], 4),
        'max_dd_2h': round(reg['max_dd'], 4),
        'slope_2h': round(reg['slope'], 4),
        'smma_crosses': smma_crosses,
        'dir_pct': round(dir_pct, 1),
        'avg_wick': round(avg_wick, 3),
        'max_pullback_pct': round(max_pullback, 4),
        'pullback_count_05': pullback_count_05,
        'step_count': step_count,
    }


def analyze_volume(volumes, closes, lookback_staircase=120, lookback_recent=30):
    """
    Analyze volume trend over the staircase period.
    """
    vol_usd = volumes * closes

    # 1. Full staircase volume slope (120 bars)
    n = min(len(vol_usd), lookback_staircase)
    v_stair = vol_usd[-n:]
    v_avg = np.mean(v_stair)
    if v_avg <= 0:
        return None
    slope_full = np.polyfit(np.arange(n), v_stair, 1)[0]
    slope_full_norm = slope_full / v_avg

    # 2. Recent 30-bar volume slope
    n30 = min(len(vol_usd), lookback_recent)
    v_recent = vol_usd[-n30:]
    v_avg_recent = np.mean(v_recent)
    slope_recent = np.polyfit(np.arange(n30), v_recent, 1)[0]
    slope_recent_norm = slope_recent / v_avg_recent if v_avg_recent > 0 else 0

    # 3. Compare first half vs second half average
    half = n // 2
    first_half_avg = np.mean(v_stair[:half])
    second_half_avg = np.mean(v_stair[half:])
    vol_ratio = second_half_avg / first_half_avg if first_half_avg > 0 else 0

    # 4. Volume spike detection (any single bar > 3x average)
    vol_spikes = np.sum(v_stair > 3 * v_avg)

    # 5. Label
    if slope_full_norm > 0.01:
        label = "increasing"
    elif slope_full_norm < -0.01:
        label = "decreasing"
    else:
        label = "flat"

    return {
        'vol_slope_full_norm': round(slope_full_norm, 5),
        'vol_slope_recent_norm': round(slope_recent_norm, 5),
        'vol_ratio_2nd_vs_1st': round(vol_ratio, 3),
        'vol_spikes_3x': int(vol_spikes),
        'vol_label': label,
        'vol_avg_usd': round(v_avg, 0),
    }


_DF_CACHE = {}

def _load_symbol_df(fname):
    """Load and cache a symbol's full dataframe."""
    if fname not in _DF_CACHE:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            _DF_CACHE[fname] = None
            return None
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        _DF_CACHE[fname] = df
    return _DF_CACHE[fname]


def load_data(symbol, date_str, entry_time_str):
    """Load 1m data for the given symbol around the entry time."""
    # Try different symbol name variations
    candidates = [
        f"{symbol}_1m.csv",
        f"1000{symbol}_1m.csv" if not symbol.startswith("1000") else None,
    ]
    # Special mappings
    if symbol == "FLOKIUSDT":
        candidates = ["1000FLOKIUSDT_1m.csv", "FLOKIUSDT_1m.csv"]
    if symbol == "SHIBUSDT":
        candidates = ["1000SHIBUSDT_1m.csv", "SHIBUSDT_1m.csv"]
    if symbol == "NEIROUSDT":
        candidates = ["NEIROUSDT_1m.csv", "NEIROETHUSDT_1m.csv"]

    candidates = [c for c in candidates if c is not None]

    for fname in candidates:
        df = _load_symbol_df(fname)
        if df is None:
            continue
        # Find the entry bar
        entry_dt = pd.Timestamp(f"{date_str} {entry_time_str}:00", tz="UTC")
        # Get 4 hours before entry (240 bars) plus 1 hour after (60 bars)
        start = entry_dt - timedelta(hours=4)
        end = entry_dt + timedelta(hours=1)
        mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
        df_slice = df[mask].copy().reset_index(drop=True)
        if len(df_slice) < 60:
            continue
        # Find the entry bar index
        diffs = (df_slice["timestamp"] - entry_dt).abs()
        entry_idx = diffs.idxmin()
        return df_slice, entry_idx, fname.replace("_1m.csv", "")
    return None, None, None


def analyze_trade(trade_tuple):
    """Full analysis of one trade."""
    num, symbol, date_str, entry_time, direction, grind_label, vol_label, stair_label = trade_tuple

    df, entry_idx, actual_symbol = load_data(symbol, date_str, entry_time)
    if df is None:
        return {"trade_num": num, "symbol": symbol, "status": "NO_DATA"}

    # Use bars up to entry point
    df_pre = df.iloc[:entry_idx + 1].copy()
    if len(df_pre) < 30:
        return {"trade_num": num, "symbol": actual_symbol, "status": "INSUFFICIENT_BARS"}

    c = df_pre["close"].values.astype(float)
    h = df_pre["high"].values.astype(float)
    l = df_pre["low"].values.astype(float)
    o = df_pre["open"].values.astype(float)
    v = df_pre["volume"].values.astype(float)

    # Compute SMMA30
    sm30 = smma(c, 30)

    # 1. Approach analysis (last 10 bars)
    approach = analyze_approach(c, h, l, direction, lookback=10)

    # 2. Staircase analysis (last 120 bars or available)
    staircase = analyze_staircase(c, h, l, o, sm30, direction, lookback=min(120, len(c)))

    # 3. Volume analysis
    volume = analyze_volume(v, c, lookback_staircase=min(120, len(c)))

    # 4. Noise classification (SMMA30 crosses in 2h)
    if staircase and staircase['smma_crosses'] >= 0:
        crosses = staircase['smma_crosses']
        if crosses <= 3:
            noise_class = "low"
        elif crosses <= 6:
            noise_class = "medium"
        else:
            noise_class = "high"
    else:
        noise_class = "unknown"

    # Also check post-entry for outcome (did trade work?)
    df_post = df.iloc[entry_idx:].copy()
    outcome = None
    if len(df_post) > 5:
        entry_price = df_post.iloc[0]["close"]
        post_c = df_post["close"].values
        post_h = df_post["high"].values
        post_l = df_post["low"].values
        if direction == "long":
            max_fav = (np.max(post_h) - entry_price) / entry_price * 100
            max_adv = (entry_price - np.min(post_l)) / entry_price * 100
        else:
            max_fav = (entry_price - np.min(post_l)) / entry_price * 100
            max_adv = (np.max(post_h) - entry_price) / entry_price * 100
        outcome = {
            'max_favorable_pct': round(max_fav, 3),
            'max_adverse_pct': round(max_adv, 3),
        }

    return {
        "trade_num": num,
        "symbol": actual_symbol,
        "date": date_str,
        "entry_time": entry_time,
        "direction": direction,
        "labels": {"grind": grind_label, "volume": vol_label, "staircase": stair_label},
        "status": "OK",
        "approach": approach,
        "staircase": staircase,
        "volume": volume,
        "noise_class": noise_class,
        "outcome": outcome,
    }


def main():
    results = []
    found = 0
    missing = 0

    for trade in TRADES:
        result = analyze_trade(trade)
        results.append(result)
        if result["status"] == "OK":
            found += 1
        else:
            missing += 1
            print(f"  Trade #{result['trade_num']} {result['symbol']}: {result['status']}")

    print(f"\n{'='*80}")
    print(f"DATA AVAILABILITY: {found}/{len(TRADES)} trades found ({missing} missing)")
    print(f"{'='*80}\n")

    ok_results = [r for r in results if r["status"] == "OK"]

    # ── APPROACH ANALYSIS ──────────────────────────────────────────────────
    print("=" * 80)
    print("APPROACH QUALITY (last 10 bars before entry)")
    print("=" * 80)
    approach_metrics = {}
    for r in ok_results:
        if r["approach"]:
            for k, v in r["approach"].items():
                approach_metrics.setdefault(k, []).append(v)

    for metric, values in approach_metrics.items():
        arr = np.array(values)
        print(f"  {metric:25s}  mean={np.mean(arr):8.4f}  std={np.std(arr):8.4f}  "
              f"p10={np.percentile(arr, 10):8.4f}  p50={np.percentile(arr, 50):8.4f}  "
              f"p90={np.percentile(arr, 90):8.4f}")

    # ── STAIRCASE ANALYSIS ─────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("STAIRCASE QUALITY (2h before entry)")
    print("=" * 80)
    stair_metrics = {}
    for r in ok_results:
        if r["staircase"]:
            for k, v in r["staircase"].items():
                stair_metrics.setdefault(k, []).append(v)

    for metric, values in stair_metrics.items():
        arr = np.array(values)
        print(f"  {metric:25s}  mean={np.mean(arr):8.4f}  std={np.std(arr):8.4f}  "
              f"p10={np.percentile(arr, 10):8.4f}  p50={np.percentile(arr, 50):8.4f}  "
              f"p90={np.percentile(arr, 90):8.4f}")

    # ── VOLUME ANALYSIS ────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("VOLUME ANALYSIS")
    print("=" * 80)
    vol_metrics = {}
    for r in ok_results:
        if r["volume"]:
            for k, v in r["volume"].items():
                if isinstance(v, (int, float)):
                    vol_metrics.setdefault(k, []).append(v)

    for metric, values in vol_metrics.items():
        arr = np.array(values)
        print(f"  {metric:25s}  mean={np.mean(arr):8.4f}  std={np.std(arr):8.4f}  "
              f"p10={np.percentile(arr, 10):8.4f}  p50={np.percentile(arr, 50):8.4f}  "
              f"p90={np.percentile(arr, 90):8.4f}")

    # Volume label vs PDF label agreement
    print(f"\n  Volume label agreement with PDF labels:")
    agree = 0
    total_vol = 0
    for r in ok_results:
        if r["volume"] and r["labels"]["volume"] != "warning":
            total_vol += 1
            computed = r["volume"]["vol_label"]
            pdf = r["labels"]["volume"]
            # Map: PDF 'yes' = increasing, 'no' = not increasing
            if (pdf == "yes" and computed == "increasing") or \
               (pdf == "no" and computed != "increasing"):
                agree += 1
    print(f"    Agreement: {agree}/{total_vol} ({100*agree/total_vol:.0f}%)" if total_vol > 0 else "    N/A")

    # ── NOISE ANALYSIS ─────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("NOISE CLASSIFICATION")
    print("=" * 80)
    from collections import Counter
    noise_counts = Counter(r["noise_class"] for r in ok_results)
    for nc, count in sorted(noise_counts.items()):
        print(f"  {nc:15s}: {count}")

    # ── APPROACH: GRIND vs SPIKE THRESHOLDS ────────────────────────────────
    print(f"\n{'='*80}")
    print("GRIND vs SPIKE: Comparing PDF-labeled grind=YES vs grind=WARNING/NO")
    print("=" * 80)
    grind_yes = [r for r in ok_results if r["labels"]["grind"] == "yes" and r["approach"]]
    grind_not = [r for r in ok_results if r["labels"]["grind"] != "yes" and r["approach"]]

    if grind_yes:
        print(f"\n  Grind = YES (n={len(grind_yes)}):")
        for metric in ['spike_ratio', 'max_bar_move_pct', 'dir_consistency_pct', 'efficiency', 'r2_approach', 'body_cv']:
            vals = [r["approach"][metric] for r in grind_yes]
            print(f"    {metric:25s}  mean={np.mean(vals):8.4f}  p25={np.percentile(vals, 25):8.4f}  p75={np.percentile(vals, 75):8.4f}")

    if grind_not:
        print(f"\n  Grind = WARNING/NO (n={len(grind_not)}):")
        for metric in ['spike_ratio', 'max_bar_move_pct', 'dir_consistency_pct', 'efficiency', 'r2_approach', 'body_cv']:
            vals = [r["approach"][metric] for r in grind_not]
            print(f"    {metric:25s}  mean={np.mean(vals):8.4f}  p25={np.percentile(vals, 25):8.4f}  p75={np.percentile(vals, 75):8.4f}")

    # ── OUTCOME ANALYSIS ───────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("TRADE OUTCOMES (1h post-entry)")
    print("=" * 80)
    outcomes = [r["outcome"] for r in ok_results if r["outcome"]]
    if outcomes:
        fav = [o["max_favorable_pct"] for o in outcomes]
        adv = [o["max_adverse_pct"] for o in outcomes]
        print(f"  Max favorable excursion:  mean={np.mean(fav):.3f}%  p50={np.percentile(fav, 50):.3f}%  p90={np.percentile(fav, 90):.3f}%")
        print(f"  Max adverse excursion:    mean={np.mean(adv):.3f}%  p50={np.percentile(adv, 50):.3f}%  p90={np.percentile(adv, 90):.3f}%")

    # ── PER-TRADE DETAIL DUMP ──────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("PER-TRADE DETAILS")
    print("=" * 80)
    print(f"{'#':>3} {'Symbol':>15} {'Date':>12} {'Dir':>6} {'G':>3} {'V':>3} {'S':>3} | "
          f"{'R2_2h':>6} {'Ch%':>6} {'DD%':>6} {'SMMAxr':>6} {'Noise':>6} | "
          f"{'SpkRat':>6} {'DirCon':>6} {'Effic':>6} {'R2apr':>6} | "
          f"{'VSlope':>8} {'VRatio':>6} {'VLbl':>6} | "
          f"{'MFE%':>6} {'MAE%':>6}")
    print("-" * 170)

    for r in ok_results:
        s = r["staircase"] or {}
        a = r["approach"] or {}
        vo = r["volume"] or {}
        oc = r["outcome"] or {}
        gl = r["labels"]

        g_sym = "Y" if gl["grind"] == "yes" else ("W" if gl["grind"] == "warning" else "N")
        v_sym = "Y" if gl["volume"] == "yes" else ("W" if gl["volume"] == "warning" else "N")
        s_sym = "Y" if gl["staircase"] == "yes" else ("W" if gl["staircase"] == "warning" else "N")

        print(f"{r['trade_num']:>3} {r['symbol']:>15} {r['date']:>12} {r['direction']:>6} {g_sym:>3} {v_sym:>3} {s_sym:>3} | "
              f"{s.get('r2_2h', 0):>6.3f} {s.get('channel_2h', 0):>6.3f} {s.get('max_dd_2h', 0):>6.3f} {s.get('smma_crosses', -1):>6} {r.get('noise_class', '?'):>6} | "
              f"{a.get('spike_ratio', 0):>6.2f} {a.get('dir_consistency_pct', 0):>6.1f} {a.get('efficiency', 0):>6.3f} {a.get('r2_approach', 0):>6.3f} | "
              f"{vo.get('vol_slope_full_norm', 0):>8.5f} {vo.get('vol_ratio_2nd_vs_1st', 0):>6.3f} {vo.get('vol_label', '?'):>6} | "
              f"{oc.get('max_favorable_pct', 0):>6.3f} {oc.get('max_adverse_pct', 0):>6.3f}")

    # Save results as JSON for further analysis
    with open("datasets/100_momo_trades_analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to datasets/100_momo_trades_analysis.json")


if __name__ == "__main__":
    main()
