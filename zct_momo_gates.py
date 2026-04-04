#!/usr/bin/env python3
"""
ZCT Momo v12 Gate Logic — ported from run_zct_momo_backtest_v12.py

Shared between backtest and live trader via strategies.py.

Signal: price breaks 6h high (long) or 6h low (short)
Gates (in order of cheapest to most expensive):
  1. Price < $100
  2. 2h price move >= 3%
  3. Duration: 2h with ≤3 SMMA30 crosses, price on correct side
  4. SMMA30 trending (change > 0.15%)
  5. Low noise: ≤6 debounced SMMA30 crosses in 2h
  6. SMMA5 alignment: >80% on correct side of SMMA30
  7. SMMA5 R² > 0.95 (trending)
  8. Staircase step detection: 5/7 segments with higher-lows/higher-closes
  9. Last 15 min grind: directional efficiency > 0.4
  10. Grind approach: spike ratio < 4.5
  11. Volume: nama30 not falling (longs) / not flat (shorts)
  12. VWAP side
  13. 30-bar wick noise < 0.55
  14. Longs only: volume acceleration >= 1.3
  15. Longs only: R² of last 30 bars >= 0.5
  16. RR guard (>= 0.95)

SL: 2nd swing low/high + buffer
TP: SL * 1.1 RR (capped at 4%)
"""

import numpy as np
import pandas as pd
from typing import Optional


# ── Excluded symbols ───────────────────────────────────────────────────────
ZCT_MOMO_EXCLUDED = {
    "BTCUSDT", "BNBUSDT", "ETHUSDT", "XRPUSDT", "LTCUSDT",
    "PAXGUSDT", "XAUUSDT", "XAGUSDT", "XPTUSDT", "BCHUSDT", "YFIUSDT",
}

MAX_PRICE = 100.0
MIN_2H_MOVE_PCT = 3.0
MIN_VOL_5M_USD = 400_000


# ── Helper functions ───────────────────────────────────────────────────────

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


def _find_second_swing(arr, entry, is_long, lookback=20, buffer_pct=0.15):
    """Find 2nd swing low/high for SL placement."""
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


def _compute_sl_tp(h_arr, l_arr, direction, entry_price,
                   min_sl=1.0, max_sl=3.0, rr=1.1, max_tp=4.0):
    """Compute SL/TP using 2nd swing low/high."""
    is_long = direction == "long"
    swing_ref = _find_second_swing(
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
# MAIN GATE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def check_zct_momo_v12_gates(df: pd.DataFrame, symbol: str,
                              depth_data: Optional[dict] = None) -> list[dict]:
    """
    Check ZCT Momo v12 gates on the latest bar of a DataFrame.

    Args:
        df: DataFrame with columns: timestamp, open, high, low, close, volume
            Must be sorted by timestamp, at least 400 bars.
        symbol: e.g. "SIRENUSDT"
        depth_data: optional order book data for wall filtering

    Returns:
        List of setup dicts (0, 1, or 2 — one per direction that passes).
        Each dict has: symbol, strategy, side, entry, sl, tp, sl_pct, tp_pct,
        rr, dps_total, dps_confidence, and gate metrics.
    """
    if symbol in ZCT_MOMO_EXCLUDED:
        return []

    n = len(df)
    if n < 400:
        return []

    # Extract arrays
    c = df["close"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    o = df["open"].values.astype(float)
    vol = df["volume"].values.astype(float)
    ts = df["timestamp"].values

    i = n - 1  # latest bar

    # ── Precompute indicators ──
    smma30 = pd.Series(c).ewm(alpha=1.0 / 30, adjust=False).mean().values
    smma5 = pd.Series(c).ewm(alpha=1.0 / 5, adjust=False).mean().values
    # vol * close = USD volume (matches backtest which uses Binance raw vol * close)
    # Bitunix quoteVol is already USD, but we multiply anyway to stay consistent
    # with backtest indicator computation on nama30
    vol_usd = vol * c
    nama30 = _nama(vol_usd, seed_len=30)

    # Wick ratio (suppress divide warnings for zero-range bars)
    rng = h - l
    body = np.abs(c - o)
    with np.errstate(divide="ignore", invalid="ignore"):
        wick = np.where(rng > 1e-12, (rng - body) / rng, 0.0)
    wick = np.nan_to_num(wick, nan=0.0)

    # 6h high/low (prior 6h, excluding current bar)
    if i < 361:
        return []
    hi6h = np.max(h[i - 361:i])
    lo6h = np.min(l[i - 361:i])

    # 5-min volume (prior 5 bars, excluding current — matches backtest)
    vol_5m = vol_usd[max(0, i - 5):i].sum()

    # VWAP (session — daily reset)
    tp_arr = (h + l + c) / 3.0
    dates = pd.to_datetime(ts).normalize()
    cum_tp_vol = 0.0
    cum_vol = 0.0
    current_date = None
    vwap_val = c[i]
    for j in range(n):
        d = dates[j]
        if d != current_date:
            cum_tp_vol = 0.0
            cum_vol = 0.0
            current_date = d
        cum_tp_vol += tp_arr[j] * vol[j]
        cum_vol += vol[j]
    if cum_vol > 0:
        vwap_val = cum_tp_vol / cum_vol

    # ── CHEAPEST: 5-min volume ──
    if vol_5m < MIN_VOL_5M_USD:
        return []

    # ── CHEAP: Price cap ──
    if c[i] > MAX_PRICE:
        return []

    # ── CHEAP: 2h price move minimum ──
    if i < 120:
        return []
    move_2h = abs(c[i] - c[i - 120]) / c[i - 120] * 100
    if move_2h < MIN_2H_MOVE_PCT:
        return []

    # ── SIGNAL DETECTION: breakout above 6h high / below 6h low ──
    signals = []
    if c[i] > hi6h:
        signals.append(("long", hi6h))
    if c[i] < lo6h:
        signals.append(("short", lo6h))
    if not signals:
        return []

    # ── CHECK GATES FOR EACH SIGNAL ──
    results = []
    min_dur_bars = 120

    for direction, level in signals:
        is_long = direction == "long"

        # ── GATE: Price on correct side of SMMA30 ──
        if is_long and c[i] < smma30[i]:
            continue
        if not is_long and c[i] > smma30[i]:
            continue

        # ── GATE: Duration — 2h with ≤3 debounced SMMA30 crosses ──
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
        if sm_change < 0.15:
            continue
        if is_long and sm_end <= sm_start:
            continue
        if not is_long and sm_end >= sm_start:
            continue

        # ── GATE: Low noise (≤6 crosses in 2h) ──
        stair_c = c[max(0, i - 119):i + 1]
        stair_sm = smma30[max(0, i - 119):i + 1]
        crosses = _count_smma_crosses(stair_c, stair_sm)
        if crosses > 6:
            continue

        # ── GATE: SMMA5 alignment + trending (nested — matches backtest) ──
        stair_sm5 = smma5[max(0, i - 119):i + 1]
        stair_sm30 = smma30[max(0, i - 119):i + 1]
        r2_sm5 = 0.0
        if len(stair_sm5) >= 60:
            # SMMA5 on correct side of SMMA30 for 80%+
            if is_long:
                sm5_pct = np.sum(stair_sm5 > stair_sm30) / len(stair_sm5) * 100
            else:
                sm5_pct = np.sum(stair_sm5 < stair_sm30) / len(stair_sm5) * 100
            if sm5_pct < 80:
                continue

            # SMMA5 R² > 0.95
            n_sm5 = len(stair_sm5)
            x_sm5 = np.arange(n_sm5, dtype=float)
            xm5, ym5 = np.mean(x_sm5), np.mean(stair_sm5)
            sxy5 = np.sum((x_sm5 - xm5) * (stair_sm5 - ym5))
            sxx5 = np.sum((x_sm5 - xm5) ** 2)
            syy5 = np.sum((stair_sm5 - ym5) ** 2)
            if sxx5 > 0 and syy5 > 0:
                r2_sm5 = (sxy5 ** 2) / (sxx5 * syy5)
                if r2_sm5 < 0.95:
                    continue
            else:
                continue

        # ── GATE: Staircase step detection (5/7 segments) ──
        n_segs = 8
        seg_size = 15
        seg_lows = []
        seg_highs = []
        seg_closes = []
        for si in range(n_segs):
            ss = i - 119 + si * seg_size
            se = ss + seg_size
            seg_lows.append(np.min(l[ss:se]))
            seg_highs.append(np.max(h[ss:se]))
            seg_closes.append(c[se - 1])

        if is_long:
            steps = sum(1 for j in range(1, n_segs)
                        if seg_lows[j] >= seg_lows[j-1] * 0.999
                        and seg_closes[j] >= seg_closes[j-1] * 0.999)
            dir_ok = seg_closes[-1] > seg_closes[0]
            no_deep_pb = all(sl >= seg_lows[0] * 0.995 for sl in seg_lows)
        else:
            steps = sum(1 for j in range(1, n_segs)
                        if seg_highs[j] <= seg_highs[j-1] * 1.001
                        and seg_closes[j] <= seg_closes[j-1] * 1.001)
            dir_ok = seg_closes[-1] < seg_closes[0]
            no_deep_pb = all(sh <= seg_highs[0] * 1.005 for sh in seg_highs)

        if steps < 5 or not dir_ok or not no_deep_pb:
            continue

        # ── GATE: Last 15 min grind (efficiency > 0.4) ──
        last15 = c[i - 14:i + 1]
        net_15 = abs(last15[-1] - last15[0])
        path_15 = np.sum(np.abs(np.diff(last15)))
        eff_15 = net_15 / path_15 if path_15 > 0 else 0
        if eff_15 < 0.4:
            continue
        if is_long and last15[-1] < last15[0]:
            continue
        if not is_long and last15[-1] > last15[0]:
            continue

        # ── GATE: Grind approach (no spike) ──
        spike_ratio = 0.0
        if i >= 10:
            tail_c = c[i - 9:i + 1]
            bar_moves = np.abs(np.diff(tail_c))
            avg_m = np.mean(bar_moves)
            max_m = np.max(bar_moves)
            spike_ratio = max_m / avg_m if avg_m > 0 else 0
            max_bar_pct = max_m / np.mean(tail_c) * 100 if np.mean(tail_c) > 0 else 0
            if spike_ratio > 4.5 and max_bar_pct > 0.5:
                continue
            if len(bar_moves) >= 7:
                accel = np.mean(bar_moves[-3:]) / np.mean(bar_moves[:-3]) if np.mean(bar_moves[:-3]) > 0 else 0
                if accel > 3.0 and max_bar_pct > 0.3:
                    continue

        # ── GATE: Volume — nama30 not falling (longs) / not flat (shorts) ──
        if i >= 11 and not np.isnan(nama30[i]) and not np.isnan(nama30[i - 10]):
            if is_long and nama30[i] < nama30[i - 10] * 0.95:
                continue
            if not is_long:
                vol_ratio = nama30[i] / nama30[i - 10] if nama30[i - 10] > 0 else 1.0
                if 0.95 <= vol_ratio <= 1.05:
                    continue

        # ── GATE: VWAP side ──
        if is_long and c[i] < vwap_val:
            continue
        if not is_long and c[i] > vwap_val:
            continue

        # ── GATE: 30-bar wick noise ──
        avg_wick = np.mean(wick[max(0, i - 29):i + 1])
        if avg_wick > 0.55:
            continue

        # ── GATE (longs only): Volume acceleration >= 1.3 ──
        vol_accel = 0.0
        if is_long and i >= 120:
            vol_usd_stair = vol_usd[max(0, i - 119):i + 1]
            recent_v = np.mean(vol_usd_stair[-30:])
            prior_v = np.mean(vol_usd_stair[:30])
            vol_accel = recent_v / prior_v if prior_v > 0 else 1.0
            if vol_accel < 1.3:
                continue

        # ── GATE (longs only): R² of last 30 bars >= 0.5 ──
        r2_30 = 0.0
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

        # ── ALL GATES PASSED — compute SL/TP ──
        entry_level = level
        sl_val, tp_val, sl_pct, tp_pct = _compute_sl_tp(
            h[:i + 1], l[:i + 1], direction, entry_level)

        # RR guard
        if is_long:
            eff_rr = (tp_val - entry_level) / (entry_level - sl_val) if entry_level > sl_val else 0
        else:
            eff_rr = (entry_level - tp_val) / (sl_val - entry_level) if sl_val > entry_level else 0
        if eff_rr < 0.95:
            continue

        # ── Depth wall filter: skip if big wall between entry and TP ──
        depth_wall_between = None
        if depth_data is not None:
            depth_wall_between = _check_wall_between_entry_tp(
                depth_data, entry_level, tp_val, direction)
            if depth_wall_between and depth_wall_between.get("blocked"):
                continue

        # ── DPS scoring ──
        dps_dur = 2 if dur_crosses == 0 else 1
        dps_app = 2 if spike_ratio < 2.5 else (1 if spike_ratio < 3.5 else 0)

        if i >= 11 and not np.isnan(nama30[i]) and not np.isnan(nama30[i - 10]):
            vol_change = (nama30[i] - nama30[i - 10]) / nama30[i - 10] if nama30[i - 10] > 0 else 0
            if vol_change > 0.05:
                dps_vol = 2; vol_label = "increasing"
            elif vol_change > -0.05:
                if is_long:
                    dps_vol = 1; vol_label = "flat"
                else:
                    dps_vol = 0; vol_label = "flat"
            else:
                if is_long:
                    dps_vol = 0; vol_label = "decreasing"
                else:
                    dps_vol = 1; vol_label = "decreasing"
        else:
            dps_vol = 1; vol_label = "flat"

        dps_total = dps_dur + dps_app + dps_vol
        dps_conf = ("max" if dps_total >= 6 else
                    ("high" if dps_total >= 4 else
                     ("low" if dps_total >= 3 else "avoid")))

        # ── MEASURE (not gate): candle metrics ──
        if i >= 15:
            last15_range = h[i - 14:i + 1] - l[i - 14:i + 1]
            last15_pct = last15_range / c[i - 14:i + 1] * 100
            max_candle_15m = float(np.max(last15_pct))
        else:
            max_candle_15m = 0.0

        # Max drawdown in 2h
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

        results.append({
            "symbol": symbol,
            "strategy": "zct_momo",
            "timestamp": str(ts[i]),
            "side": direction,
            "level": round(level, 8),
            "entry": round(entry_level, 8),
            "sl": round(sl_val, 8),
            "tp": round(tp_val, 8),
            "sl_pct": round(sl_pct, 3),
            "tp_pct": round(tp_pct, 3),
            "rr": round(eff_rr, 2),
            "dps_total": dps_total,
            "dps_confidence": dps_conf,
            "dps_dur": dps_dur,
            "dps_app": dps_app,
            "dps_vol": dps_vol,
            "vol_trend": vol_label,
            "smma_crosses": crosses,
            "r2_sm5": round(r2_sm5, 4),
            "steps": steps,
            "eff_15m": round(eff_15, 3),
            "spike_ratio": round(spike_ratio, 2),
            "vol_accel": round(vol_accel, 3),
            "r2_30": round(r2_30, 4),
            "max_candle_15m": round(max_candle_15m, 3),
            "max_dd_2h": round(dd_pct, 3),
            "depth_wall": depth_wall_between,
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════
# DEPTH WALL FILTER
# ═══════════════════════════════════════════════════════════════════════════

def _check_wall_between_entry_tp(depth_data: dict, entry: float, tp: float,
                                  side: str,
                                  min_wall_strength: float = 3.0) -> dict:
    """
    Check if there's a significant wall between entry and TP that would
    block price from reaching the target.

    For longs: check ask-side walls between entry and tp (entry < tp)
    For shorts: check bid-side walls between tp and entry (tp < entry)

    Returns dict with wall info, or None if no wall found.
    'blocked' key is True if wall is strong enough to skip the trade.
    """
    if not depth_data:
        return None

    asks = depth_data.get("asks", [])
    bids = depth_data.get("bids", [])

    if side == "long":
        # Walls above entry, below TP = resistance in the way
        walls_in_path = []
        for level in asks:
            price = float(level[0])
            qty = float(level[1])
            if entry < price < tp:
                usd_val = price * qty
                walls_in_path.append({"price": price, "qty": qty, "usd": usd_val})
    else:
        # Walls below entry, above TP = support in the way
        walls_in_path = []
        for level in bids:
            price = float(level[0])
            qty = float(level[1])
            if tp < price < entry:
                usd_val = price * qty
                walls_in_path.append({"price": price, "qty": qty, "usd": usd_val})

    if not walls_in_path:
        return {"blocked": False, "walls": 0, "max_wall_usd": 0}

    # Find the biggest wall
    biggest = max(walls_in_path, key=lambda w: w["usd"])

    # Compute wall strength: biggest wall USD vs average level USD
    if side == "long":
        all_levels = [{"price": float(a[0]), "qty": float(a[1]),
                       "usd": float(a[0]) * float(a[1])} for a in asks[:20]]
    else:
        all_levels = [{"price": float(b[0]), "qty": float(b[1]),
                       "usd": float(b[0]) * float(b[1])} for b in bids[:20]]

    avg_level_usd = np.mean([lv["usd"] for lv in all_levels]) if all_levels else 1.0
    wall_strength = biggest["usd"] / avg_level_usd if avg_level_usd > 0 else 0

    blocked = wall_strength >= min_wall_strength

    return {
        "blocked": blocked,
        "walls": len(walls_in_path),
        "max_wall_usd": round(biggest["usd"], 0),
        "max_wall_price": biggest["price"],
        "max_wall_strength": round(wall_strength, 1),
        "avg_level_usd": round(avg_level_usd, 0),
    }
