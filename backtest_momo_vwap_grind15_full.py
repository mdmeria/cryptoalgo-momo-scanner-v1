#!/usr/bin/env python3
"""
Gate-based momentum backtest engine.

Loads 1-minute OHLCV data from a pre-built dataset, scans every eligible bar
for a momentum signal, applies the full gate stack from momo_gate_settings.json,
computes entry/SL/TP, then resolves the trade outcome forward in the same dataset.

Usage:
  python backtest_momo_vwap_grind15_full.py \
    --dataset-dir datasets/momo_1m_7d_top100_midcap_30d \
    --gates-config momo_gate_settings.json \
    --out-prefix momo_run

Outputs (all CSV):
  {out_prefix}_overall.csv      – single-row summary with settings + aggregate stats
  {out_prefix}_by_symbol.csv    – per-symbol win/loss/count breakdown
  {out_prefix}_trade_list.csv   – every trade with entry, SL, TP, outcome, duration
  {out_prefix}_skipped.csv      – bars that passed early gates but failed later ones (why)
"""

from __future__ import annotations

import argparse
import json
import math
import sys

# Force UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple  # noqa: F401

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Gate settings dataclass
# ---------------------------------------------------------------------------

@dataclass
class GateSettings:
    # Gate ON/OFF
    enable_2h_gate: bool = True
    enable_10m_gate: bool = True
    enable_30m_noise_gate: bool = True
    enable_5m_antispike_gate: bool = True
    enable_volusd_gate: bool = True
    enable_regime_breakout_gate: bool = True
    enable_vwap_side_gate: bool = True
    enable_tp_sl_keylevel_gate: bool = True
    enable_rr_guard: bool = True
    enable_min_tp_sl_pct_gate: bool = True

    # 10m gate params
    grind10_min: float = 0.52          # directional efficiency floor
    wick_ratio_10m_max: float = 0.55   # max wick-to-range ratio
    angle_min_deg: float = 45.0
    angle_max_deg: float = 60.0

    # 2h directional gate params
    dir_lookback_bars: int = 120
    dir_move_min_pct: float = 1.0      # min net directional move %
    dir_eff_min: float = 0.35          # min directional efficiency
    noise_wick_max: float = 0.60       # max wick ratio over 2h

    # 30m / 5m noise and spike params
    noise_wick_30m_max: float = 0.55
    spike5_max_abs_ret_pct: float = 1.5
    spike5_max_range_pct: float = 2.5

    # Core thresholds
    day_change_pct: float = 5.0        # min abs 24h change % for regime gate
    min_profit_pct: float = 1.0        # min TP and SL distance from entry
    chop_range_pct: float = 0.3        # min net move % required in any 30-bar window within 2h

    # Volume USD filter
    min_vol_usd_5m: float = 0.0        # min total USD volume over last 5 bars (0 = disabled)

    # 7 EMA cross gate
    enable_ema7_cross_gate: bool = True
    ema7_cross_lookback: int = 30      # bars to look back for the cross event

    # Staircase quality gate
    enable_staircase_gate: bool = True
    staircase_lookback: int = 120      # bars to evaluate (2h)
    staircase_dir_pct_min: float = 60.0   # min % of candles in trend direction
    staircase_wick_max: float = 0.35      # max avg wick ratio over lookback
    staircase_max_crosses: int = 3        # max debounced SMMA30 crosses
    staircase_max_ret_std: float = 0.20   # max std dev of per-bar returns (smoothness)
    staircase_min_net_move_pct: float = 1.0  # min net directional move %
    staircase_abv_ma_min: float = 80.0   # min % bars on correct side of SMMA30 over 2h
    staircase_max_crosses_smma: int = 2  # max debounced SMMA30 crosses (FET-style)
    staircase_min_consistent_segs: int = 5  # min 15-min segments with dir>=53% (out of 8)

    # Last-15min momentum gate (PENGU-style staircase into entry)
    enable_last15m_gate: bool = False
    last15m_dir_pct_min: float = 60.0    # min % candles in trend direction
    last15m_dir_pct_max: float = 80.0    # max % candles (filters spikes/waterfalls)
    last15m_net_move_min: float = 0.2    # min net move % in trade direction
    last15m_abv_ma_min: float = 80.0     # min % of bars above/below SMMA30
    last15m_hl_min: int = 8              # min higher-lows (long) or lower-highs (short) out of 14
    last15m_max_pullback_pct: float = 0.5  # max adverse pullback within 15 bars
    last15m_min_pullback_pct: float = 0.2  # min pullback (filters dead/flat zones)
    last15m_max_consecutive: int = 5     # max same-direction run (filters spikes)
    last15m_vol_steady_min: float = 60.0 # min % bars with volume >= 50% of avg (ZCT steady vol)

    # Trade management (not in JSON — hardcoded defaults)
    rr: float = 1.1                    # reward-to-risk ratio
    min_sl_pct: float = 1.0            # floor on SL distance (ZCT: always at least 1%)
    max_sl_pct: float = 2.0            # cap on SL distance (ZCT: never more than 2%)
    max_tp_pct: float = 4.0            # ceiling on TP distance (ZCT: never more than 4%)
    # 6h level freshness gate
    entry_not_crossed_6h: bool = True  # entry level must be untouched in prior 6h
    # Trigger bar: wick rejection requirements
    min_wick_ratio_trigger: float = 0.30  # trigger bar must have wick >= 30% of its range

    # Retest entry parameters
    max_bars_to_confirm: int = 8       # bars to wait for 2 closes above level
    max_bars_to_retest: int = 8        # bars after confirmation to get retest
    cancel_r_extension: float = 0.75  # cancel trade if price runs >0.75R before retest
    # Trade resolution
    max_hold_bars: int = 60            # ZCT: setups resolve within ~1h max

    @classmethod
    def from_json(cls, path: str) -> "GateSettings":
        with open(path) as f:
            data = json.load(f)
        obj = cls()
        for k, v in data.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
        return obj

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _smma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1.0 / length, adjust=False).mean()


def _nama(src: pd.Series, seed_len: int = 60) -> pd.Series:
    """Pine Script nama(): SMA seed, then fixed-alpha EWM (alpha=1.618/100)."""
    alpha = 1.618 / 100.0
    out = src.copy().astype(float)
    sma_seed = src.rolling(seed_len).mean()
    first_valid = sma_seed.first_valid_index()
    if first_valid is None:
        return out * float("nan")
    iloc_start = src.index.get_loc(first_valid)
    out.iloc[:iloc_start] = float("nan")
    out.iloc[iloc_start] = float(sma_seed.iloc[iloc_start])
    for i in range(iloc_start + 1, len(src)):
        out.iloc[i] = alpha * float(src.iloc[i]) + (1.0 - alpha) * float(out.iloc[i - 1])
    return out


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def _linear_slope(values: pd.Series) -> float:
    clean = values.dropna()
    if len(clean) < 5:
        return 0.0
    x = np.arange(len(clean), dtype=float)
    y = clean.to_numpy(dtype=float)
    if np.allclose(y, y[0]):
        return 0.0
    slope, _ = np.polyfit(x, y, 1)
    return float(slope / max(abs(np.mean(y)), 1e-9))


def _wick_ratio(df: pd.DataFrame) -> pd.Series:
    """Upper + lower wick total as fraction of bar range."""
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    body = (df["close"] - df["open"]).abs()
    return ((rng - body) / rng).fillna(0.0)


def _vwap(df: pd.DataFrame) -> float:
    if df.empty:
        return float("nan")
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    denom = float(df["volume"].sum())
    return float((tp * df["volume"]).sum() / denom) if denom > 0 else float("nan")


# ---------------------------------------------------------------------------
# Prepare features on a symbol DataFrame (called once per symbol)
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add indicator columns to a symbol's full 1m DataFrame."""
    df = df.copy()
    df["ema7"] = df["close"].ewm(span=7, adjust=False).mean()
    df["smma30"] = _smma(df["close"], 30)
    df["smma120"] = _smma(df["close"], 120)
    df["atr14"] = _atr(df, 14)
    df["vol_usd"] = df["volume"] * df["close"]
    df["nama60"] = _nama(df["vol_usd"], seed_len=60)
    df["wick_ratio"] = _wick_ratio(df)
    # 24h rolling change % (approx using 1440-bar rolling window)
    df["day_open"] = df["close"].shift(1440)
    df["day_change_pct"] = (df["close"] - df["day_open"]) / df["day_open"].replace(0, np.nan) * 100.0
    # Session VWAP: cumulative from start of each UTC day
    df["date"] = df.index.normalize()
    df["tp"] = (df["high"] + df["low"] + df["close"]) / 3.0
    df["tp_vol"] = df["tp"] * df["volume"]
    df["cum_tp_vol"] = df.groupby("date")["tp_vol"].cumsum()
    df["cum_vol"] = df.groupby("date")["volume"].cumsum()
    df["vwap"] = df["cum_tp_vol"] / df["cum_vol"].replace(0, np.nan)
    # 6h rolling high/low
    df["high_6h"] = df["high"].rolling(360).max()
    df["low_6h"] = df["low"].rolling(360).min()
    return df


# ---------------------------------------------------------------------------
# Gate functions — each takes a slice ending at the candidate bar
# ---------------------------------------------------------------------------

def gate_volusd(df_slice: pd.DataFrame, lookback: int = 10) -> bool:
    """VolUsd MA60 must be higher than it was 10 bars ago."""
    nama = df_slice["nama60"]
    if nama.isna().all() or len(nama) < lookback + 2:
        return True  # not enough data — pass
    v_now = float(nama.iloc[-1])
    v_prev = float(nama.iloc[-(lookback + 1)])
    if math.isnan(v_now) or math.isnan(v_prev):
        return True
    return v_now > v_prev


def gate_vol_usd_5m(df_slice: pd.DataFrame, cfg: GateSettings) -> bool:
    """Total USD volume over last 5 bars must exceed min_vol_usd_5m."""
    if cfg.min_vol_usd_5m <= 0:
        return True
    vol5 = float(df_slice["vol_usd"].iloc[-5:].sum())
    return vol5 >= cfg.min_vol_usd_5m


def gate_30m_noise(df_slice: pd.DataFrame, cfg: GateSettings) -> bool:
    """Average wick ratio over last 30 bars must be below threshold."""
    wr = df_slice["wick_ratio"].iloc[-30:]
    if len(wr) < 20:
        return True
    return float(wr.mean()) <= cfg.noise_wick_30m_max


def gate_5m_antispike(df_slice: pd.DataFrame, cfg: GateSettings) -> bool:
    """Last 5 bars must not show an outsized return or range."""
    tail5 = df_slice.iloc[-5:]
    if len(tail5) < 5:
        return True
    abs_ret = abs(float(tail5["close"].iloc[-1]) - float(tail5["close"].iloc[0])) / max(float(tail5["close"].iloc[0]), 1e-9) * 100.0
    rng = (float(tail5["high"].max()) - float(tail5["low"].min())) / max(float(tail5["close"].iloc[0]), 1e-9) * 100.0
    return abs_ret <= cfg.spike5_max_abs_ret_pct and rng <= cfg.spike5_max_range_pct


def gate_10m_directional(df_slice: pd.DataFrame, side: str, cfg: GateSettings) -> Tuple[bool, str]:
    """10-bar directional efficiency, wick cap, and angle band."""
    tail10 = df_slice.iloc[-10:]
    close = tail10["close"]
    is_long = side == "long"

    start, end = float(close.iloc[0]), float(close.iloc[-1])
    raw_move = (end - start) / max(abs(start), 1e-9)
    dir_move = raw_move if is_long else -raw_move

    path = close.pct_change().abs().fillna(0.0).sum()
    efficiency = float(dir_move / max(float(path), 1e-9))

    avg_wick = float(df_slice["wick_ratio"].iloc[-10:].mean())

    # Angle: atan of total % move over 10 bars.
    # 1% move  → atan(1.0)  = 45°   (minimum grind)
    # 1.73% move → atan(1.73) = 60°  (maximum before spike territory)
    # Uses dir_move * 100 to convert fraction → percent.
    angle_deg = float(math.degrees(math.atan(abs(dir_move) * 100.0)))

    dir_ok = efficiency >= cfg.grind10_min
    wick_ok = avg_wick <= cfg.wick_ratio_10m_max
    angle_ok = cfg.angle_min_deg <= abs(angle_deg) <= cfg.angle_max_deg

    passed = dir_ok and wick_ok and angle_ok
    skip_reason = ""
    if not passed:
        reasons = []
        if not dir_ok:
            reasons.append(f"efficiency={efficiency:.2f}<{cfg.grind10_min}")
        if not wick_ok:
            reasons.append(f"wick={avg_wick:.2f}>{cfg.wick_ratio_10m_max}")
        if not angle_ok:
            reasons.append(f"angle={angle_deg:.1f}deg")
        skip_reason = ";".join(reasons)
    return passed, skip_reason


def gate_regime_breakout(df_slice: pd.DataFrame, side: str, cfg: GateSettings) -> Tuple[bool, str]:
    """Regime: 24h change >= threshold AND close beyond prior 6h extreme."""
    day_chg = float(df_slice["day_change_pct"].iloc[-1])
    close = float(df_slice["close"].iloc[-1])
    is_long = side == "long"

    if math.isnan(day_chg):
        return True, ""  # not enough history — pass

    if is_long:
        required_chg = day_chg >= cfg.day_change_pct
        # Compare vs prior 6h high (use bar at -361 to avoid current 6h window)
        prior_6h_hi = float(df_slice["high"].iloc[-361:-1].max()) if len(df_slice) > 362 else float("nan")
        breakout_ok = math.isnan(prior_6h_hi) or close > prior_6h_hi
        passed = required_chg and breakout_ok
        if not passed:
            if not required_chg:
                return False, f"day_chg={day_chg:.1f}%<{cfg.day_change_pct}"
            return False, f"no_6h_breakout close={close:.4f} hi6h={prior_6h_hi:.4f}"
    else:
        required_chg = day_chg <= -cfg.day_change_pct
        prior_6h_lo = float(df_slice["low"].iloc[-361:-1].min()) if len(df_slice) > 362 else float("nan")
        breakout_ok = math.isnan(prior_6h_lo) or close < prior_6h_lo
        passed = required_chg and breakout_ok
        if not passed:
            if not required_chg:
                return False, f"day_chg={day_chg:.1f}%>{-cfg.day_change_pct}"
            return False, f"no_6h_breakout close={close:.4f} lo6h={prior_6h_lo:.4f}"

    return True, ""


def gate_vwap_side(df_slice: pd.DataFrame, side: str) -> bool:
    """Close must be on the correct side of session VWAP."""
    vwap = float(df_slice["vwap"].iloc[-1])
    close = float(df_slice["close"].iloc[-1])
    if math.isnan(vwap):
        return True
    if side == "long":
        return close > vwap
    return close < vwap


def gate_smma_trend(df_slice: pd.DataFrame, side: str, cfg: GateSettings) -> Tuple[bool, str]:
    """
    ZCT Noise gate — Low Noise required for momentum (per ZCT Quantifying Noise doc).

    Three checks over the last 120 bars (2h):

    1. smma30 is TRENDING in the right direction (not sideways).
       Long: smma30[-1] > smma30[0].  Short: smma30[-1] < smma30[0].

    2. Price crosses smma30 at most 3 times = Low Noise per ZCT table.
       A cross = candle close switches from one side of smma30 to the other.
       0-3 = Low, 4-6 = Medium, 7+ = High (only Low passes here).

    3. smma30 and smma120 do not criss-cross — they stay in order throughout
       (smma30 above smma120 for long, below for short).
    """
    lb = cfg.dir_lookback_bars  # 120 bars = 2h
    if len(df_slice) < lb:
        return True, ""

    left = df_slice.iloc[-lb:]
    is_long = side == "long"

    close_s  = left["close"].to_numpy(dtype=float)
    smma30_s = left["smma30"].to_numpy(dtype=float)
    smma120_s = left["smma120"].to_numpy(dtype=float)

    # Drop leading NaN warmup bars
    valid = ~(np.isnan(smma30_s) | np.isnan(smma120_s) | np.isnan(close_s))
    if valid.sum() < 30:
        return True, ""
    close_s   = close_s[valid]
    smma30_s  = smma30_s[valid]
    smma120_s = smma120_s[valid]

    # --- Check 1: smma30 trending in the right direction ---
    smma_trending = bool(smma30_s[-1] > smma30_s[0]) if is_long else bool(smma30_s[-1] < smma30_s[0])

    # --- Check 2: count debounced crossovers of smma30 (0-3 = Low Noise) ---
    # A new cross is only counted after price has traveled >= 5 bars on one side.
    # Choppy back-and-forth within 5 bars of the MA counts as one touch, not multiple.
    above = close_s > smma30_s
    min_bars_away = 5
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
                # bounced back before 5 bars — chop, reset tentative
                tentative_side = None
                tentative_run = 0
                if above[i] != confirmed_side:
                    tentative_side = above[i]
                    tentative_run = 1
    noise_ok = crossovers <= 3

    # --- Check 3: smma30 and smma120 stay in correct order (no criss-cross) ---
    gap = smma30_s - smma120_s
    no_smma_cross = bool((gap > 0).all()) if is_long else bool((gap < 0).all())

    passed = smma_trending and noise_ok and no_smma_cross
    if not passed:
        reasons = []
        if not smma_trending:
            reasons.append("smma30_sideways")
        if not noise_ok:
            reasons.append(f"crosses={crossovers}>3")
        if not no_smma_cross:
            reasons.append("smma30_120_cross")
        return False, ";".join(reasons)
    return True, ""


def gate_ema7_cross(df_slice: pd.DataFrame, side: str, cfg: GateSettings) -> Tuple[bool, str]:
    """
    ZCT 7 EMA cross confirmation gate.

    For longs:  7 EMA must be currently above SMMA30, AND the cross (from below
                to above) must have happened within the last `ema7_cross_lookback` bars.
    For shorts: 7 EMA must be currently below SMMA30, AND the cross happened recently.

    This confirms short-term momentum is accelerating in the trade direction.
    """
    lb = cfg.ema7_cross_lookback
    if len(df_slice) < lb + 2:
        return True, ""

    tail = df_slice.iloc[-lb:]
    ema7 = tail["ema7"].to_numpy(dtype=float)
    smma30 = tail["smma30"].to_numpy(dtype=float)

    valid = ~(np.isnan(ema7) | np.isnan(smma30))
    if valid.sum() < 10:
        return True, ""
    ema7 = ema7[valid]
    smma30 = smma30[valid]

    is_long = side == "long"

    # Check 1: EMA7 currently on correct side of SMMA30
    if is_long:
        current_ok = ema7[-1] > smma30[-1]
    else:
        current_ok = ema7[-1] < smma30[-1]

    if not current_ok:
        return False, "ema7_wrong_side"

    # Check 2: cross event happened within the lookback window
    above = ema7 > smma30
    cross_found = False
    for i in range(1, len(above)):
        if is_long and above[i] and not above[i - 1]:
            cross_found = True
            break
        if not is_long and not above[i] and above[i - 1]:
            cross_found = True
            break

    if not cross_found:
        return False, "ema7_no_recent_cross"

    return True, ""


def gate_last15m(df_slice: pd.DataFrame, side: str, cfg: GateSettings) -> Tuple[bool, str]:
    """
    Last-15min momentum gate — PENGU-style staircase into entry.

    Checks the final 15 bars before entry for:
    1. Directional consistency >= last15m_dir_pct_min
    2. Net move in trade direction >= last15m_net_move_min
    3. % of bars on correct side of SMMA30 >= last15m_abv_ma_min
    4. Staircase structure: higher-lows (long) / lower-highs (short) >= last15m_hl_min
    5. Shallow pullbacks: max adverse pullback <= last15m_max_pullback_pct
    6. No spike runs: max consecutive same-dir candles <= last15m_max_consecutive
    """
    if len(df_slice) < 20:
        return True, ""

    tail = df_slice.iloc[-15:]
    is_long = side == "long"

    closes = tail["close"].to_numpy(dtype=float)
    opens = tail["open"].to_numpy(dtype=float)
    highs = tail["high"].to_numpy(dtype=float)
    lows = tail["low"].to_numpy(dtype=float)
    smma30 = tail["smma30"].to_numpy(dtype=float)

    # 1. Dir consistency
    if is_long:
        dir_pct = np.sum(closes > opens) / 15 * 100.0
    else:
        dir_pct = np.sum(closes < opens) / 15 * 100.0

    # 2. Net move in trade direction
    net = (closes[-1] - closes[0]) / max(abs(closes[0]), 1e-12) * 100.0
    net_dir = net if is_long else -net

    # 3. % above/below SMMA30
    valid = ~np.isnan(smma30)
    if valid.sum() > 0:
        if is_long:
            abv_pct = np.sum(closes[valid] > smma30[valid]) / valid.sum() * 100.0
        else:
            abv_pct = np.sum(closes[valid] < smma30[valid]) / valid.sum() * 100.0
    else:
        abv_pct = 100.0

    # 4. Staircase structure: higher-lows (long) or lower-highs (short)
    if is_long:
        hl_count = int(np.sum(lows[1:] > lows[:-1]))
    else:
        hl_count = int(np.sum(highs[1:] < highs[:-1]))

    # 5. Max pullback depth within the 15 bars
    if is_long:
        running_high = np.maximum.accumulate(closes)
        pullbacks = (running_high - lows) / np.maximum(running_high, 1e-12) * 100.0
    else:
        running_low = np.minimum.accumulate(closes)
        pullbacks = (highs - running_low) / np.maximum(running_low, 1e-12) * 100.0
    max_pullback = float(np.max(pullbacks))

    # 6. Max consecutive same-direction candles (spike filter)
    dirs = closes > opens if is_long else closes < opens
    max_run = 0
    cur_run = 0
    for d in dirs:
        if d:
            cur_run += 1
            if cur_run > max_run:
                max_run = cur_run
        else:
            cur_run = 0

    # 7. Volume steadiness (ZCT Increasing/Steady Volume)
    volumes = tail["volume"].to_numpy(dtype=float)
    vol_mean = float(np.mean(volumes))
    if vol_mean > 0:
        vol_steady = float(np.sum(volumes >= vol_mean * 0.5) / 15 * 100.0)
    else:
        vol_steady = 100.0

    reasons = []
    if dir_pct < cfg.last15m_dir_pct_min:
        reasons.append(f"l15_dir={dir_pct:.0f}%")
    if dir_pct > cfg.last15m_dir_pct_max:
        reasons.append(f"l15_dir={dir_pct:.0f}%>{cfg.last15m_dir_pct_max:.0f}%")
    if net_dir < cfg.last15m_net_move_min:
        reasons.append(f"l15_net={net_dir:+.2f}%")
    if abv_pct < cfg.last15m_abv_ma_min:
        reasons.append(f"l15_abvMA={abv_pct:.0f}%")
    if hl_count < cfg.last15m_hl_min:
        reasons.append(f"l15_hl={hl_count}<{cfg.last15m_hl_min}")
    if max_pullback > cfg.last15m_max_pullback_pct:
        reasons.append(f"l15_pb={max_pullback:.2f}%>{cfg.last15m_max_pullback_pct}")
    if max_pullback < cfg.last15m_min_pullback_pct:
        reasons.append(f"l15_pb={max_pullback:.2f}%<{cfg.last15m_min_pullback_pct}")
    if max_run > cfg.last15m_max_consecutive:
        reasons.append(f"l15_run={max_run}>{cfg.last15m_max_consecutive}")
    if vol_steady < cfg.last15m_vol_steady_min:
        reasons.append(f"l15_volSt={vol_steady:.0f}%<{cfg.last15m_vol_steady_min:.0f}%")

    if reasons:
        return False, ";".join(reasons)
    return True, ""


def gate_staircase_quality(df_slice: pd.DataFrame, side: str, cfg: GateSettings) -> Tuple[bool, str]:
    """
    ZCT staircase quality gate — enforces clean, consistent momentum over 2h.

    Checks:
    1. Directional consistency: >= staircase_dir_pct_min % candles in trend direction
    2. Low wicks: avg wick ratio <= staircase_wick_max (clean bodies)
    3. Low noise: debounced SMMA30 crosses <= staircase_max_crosses
    4. Smooth returns: std dev of per-bar returns <= staircase_max_ret_std
    5. Net move: >= staircase_min_net_move_pct in the right direction
    """
    lb = cfg.staircase_lookback
    if len(df_slice) < lb + 5:
        return True, ""

    tail = df_slice.iloc[-lb:]
    is_long = side == "long"

    opens = tail["open"].to_numpy(dtype=float)
    closes = tail["close"].to_numpy(dtype=float)
    highs = tail["high"].to_numpy(dtype=float)
    lows = tail["low"].to_numpy(dtype=float)
    smma30_s = tail["smma30"].to_numpy(dtype=float)

    valid = ~np.isnan(smma30_s)
    if valid.sum() < 30:
        return True, ""

    reasons = []

    # 1. Directional consistency
    if is_long:
        dir_candles = np.sum(closes > opens)
    else:
        dir_candles = np.sum(closes < opens)
    dir_pct = dir_candles / lb * 100.0
    if dir_pct < cfg.staircase_dir_pct_min:
        reasons.append(f"dir={dir_pct:.0f}%<{cfg.staircase_dir_pct_min:.0f}%")

    # 2. Low wick ratio
    ranges = highs - lows
    bodies = np.abs(closes - opens)
    safe_ranges = np.where(ranges < 1e-12, np.nan, ranges)
    wicks = (ranges - bodies) / safe_ranges
    avg_wick = float(np.nanmean(wicks))
    if avg_wick > cfg.staircase_wick_max:
        reasons.append(f"wick={avg_wick:.2f}>{cfg.staircase_wick_max}")

    # 3. Debounced SMMA30 crosses
    close_v = closes[valid]
    smma_v = smma30_s[valid]
    above = close_v > smma_v
    min_bars_away = 5
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
                if above[i] != confirmed_side:
                    tentative_side = above[i]
                    tentative_run = 1
    if crossovers > cfg.staircase_max_crosses:
        reasons.append(f"crosses={crossovers}>{cfg.staircase_max_crosses}")

    # 4. Return smoothness
    rets = np.diff(closes) / np.maximum(closes[:-1], 1e-12) * 100.0
    ret_std = float(np.std(rets))
    if ret_std > cfg.staircase_max_ret_std:
        reasons.append(f"ret_std={ret_std:.2f}>{cfg.staircase_max_ret_std}")

    # 5. Net directional move
    net_move = (closes[-1] - closes[0]) / max(abs(closes[0]), 1e-12) * 100.0
    if is_long:
        move_ok = net_move >= cfg.staircase_min_net_move_pct
    else:
        move_ok = -net_move >= cfg.staircase_min_net_move_pct
    if not move_ok:
        reasons.append(f"net_move={net_move:.2f}%")

    # 6. AbvMA: % of bars on correct side of SMMA30 over 2h
    close_valid = closes[valid]
    smma_valid = smma30_s[valid]
    if is_long:
        abv_pct = float(np.sum(close_valid > smma_valid) / len(close_valid) * 100.0)
    else:
        abv_pct = float(np.sum(close_valid < smma_valid) / len(close_valid) * 100.0)
    if abv_pct < cfg.staircase_abv_ma_min:
        reasons.append(f"abvMA={abv_pct:.0f}%<{cfg.staircase_abv_ma_min:.0f}%")

    # 7. SMMA30 crosses (tighter FET-style check, separate from check #3)
    if crossovers > cfg.staircase_max_crosses_smma:
        reasons.append(f"xing={crossovers}>{cfg.staircase_max_crosses_smma}")

    # 8. Consistent segments: how many 15-min segments have dir >= 53%
    n_segs = lb // 15
    consistent_segs = 0
    for seg_i in range(n_segs):
        seg_c = closes[seg_i * 15 : (seg_i + 1) * 15]
        seg_o = opens[seg_i * 15 : (seg_i + 1) * 15]
        if len(seg_c) < 15:
            continue
        if is_long:
            seg_dir = np.sum(seg_c > seg_o) / len(seg_c) * 100.0
        else:
            seg_dir = np.sum(seg_c < seg_o) / len(seg_c) * 100.0
        if seg_dir >= 53.0:
            consistent_segs += 1
    if consistent_segs < cfg.staircase_min_consistent_segs:
        reasons.append(f"segs={consistent_segs}<{cfg.staircase_min_consistent_segs}")

    if reasons:
        return False, ";".join(reasons)
    return True, ""


def gate_entry_not_crossed_6h(df_slice: pd.DataFrame, side: str) -> bool:
    """
    Entry level must be fresh — not crossed by any bar in the prior ~6h window.

    For long: entry_level = high of the last bar (trigger level).
    For short: entry_level = low of the last bar.

    Checks bars from -370 to -10 (roughly 6h ago up to 10 bars ago).
    If any bar's range encompasses the level → stale level → reject.
    """
    is_long = side == "long"
    entry_level = float(df_slice["high"].iloc[-1]) if is_long else float(df_slice["low"].iloc[-1])

    n = len(df_slice)
    # Need at least 10 bars of recent data before the 6h lookback
    if n < 12:
        return True  # not enough history — pass

    window_end = n - 10          # exclude last 10 bars (recent action)
    window_start = max(0, n - 370)  # ~6h back
    prior = df_slice.iloc[window_start:window_end]

    if prior.empty:
        return True

    crossed = (prior["low"] <= entry_level) & (prior["high"] >= entry_level)
    return not crossed.any()


# ---------------------------------------------------------------------------
# SL / TP calculation
# ---------------------------------------------------------------------------

def _find_second_swing_low(
    lows: pd.Series,
    entry: float,
    is_long: bool,
    lookback: int = 20,
    buffer_pct: float = 0.15,
) -> Optional[float]:
    """
    Find the second swing low (for longs) or swing high (for shorts) below/above entry
    within the last `lookback` bars.

    ZCT rule: look back ~10 min for V-shape structure, use the SECOND level
    below entry (not the nearest) as the SL reference.

    A V-shape swing low: lows[i] < lows[i-1] and lows[i] < lows[i+1]

    Returns the price level for SL reference, or None if not enough swings found.
    """
    tail = lows.iloc[-lookback:]
    arr = tail.to_numpy(dtype=float)
    n = len(arr)

    swing_prices = []
    for i in range(1, n - 1):
        if is_long:
            # V-shape low: lower than both neighbours
            if arr[i] < arr[i - 1] and arr[i] < arr[i + 1] and arr[i] < entry:
                swing_prices.append(arr[i])
        else:
            # inverted V-shape high: higher than both neighbours
            if arr[i] > arr[i - 1] and arr[i] > arr[i + 1] and arr[i] > entry:
                swing_prices.append(arr[i])

    if not swing_prices:
        return None

    if is_long:
        # Sort descending (closest to entry first), take second
        swing_prices.sort(reverse=True)
    else:
        # Sort ascending (closest to entry first), take second
        swing_prices.sort()

    # Use second swing if available, else first
    ref = swing_prices[1] if len(swing_prices) >= 2 else swing_prices[0]

    # Add buffer below (for long) or above (for short)
    if is_long:
        return ref * (1.0 - buffer_pct / 100.0)
    else:
        return ref * (1.0 + buffer_pct / 100.0)


def compute_sl_tp(
    df_slice: pd.DataFrame,
    side: str,
    cfg: GateSettings,
    entry_price: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    Return (entry, sl, tp).

    SL placement (ZCT rule):
      1. Find the second V-shape swing low in the last 20 bars below entry.
      2. Place SL 0.15% below that level.
      3. Floor at min_sl_pct (default 1.0%) from entry.
      4. Cap at max_sl_pct to avoid over-wide stops.

    TP = entry ± sl_distance × rr, capped at max_tp_pct.

    entry_price: if provided, use this as entry (e.g. retest level). Otherwise use last close.
    """
    is_long = side == "long"
    entry = entry_price if entry_price is not None else float(df_slice["close"].iloc[-1])

    # --- SL: second swing low below entry ---
    swing_sl = _find_second_swing_low(
        lows=df_slice["low"] if is_long else df_slice["high"],
        entry=entry,
        is_long=is_long,
        lookback=20,
        buffer_pct=0.15,
    )

    if swing_sl is not None:
        if is_long:
            struct_pct = (entry - swing_sl) / max(entry, 1e-9) * 100.0
        else:
            struct_pct = (swing_sl - entry) / max(entry, 1e-9) * 100.0
    else:
        struct_pct = cfg.min_sl_pct  # fallback

    # Apply floors and caps
    sl_pct = max(struct_pct, cfg.min_sl_pct)   # never tighter than 1%
    sl_pct = min(sl_pct, cfg.max_sl_pct)        # never wider than max

    tp_pct = min(sl_pct * cfg.rr, cfg.max_tp_pct)

    if is_long:
        sl = entry * (1.0 - sl_pct / 100.0)
        tp = entry * (1.0 + tp_pct / 100.0)
    else:
        sl = entry * (1.0 + sl_pct / 100.0)
        tp = entry * (1.0 - tp_pct / 100.0)

    return entry, sl, tp


def rr_guard(entry: float, sl: float, tp: float, side: str, min_rr: float = 0.95) -> Tuple[bool, float]:
    """Effective RR must be >= min_rr (default 0.95 to allow 1R with rounding)."""
    if side == "long":
        risk = entry - sl
        reward = tp - entry
    else:
        risk = sl - entry
        reward = entry - tp
    if risk <= 0:
        return False, 0.0
    eff_rr = reward / risk
    return eff_rr >= min_rr, eff_rr


def min_tp_sl_gate(entry: float, sl: float, tp: float, side: str, min_pct: float) -> bool:
    """Both TP and SL must be at least min_pct% from entry."""
    if side == "long":
        sl_pct = (entry - sl) / entry * 100.0
        tp_pct = (tp - entry) / entry * 100.0
    else:
        sl_pct = (sl - entry) / entry * 100.0
        tp_pct = (entry - tp) / entry * 100.0
    return sl_pct >= min_pct and tp_pct >= min_pct


# ---------------------------------------------------------------------------
# Forward trade resolution
# ---------------------------------------------------------------------------

def find_retest_entry(
    df: pd.DataFrame,
    trigger_iloc: int,
    side: str,
    sl_pct: float,
    max_bars_to_confirm: int = 8,
    max_bars_to_retest: int = 8,
    cancel_r_extension: float = 0.75,
    level_override: Optional[float] = None,
) -> Optional[Tuple[int, float]]:
    """
    After trigger bar (all gates passed), find ZCT-style retest entry.

    ZCT procedure:
      1. Level = level_override if provided, else high of trigger bar (long) / low (short).
      2. Wait for 2 consecutive 1m closes above/below level = breakout confirmed.
      3. If price extends > level + 0.75*SL before retest → cancel (missed trade).
      4. After confirmation, wait for price to retrace back to level → fill at level.

    Returns (entry_bar_iloc, entry_price) or None if missed/cancelled.
    """
    is_long = side == "long"

    if level_override is not None:
        level = level_override
    elif is_long:
        level = float(df["high"].iloc[trigger_iloc])
    else:
        level = float(df["low"].iloc[trigger_iloc])

    if is_long:
        cancel_price = level * (1.0 + cancel_r_extension * sl_pct / 100.0)
    else:
        cancel_price = level * (1.0 - cancel_r_extension * sl_pct / 100.0)

    # Phase 1: wait for 2 consecutive closes above (long) / below (short) the level
    consec = 0
    confirm_bar: Optional[int] = None
    search_end = min(trigger_iloc + max_bars_to_confirm + 1, len(df))

    for j in range(trigger_iloc + 1, search_end):
        row = df.iloc[j]
        hi_j, lo_j, cl_j = float(row["high"]), float(row["low"]), float(row["close"])

        # Cancel if price already ran 0.75R before retest
        if is_long and hi_j >= cancel_price:
            return None
        if not is_long and lo_j <= cancel_price:
            return None

        if is_long:
            consec = consec + 1 if cl_j > level else 0
        else:
            consec = consec + 1 if cl_j < level else 0

        if consec >= 2:
            confirm_bar = j
            break

    if confirm_bar is None:
        return None  # breakout never confirmed

    # Phase 2: wait for retest (price pulls back to level)
    search_end2 = min(confirm_bar + max_bars_to_retest + 1, len(df))
    for j in range(confirm_bar + 1, search_end2):
        row = df.iloc[j]
        hi_j, lo_j = float(row["high"]), float(row["low"])

        # Cancel if still extending without retest
        if is_long and hi_j >= cancel_price:
            return None
        if not is_long and lo_j <= cancel_price:
            return None

        # Retest: price touches back to the level
        if is_long and lo_j <= level:
            return j, level
        if not is_long and hi_j >= level:
            return j, level

    return None  # retest never came → missed trade


def resolve_trade(
    df_full: pd.DataFrame,
    entry_iloc: int,
    entry: float,
    sl: float,
    tp: float,
    side: str,
    max_bars: int = 30,
) -> Tuple[str, int, float]:
    """
    Walk forward bar by bar from entry_iloc+1 to determine outcome.
    SL takes priority when both are hit in the same bar (conservative).
    Returns (outcome, bars_held, exit_price).
    outcome: 'TP' | 'SL' | 'OPEN'
    """
    is_long = side == "long"
    end_iloc = min(entry_iloc + max_bars, len(df_full))

    for i in range(entry_iloc + 1, end_iloc):
        bar = df_full.iloc[i]
        lo, hi = float(bar["low"]), float(bar["high"])

        if is_long:
            sl_hit = lo <= sl
            tp_hit = hi >= tp
        else:
            sl_hit = hi >= sl
            tp_hit = lo <= tp

        bars_held = i - entry_iloc
        if sl_hit:
            return "SL", bars_held, sl
        if tp_hit:
            return "TP", bars_held, tp

    return "OPEN", end_iloc - entry_iloc, float(df_full.iloc[end_iloc - 1]["close"])


# ---------------------------------------------------------------------------
# Per-symbol scan
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    symbol: str
    timestamp: str
    side: str
    entry: float
    sl: float
    tp: float
    sl_pct: float
    tp_pct: float
    eff_rr: float
    outcome: str
    bars_held: int
    exit_price: float
    pnl_pct: float
    skip_gate: str = ""


def scan_symbol(
    symbol: str,
    df: pd.DataFrame,
    cfg: GateSettings,
    warmup_bars: int = 500,
    min_bars_between_trades: int = 60,
) -> Tuple[List[TradeRecord], List[dict], dict]:
    """
    Scan every bar from warmup_bars onward, apply all enabled gates,
    compute entry/SL/TP, resolve outcome.
    Returns (trades, skipped_records, gate_fail_counts).
    """
    trades: List[TradeRecord] = []
    skipped: List[dict] = []
    last_trade_bar = -min_bars_between_trades

    # Per-gate fail counters (counted once per bar, not per side)
    gate_fails: dict = {
        "volusd": 0, "vol_usd_5m": 0, "30m_noise": 0, "5m_spike": 0,
        "regime_long": 0, "regime_short": 0,
        "not_crossed_6h_long": 0, "not_crossed_6h_short": 0,
        "10m_long": 0, "10m_short": 0,
        "vwap_long": 0, "vwap_short": 0,
        "2h_long": 0, "2h_short": 0,
        "rr": 0, "min_tp_sl": 0,
        "no_wick_trigger": 0,
    }

    for i in range(warmup_bars, len(df)):
        if i - last_trade_bar < min_bars_between_trades:
            continue

        df_slice = df.iloc[: i + 1]
        bar_ts = str(df.index[i])
        trade_found = False

        # --- Side-independent (global) gates first ---
        global_skip = ""

        if cfg.enable_volusd_gate and not gate_volusd(df_slice):
            gate_fails["volusd"] += 1
            global_skip = "volusd_falling"

        elif not gate_vol_usd_5m(df_slice, cfg):
            gate_fails["vol_usd_5m"] += 1
            global_skip = "vol_usd_5m_low"

        elif cfg.enable_30m_noise_gate and not gate_30m_noise(df_slice, cfg):
            gate_fails["30m_noise"] += 1
            global_skip = "30m_noise"

        elif cfg.enable_5m_antispike_gate and not gate_5m_antispike(df_slice, cfg):
            gate_fails["5m_spike"] += 1
            global_skip = "5m_spike"

        if global_skip:
            skipped.append({"symbol": symbol, "timestamp": bar_ts, "side": "any", "reason": global_skip})
            continue

        # --- Side-specific gates ---
        for side in ("long", "short"):
            side_key = side  # "long" or "short"

            if cfg.enable_regime_breakout_gate:
                passed_regime, regime_reason = gate_regime_breakout(df_slice, side, cfg)
                if not passed_regime:
                    gate_fails[f"regime_{side}"] += 1
                    continue

            if cfg.entry_not_crossed_6h:
                if not gate_entry_not_crossed_6h(df_slice, side):
                    gate_fails[f"not_crossed_6h_{side}"] += 1
                    continue

            if cfg.enable_10m_gate:
                passed_10m, reason_10m = gate_10m_directional(df_slice, side, cfg)
                if not passed_10m:
                    gate_fails[f"10m_{side}"] += 1
                    continue

            if cfg.enable_vwap_side_gate:
                if not gate_vwap_side(df_slice, side):
                    gate_fails[f"vwap_{side}"] += 1
                    continue

            if cfg.enable_ema7_cross_gate:
                passed_ema7, reason_ema7 = gate_ema7_cross(df_slice, side, cfg)
                if not passed_ema7:
                    gate_fails[f"ema7_{side}"] = gate_fails.get(f"ema7_{side}", 0) + 1
                    continue

            if cfg.enable_staircase_gate:
                passed_stair, reason_stair = gate_staircase_quality(df_slice, side, cfg)
                if not passed_stair:
                    gate_fails[f"staircase_{side}"] = gate_fails.get(f"staircase_{side}", 0) + 1
                    continue

            if cfg.enable_last15m_gate:
                passed_l15, reason_l15 = gate_last15m(df_slice, side, cfg)
                if not passed_l15:
                    gate_fails[f"last15m_{side}"] = gate_fails.get(f"last15m_{side}", 0) + 1
                    continue

            if cfg.enable_2h_gate:
                passed_2h, reason_2h = gate_smma_trend(df_slice, side, cfg)
                if not passed_2h:
                    gate_fails[f"2h_{side}"] += 1
                    continue

            # --- Direct entry at current close (all quality gates passed) ---
            entry_iloc = i
            entry_price = float(df.iloc[i]["close"])

            entry, sl, tp = compute_sl_tp(df_slice, side, cfg, entry_price=entry_price)

            if cfg.enable_rr_guard:
                rr_ok, eff_rr = rr_guard(entry, sl, tp, side)
                if not rr_ok:
                    gate_fails["rr"] += 1
                    skipped.append({"symbol": symbol, "timestamp": bar_ts, "side": side, "reason": f"rr={eff_rr:.2f}"})
                    continue
            else:
                _, eff_rr = rr_guard(entry, sl, tp, side)

            if cfg.enable_min_tp_sl_pct_gate:
                if not min_tp_sl_gate(entry, sl, tp, side, cfg.min_profit_pct):
                    gate_fails["min_tp_sl"] += 1
                    skipped.append({"symbol": symbol, "timestamp": bar_ts, "side": side, "reason": "min_tp_sl"})
                    continue

            outcome, bars_held, exit_price = resolve_trade(
                df, entry_iloc, entry, sl, tp, side, max_bars=cfg.max_hold_bars
            )

            if side == "long":
                sl_pct = (entry - sl) / entry * 100.0
                tp_pct = (tp - entry) / entry * 100.0
                pnl_pct = (exit_price - entry) / entry * 100.0 if outcome != "SL" else -sl_pct
            else:
                sl_pct = (sl - entry) / entry * 100.0
                tp_pct = (entry - tp) / entry * 100.0
                pnl_pct = (entry - exit_price) / entry * 100.0 if outcome != "SL" else -sl_pct

            trades.append(TradeRecord(
                symbol=symbol,
                timestamp=str(df.index[entry_iloc]),
                side=side,
                entry=entry,
                sl=sl,
                tp=tp,
                sl_pct=round(sl_pct, 3),
                tp_pct=round(tp_pct, 3),
                eff_rr=round(eff_rr, 2),
                outcome=outcome,
                bars_held=bars_held,
                exit_price=exit_price,
                pnl_pct=round(pnl_pct, 3),
            ))
            last_trade_bar = entry_iloc
            trade_found = True
            break

    return trades, skipped, gate_fails


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_manifest(dataset_dir: Path) -> pd.DataFrame:
    manifest_path = dataset_dir / "dataset_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    df = pd.read_csv(manifest_path)
    return df[df["ok"].astype(str).str.lower() == "true"].reset_index(drop=True)


def load_symbol_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_index()


# ---------------------------------------------------------------------------
# Aggregate stats
# ---------------------------------------------------------------------------

def compute_stats(trades: List[TradeRecord]) -> dict:
    if not trades:
        return {
            "total_trades": 0, "tp_count": 0, "sl_count": 0, "open_count": 0,
            "win_rate_pct": 0.0, "avg_pnl_pct": 0.0, "total_pnl_pct": 0.0,
            "avg_bars_held": 0.0,
        }
    total = len(trades)
    tp = sum(1 for t in trades if t.outcome == "TP")
    sl = sum(1 for t in trades if t.outcome == "SL")
    op = sum(1 for t in trades if t.outcome == "OPEN")
    pnls = [t.pnl_pct for t in trades]
    return {
        "total_trades": total,
        "tp_count": tp,
        "sl_count": sl,
        "open_count": op,
        "win_rate_pct": round(tp / total * 100, 1),
        "avg_pnl_pct": round(float(np.mean(pnls)), 3),
        "total_pnl_pct": round(float(np.sum(pnls)), 3),
        "avg_bars_held": round(float(np.mean([t.bars_held for t in trades])), 1),
    }


def by_symbol_stats(trades: List[TradeRecord]) -> pd.DataFrame:
    rows = []
    symbols = sorted({t.symbol for t in trades})
    for sym in symbols:
        sym_trades = [t for t in trades if t.symbol == sym]
        s = compute_stats(sym_trades)
        s["symbol"] = sym
        rows.append(s)
    if not rows:
        return pd.DataFrame(columns=["symbol", "total_trades", "tp_count", "sl_count",
                                     "open_count", "win_rate_pct", "avg_pnl_pct", "total_pnl_pct"])
    return pd.DataFrame(rows)[["symbol", "total_trades", "tp_count", "sl_count",
                                "open_count", "win_rate_pct", "avg_pnl_pct", "total_pnl_pct"]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _process_symbol(args_tuple):
    """Worker function for parallel symbol processing."""
    symbol, csv_path, cfg, warmup, min_gap = args_tuple
    p = Path(csv_path)
    if not p.is_absolute():
        p = Path.cwd() / p
    if not p.exists():
        return symbol, [], [], {}, "not_found"
    try:
        raw_df = load_symbol_df(str(p))
        df = prepare_features(raw_df)
        trades, skipped, gate_fails = scan_symbol(
            symbol=symbol, df=df, cfg=cfg,
            warmup_bars=warmup, min_bars_between_trades=min_gap,
        )
        return symbol, trades, skipped, gate_fails, "ok"
    except Exception as exc:
        return symbol, [], [], {}, str(exc)[:80]


def main():
    parser = argparse.ArgumentParser(description="Gate-based momentum backtest")
    parser.add_argument("--dataset-dir", required=True, help="Path to dataset directory")
    parser.add_argument("--gates-config", default="momo_gate_settings.json",
                        help="Path to gate settings JSON (default: momo_gate_settings.json)")
    parser.add_argument("--out-prefix", default="momo_run", help="Output CSV filename prefix")
    parser.add_argument("--symbols", nargs="+", help="Limit to specific symbols (optional)")
    parser.add_argument("--exclude-symbols", default=None,
                        help="Comma-separated symbols to exclude (e.g. BTCUSDT,ETHUSDT)")
    parser.add_argument("--warmup", type=int, default=500,
                        help="Bars to skip before scanning (default: 500)")
    parser.add_argument("--min-gap", type=int, default=60,
                        help="Min bars between trades per symbol (default: 60)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel workers (0 = auto, 1 = serial)")
    args = parser.parse_args()

    cfg = GateSettings.from_json(args.gates_config)
    dataset_dir = Path(args.dataset_dir)
    manifest = load_manifest(dataset_dir)

    if args.symbols:
        manifest = manifest[manifest["symbol"].isin(args.symbols)].reset_index(drop=True)

    if args.exclude_symbols:
        exclude_set = {s.strip() for s in args.exclude_symbols.split(",")}
        before = len(manifest)
        manifest = manifest[~manifest["symbol"].isin(exclude_set)].reset_index(drop=True)
        print(f"Excluded {before - len(manifest)} symbols: {', '.join(sorted(exclude_set))}")

    all_trades: List[TradeRecord] = []
    all_skipped: List[dict] = []
    total_gate_fails: dict = {}

    # Build work items
    work_items = []
    for _, row in manifest.iterrows():
        work_items.append((row["symbol"], row["path"], cfg, args.warmup, args.min_gap))

    n_workers = args.workers if args.workers > 0 else min(8, len(work_items))
    print(f"Scanning {len(work_items)} symbols with {n_workers} workers...")

    if n_workers <= 1:
        # Serial mode
        for i, item in enumerate(work_items, 1):
            symbol, trades, skipped, gate_fails, status = _process_symbol(item)
            all_trades.extend(trades)
            all_skipped.extend(skipped)
            for k, v in gate_fails.items():
                total_gate_fails[k] = total_gate_fails.get(k, 0) + v
            print(f"  [{i:3d}/{len(work_items)}] {symbol:20s} {len(trades)} trades")
    else:
        # Parallel mode
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_process_symbol, item): item[0] for item in work_items}
            for i, fut in enumerate(as_completed(futures), 1):
                symbol, trades, skipped, gate_fails, status = fut.result()
                all_trades.extend(trades)
                all_skipped.extend(skipped)
                for k, v in gate_fails.items():
                    total_gate_fails[k] = total_gate_fails.get(k, 0) + v
                print(f"  [{i:3d}/{len(work_items)}] {symbol:20s} {len(trades)} trades")

    # --- Write outputs ---
    prefix = args.out_prefix

    # Trade list
    if all_trades:
        trade_df = pd.DataFrame([asdict(t) for t in all_trades])
        trade_df.to_csv(f"{prefix}_trade_list.csv", index=False)
        print(f"\nWrote {prefix}_trade_list.csv ({len(trade_df)} trades)")
    else:
        pd.DataFrame().to_csv(f"{prefix}_trade_list.csv", index=False)
        print(f"\nNo trades found — wrote empty {prefix}_trade_list.csv")

    # By-symbol
    by_sym = by_symbol_stats(all_trades)
    by_sym.to_csv(f"{prefix}_by_symbol.csv", index=False)
    print(f"Wrote {prefix}_by_symbol.csv")

    # Skipped
    if all_skipped:
        pd.DataFrame(all_skipped).to_csv(f"{prefix}_skipped.csv", index=False)
    else:
        pd.DataFrame(columns=["symbol", "timestamp", "side", "reason"]).to_csv(
            f"{prefix}_skipped.csv", index=False
        )
    print(f"Wrote {prefix}_skipped.csv ({len(all_skipped)} skipped bars)")

    # Overall
    stats = compute_stats(all_trades)
    stats.update(cfg.to_dict())
    stats["dataset_dir"] = str(dataset_dir)
    stats["symbols_scanned"] = len(manifest)
    overall_df = pd.DataFrame([stats])
    overall_df.to_csv(f"{prefix}_overall.csv", index=False)
    print(f"Wrote {prefix}_overall.csv")

    # Print summary
    print(f"\n{'='*60}")
    print(f"BACKTEST SUMMARY")
    print(f"{'='*60}")
    print(f"Symbols scanned : {stats['symbols_scanned']}")
    print(f"Total trades    : {stats['total_trades']}")
    print(f"TP / SL / Open  : {stats['tp_count']} / {stats['sl_count']} / {stats['open_count']}")
    print(f"Win rate        : {stats['win_rate_pct']:.1f}%")
    print(f"Avg PnL/trade   : {stats['avg_pnl_pct']:.3f}%")
    print(f"Total PnL       : {stats['total_pnl_pct']:.3f}%")
    print(f"Avg bars held   : {stats['avg_bars_held']:.1f}")

    if total_gate_fails:
        print(f"\nGATE FAIL COUNTS (bars rejected per gate):")
        for gate, count in sorted(total_gate_fails.items(), key=lambda x: -x[1]):
            print(f"  {gate:<20}: {count:>8}")


if __name__ == "__main__":
    main()
