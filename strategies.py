#!/usr/bin/env python3
"""
Unified Strategy Module

Central registry for all trading strategies. Each strategy has:
  - A shared gate function (used by both backtest and live trader)
  - A settings/config class
  - Enable/disable toggle via StrategyConfig

Adding a new strategy:
  1. Create the gate function in its own file (e.g., scan_new_strategy.py)
  2. Import and register it here
  3. Add enable flag to StrategyConfig
  4. Add detection call in live_dummy_trader.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Strategy Config — enable/disable strategies + global settings
# ---------------------------------------------------------------------------

STRATEGY_CONFIG_FILE = "strategy_config.json"


@dataclass
class StrategyConfig:
    """Master config for enabling/disabling strategies and global settings."""

    # Strategy toggles
    enable_mean_reversion: bool = True
    enable_strict_mr: bool = True
    enable_momentum: bool = True
    enable_depth: bool = True
    enable_depth_bounce: bool = True
    enable_bouncy_ball: bool = True
    enable_mr_chop: bool = True

    # Global filters
    min_dps_live: int = 3           # min DPS score to take a trade in live
    dummy_risk_pct: float = 0.1     # risk % for low-confidence / dummy trades
    standard_risk_pct: float = 1.0  # risk % for high-confidence trades
    dummy_balance: float = 10000.0  # dummy account balance

    # Max concurrent positions per strategy
    max_positions_mr: int = 10
    max_positions_strict_mr: int = 10
    max_positions_momo: int = 10
    max_positions_depth: int = 10
    max_positions_depth_bounce: int = 10
    max_positions_bouncy_ball: int = 10

    @classmethod
    def from_json(cls, path: str = STRATEGY_CONFIG_FILE) -> "StrategyConfig":
        if not Path(path).exists():
            return cls()
        with open(path) as f:
            data = json.load(f)
        obj = cls()
        for k, v in data.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
        return obj

    def to_json(self, path: str = STRATEGY_CONFIG_FILE):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Mean Reversion — gate function + settings
# ---------------------------------------------------------------------------

from scan_mean_reversion import (
    MRSettings,
    check_mr_gates_at_bar,
    check_strict_mr_gates_at_bar,
    count_level_touches_strict,
    # Re-export individual helpers for backtest access
    detect_choppy_range,
    detect_pre_chop_trend,
    detect_range_entry,
    count_level_touches,
    check_opposite_bound_intact,
    compute_vwap_bands,
    compute_sl_tp,
    check_vwap_clear_path,
    classify_noise,
    evaluate_dps,
)

# ---------------------------------------------------------------------------
# Momentum — gate function + settings
# ---------------------------------------------------------------------------

from backtest_momo_vwap_grind15_full import (
    GateSettings as MomoGateSettings,
    check_momo_gates_at_bar,
    prepare_features as prepare_momo_features,
)
from scan_momo_quality import calc_regression as _momo_calc_regression

# ---------------------------------------------------------------------------
# Depth-of-Book — wall-based strategy
# ---------------------------------------------------------------------------

from strategy_depth import (
    DepthStrategySettings,
    check_depth_setup,
    evaluate_zct_alignment,
    DEPTH_EXCLUDED_SYMBOLS,
)

# ---------------------------------------------------------------------------
# Depth Wall Bounce — touch + bounce off wall
# ---------------------------------------------------------------------------

from strategy_depth_bounce import (
    DepthBounceSettings,
    check_depth_bounce_setup,
    check_75pct_tp_rule,
)

# ---------------------------------------------------------------------------
# Bouncy Ball MR — oscillating range strategy
# ---------------------------------------------------------------------------

from strategy_bouncy_ball import (
    BouncyBallSettings,
    check_bouncy_ball_setup,
)

from strategy_mr_chop import (
    MRChopSettings,
    check_range_shift_setup,
    check_one_sided_chop_setup,
)


# ---------------------------------------------------------------------------
# Strategy runner — detects setups for all enabled strategies
# ---------------------------------------------------------------------------

def _momo_quality_filter(df_sorted: pd.DataFrame, side: str) -> Optional[dict]:
    """
    Momentum quality filter: 2h regression, split maxDD, 15m channel+slope, DPS.
    Returns dict with quality metrics and DPS scores, or None if filtered out.
    """
    n = len(df_sorted)
    if n < 150:
        return None

    c = df_sorted["close"].values.astype(float)
    h = df_sorted["high"].values.astype(float)
    lo = df_sorted["low"].values.astype(float)
    v = df_sorted["volume"].values.astype(float)

    end = n  # work with the full array (end is exclusive index)

    # --- 2h regression (last 120 bars) ---
    if end < 120:
        return None
    res_2h = _momo_calc_regression(c[end-120:end], h[end-120:end], lo[end-120:end])
    if res_2h is None:
        return None
    if res_2h["r2"] <= 0.85 or res_2h["channel"] >= 1.25:
        return None

    # --- Split maxDD: first 60 bars < 1.15%, last 60 bars < 0.7% ---
    mid = end - 60
    res_1h_first = _momo_calc_regression(c[end-120:mid], h[end-120:mid], lo[end-120:mid])
    res_1h_last = _momo_calc_regression(c[mid:end], h[mid:end], lo[mid:end])
    if res_1h_first is None or res_1h_last is None:
        return None
    if res_1h_first["max_dd"] >= 1.15 or res_1h_last["max_dd"] >= 0.7:
        return None

    # --- 15m regression: disabled — 2-bar confirmation handles last 15m quality ---
    # res_15 = _momo_calc_regression(c[end-15:end], h[end-15:end], lo[end-15:end])
    # if res_15 is None:
    #     return None
    # if res_15["channel"] >= 0.20:
    #     return None
    # if side == "long" and res_15["slope"] <= 0:
    #     return None
    # if side == "short" and res_15["slope"] >= 0:
    #     return None

    # --- ZCT DPS: Duration ---
    dps_dur = 1  # 2hr staircase = 1
    if end >= 240:
        res_4h = _momo_calc_regression(c[end-240:end], h[end-240:end], lo[end-240:end])
        if res_4h is not None and res_4h["r2"] > 0.85:
            dps_dur = 2

    # --- ZCT DPS: Approach (always 2 — 15m channel filter = grind) ---
    dps_app = 2

    # --- ZCT DPS: Volume (side-specific) ---
    vol_usd = v[end-30:end] * c[end-30:end]
    vol_avg = np.mean(vol_usd)
    if vol_avg <= 0:
        return None
    vol_slope = np.polyfit(np.arange(len(vol_usd)), vol_usd, 1)[0]
    vol_norm = vol_slope / vol_avg
    if vol_norm > 0.01:
        vol_label = "increasing"
    elif vol_norm < -0.01:
        vol_label = "decreasing"
    else:
        vol_label = "flat"

    if side == "long":
        dps_vol = 2 if vol_label == "increasing" else (1 if vol_label == "flat" else 0)
    else:
        dps_vol = 2 if vol_label == "increasing" else (1 if vol_label == "decreasing" else 0)

    dps_total = dps_dur + dps_app + dps_vol

    return {
        "r2_2h": round(res_2h["r2"], 4),
        "channel_2h": round(res_2h["channel"], 4),
        "channel_15m": 0,  # 15m filter disabled
        "max_dd_1h_first": round(res_1h_first["max_dd"], 4),
        "max_dd_1h_last": round(res_1h_last["max_dd"], 4),
        "dps_dur": dps_dur,
        "dps_app": dps_app,
        "dps_vol": dps_vol,
        "dps_total_quality": dps_total,
        "vol_trend_quality": vol_label,
    }


def detect_setups(df: pd.DataFrame, symbol: str,
                  config: StrategyConfig,
                  mr_cfg: Optional[MRSettings] = None,
                  momo_cfg: Optional[MomoGateSettings] = None,
                  depth_data: Optional[dict] = None,
                  depth_cfg: Optional[DepthStrategySettings] = None) -> list[dict]:
    """
    Run all enabled strategies on the latest bar.
    Returns a list of setup dicts (0, 1, or 2 — one per strategy).
    """
    setups = []

    # Skip symbols used only for market condition evaluation (not traded)
    if symbol in DEPTH_EXCLUDED_SYMBOLS:
        return setups

    # --- Mean Reversion ---
    if config.enable_mean_reversion and mr_cfg is not None:
        min_bars = max(mr_cfg.range_max_bars, mr_cfg.noise_lookback_bars, 720) + 10
        if len(df) >= min_bars:
            df_sorted = df.sort_values("timestamp").reset_index(drop=True)
            i = len(df_sorted) - 1
            result = check_mr_gates_at_bar(df_sorted, i, mr_cfg)
            if result["passed"] and result["dps_total"] >= config.min_dps_live:
                setup = {
                    "symbol": symbol,
                    "strategy": "mean_reversion",
                    "timestamp": str(df_sorted.iloc[i]["timestamp"]),
                    "side": result["side"],
                    "entry": result["entry"],
                    "sl": result["sl"],
                    "tp": result["tp"],
                    "sl_pct": result["sl_pct"],
                    "tp_pct": result["tp_pct"],
                    "rr": result["rr"],
                    "dps_total": result["dps_total"],
                    "dps_confidence": result["dps_confidence"],
                    "noise_level": result["noise_level"],
                    "touches": result["touches"],
                    "break_pct": result["break_pct"],
                    "range_width_pct": result["range_width_pct"],
                    "range_duration_hrs": result["range_duration_hrs"],
                    "pre_chop_trend": result["pre_chop_trend"],
                    "approach": result["dps_v2_label"],
                    "vol_trend": result["dps_v3_vol_trend"],
                    "range_upper": result["range_upper"],
                    "range_lower": result["range_lower"],
                    "touch_timestamps": result["touch_timestamps"],
                }
                setups.append(setup)

    # --- Strict Mean Reversion ---
    if config.enable_strict_mr and mr_cfg is not None:
        min_bars = max(mr_cfg.range_max_bars, mr_cfg.noise_lookback_bars, 720) + 10
        if len(df) >= min_bars:
            df_sorted = df.sort_values("timestamp").reset_index(drop=True)
            i = len(df_sorted) - 1
            result = check_strict_mr_gates_at_bar(df_sorted, i, mr_cfg)
            if result["passed"] and result["dps_total"] >= config.min_dps_live:
                setup = {
                    "symbol": symbol,
                    "strategy": "strict_mr",
                    "timestamp": str(df_sorted.iloc[i]["timestamp"]),
                    "side": result["side"],
                    "entry": result["entry"],
                    "sl": result["sl"],
                    "tp": result["tp"],
                    "sl_pct": result["sl_pct"],
                    "tp_pct": result["tp_pct"],
                    "rr": result["rr"],
                    "dps_total": result["dps_total"],
                    "dps_confidence": result["dps_confidence"],
                    "noise_level": result["noise_level"],
                    "touches": result["touches"],
                    "break_pct": result["break_pct"],
                    "range_width_pct": result["range_width_pct"],
                    "range_duration_hrs": result["range_duration_hrs"],
                    "pre_chop_trend": result["pre_chop_trend"],
                    "approach": result["dps_v2_label"],
                    "vol_trend": result["dps_v3_vol_trend"],
                    "range_upper": result["range_upper"],
                    "range_lower": result["range_lower"],
                    "touch_timestamps": result["touch_timestamps"],
                    "closes_beyond_count": result["closes_beyond_count"],
                }
                setups.append(setup)

    # --- Momentum ---
    if config.enable_momentum and momo_cfg is not None:
        if len(df) >= 150:
            df_sorted = df.sort_values("timestamp").reset_index(drop=True)
            df_indexed = df_sorted.set_index("timestamp").copy()
            df_indexed.index = pd.to_datetime(df_indexed.index, utc=True)
            df_prepped = prepare_momo_features(df_indexed)

            for direction in ["long", "short"]:
                result = check_momo_gates_at_bar(df_prepped, direction, momo_cfg)
                if not result["passed"]:
                    continue

                # --- Momentum quality filter (2h R², channel, maxDD, 15m) ---
                qf = _momo_quality_filter(df_sorted, direction)
                if qf is None:
                    continue

                # Override DPS with quality-filter DPS
                result["dps_total"] = qf["dps_total_quality"]
                dps_t = qf["dps_total_quality"]
                result["dps_confidence"] = (
                    "max" if dps_t >= 6 else
                    ("high" if dps_t >= 4 else
                     ("low" if dps_t >= 3 else "avoid")))

                if result["dps_total"] < config.min_dps_live:
                    continue

                entry = round(result["entry"], 8)
                sl = round(result["sl"], 8)
                tp = round(result["tp"], 8)
                sl_pct = result["sl_pct"]
                tp_pct = result["tp_pct"]
                rr = result["rr"]
                tp_source = "strategy"

                # Log depth wall info for evaluation (don't use for TP placement)
                depth_wall_info = {}
                if depth_data is not None:
                    try:
                        from depth_tp_sl_analyzer import compute_depth_tp_sl
                        depth_result = compute_depth_tp_sl(
                            depth_data, entry, side=direction,
                            strategy="momentum", min_tp_pct=1.0, min_rr=1.0)
                        best = depth_result.get("best_combo")
                        if best:
                            depth_wall_info = {
                                "depth_tp": round(best["tp"]["price"], 8),
                                "depth_tp_pct": round(best["tp"]["dist_pct"], 3),
                                "depth_tp_wall_usd": round(best["tp"].get("wall_usd", 0), 0),
                                "depth_tp_wall_strength": round(best["tp"].get("wall_strength", 0), 1),
                                "depth_sl_wall_usd": round(best["sl"].get("wall_usd", 0), 0),
                                "depth_sl_wall_strength": round(best["sl"].get("wall_strength", 0), 1),
                            }
                    except Exception:
                        pass

                setup = {
                    "symbol": symbol,
                    "strategy": "momentum",
                    "timestamp": str(df_sorted.iloc[-1]["timestamp"]),
                    "side": direction,
                    "entry": entry,
                    "sl": sl,
                    "tp": tp,
                    "sl_pct": sl_pct,
                    "tp_pct": tp_pct,
                    "rr": rr,
                    "dps_total": result["dps_total"],
                    "dps_confidence": result["dps_confidence"],
                    "dps_dur": qf["dps_dur"],
                    "dps_app": qf["dps_app"],
                    "dps_vol": qf["dps_vol"],
                    "approach": result["approach"],
                    "vol_trend": qf["vol_trend_quality"],
                    "duration_hrs": result["duration_hrs"],
                    "r2_2h": qf["r2_2h"],
                    "channel_2h": qf["channel_2h"],
                    "channel_15m": qf["channel_15m"],
                    "max_dd_1h_first": qf["max_dd_1h_first"],
                    "max_dd_1h_last": qf["max_dd_1h_last"],
                    "tp_source": tp_source,
                    **depth_wall_info,
                }
                setups.append(setup)
                break  # Only take one direction per symbol

    # --- Depth-of-Book ---
    if config.enable_depth and depth_data is not None:
        # Skip excluded symbols
        if symbol not in DEPTH_EXCLUDED_SYMBOLS:
            d_cfg = depth_cfg or DepthStrategySettings()
            df_sorted = df.sort_values("timestamp").reset_index(drop=True)
            current_price = float(df_sorted.iloc[-1]["close"])

            result = check_depth_setup(depth_data, current_price, d_cfg)
            if result["passed"]:
                # Evaluate ZCT alignment
                zct = evaluate_zct_alignment(
                    df, result["side"], mr_cfg, momo_cfg)

                setup = {
                    "symbol": symbol,
                    "strategy": "depth",
                    "timestamp": str(df_sorted.iloc[-1]["timestamp"]),
                    "side": result["side"],
                    "entry": result["entry"],
                    "sl": result["sl"],
                    "tp": result["tp"],
                    "sl_pct": result["sl_pct"],
                    "tp_pct": result["tp_pct"],
                    "rr": result["rr"],
                    "sl_wall_usd": result["sl_wall_usd"],
                    "sl_wall_strength": result["sl_wall_strength"],
                    "tp_wall_usd": result["tp_wall_usd"],
                    "tp_wall_strength": result["tp_wall_strength"],
                    "imbalance_1pct": result["imbalance_1pct"],
                    "imbalance_2pct": result["imbalance_2pct"],
                    "dps_total": zct["dps_total"],
                    "dps_confidence": zct["dps_confidence"],
                    "zct_alignment": zct["alignment"],
                    "zct_mr_reason": zct["mr_reason"],
                    "zct_momo_reason": zct["momo_reason"],
                }
                setups.append(setup)

    # --- Depth Wall Bounce ---
    if config.enable_depth_bounce and depth_data is not None:
        if symbol not in DEPTH_EXCLUDED_SYMBOLS:
            db_cfg = DepthBounceSettings()
            df_sorted = df.sort_values("timestamp").reset_index(drop=True)

            result = check_depth_bounce_setup(depth_data, df_sorted, db_cfg)
            if result["passed"]:
                # Evaluate ZCT alignment
                zct = evaluate_zct_alignment(
                    df, result["side"], mr_cfg, momo_cfg)

                setup = {
                    "symbol": symbol,
                    "strategy": "depth_bounce",
                    "timestamp": str(df_sorted.iloc[-1]["timestamp"]),
                    "side": result["side"],
                    "entry": result["entry"],
                    "sl": result["sl"],
                    "tp": result["tp"],
                    "sl_pct": result["sl_pct"],
                    "tp_pct": result["tp_pct"],
                    "rr": result["rr"],
                    "sl_wall_usd": result["entry_wall_usd"],
                    "sl_wall_strength": result["entry_wall_strength"],
                    "tp_wall_usd": result["tp_wall_usd"],
                    "tp_wall_strength": result["tp_wall_strength"],
                    "bounce_type": result["bounce_type"],
                    "wall_price": result["wall_price"],
                    "wick_price": result["wick_price"],
                    "dps_total": zct["dps_total"],
                    "dps_confidence": zct["dps_confidence"],
                    "zct_alignment": zct["alignment"],
                    "zct_mr_reason": zct["mr_reason"],
                    "zct_momo_reason": zct["momo_reason"],
                }
                setups.append(setup)

    # --- Bouncy Ball MR ---
    if config.enable_bouncy_ball:
        min_bars_bb = 480 + 240 + 10  # range + pre-trend lookback
        if len(df) >= min_bars_bb:
            df_sorted = df.sort_values("timestamp").reset_index(drop=True)
            i = len(df_sorted) - 1
            bb_cfg = BouncyBallSettings()
            result = check_bouncy_ball_setup(df_sorted, i, bb_cfg)
            if result["passed"]:
                setup = {
                    "symbol": symbol,
                    "strategy": "bouncy_ball",
                    "timestamp": str(df_sorted.iloc[i]["timestamp"]),
                    "side": result["side"],
                    "entry": result["entry"],
                    "sl": result["sl"],
                    "tp": result["tp"],
                    "sl_pct": result["sl_pct"],
                    "tp_pct": result["tp_pct"],
                    "rr": result["rr"],
                    "range_upper": result["range_upper"],
                    "range_lower": result["range_lower"],
                    "range_pct": result["range_pct"],
                    "upper_touches": result["upper_touches"],
                    "lower_touches": result["lower_touches"],
                    "clean_score": result["clean_score"],
                    "inside_pct": result["inside_pct"],
                    "pre_trend": result["pre_trend"],
                    "pre_trend_pct": result["pre_trend_pct"],
                    "range_duration_bars": result["range_duration_bars"],
                    "dps_total": result["clean_score"],  # use clean score as quality
                    "dps_confidence": "high" if result["clean_score"] >= 8 else "medium",
                }
                setups.append(setup)

    # --- MR Chop (Range Shift + One-Sided Chop) ---
    if config.enable_mr_chop:
        min_bars_chop = 600 + 10
        if len(df) >= min_bars_chop:
            df_sorted = df.sort_values("timestamp").reset_index(drop=True)
            i = len(df_sorted) - 1
            chop_cfg = MRChopSettings()

            # Try Range Shift first
            result = check_range_shift_setup(df_sorted, i, chop_cfg)
            if not result["passed"]:
                # Try One-Sided Chop
                result = check_one_sided_chop_setup(df_sorted, i, chop_cfg)

            if result["passed"]:
                variant = result.get("strategy_variant", "mr_chop")
                setup = {
                    "symbol": symbol,
                    "strategy": "mr_chop",
                    "strategy_variant": variant,
                    "timestamp": str(df_sorted.iloc[i]["timestamp"]),
                    "side": result["side"],
                    "entry": result["entry"],
                    "sl": result["sl"],
                    "tp": result["tp"],
                    "sl_pct": result["sl_pct"],
                    "tp_pct": result["tp_pct"],
                    "rr": result["rr"],
                    "n_swings": result.get("n_swings", 0),
                    "last_swing_mins": result.get("last_swing_mins", 0),
                    "chop_hrs": result.get("chop_hrs", 0),
                    "dps_total": result.get("dps_total", 0),
                    "dps_confidence": "high" if result.get("dps_total", 0) >= 5 else "medium",
                    "vol_type": result.get("vol_type", ""),
                    "pre_trend": result.get("pre_trend", ""),
                    "shift_pct": result.get("shift_pct", 0),
                    "level_touches": result.get("level_touches", 0),
                }
                setups.append(setup)

    return setups


def get_risk_pct(setup: dict, config: StrategyConfig) -> float:
    """Determine risk % for a setup based on DPS confidence and pre-chop trend."""
    # Depth/bouncy_ball strategies: always dummy risk (experimental)
    if setup.get("strategy") in ("depth", "depth_bounce", "bouncy_ball", "mr_chop"):
        return config.dummy_risk_pct

    # MR: force dummy on unclear pre-chop trend
    if setup.get("strategy") == "mean_reversion":
        if setup.get("pre_chop_trend") == "unclear":
            return config.dummy_risk_pct

    if setup.get("dps_confidence") in ("max", "high"):
        return config.standard_risk_pct
    return config.dummy_risk_pct


# ---------------------------------------------------------------------------
# Market Condition Evaluator
# ---------------------------------------------------------------------------

class MarketCondition:
    """
    Evaluates market conditions using a Combined Direction Score (-3 to +3).

    Signals:
      1. BTC SMMA30 slope (120-bar lookback): up=+1, down=-1
      2. BTC close vs session VWAP: above=+1, below=-1
      3. Market breadth (% of coins above SMMA30): >60%=+1, <40%=-1

    Usage:
      mc = MarketCondition()
      mc.update_btc(btc_df)           # call once per cycle with BTC candles
      mc.update_breadth(all_dfs)      # call once per cycle with all symbol candles
      score = mc.score                # -3 to +3
      mc.is_allowed("depth", "short") # check if trade is allowed
    """

    def __init__(self):
        self.score = 0
        self.btc_smma_signal = 0
        self.btc_vwap_signal = 0
        self.breadth_signal = 0
        self.breadth_pct = 50.0

    def update_btc(self, df: pd.DataFrame):
        """Update BTC signals from candle data. df must have close, high, low, volume, timestamp."""
        if len(df) < 150:
            return

        closes = df["close"].values
        # SMMA30
        smma30 = pd.Series(closes).ewm(alpha=1/30, adjust=False).mean().values

        # Slope over last 120 bars
        if len(smma30) > 120:
            current = smma30[-1]
            prev = smma30[-121]
            if prev > 0:
                slope_pct = (current - prev) / prev * 100
                if slope_pct > 0.01:
                    self.btc_smma_signal = 1
                elif slope_pct < -0.01:
                    self.btc_smma_signal = -1
                else:
                    self.btc_smma_signal = 0

        # Session VWAP (from start of current UTC day)
        timestamps = df["timestamp"]
        if hasattr(timestamps.iloc[-1], "normalize"):
            today_start = timestamps.iloc[-1].normalize()
        else:
            today_start = pd.Timestamp(str(timestamps.iloc[-1])[:10], tz="UTC")

        today_mask = timestamps >= today_start
        if today_mask.sum() > 5:
            today = df[today_mask]
            tp = (today["high"] + today["low"] + today["close"]) / 3
            cum_tp_vol = (tp * today["volume"]).cumsum()
            cum_vol = today["volume"].cumsum()
            vwap = cum_tp_vol / cum_vol.replace(0, float("nan"))
            last_vwap = float(vwap.iloc[-1])
            last_close = float(today["close"].iloc[-1])

            if not pd.isna(last_vwap) and last_vwap > 0:
                if last_close > last_vwap:
                    self.btc_vwap_signal = 1
                else:
                    self.btc_vwap_signal = -1

        self.score = self.btc_smma_signal + self.btc_vwap_signal + self.breadth_signal

    def update_breadth(self, symbol_dfs: dict[str, pd.DataFrame]):
        """
        Update breadth signal from a dict of {symbol: DataFrame}.
        Each df must have 'close' column with at least 30 bars.
        """
        if not symbol_dfs:
            return

        above = 0
        total = 0
        for sym, df in symbol_dfs.items():
            if len(df) < 30:
                continue
            closes = df["close"].values
            smma30 = pd.Series(closes).ewm(alpha=1/30, adjust=False).mean().values
            if closes[-1] > smma30[-1]:
                above += 1
            total += 1

        if total > 10:
            self.breadth_pct = above / total * 100
            if self.breadth_pct > 60:
                self.breadth_signal = 1
            elif self.breadth_pct < 40:
                self.breadth_signal = -1
            else:
                self.breadth_signal = 0

        self.score = self.btc_smma_signal + self.btc_vwap_signal + self.breadth_signal

    def is_allowed(self, strategy: str, side: str) -> bool:
        """
        Check if a trade is allowed given current market conditions.

        Rules:
          - Momo: only when DPS >= 4 enforced elsewhere; direction must align
            Score >= 2: only longs. Score <= -2: only shorts.
            Score 0: skip Momo entirely.
          - Depth/Depth_bounce: skip at ±3 (extreme). Direction filter at ±2.
          - MR/Strict_MR: no market filter (it hurts MR).
        """
        s = self.score

        if strategy == "momentum":
            # Skip momo at neutral
            if s == 0:
                return False
            # Direction filter
            if s >= 2 and side == "short":
                return False
            if s <= -2 and side == "long":
                return False
            return True

        elif strategy in ("depth", "depth_bounce", "bouncy_ball", "mr_chop"):
            # At extremes: only allow direction-aligned trades
            # Score +3: only longs. Score -3: only shorts.
            if s >= 3 and side == "short":
                return False
            if s >= 3 and side == "long":
                return True
            if s <= -3 and side == "long":
                return False
            if s <= -3 and side == "short":
                return True
            # Direction filter at ±2
            if s >= 2 and side == "short":
                return False
            if s <= -2 and side == "long":
                return False
            return True

        # MR / Strict MR — no market filter
        return True

    def summary(self) -> str:
        """One-line summary for logging."""
        labels = {-3: "STRONG_SHORT", -2: "SHORT", -1: "LEAN_SHORT",
                  0: "NEUTRAL", 1: "LEAN_LONG", 2: "LONG", 3: "STRONG_LONG"}
        return (f"MKT={labels.get(self.score, '?')}({self.score:+d}) "
                f"[SMMA={self.btc_smma_signal:+d} VWAP={self.btc_vwap_signal:+d} "
                f"Breadth={self.breadth_pct:.0f}%({self.breadth_signal:+d})]")
