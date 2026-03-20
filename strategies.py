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
# Strategy runner — detects setups for all enabled strategies
# ---------------------------------------------------------------------------

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
        if len(df) >= 500:
            df_sorted = df.sort_values("timestamp").reset_index(drop=True)
            df_indexed = df_sorted.set_index("timestamp").copy()
            df_indexed.index = pd.to_datetime(df_indexed.index, utc=True)
            df_prepped = prepare_momo_features(df_indexed)

            for direction in ["long", "short"]:
                result = check_momo_gates_at_bar(df_prepped, direction, momo_cfg)
                if not result["passed"]:
                    continue
                if result["dps_total"] < config.min_dps_live:
                    continue

                entry = round(result["entry"], 8)
                sl = round(result["sl"], 8)
                tp = round(result["tp"], 8)
                sl_pct = result["sl_pct"]
                tp_pct = result["tp_pct"]
                rr = result["rr"]
                tp_source = "strategy"

                # Override TP with depth wall if available and better
                if depth_data is not None:
                    try:
                        from depth_tp_sl_analyzer import compute_depth_tp_sl
                        depth_result = compute_depth_tp_sl(
                            depth_data, entry, side=direction,
                            strategy="momentum", min_tp_pct=1.0, min_rr=1.0)
                        best = depth_result.get("best_combo")
                        if best and best["tp"]["dist_pct"] > tp_pct:
                            # Depth TP is wider — use it
                            depth_tp = best["tp"]["price"]
                            depth_tp_pct = best["tp"]["dist_pct"]
                            depth_rr = depth_tp_pct / sl_pct if sl_pct > 0 else rr
                            tp = round(depth_tp, 8)
                            tp_pct = round(depth_tp_pct, 3)
                            rr = round(depth_rr, 2)
                            tp_source = "depth"
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
                    "approach": result["approach"],
                    "vol_trend": result["vol_trend"],
                    "duration_hrs": result["duration_hrs"],
                    "tp_source": tp_source,
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

    return setups


def get_risk_pct(setup: dict, config: StrategyConfig) -> float:
    """Determine risk % for a setup based on DPS confidence and pre-chop trend."""
    # Depth strategies: always dummy risk (experimental)
    if setup.get("strategy") in ("depth", "depth_bounce"):
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

        elif strategy in ("depth", "depth_bounce"):
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
