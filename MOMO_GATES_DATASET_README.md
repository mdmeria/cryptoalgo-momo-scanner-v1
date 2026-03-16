# Momo Gates, Dataset, and Script Guide

This document describes the current momentum backtest workflow centered on:
- `backtest_momo_vwap_grind15_full.py`
- `momo_gate_settings.json`
- dataset folder `datasets/momo_1m_7d_top100_midcap_30d`

## 1) Core Strategy Files

- `backtest_momo_vwap_grind15_full.py`
  - Main backtest engine for the gate-based momentum strategy.
  - Loads gate toggles and thresholds from external JSON config.
  - Produces `_overall.csv`, `_by_symbol.csv`, `_trade_list.csv`, `_skipped.csv` outputs.

- `momo_gate_settings.json`
  - Source of truth for gate ON/OFF switches and gate parameters.
  - Default file currently enables all major gates, with loose defaults for 30m noise and 5m anti-spike.

- `backtest_momentum_pullback_sweep.py`
  - Shared utilities used by the main backtest:
    - `prepare_features`
    - `DailyKeyLevels`
    - `fetch_daily_klines`
    - `extract_key_levels`
    - `adjust_tp_by_key_levels`

## 2) Gate Configuration (Current Keys)

Gate switches:
- `enable_2h_gate`
- `enable_10m_gate`
- `enable_30m_noise_gate`
- `enable_5m_antispike_gate`
- `enable_volusd_gate`
- `enable_regime_breakout_gate`
- `enable_vwap_side_gate`
- `enable_tp_sl_keylevel_gate`
- `enable_rr_guard`
- `enable_min_tp_sl_pct_gate`

10m gate parameters:
- `grind10_min`
- `wick_ratio_10m_max`
- `angle_min_deg`
- `angle_max_deg`

2h directional gate parameters:
- `dir_lookback_bars`
- `dir_move_min_pct`
- `dir_eff_min`
- `noise_wick_max`

30m / 5m noise and spike parameters:
- `noise_wick_30m_max`
- `spike5_max_abs_ret_pct`
- `spike5_max_range_pct`

Core threshold parameters:
- `day_change_pct`
- `min_profit_pct`

## 3) Gate Execution Order (Runtime)

For each candidate minute bar:
1. VolUsd MA60 up vs 10m ago (if `enable_volusd_gate`).
2. 30m wick-noise cap (if `enable_30m_noise_gate`).
3. 5m anti-spike caps (if `enable_5m_antispike_gate`).
4. 10m directional gate (if `enable_10m_gate`):
   - directional efficiency (`grind10_min`)
   - wick cap (`wick_ratio_10m_max`)
   - angle band (`angle_min_deg..angle_max_deg`)
5. Regime + 6h breakout direction (if `enable_regime_breakout_gate`):
   - long: `day_change >= day_change_pct` and close above prior 6h high
   - short: `day_change <= -day_change_pct` and close below prior 6h low
6. VWAP side (if `enable_vwap_side_gate`).
7. 10m direction consistency with chosen side (if `enable_10m_gate`).
8. 2h directional + low-noise gate (if `enable_2h_gate`).
9. Entry at current close.
10. SL from min(ATR distance, 6h structure distance), bounded by `min_sl_pct`.
11. TP from RR target (`rr`).
12. Daily key-level TP cap and viability (if `enable_tp_sl_keylevel_gate`).
13. Min TP and SL distance from entry (if `enable_min_tp_sl_pct_gate`, uses `min_profit_pct`).
14. Final RR guard `effective_rr > 1` (if `enable_rr_guard`).
15. Forward resolve TP/SL outcome with conservative SL-first tie-break.

## 4) Dataset We Use

Primary dataset for current runs:
- `datasets/momo_1m_7d_top100_midcap_30d/`

Important files inside:
- `dataset_manifest.csv`
  - Columns: `symbol, ok, reason, bars, path, source`
  - Used to decide which symbols are loaded and where each symbol CSV lives.
- `*_1m_7d.csv`
  - One file per symbol, standard OHLCV at 1-minute resolution.

Dataset creation script:
- `build_momentum_dataset.py`
  - Fetches 1m data (7d default), builds per-symbol files, writes manifest.
  - Supports manual list or `top100_midcap_30d` universe mode.

Example dataset build:
```bash
python build_momentum_dataset.py --universe top100_midcap_30d --midcap-count 100 --midcap-skip-top 20 --days 7 --out-dir datasets/momo_1m_7d_top100_midcap_30d
```

## 5) Main Scripts Used in This Workflow

Data and features:
- `build_momentum_dataset.py`
- `backtest_momentum_pullback_sweep.py` (feature prep + key-level helpers)

Backtest engine:
- `backtest_momo_vwap_grind15_full.py`

Analysis/export helpers commonly used with current outputs:
- `analyze_improved_scan.py`
- `analyze_improved_v2.py`
- `analyze_live_trades.py`
- `diagnose_failures.py`
- `export_combo_trade_details.py`
- `build_review_lists.py`
- `build_review_lists_from_file.py`
- `inspect_csv.py`
- `calc_avg_time_trade.py`

Monitoring/live-adjacent scripts often referenced around this workflow:
- `monitor_momo_passes.py`
- `simulate_live_trades.py`
- `start_live_trading.py`
- `live_trade_monitor.py`
- `live_trade_manager.py`

## 6) Running the Current Gate-Based Backtest

Use gate config file:
```bash
python backtest_momo_vwap_grind15_full.py \
  --dataset-dir datasets/momo_1m_7d_top100_midcap_30d \
  --gates-config momo_gate_settings.json \
  --out-prefix momo_run
```

Outputs:
- `momo_run_overall.csv`
- `momo_run_by_symbol.csv`
- `momo_run_trade_list.csv`
- `momo_run_skipped.csv`

## 7) Notes

- All gate defaults are defined in code (`GateSettings`) and can be overridden in `momo_gate_settings.json`.
- The backtest always writes the effective gate settings into `_overall.csv` so each result is self-describing.
