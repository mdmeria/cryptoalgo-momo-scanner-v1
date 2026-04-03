#!/usr/bin/env python3
"""
Debug: run existing momo gates + ZCT gates against actual B+ trades from the PDF.
Shows which gate blocks each trade.
"""
import sys, os, math
import pandas as pd, numpy as np
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backtest_momo_vwap_grind15_full import (
    GateSettings as MomoGateSettings,
    prepare_features as prepare_momo_features,
    gate_volusd, gate_vol_usd_5m, gate_30m_noise, gate_5m_antispike,
    gate_regime_breakout, gate_entry_not_crossed_6h, gate_10m_directional,
    gate_vwap_side, gate_ema7_cross, gate_staircase_quality, gate_smma_trend,
    gate_last15m, compute_sl_tp, rr_guard, min_tp_sl_gate,
)

DATASET_DIR = Path("datasets/binance_futures_1m")

# B+ trades from the PDF (subset we have data for)
PDF_TRADES = [
    (2, "ENAUSDT", "2024-08-02", "15:19", "short"),
    (7, "TRXUSDT", "2024-08-20", "14:00", "long"),
    (8, "TRXUSDT", "2024-08-21", "14:30", "short"),
    (9, "AAVEUSDT", "2024-08-28", "10:00", "long"),
    (10, "TRXUSDT", "2024-08-29", "07:14", "short"),
    (23, "LINKUSDT", "2024-11-22", "08:30", "long"),
    (24, "SOLUSDT", "2024-11-23", "18:00", "short"),
    (27, "VIRTUALUSDT", "2024-12-20", "18:30", "long"),
    (34, "XRPUSDT", "2025-01-27", "14:16", "long"),
    (36, "DOGEUSDT", "2025-01-28", "08:44", "short"),
    (59, "ENAUSDT", "2024-08-10", "10:30", "long"),
    (62, "ENAUSDT", "2024-04-16", "08:00", "long"),
    (65, "SUIUSDT", "2024-10-24", "05:00", "short"),
    (70, "ONDOUSDT", "2024-07-18", "16:15", "long"),
    (86, "ARCUSDT", "2025-03-08", "14:21", "short"),
    (90, "LAYERUSDT", "2025-02-21", "04:10", "short"),
]

_DF_CACHE = {}

def load_symbol(symbol):
    if symbol in _DF_CACHE:
        return _DF_CACHE[symbol]
    candidates = [f"{symbol}_1m.csv"]
    if not symbol.startswith("1000"):
        candidates.append(f"1000{symbol}_1m.csv")
    for fname in candidates:
        fpath = DATASET_DIR / fname
        if fpath.exists():
            df = pd.read_csv(str(fpath), parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            _DF_CACHE[symbol] = df
            return df
    return None


def test_trade(trade_num, symbol, date_str, entry_time, direction):
    df_raw = load_symbol(symbol)
    if df_raw is None:
        print(f"  Trade #{trade_num} {symbol}: NO DATA")
        return

    cfg = MomoGateSettings.from_json("zct_momo_gate_settings.json") if Path("zct_momo_gate_settings.json").exists() else MomoGateSettings()

    # Find the bar
    entry_dt = pd.Timestamp(f"{date_str} {entry_time}:00", tz="UTC")
    diffs = (df_raw["timestamp"] - entry_dt).abs()
    bar_idx = diffs.idxmin()

    # Prepare features
    df_indexed = df_raw.set_index("timestamp").copy()
    df_indexed.index = pd.to_datetime(df_indexed.index, utc=True)
    df_prepped = prepare_momo_features(df_indexed)

    # Get slice up to bar
    ts_at_bar = df_raw.iloc[bar_idx]["timestamp"]
    slice_end = df_prepped.index.searchsorted(ts_at_bar, side="right")
    df_slice = df_prepped.iloc[max(0, slice_end - 500):slice_end]

    if len(df_slice) < 50:
        print(f"  Trade #{trade_num} {symbol}: INSUFFICIENT DATA at {entry_dt}")
        return

    print(f"\n{'='*80}")
    print(f"Trade #{trade_num}: {symbol} {direction} @ {date_str} {entry_time} UTC")
    print(f"{'='*80}")

    is_long = direction == "long"
    close = float(df_slice["close"].iloc[-1])
    print(f"  Price at bar: {close}")

    # Test each gate individually
    gates = []

    # Global gates
    g1 = gate_volusd(df_slice) if cfg.enable_volusd_gate else True
    gates.append(("volusd (nama60 rising)", g1, ""))

    g2 = gate_vol_usd_5m(df_slice, cfg)
    gates.append(("vol_usd_5m", g2, f"5m_vol=${float(df_slice['vol_usd'].iloc[-5:].sum()):,.0f}"))

    g3 = gate_30m_noise(df_slice, cfg) if cfg.enable_30m_noise_gate else True
    avg_wick = float(df_slice["wick_ratio"].iloc[-30:].mean()) if len(df_slice) >= 30 else 0
    gates.append(("30m_noise", g3, f"avg_wick={avg_wick:.3f}"))

    g4 = gate_5m_antispike(df_slice, cfg) if cfg.enable_5m_antispike_gate else True
    gates.append(("5m_antispike", g4, ""))

    # Side-specific gates
    if cfg.enable_regime_breakout_gate:
        passed, reason = gate_regime_breakout(df_slice, direction, cfg)
        gates.append(("regime_breakout", passed, reason))

    if cfg.entry_not_crossed_6h:
        passed = gate_entry_not_crossed_6h(df_slice, direction)
        if is_long:
            hi6h = float(df_slice["high_6h"].iloc[-1]) if "high_6h" in df_slice.columns else 0
            gates.append(("entry_not_crossed_6h", passed, f"close={close:.4f} hi6h={hi6h:.4f}"))
        else:
            lo6h = float(df_slice["low_6h"].iloc[-1]) if "low_6h" in df_slice.columns else 0
            gates.append(("entry_not_crossed_6h", passed, f"close={close:.4f} lo6h={lo6h:.4f}"))

    if cfg.enable_10m_gate:
        passed, reason = gate_10m_directional(df_slice, direction, cfg)
        gates.append(("10m_directional", passed, reason))

    if cfg.enable_vwap_side_gate:
        vwap = float(df_slice["vwap"].iloc[-1]) if "vwap" in df_slice.columns else 0
        passed = gate_vwap_side(df_slice, direction)
        gates.append(("vwap_side", passed, f"close={close:.4f} vwap={vwap:.4f}"))

    if cfg.enable_ema7_cross_gate:
        passed, reason = gate_ema7_cross(df_slice, direction, cfg)
        ema7 = float(df_slice["ema7"].iloc[-1]) if "ema7" in df_slice.columns else 0
        smma30 = float(df_slice["smma30"].iloc[-1]) if "smma30" in df_slice.columns else 0
        gates.append(("ema7_cross", passed, f"ema7={ema7:.4f} smma30={smma30:.4f} {reason}"))

    if cfg.enable_staircase_gate:
        passed, reason = gate_staircase_quality(df_slice, direction, cfg)
        gates.append(("staircase_quality", passed, reason))

    if cfg.enable_2h_gate:
        passed, reason = gate_smma_trend(df_slice, direction, cfg)
        gates.append(("2h_smma_trend", passed, reason))

    # ZCT duration check
    c_arr = df_raw["close"].values[:bar_idx + 1].astype(float)
    smma30_arr = pd.Series(c_arr).ewm(alpha=1.0 / 30, adjust=False).mean().values
    streak = 0
    for j in range(len(c_arr) - 1, -1, -1):
        if is_long and c_arr[j] > smma30_arr[j]:
            streak += 1
        elif not is_long and c_arr[j] < smma30_arr[j]:
            streak += 1
        else:
            break
    dur_hrs = streak / 60.0
    gates.append(("ZCT_duration_2h", dur_hrs >= 2.0, f"duration={dur_hrs:.1f}h ({streak} bars)"))

    # ZCT grind check (last 10 bars)
    if len(c_arr) >= 15:
        tail = c_arr[-10:]
        bm = np.abs(np.diff(tail))
        avg_m = np.mean(bm)
        max_m = np.max(bm)
        sr = max_m / avg_m if avg_m > 0 else 0
        mbp = max_m / np.mean(tail) * 100 if np.mean(tail) > 0 else 0
        is_grind = not (sr > 4.5 and mbp > 0.5)
        gates.append(("ZCT_grind", is_grind, f"spike_ratio={sr:.2f} max_bar_pct={mbp:.3f}%"))

    # Market condition
    cache_path = DATASET_DIR / "market_conditions_cache.csv"
    if cache_path.exists():
        cache_df = pd.read_csv(str(cache_path), parse_dates=["timestamp"])
        cache_ts = cache_df["timestamp"].values
        idx = int(np.searchsorted(cache_ts, np.datetime64(entry_dt.tz_localize(None)), side="right")) - 1
        mkt_score = int(cache_df.iloc[idx]["score"]) if idx >= 0 else 0
        if direction == "long":
            mkt_ok = mkt_score >= 2
        else:
            mkt_ok = mkt_score <= -2
        gates.append(("market_condition", mkt_ok, f"score={mkt_score} (need {'>=2' if direction == 'long' else '<=-2'})"))

    # Print results
    all_pass = True
    for name, passed, detail in gates:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {'[PASS]' if passed else '[FAIL]':>6} {name:30s} {detail}")

    print(f"\n  OVERALL: {'ALL GATES PASS' if all_pass else 'BLOCKED'}")
    blocked = [name for name, passed, _ in gates if not passed]
    if blocked:
        print(f"  Blocked by: {', '.join(blocked)}")


def main():
    print("Testing existing momo gates against PDF B+ trades...")
    print("Config: zct_momo_gate_settings.json")
    print()

    for trade in PDF_TRADES:
        test_trade(*trade)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
