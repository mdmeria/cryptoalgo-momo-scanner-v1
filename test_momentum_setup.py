"""Standalone momentum setup evaluator (no screener / no market-state gate)."""

from __future__ import annotations

import argparse
from typing import List

import pandas as pd
import requests

from momentum_quality import evaluate_momentum_setup


def fetch_klines(symbol: str, interval: str = "1m", limit: int = 160) -> pd.DataFrame | None:
    """Fetch Binance Vision klines for a symbol in spot format (e.g., BTCUSDT)."""
    url = "https://data-api.binance.vision/api/v3/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return None

        raw = response.json()
        if not isinstance(raw, list) or len(raw) < 120:
            return None

        df = pd.DataFrame(
            [
                {
                    "timestamp": pd.to_datetime(k[0], unit="ms"),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                }
                for k in raw
            ]
        )
        return df.set_index("timestamp")
    except Exception:
        return None


def infer_direction(df: pd.DataFrame) -> str:
    """Infer likely momentum side from EMA alignment."""
    ema20 = df["close"].ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = df["close"].ewm(span=50, adjust=False).mean().iloc[-1]
    close = float(df["close"].iloc[-1])

    if close > ema20 > ema50:
        return "long"
    if close < ema20 < ema50:
        return "short"
    return "long"


def parse_symbols(symbols_raw: str) -> List[str]:
    return [s.strip().upper() for s in symbols_raw.split(",") if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate momentum quality for specific symbols")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols, e.g. FFUSDT,XLMUSDT")
    parser.add_argument("--direction", choices=["auto", "long", "short"], default="auto")
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--min-score", type=float, default=0.60)
    args = parser.parse_args()

    symbols = parse_symbols(args.symbols)

    print("=" * 84)
    print("Momentum Setup Evaluation (Direct Symbol Test)")
    print("No screener. No market-state gate. No min-volume filter.")
    print("=" * 84)

    for symbol in symbols:
        df = fetch_klines(symbol=symbol, interval=args.interval, limit=args.limit)
        if df is None:
            print(f"\n{symbol:<12} -> NO DATA (symbol not available or insufficient candles)")
            continue

        direction = infer_direction(df) if args.direction == "auto" else args.direction
        result = evaluate_momentum_setup(
            df=df,
            direction=direction,
            min_quality_score=args.min_score,
            symbol=symbol,
            enforce_extended_rules=True,
        )

        print(f"\n{symbol:<12} -> direction={direction:<5} quality={'PASS' if result.passed else 'FAIL'} score={result.score:.2f} tier={result.quality_tier}")
        print(
            "  checks: "
            f"approach={result.checks['slow_grind_approach']} "
            f"pre_entry_30m={result.checks['pre_entry_directional_30m']} "
            f"staircase={result.checks['left_side_staircase']} "
            f"volume={result.checks['volume_not_decreasing']} "
            f"not_choppy={result.checks['not_choppy']} "
            f"balanced_2h={result.checks['balanced_momo_2h']} "
            f"parallel_30smma_2h={result.checks['parallel_to_smma30_2h']} "
            f"spread_30_120_inc_2h={result.checks['smma_spread_increasing_2h']} "
            f"day_change={result.checks['day_change_ok']} "
            f"vwap_side={result.checks['vwap_side_ok']} "
            f"first2h_prev_vwap={result.checks['first_2h_prev_day_vwap_ok']} "
            f"entry_fresh_6h={result.checks['entry_not_crossed_6h']}"
        )
        print(
            "  metrics: "
            f"net_move_10={result.metrics['net_move_10']:.4f} "
            f"opp_candles_10={int(result.metrics['opposite_candles_10'])} "
            f"pre_entry_move_30m_pct={result.metrics.get('pre_entry_move_30m_pct', 0.0):.2f} "
            f"pre_entry_eff_30m={result.metrics.get('pre_entry_efficiency_30m', 0.0):.2f} "
            f"pre_entry_dir_ratio_30m={result.metrics.get('pre_entry_dir_bar_ratio_30m', 0.0):.2f} "
            f"smma30_slope={result.metrics['smma30_slope']:.5f} "
            f"smma120_slope={result.metrics['smma120_slope']:.5f} "
            f"staircase_bars_120={int(result.metrics['staircase_bars_120'])} "
            f"trend_stack_bars_120={int(result.metrics['trend_stack_bars_120'])} "
            f"vol_slope={result.metrics['vol_slope']:.5f} "
            f"smma_side_changes_60={int(result.metrics['smma_side_changes_60'])} "
            f"day_change_pct={result.metrics['day_change_pct']:.2f} "
            f"entry_cross_count_6h={int(result.metrics['entry_cross_count_6h'])} "
            f"retracements_found={int(result.metrics.get('retracements_found', 0))} "
            f"dir_move_2h_pct={result.metrics.get('dir_move_2h_pct', 0.0):.2f} "
            f"dir_bar_ratio_2h={result.metrics.get('dir_bar_ratio_2h', 0.0):.2f} "
            f"max_dir_impulse_8m_pct={result.metrics.get('max_dir_impulse_8m_pct', 0.0):.2f} "
            f"efficiency_2h={result.metrics.get('efficiency_2h', 0.0):.2f} "
            f"grind_windows={int(result.metrics.get('grind_windows_count', 0))} "
            f"avg_grind_impulse={result.metrics.get('avg_grind_impulse_pct', 0.0):.2f} "
            f"grind_std={result.metrics.get('grind_impulse_std', 0.0):.4f} "
            f"grind_bar_participation={result.metrics.get('grind_bar_participation', 0.0):.2f} "
            f"smma30_crosses_2h={int(result.metrics.get('smma30_crosses_2h', 0))} "
            f"smma30_trend_side_ratio_2h={result.metrics.get('smma30_trend_side_ratio_2h', 0.0):.2f} "
            f"smma_spread_slope_2h={result.metrics.get('smma_spread_slope_2h', 0.0):.5f} "
            f"smma_spread_up_ratio_2h={result.metrics.get('smma_spread_up_ratio_2h', 0.0):.2f}"
        )


if __name__ == "__main__":
    main()
