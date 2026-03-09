"""Enrich momentum setups with entry/SL/TP and forward backtest results."""

from __future__ import annotations

import argparse
import os
import pandas as pd
import requests
from entry_sl_tp import calculate_order_setup
from backtest_forward import backtest_forward


def process_setups(input_csv: str, output_csv: str) -> None:
    """
    Read momentum setups, calculate SL/TP, run forward backtest, write enriched CSV.
    """
    
    df = pd.read_csv(input_csv)
    print(f"Processing {len(df)} setups from {input_csv}")
    
    enriched_rows = []
    
    for idx, row in df.iterrows():
        symbol = row["symbol"]
        direction = row["direction"]
        timestamp_utc = row["timestamp_utc"]
        
        # Fetch 30m kline at this timestamp to get actual close price
        ts_utc = pd.to_datetime(timestamp_utc, utc=True)
        ts_ms = int(ts_utc.timestamp() * 1000)
        
        try:
            # Fetch the 30m bar that includes this timestamp
            url = "https://data-api.binance.vision/api/v3/klines"
            params = {
                "symbol": symbol.upper(),
                "interval": "30m",
                "limit": 1,
                "endTime": ts_ms,
            }
            response = requests.get(url, params=params, timeout=8)
            if response.status_code != 200:
                print(f"  Skipping {symbol} {timestamp_utc}: API error fetching kline")
                continue
            
            kline_data = response.json()
            if not kline_data or len(kline_data) == 0:
                print(f"  Skipping {symbol} {timestamp_utc}: no kline data")
                continue
            
            entry_price = float(kline_data[0][4])  # Close price
            
        except Exception as e:
            print(f"  Skipping {symbol} {timestamp_utc}: {e}")
            continue
        
        # Step 1: Calculate entry/SL/TP
        try:
            setup = calculate_order_setup(
                symbol=symbol,
                direction=direction,
                timestamp_utc=timestamp_utc,
                entry_price=entry_price,
            )
        except Exception as e:
            print(f"  Error calculating setup for {symbol} {timestamp_utc}: {e}")
            continue
        
        # Step 2: Backtest forward 4h
        try:
            backtest = backtest_forward(
                symbol=symbol,
                timestamp_utc=timestamp_utc,
                entry_price=setup.entry_price,
                sl_price=setup.sl_price,
                tp_price=setup.tp_price,
                direction=direction,
            )
        except Exception as e:
            print(f"  Error backtesting {symbol} {timestamp_utc}: {e}")
            continue
        
        # Step 3: Enrich row with new data
        enriched_row = row.to_dict()
        enriched_row.update({
            "entry_price": round(setup.entry_price, 8),
            "sl_price": round(setup.sl_price, 8),
            "tp_price": round(setup.tp_price, 8),
            "risk_usd": round(setup.risk_amount, 8),
            "reward_usd": round(setup.reward_amount, 8),
            "rr_ratio": round(setup.rr_ratio, 2),
            "tp_distance_pct": round(setup.entry_distance_pct, 2),
            "order_types": setup.order_types,
            
            "sl_hit_1h": backtest.sl_hit_1h,
            "tp_hit_1h": backtest.tp_hit_1h,
            "mae_1h_pct": round(backtest.mae_1h_pct, 2),
            "mfe_1h_pct": round(backtest.mfe_1h_pct, 2),
            "pnl_1h_pct": round(backtest.pnl_1h_pct, 2),
            "time_to_exit_1h_bars": backtest.time_to_exit_1h,
            
            "sl_hit_2h": backtest.sl_hit_2h,
            "tp_hit_2h": backtest.tp_hit_2h,
            "mae_2h_pct": round(backtest.mae_2h_pct, 2),
            "mfe_2h_pct": round(backtest.mfe_2h_pct, 2),
            "pnl_2h_pct": round(backtest.pnl_2h_pct, 2),
            "time_to_exit_2h_bars": backtest.time_to_exit_2h,
            
            "sl_hit_4h": backtest.sl_hit_4h,
            "tp_hit_4h": backtest.tp_hit_4h,
            "mae_4h_pct": round(backtest.mae_4h_pct, 2),
            "mfe_4h_pct": round(backtest.mfe_4h_pct, 2),
            "pnl_4h_pct": round(backtest.pnl_4h_pct, 2),
            "time_to_exit_4h_bars": backtest.time_to_exit_4h,
            
            "trade_outcome": backtest.trade_outcome,
        })
        
        enriched_rows.append(enriched_row)
        
        if (idx + 1) % 5 == 0:
            print(f"  Processed {idx + 1} setups...")
    
    # Write enriched CSV
    enriched_df = pd.DataFrame(enriched_rows)
    enriched_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\nWrote {len(enriched_df)} enriched setups to {output_csv}")
    
    # Print summary statistics
    print_analysis(enriched_df)


def print_analysis(df: pd.DataFrame) -> None:
    """Print backtest performance summary."""
    
    if df.empty:
        print("No data to analyze.")
        return
    
    print("\n" + "="*80)
    print("FORWARD BACKTEST ANALYSIS (4h+ lookback)")
    print("="*80)
    
    # 4h outcome analysis
    tp_hit_4h = len(df[df["tp_hit_4h"] == True])
    sl_hit_4h = len(df[df["sl_hit_4h"] == True])
    open_4h = len(df[(df["tp_hit_4h"] == False) & (df["sl_hit_4h"] == False)])
    
    print(f"\n4H OUTCOMES (n={len(df)}):")
    print(f"  TP Hit:  {tp_hit_4h:3d} ({100*tp_hit_4h/len(df):5.1f}%)")
    print(f"  SL Hit:  {sl_hit_4h:3d} ({100*sl_hit_4h/len(df):5.1f}%)")
    print(f"  Open:    {open_4h:3d} ({100*open_4h/len(df):5.1f}%)")
    
    # Win rate (TP before SL)
    win_rate = tp_hit_4h / (tp_hit_4h + sl_hit_4h) if (tp_hit_4h + sl_hit_4h) > 0 else 0
    print(f"  Win Rate (TP% of closed): {100*win_rate:5.1f}%")
    
    # P&L analysis at 4h
    avg_pnl_4h = df["pnl_4h_pct"].mean()
    avg_mae_4h = df["mae_4h_pct"].mean()
    avg_mfe_4h = df["mfe_4h_pct"].mean()
    
    print(f"\n4H PRICE ACTION:")
    print(f"  Avg PnL:  {avg_pnl_4h:7.2f}%")
    print(f"  Avg MAE:  {avg_mae_4h:7.2f}%")
    print(f"  Avg MFE:  {avg_mfe_4h:7.2f}%")
    
    # Risk/Reward
    avg_rr = df["rr_ratio"].mean()
    avg_risk = df["risk_usd"].mean()
    
    print(f"\nRISK/REWARD:")
    print(f"  Avg R:R Ratio:  {avg_rr:.2f}")
    print(f"  Avg Risk (USD): ${avg_risk:.8f}")
    
    # 1h and 2h early exits
    tp_1h = len(df[df["tp_hit_1h"] == True])
    tp_2h = len(df[df["tp_hit_2h"] == True])
    
    print(f"\nEARLY EXITS:")
    print(f"  TP Hit by 1h: {tp_1h:3d} ({100*tp_1h/len(df):5.1f}%)")
    print(f"  TP Hit by 2h: {tp_2h:3d} ({100*tp_2h/len(df):5.1f}%)")
    
    # P&L distribution
    print(f"\nP&L DISTRIBUTION (4h):")
    print(f"  Best:  {df['pnl_4h_pct'].max():7.2f}%")
    print(f"  Worst: {df['pnl_4h_pct'].min():7.2f}%")
    print(f"  Median: {df['pnl_4h_pct'].median():7.2f}%")
    
    # Winners vs losers
    winners = len(df[df["pnl_4h_pct"] > 0])
    losers = len(df[df["pnl_4h_pct"] < 0])
    print(f"\nTRADE OUTCOMES (4h):")
    print(f"  Winners: {winners:3d} ({100*winners/len(df):5.1f}%)")
    print(f"  Losers:  {losers:3d} ({100*losers/len(df):5.1f}%)")
    
    # Top/bottom performers
    print(f"\nTOP 3 PERFORMERS:")
    top3 = df.nlargest(3, "pnl_4h_pct")[["timestamp_est", "symbol", "direction", "pnl_4h_pct"]]
    for _, row in top3.iterrows():
        print(f"  {row['symbol']:12s} {row['direction']:5s} {row['pnl_4h_pct']:7.2f}%")
    
    print(f"\nBOTTOM 3 PERFORMERS:")
    bot3 = df.nsmallest(3, "pnl_4h_pct")[["timestamp_est", "symbol", "direction", "pnl_4h_pct"]]
    for _, row in bot3.iterrows():
        print(f"  {row['symbol']:12s} {row['direction']:5s} {row['pnl_4h_pct']:7.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="final_momo_passes.csv", help="Input CSV with momentum setups")
    parser.add_argument("--output", default="momo_with_backtest.csv", help="Output enriched CSV")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found")
        return
    
    process_setups(args.input, args.output)


if __name__ == "__main__":
    main()
