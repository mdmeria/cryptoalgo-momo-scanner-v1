"""
Live Trade Performance Analysis
Analyzes results from live_trades.csv
"""

import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime


def analyze_live_trades(trade_log: str = "live_trades.csv"):
    """Analyze performance from live trade log"""
    
    if not Path(trade_log).exists():
        print(f"✗ Trade log not found: {trade_log}")
        return
    
    df = pd.read_csv(trade_log)
    
    if len(df) == 0:
        print(f"No trades in {trade_log}")
        return
    
    # Separate closed and open trades
    closed = df[df['status'] == 'CLOSED'].copy()
    open_trades = df[df['status'] == 'OPEN'].copy()
    
    print("=" * 80)
    print("LIVE TRADE PERFORMANCE ANALYSIS")
    print("=" * 80)
    print()
    
    # Overall stats
    print(f"TRADE COUNT:")
    print(f"  Total Trades:        {len(df)}")
    print(f"  Closed Trades:       {len(closed)}")
    print(f"  Open Trades:         {len(open_trades)}")
    print()
    
    if len(closed) == 0:
        print("No closed trades yet. Come back later!")
        return
    
    # Convert columns to numeric
    closed['pnl_pct'] = pd.to_numeric(closed['pnl_pct'], errors='coerce')
    closed['pnl_usd'] = pd.to_numeric(closed['pnl_usd'], errors='coerce')
    closed['time_in_trade_min'] = pd.to_numeric(closed['time_in_trade_min'], errors='coerce')
    closed['tp_distance_pct'] = pd.to_numeric(closed['tp_distance_pct'], errors='coerce')
    
    # Win/Loss stats
    wins = len(closed[closed['pnl_pct'] > 0])
    losses = len(closed[closed['pnl_pct'] < 0])
    breakeven = len(closed[closed['pnl_pct'] == 0])
    win_rate = (wins / len(closed) * 100) if len(closed) > 0 else 0
    
    print(f"WIN/LOSS STATS:")
    print(f"  Wins:                {wins}")
    print(f"  Losses:              {losses}")
    print(f"  Breakeven:           {breakeven}")
    print(f"  Win Rate:            {win_rate:.1f}%")
    print()
    
    # P&L stats
    total_pnl = closed['pnl_pct'].sum()
    avg_pnl = closed['pnl_pct'].mean()
    median_pnl = closed['pnl_pct'].median()
    max_win = closed['pnl_pct'].max()
    max_loss = closed['pnl_pct'].min()
    
    print(f"P&L STATISTICS (%):")
    print(f"  Total P&L:           {total_pnl:.2f}%")
    print(f"  Average Trade:       {avg_pnl:.2f}%")
    print(f"  Median Trade:        {median_pnl:.2f}%")
    print(f"  Best Trade:          {max_win:.2f}%")
    print(f"  Worst Trade:         {max_loss:.2f}%")
    print()
    
    # Time in trade
    avg_time = closed['time_in_trade_min'].mean()
    min_time = closed['time_in_trade_min'].min()
    max_time = closed['time_in_trade_min'].max()
    
    print(f"TIME IN TRADE:")
    print(f"  Average:             {avg_time:.1f} minutes")
    print(f"  Min:                 {min_time:.1f} minutes")
    print(f"  Max:                 {max_time:.1f} minutes")
    print()
    
    # Exit reasons
    tp_hits = len(closed[closed['exit_reason'] == 'TP_HIT'])
    sl_hits = len(closed[closed['exit_reason'] == 'SL_HIT'])
    
    print(f"EXIT BREAKDOWN:")
    print(f"  TP Hits:             {tp_hits} ({tp_hits/len(closed)*100:.1f}%)")
    print(f"  SL Hits:             {sl_hits} ({sl_hits/len(closed)*100:.1f}%)")
    print()
    
    # Direction breakdown
    longs = closed[closed['direction'] == 'long']
    shorts = closed[closed['direction'] == 'short']
    
    if len(longs) > 0:
        long_wr = (len(longs[longs['pnl_pct'] > 0]) / len(longs) * 100)
        long_avg = longs['pnl_pct'].mean()
        print(f"LONG TRADES:")
        print(f"  Count:               {len(longs)}")
        print(f"  Win Rate:            {long_wr:.1f}%")
        print(f"  Avg P&L:             {long_avg:.2f}%")
    
    if len(shorts) > 0:
        short_wr = (len(shorts[shorts['pnl_pct'] > 0]) / len(shorts) * 100)
        short_avg = shorts['pnl_pct'].mean()
        print(f"SHORT TRADES:")
        print(f"  Count:               {len(shorts)}")
        print(f"  Win Rate:            {short_wr:.1f}%")
        print(f"  Avg P&L:             {short_avg:.2f}%")
    print()
    
    # Best and worst performers
    print(f"TOP PERFORMERS:")
    top_3 = closed.nlargest(3, 'pnl_pct')[['symbol', 'direction', 'entry_time_est', 'pnl_pct']]
    for idx, row in top_3.iterrows():
        print(f"  {row['symbol']:12} {row['direction']:5} @ {row['entry_time_est']:30} → +{row['pnl_pct']:.2f}%")
    print()
    
    print(f"BOTTOM PERFORMERS:")
    bottom_3 = closed.nsmallest(3, 'pnl_pct')[['symbol', 'direction', 'entry_time_est', 'pnl_pct']]
    for idx, row in bottom_3.iterrows():
        print(f"  {row['symbol']:12} {row['direction']:5} @ {row['entry_time_est']:30} → {row['pnl_pct']:.2f}%")
    print()
    
    # Symbol performance
    print(f"PERFORMANCE BY SYMBOL:")
    symbol_stats = closed.groupby('symbol').agg({
        'pnl_pct': ['count', 'mean', lambda x: (x > 0).sum() / len(x) * 100]
    }).round(2)
    symbol_stats.columns = ['Trades', 'Avg P&L %', 'Win Rate %']
    symbol_stats = symbol_stats.sort_values('Avg P&L %', ascending=False)
    print(symbol_stats.to_string())
    print()
    
    print("=" * 80)
    
    # Recent trades
    print(f"\nRECENT TRADES (last 10):")
    recent = closed.tail(10)[['entry_time_est', 'symbol', 'direction', 'entry_price', 
                              'exit_reason', 'exit_price', 'pnl_pct', 'time_in_trade_min']]
    print(recent.to_string(index=False))
    print()


def monitor_live_performance(trade_log: str = "live_trades.csv", refresh_interval: int = 30):
    """Continuously monitor live performance"""
    import time
    
    print(f"Monitoring {trade_log} (refreshing every {refresh_interval}s)")
    print("Press Ctrl+C to stop")
    print()
    
    while True:
        try:
            # Clear screen
            import os
            os.system('cls' if os.name == 'nt' else 'clear')
            
            analyze_live_trades(trade_log)
            
            print(f"\nNext update in {refresh_interval}s (Ctrl+C to stop)...")
            time.sleep(refresh_interval)
        
        except KeyboardInterrupt:
            print("\nMonitoring stopped")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Trade Performance Analysis")
    parser.add_argument("--trade-log", default="live_trades.csv", help="Trade log file to analyze")
    parser.add_argument("--monitor", action="store_true", help="Continuously monitor performance")
    parser.add_argument("--refresh", type=int, default=30, help="Refresh interval for monitoring (seconds)")
    
    args = parser.parse_args()
    
    if args.monitor:
        monitor_live_performance(args.trade_log, args.refresh)
    else:
        analyze_live_trades(args.trade_log)
