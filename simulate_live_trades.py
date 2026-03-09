"""
Live Trade Simulator - Demo/Testing Version
Uses backtest data to simulate live trading when Binance API is unavailable
"""

import pandas as pd
import os
from datetime import datetime
import time
import argparse


def simulate_live_trades_from_backtest(backtest_file: str = "momo_with_backtest_structural.csv",
                                       output_file: str = "live_trades.csv",
                                       speed_factor: int = 60):
    """
    Simulate live trading by replaying backtest data with delays
    
    Args:
        backtest_file: Source backtest CSV with entry/SL/TP info
        output_file: Output file for trade logs
        speed_factor: Speed multiplier (60 = run at 60x speed, 1h becomes 1m)
    """
    
    # Load backtest data
    df_backtest = pd.read_csv(backtest_file)
    
    print("=" * 70)
    print("LIVE TRADE SIMULATOR (Demo/Test Mode)")
    print("=" * 70)
    print(f"Using backtest results from: {backtest_file}")
    print(f"Output file: {output_file}")
    print(f"Simulating {len(df_backtest)} trades")
    print(f"Speed factor: {speed_factor}x (1h = {60/speed_factor:.0f}s)")
    print()
    
    # Initialize output file
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # Write headers
    headers = ['trade_id', 'entry_time_utc', 'entry_time_est', 'symbol', 'direction',
               'entry_price', 'sl_price', 'tp_price', 'risk_usd', 'reward_usd', 'rr_ratio',
               'tp_distance_pct', 'status', 'exit_time_utc', 'exit_time_est', 
               'exit_price', 'exit_reason', 'pnl_usd', 'pnl_pct', 'time_in_trade_min']
    
    with open(output_file, 'w') as f:
        f.write(','.join(headers) + '\n')
    
    trades_created = 0
    trades_closed = 0
    
    # Simulate each trade
    for idx, row in df_backtest.iterrows():
        if trades_created >= 20:  # Limit to first 20 trades for demo
            break
        
        try:
            trade_id = f"{row['symbol']}_{row['timestamp_est'].replace(':', '').replace('-', '')}"
            
            # Calculate realistic delay before close
            if row['sl_hit_4h'] or row['tp_hit_4h']:
                # Trade closes - use simulated time to close
                exit_reason = 'TP_HIT' if row['tp_hit_4h'] else 'SL_HIT'
                
                # Estimate time to close (in hours, then convert to delay seconds)
                if row['sl_hit_1h'] or row['tp_hit_1h']:
                    time_hours = 0.5  # ~30 minutes
                elif row['sl_hit_2h'] or row['tp_hit_2h']:
                    time_hours = 1.0  # ~1 hour
                else:
                    time_hours = 2.0  # ~2 hours
                
                delay_seconds = (time_hours * 3600) / speed_factor
                
                print(f"[{idx+1}/{len(df_backtest)}] {trade_id}")
                print(f"  → Entry: {row['symbol']} {row['direction'].upper()} @ ${row['entry_price']:.8f}")
                print(f"  → SL: ${row['sl_price']:.8f} | TP: ${row['tp_price']:.8f}")
                print(f"  → Will close in {time_hours:.1f}h (delay: {delay_seconds:.0f}s)")
                print()
                
                # Create entry
                entry_time_utc = row['timestamp_utc']
                entry_time_est = row['timestamp_est']
                
                entry_line = f"{trade_id},{entry_time_utc},{entry_time_est},{row['symbol']},{row['direction']},{row['entry_price']},{row['sl_price']},{row['tp_price']},{row['risk_usd']},{row['reward_usd']},{row['rr_ratio']},{row['tp_distance_pct']},OPEN,,,,,,,"
                
                with open(output_file, 'a') as f:
                    f.write(entry_line + '\n')
                
                trades_created += 1
                
                # Wait before closing
                time.sleep(delay_seconds)
                
                # Calculate exit
                from datetime import datetime, timedelta
                entry_dt = pd.to_datetime(entry_time_utc)
                close_dt = entry_dt + timedelta(hours=time_hours)
                exit_time_utc = close_dt.isoformat()
                exit_time_est = close_dt.isoformat()
                
                # Read and update the file
                df_live = pd.read_csv(output_file)
                trade_mask = df_live['trade_id'] == trade_id
                
                if trade_mask.any():
                    df_live.loc[trade_mask, 'status'] = 'CLOSED'
                    df_live.loc[trade_mask, 'exit_time_utc'] = exit_time_utc
                    df_live.loc[trade_mask, 'exit_time_est'] = exit_time_est
                    df_live.loc[trade_mask, 'exit_price'] = row['tp_price'] if exit_reason == 'TP_HIT' else row['sl_price']
                    df_live.loc[trade_mask, 'exit_reason'] = exit_reason
                    df_live.loc[trade_mask, 'pnl_usd'] = row['reward_usd'] if exit_reason == 'TP_HIT' else -row['risk_usd']
                    df_live.loc[trade_mask, 'pnl_pct'] = row['pnl_4h_pct']
                    df_live.loc[trade_mask, 'time_in_trade_min'] = time_hours * 60
                    
                    df_live.to_csv(output_file, index=False)
                    trades_closed += 1
                    
                    # Print result
                    outcome = "✓ TP HIT" if exit_reason == "TP_HIT" else "✗ SL HIT"
                    print(f"{outcome}: {row['symbol']} | PnL: {row['pnl_4h_pct']:+.2f}% | Time: {time_hours:.1f}h")
                    print()
        
        except Exception as e:
            print(f"✗ Error simulating trade: {str(e)}")
    
    print()
    print("=" * 70)
    print(f"SIMULATION COMPLETE")
    print(f"Trades created: {trades_created}")
    print(f"Trades closed: {trades_closed}")
    print(f"Output saved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Trade Simulator")
    parser.add_argument("--backtest", default="momo_with_backtest_structural.csv", help="Backtest data file")
    parser.add_argument("--output", default="live_trades.csv", help="Output trade log")
    parser.add_argument("--speed", type=int, default=60, help="Speed factor (60 = 1h becomes 1m)")
    
    args = parser.parse_args()
    
    simulate_live_trades_from_backtest(args.backtest, args.output, args.speed)
