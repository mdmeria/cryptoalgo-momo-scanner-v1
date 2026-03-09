"""
Live Trade Monitor
Monitors active trades for SL/TP hits and logs exits
"""

import pandas as pd
import requests
import json
from datetime import datetime
import argparse
import time
from pathlib import Path


def fetch_current_prices(symbols: list) -> dict:
    """Fetch current prices from Binance"""
    if not symbols:
        return {}
    
    try:
        # Fetch all prices at once
        response = requests.get(
            "https://api.binance.com/api/v3/ticker/price",
            params={"symbols": json.dumps([f"{s}USDT" if not s.endswith('USDT') else s for s in symbols])}
        )
        
        if response.status_code == 200:
            prices = {}
            for item in response.json():
                symbol = item['symbol'].replace('USDT', '')
                prices[symbol] = float(item['price'])
            return prices
    except Exception as e:
        print(f"✗ Error fetching prices: {str(e)}")
    
    return {}


def check_trade_exits(trade_log: str = "live_trades.csv"):
    """
    Check for trade exits based on current prices
    """
    
    print("=" * 70)
    print("LIVE TRADE MONITOR")
    print("=" * 70)
    print(f"Monitoring trades from {trade_log}")
    print()
    
    if not Path(trade_log).exists():
        print(f"✗ Trade log file not found: {trade_log}")
        return
    
    update_count = 0
    
    while True:
        try:
            # Load current trades
            df = pd.read_csv(trade_log)
            
            # Filter for open trades
            open_trades = df[df['status'] == 'OPEN'].copy()
            
            if len(open_trades) == 0:
                print(f"[{datetime.now().isoformat()}] No open trades to monitor")
                time.sleep(60)
                continue
            
            # Get unique symbols
            symbols = open_trades['symbol'].unique().tolist()
            
            # Fetch current prices
            current_prices = fetch_current_prices(symbols)
            
            if not current_prices:
                print(f"[{datetime.now().isoformat()}] Could not fetch prices")
                time.sleep(60)
                continue
            
            current_time_utc = datetime.utcnow().isoformat() + "Z"
            current_time_est = datetime.now().isoformat()
            
            # Check each open trade
            closed_count = 0
            for idx, trade in open_trades.iterrows():
                symbol = trade['symbol']
                
                if symbol not in current_prices:
                    continue
                
                current_price = current_prices[symbol]
                direction = trade['direction']
                sl_price = float(trade['sl_price'])
                tp_price = float(trade['tp_price'])
                entry_price = float(trade['entry_price'])
                
                # Determine if SL or TP was hit
                exit_reason = None
                exit_price = None
                
                if direction == 'long':
                    if current_price >= tp_price:
                        exit_reason = 'TP_HIT'
                        exit_price = tp_price
                    elif current_price <= sl_price:
                        exit_reason = 'SL_HIT'
                        exit_price = sl_price
                
                else:  # short
                    if current_price <= tp_price:
                        exit_reason = 'TP_HIT'
                        exit_price = tp_price
                    elif current_price >= sl_price:
                        exit_reason = 'SL_HIT'
                        exit_price = sl_price
                
                # Update trade if exit condition met
                if exit_reason:
                    df.loc[idx, 'status'] = 'CLOSED'
                    df.loc[idx, 'exit_time_utc'] = current_time_utc
                    df.loc[idx, 'exit_time_est'] = current_time_est
                    df.loc[idx, 'exit_price'] = exit_price
                    df.loc[idx, 'exit_reason'] = exit_reason
                    
                    # Calculate P&L
                    if direction == 'long':
                        pnl_usd = (exit_price - entry_price) * 1
                        pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                    else:
                        pnl_usd = (entry_price - exit_price) * 1
                        pnl_pct = ((entry_price - exit_price) / entry_price) * 100
                    
                    df.loc[idx, 'pnl_usd'] = pnl_usd
                    df.loc[idx, 'pnl_pct'] = pnl_pct
                    
                    # Time in trade
                    entry_time = pd.to_datetime(trade['entry_time_utc'])
                    exit_time = pd.to_datetime(current_time_utc)
                    time_in_trade_min = (exit_time - entry_time).total_seconds() / 60
                    df.loc[idx, 'time_in_trade_min'] = time_in_trade_min
                    
                    # Print exit
                    outcome = "✓ TP HIT" if exit_reason == "TP_HIT" else "✗ SL HIT"
                    print(f"{outcome}: {symbol} {direction.upper()} | Exit: {exit_price:.8f} | PnL: {pnl_pct:.2f}% | Time: {time_in_trade_min:.1f}m")
                    closed_count += 1
            
            # Save updated trades
            df.to_csv(trade_log, index=False)
            
            if closed_count > 0:
                update_count += closed_count
                print(f"Updated {closed_count} trade exit(s)")
            
            # Print status
            open_count = len(df[df['status'] == 'OPEN'])
            closed_count_total = len(df[df['status'] == 'CLOSED'])
            
            if closed_count_total > 0:
                closed_df = df[df['status'] == 'CLOSED']
                wins = len(closed_df[closed_df['pnl_pct'] > 0])
                losses = len(closed_df[closed_df['pnl_pct'] < 0])
                win_rate = (wins / closed_count_total * 100) if closed_count_total > 0 else 0
                avg_pnl = closed_df['pnl_pct'].mean()
                
                print(f"  Open: {open_count} | Closed: {closed_count_total} ({wins}W {losses}L) | Win%: {win_rate:.1f}% | Avg P&L: {avg_pnl:.2f}%")
            else:
                print(f"  Open: {open_count} | No closed trades yet")
            
            # Wait before next check
            time.sleep(60)
        
        except KeyboardInterrupt:
            print("\n\nMonitor stopped by user")
            break
        
        except Exception as e:
            print(f"✗ Monitor error: {str(e)}")
            time.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Trade Monitor")
    parser.add_argument("--trade-log", default="live_trades.csv", help="Trade log file to monitor")
    
    args = parser.parse_args()
    
    check_trade_exits(trade_log=args.trade_log)
