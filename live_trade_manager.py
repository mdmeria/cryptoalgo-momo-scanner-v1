"""
Live Momentum Trading System
Scans for momentum passes, places trades with structural SL/TP, and logs results
"""

import pandas as pd
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import time
import requests

from entry_sl_tp import calculate_order_setup


class LiveTradeLogger:
    """Manages trade entry and exit logging"""
    
    def __init__(self, log_file: str = "live_trades.csv"):
        self.log_file = log_file
        self.trades = []
        self._init_log()
    
    def _init_log(self):
        """Initialize or load existing log"""
        if os.path.exists(self.log_file):
            self.trades = pd.read_csv(self.log_file).to_dict('records')
        else:
            # Create with headers
            columns = [
                'trade_id', 'entry_time_utc', 'entry_time_est', 'symbol', 'direction',
                'entry_price', 'sl_price', 'tp_price', 'risk_usd', 'reward_usd', 'rr_ratio',
                'tp_distance_pct', 'status', 'exit_time_utc', 'exit_time_est', 
                'exit_price', 'exit_reason', 'pnl_usd', 'pnl_pct', 'time_in_trade_min'
            ]
            pd.DataFrame(columns=columns).to_csv(self.log_file, index=False)
            self.trades = []
    
    def log_entry(self, symbol: str, direction: str, entry_price: float, 
                  sl_price: float, tp_price: float, risk_usd: float, 
                  reward_usd: float, rr_ratio: float, tp_distance_pct: float,
                  timestamp_utc: str, timestamp_est: str) -> str:
        """Log a new trade entry"""
        trade_id = f"{symbol}_{timestamp_utc.replace(':', '').replace('-', '').replace('T', '_').replace('Z', '')}"
        
        trade = {
            'trade_id': trade_id,
            'entry_time_utc': timestamp_utc,
            'entry_time_est': timestamp_est,
            'symbol': symbol,
            'direction': direction,
            'entry_price': round(entry_price, 8),
            'sl_price': round(sl_price, 8),
            'tp_price': round(tp_price, 8),
            'risk_usd': round(risk_usd, 8),
            'reward_usd': round(reward_usd, 8),
            'rr_ratio': round(rr_ratio, 2),
            'tp_distance_pct': round(tp_distance_pct, 2),
            'status': 'OPEN',
            'exit_time_utc': '',
            'exit_time_est': '',
            'exit_price': '',
            'exit_reason': '',
            'pnl_usd': '',
            'pnl_pct': '',
            'time_in_trade_min': ''
        }
        
        self.trades.append(trade)
        self._save()
        print(f"✓ ENTRY: {symbol} {direction.upper():5} @ ${entry_price:.8f} | SL: ${sl_price:.8f} | TP: ${tp_price:.8f}")
        return trade_id
    
    def log_exit(self, trade_id: str, exit_reason: str, exit_price: float,
                 timestamp_utc: str, timestamp_est: str):
        """Log a trade exit"""
        for trade in self.trades:
            if trade['trade_id'] == trade_id:
                trade['status'] = 'CLOSED'
                trade['exit_time_utc'] = timestamp_utc
                trade['exit_time_est'] = timestamp_est
                trade['exit_price'] = round(exit_price, 8)
                trade['exit_reason'] = exit_reason
                
                # Calculate P&L
                entry_price = float(trade['entry_price'])
                direction = trade['direction']
                
                if direction == 'long':
                    pnl_usd = (exit_price - entry_price) * 1  # Assuming 1 unit
                    pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                else:  # short
                    pnl_usd = (entry_price - exit_price) * 1
                    pnl_pct = ((entry_price - exit_price) / entry_price) * 100
                
                trade['pnl_usd'] = round(pnl_usd, 8)
                trade['pnl_pct'] = round(pnl_pct, 2)
                
                # Calculate time in trade
                entry_time = pd.to_datetime(trade['entry_time_utc'])
                exit_time = pd.to_datetime(timestamp_utc)
                time_in_trade = (exit_time - entry_time).total_seconds() / 60
                trade['time_in_trade_min'] = round(time_in_trade, 1)
                
                self._save()
                
                outcome = "✓ TP HIT" if exit_reason == "TP_HIT" else "✗ SL HIT"
                print(f"{outcome}: {trade['symbol']} {trade['direction'].upper():5} | Exit: ${exit_price:.8f} | PnL: {pnl_pct:+.2f}% (${pnl_usd:+.8f}) | Time: {time_in_trade:.1f}m")
                break
    
    def _save(self):
        """Save trades to CSV"""
        df = pd.DataFrame(self.trades)
        df.to_csv(self.log_file, index=False)
    
    def get_summary(self):
        """Get performance summary"""
        if not self.trades:
            return None
        
        closed = [t for t in self.trades if t['status'] == 'CLOSED']
        open_trades = [t for t in self.trades if t['status'] == 'OPEN']
        
        if not closed:
            return {
                'total_trades': len(self.trades),
                'closed_trades': 0,
                'open_trades': len(open_trades),
                'win_rate': 0,
                'avg_pnl_pct': 0,
                'total_pnl_pct': 0
            }
        
        wins = len([t for t in closed if t['pnl_pct'] > 0])
        losses = len([t for t in closed if t['pnl_pct'] < 0])
        
        return {
            'total_trades': len(self.trades),
            'closed_trades': len(closed),
            'open_trades': len(open_trades),
            'wins': wins,
            'losses': losses,
            'win_rate': (wins / len(closed) * 100) if closed else 0,
            'avg_pnl_pct': sum([t['pnl_pct'] for t in closed]) / len(closed) if closed else 0,
            'total_pnl_pct': sum([t['pnl_pct'] for t in closed])
        }


def fetch_current_price(symbol: str) -> float:
    """Fetch current price from Binance"""
    try:
        response = requests.get(
            "https://api.binance.com/api/v3/ticker/price",
            params={"symbol": f"{symbol}USDT" if not symbol.endswith('USDT') else symbol}
        )
        if response.status_code == 200:
            return float(response.json()['price'])
    except Exception as e:
        print(f"  Warning: Could not fetch price for {symbol}: {str(e)}")
    return None


def run_live_scan(input_file: str, trade_log: str = "live_trades.csv", 
                  interval_min: int = 5, max_trades_per_scan: int = 5):
    """
    Run live scanning for momentum passes and manage trades
    
    Args:
        input_file: CSV file with momentum passes to place trades from
        trade_log: Output CSV file for trade logging
        interval_min: Interval in minutes between scans
        max_trades_per_scan: Maximum new trades to place per scan
    """
    
    logger = LiveTradeLogger(trade_log)
    
    print(f"=" * 70)
    print(f"LIVE MOMENTUM TRADING SYSTEM")
    print(f"=" * 70)
    print(f"Input file: {input_file}")
    print(f"Trade log: {trade_log}")
    print(f"Scan interval: {interval_min} minutes")
    print(f"Max trades per scan: {max_trades_per_scan}")
    print()
    
    # Load momentum passes
    if not os.path.exists(input_file):
        print(f"✗ Input file not found: {input_file}")
        return
    
    df_passes = pd.read_csv(input_file)
    print(f"Loaded {len(df_passes)} momentum passes from {input_file}")
    print()
    
    placed_trades = set()  # Track which setups have been placed
    
    while True:
        try:
            current_time_utc = datetime.utcnow().isoformat() + "Z"
            current_time_est = datetime.now().isoformat()
            
            print(f"\n[{current_time_est}] Scan started...")
            
            # Check for new momentum passes to place
            trades_to_place = []
            for idx, row in df_passes.iterrows():
                setup_id = f"{row['symbol']}_{row['timestamp_utc']}"
                
                if setup_id not in placed_trades:
                    trades_to_place.append(row)
                    placed_trades.add(setup_id)
                    
                    if len(trades_to_place) >= max_trades_per_scan:
                        break
            
            # Place new trades
            for row in trades_to_place:
                try:
                    symbol = row['symbol']
                    direction = row['direction']
                    timestamp_utc = row['timestamp_utc']
                    timestamp_est = row['timestamp_est']
                    
                    # Fetch current price
                    entry_price = fetch_current_price(symbol)
                    if entry_price is None:
                        print(f"✗ Could not fetch price for {symbol}, skipping")
                        continue
                    
                    # Calculate structural SL/TP
                    setup = calculate_order_setup(
                        symbol=symbol,
                        direction=direction,
                        timestamp_utc=timestamp_utc,
                        entry_price=entry_price
                    )
                    
                    # Log entry
                    trade_id = logger.log_entry(
                        symbol=symbol,
                        direction=direction,
                        entry_price=setup.entry_price,
                        sl_price=setup.sl_price,
                        tp_price=setup.tp_price,
                        risk_usd=setup.risk_amount,
                        reward_usd=setup.reward_amount,
                        rr_ratio=setup.rr_ratio,
                        tp_distance_pct=setup.entry_distance_pct,
                        timestamp_utc=timestamp_utc,
                        timestamp_est=timestamp_est
                    )
                
                except Exception as e:
                    print(f"✗ Error placing trade for {row.get('symbol', '?')}: {str(e)}")
            
            if trades_to_place:
                print(f"Placed {len(trades_to_place)} new trades")
            
            # Print status
            summary = logger.get_summary()
            if summary:
                print(f"\nPerformance Summary:")
                print(f"  Total Trades: {summary['total_trades']} | Open: {summary['open_trades']} | Closed: {summary['closed_trades']}")
                if summary['closed_trades'] > 0:
                    print(f"  Win Rate: {summary['win_rate']:.1f}% ({summary['wins']}W {summary['losses']}L)")
                    print(f"  Avg P&L: {summary['avg_pnl_pct']:.2f}% | Total P&L: {summary['total_pnl_pct']:.2f}%")
            
            # Wait before next scan
            print(f"Next scan in {interval_min} minutes...")
            time.sleep(interval_min * 60)
        
        except KeyboardInterrupt:
            print("\n\nLive scan stopped by user")
            
            # Print final summary
            summary = logger.get_summary()
            if summary:
                print(f"\n{'='*70}")
                print(f"FINAL PERFORMANCE SUMMARY")
                print(f"{'='*70}")
                print(f"Total Trades: {summary['total_trades']}")
                print(f"Open Trades: {summary['open_trades']}")
                print(f"Closed Trades: {summary['closed_trades']}")
                if summary['closed_trades'] > 0:
                    print(f"Win Rate: {summary['win_rate']:.1f}%")
                    print(f"Wins/Losses: {summary['wins']}/{summary['losses']}")
                    print(f"Avg P&L: {summary['avg_pnl_pct']:.2f}%")
                    print(f"Total P&L: {summary['total_pnl_pct']:.2f}%")
            
            break
        
        except Exception as e:
            print(f"✗ Scan error: {str(e)}")
            print(f"Retrying in {interval_min} minutes...")
            time.sleep(interval_min * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Momentum Trading System")
    parser.add_argument("--input", default="final_momo_passes.csv", help="Input file with momentum passes")
    parser.add_argument("--output", default="live_trades.csv", help="Output CSV file for trade logging")
    parser.add_argument("--interval-min", type=int, default=5, help="Scan interval in minutes")
    parser.add_argument("--max-trades", type=int, default=5, help="Maximum new trades per scan")
    
    args = parser.parse_args()
    
    run_live_scan(
        input_file=args.input,
        trade_log=args.output,
        interval_min=args.interval_min,
        max_trades_per_scan=args.max_trades
    )
