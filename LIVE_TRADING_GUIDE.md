# Live Momentum Trading System - Quick Start Guide

## Overview

This system continuously scans for momentum passes, places trades with structural SL/TP, and logs all results for performance evaluation.

### Components

1. **live_trade_manager.py** - Places trades from momentum passes with structural SL/TP
2. **live_trade_monitor.py** - Monitors active trades for exit conditions (SL/TP hits)
3. **analyze_live_trades.py** - Analyzes performance from the trade log
4. **start_live_trading.py** - Master startup script (optional, coordinates all components)

---

## Quick Start (3 Steps)

### Step 1: Generate Momentum Passes (If Needed)

If you don't have `final_momo_passes.csv`:

```bash
cd C:\Projects\CryptoAlgo
python monitor_momo_passes.py --once --output final_momo_passes.csv --interval-min 5
```

This will scan the market once and save all current momentum passes to `final_momo_passes.csv`.

**OR** Run continuous scanning:

```bash
python monitor_momo_passes.py --interval-min 5 --output final_momo_passes.csv
```

This will keep updating `final_momo_passes.csv` with new passes every 5 minutes.

---

### Step 2: Start Live Trade Manager

In a new terminal:

```bash
cd C:\Projects\CryptoAlgo
python live_trade_manager.py --input final_momo_passes.csv --output live_trades.csv --interval-min 5 --max-trades 5
```

**Parameters:**
- `--input`: CSV file with momentum passes (will be scanned for new entries)
- `--output`: CSV file to log all trades and results
- `--interval-min`: Scan interval in minutes (default: 5)
- `--max-trades`: Max new trades to place per scan (default: 5)

**Output:** Logs each trade entry with:
- Entry time (UTC & EST)
- Symbol & direction
- Entry price, SL, TP
- Risk/reward ratio
- TP distance %

---

### Step 3: Start Live Trade Monitor

In another terminal:

```bash
cd C:\Projects\CryptoAlgo
python live_trade_monitor.py --trade-log live_trades.csv
```

This monitors active trades for SL/TP hits and updates the log with:
- Exit time
- Exit price & reason (TP_HIT or SL_HIT)
- P&L % and $
- Time in trade (minutes)

---

## Monitoring Performance

### View Results in Real-Time

**One-time analysis:**
```bash
python analyze_live_trades.py --trade-log live_trades.csv
```

**Continuous monitoring (updates every 30s):**
```bash
python analyze_live_trades.py --trade-log live_trades.csv --monitor
```

**Custom refresh interval:**
```bash
python analyze_live_trades.py --trade-log live_trades.csv --monitor --refresh 60
```

---

## All-in-One Startup (Optional)

If you want to start everything at once:

```bash
python start_live_trading.py --scan-interval 5 --max-trades 5
```

This starts both the trade manager and exit monitor in parallel and displays their output in one terminal.

---

## Output Files

### live_trades.csv

Main trading log with columns:
- `trade_id`: Unique trade identifier
- `entry_time_utc`, `entry_time_est`: Entry timestamps
- `symbol`, `direction`: Coin and direction (long/short)
- `entry_price`, `sl_price`, `tp_price`: Order prices
- `risk_usd`, `reward_usd`, `rr_ratio`: Risk/reward metrics
- `tp_distance_pct`: Distance to TP (%)
- `status`: OPEN or CLOSED
- `exit_time_utc`, `exit_time_est`: Exit timestamps
- `exit_price`, `exit_reason`: TP_HIT or SL_HIT
- `pnl_usd`, `pnl_pct`: Profit/loss
- `time_in_trade_min`: Minutes from entry to exit

---

## Example Workflow

**Terminal 1 - Momentum Scanner:**
```bash
cd C:\Projects\CryptoAlgo
python monitor_momo_passes.py --interval-min 5 --output final_momo_passes.csv
```

**Terminal 2 - Trade Manager:**
```bash
cd C:\Projects\CryptoAlgo
python live_trade_manager.py --input final_momo_passes.csv --output live_trades.csv --interval-min 5
```

**Terminal 3 - Exit Monitor:**
```bash
cd C:\Projects\CryptoAlgo
python live_trade_monitor.py --trade-log live_trades.csv
```

**Terminal 4 - Performance Monitor:**
```bash
cd C:\Projects\CryptoAlgo
python analyze_live_trades.py --trade-log live_trades.csv --monitor
```

---

## Performance Metrics Tracked

### Win/Loss Stats
- Total trades, closed, and open
- Wins, losses, and win rate %

### P&L Analysis
- Total P&L %
- Average, median, best, and worst trades
- Profit factor (total wins / total losses)

### Time Analysis
- Average time in trade (minutes)
- Time distribution

### Exit Breakdown
- TP hits vs SL hits
- Percentage of trades exiting via TP

### Symbol Performance
- Performance stats per symbol
- Top and bottom performers

### Direction Breakdown
- Long vs short performance
- Direction-specific win rates

---

## Important Notes

### Paper Trading (Not Live)
Currently, this system:
- **Paper trades** (simulates entries at kline close)
- Monitors price action for SL/TP hits
- Logs everything for analysis

To go live with real orders, you would need to:
1. Add Binance order placement to `live_trade_manager.py` (using binance-futures-connector)
2. Track positions instead of simulating exits

### Data Requirements
- `final_momo_passes.csv` must have columns:
  - `timestamp_utc`, `timestamp_est`
  - `symbol`, `direction`
  - `entry_price` (optional; will fetch latest)

- Requires live Binance price data (via public API)

### Performance Expectations
Based on backtest:
- **TP Hit Rate**: 77.8%
- **Win Rate**: 91.3% (on closed trades)
- **Avg Time in Trade**: ~70 minutes
- **Average P&L**: -0.77% (actual market conditions vary)

---

## Troubleshooting

### "Input file not found"
Make sure you've run the momentum scanner first to generate `final_momo_passes.csv`.

### "Error fetching prices"
This usually means the Binance API is temporarily unavailable. The monitor will retry in 60 seconds.

### Trades not closing
Make sure `live_trade_monitor.py` is running. It checks for exits every minute by default.

### Reset / Start Fresh
To start over with a clean trade log:
```bash
del live_trades.csv
python live_trade_manager.py --input final_momo_passes.csv --output live_trades.csv
```

---

## Next Steps

1. **Run the system** for at least 10-20 trades to evaluate performance
2. **Compare results** to backtest expectations (77.8% TP hit, 91.3% win rate)
3. **Identify patterns** - which symbols/times perform best
4. **Adjust parameters** if needed (time windows, TP constraints, etc.)
5. **Go live** once confident in results (requires adding real order placement)

---

## Getting Help

If you encounter issues:
1. Check the Python output messages - they're descriptive
2. Verify CSV files exist: `final_momo_passes.csv` and `live_trades.csv`
3. Review live_trades.csv for format

Good luck! 🚀
