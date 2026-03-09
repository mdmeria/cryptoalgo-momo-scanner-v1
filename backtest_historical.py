#!/usr/bin/env python3
"""
Backtest historical momentum setups by re-evaluating them at current time.
Allows rapid testing of filter changes against symbols from CSV log.

Usage:
  python backtest_historical.py --from-csv momo_passes_day1_est_log.csv --limit 20
  python backtest_historical.py --from-csv momo_passes_day1_est_log.csv --days 7
"""
import argparse
import csv
from datetime import datetime, timezone, timedelta
import pytz
import pandas as pd
import requests
from momentum_quality import evaluate_momentum_setup

# Eastern timezone for consistency
EST fetch_klines(symbol: str, interval: str = "1m", limit: int = 160) -> pd.DataFrame | None:
    """Fetch Binance Vision klines for a symbol."""
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
    # Fetch current datadf = fetch_klines(symbol, interval="1m", limit=160)
    if df is None or len(df) < 120:
        print("❌ FAILED TO FETCH DATA")
        return False, None
    
    result = evaluate_momentum_setup(df, direction=direction            for k in raw
            ]
        )
        return df.set_index("timestamp")
    except Exception:
        return None

def = pytz.timezone('America/New_York')

def parse_timestamp(ts_str: str) -> datetime:
    """Parse ISO timestamp with timezone."""
    # Handle format: 2026-03-08T12:21:36-05:00
    dt = datetime.fromisoformat(ts_str)
    if dt.tzinfo is None:
        dt = EST.localize(dt)
    return dt

def _check_symbol(passed: bool) -> str:
    """Return visual symbol for check status."""
    return '✅' if passed else '❌'

def evaluate_current(symbol: str, direction: str, original_timestamp: datetime, show_metrics: bool = False):
    """Re-evaluate a symbol at current time (not historical)."""
    print(f"\n{'='*80}")
    print(f"Symbol: {symbol} | Direction: {direction.upper()} | Originally: {original_timestamp.strftime('%Y-%m-%d %H:%M %Z')}")
    print(f"{'='*80}")
    
    # Fetch current data
    df = fetch_klines(symbol, interval="1m", limit=160)
    if df is None or len(df) < 120:
        print("❌ FAILED TO FETCH DATA")
        return False, None
    
    result = evaluate_momentum_setup(df, direction=direction)
    
    # Show pass/fail
    core_passed = all([
        result.checks.get('on_approach'),
        result.checks.get('staircase_structure'),
        result.checks.get('volume_normal'),
        result.checks.get('not_choppy'),
        result.checks.get('balanced_momo_2h'),
        result.checks.get('parallel_30smma_2h'),
        result.checks.get('smma_spread_increasing_2h'),
        result.checks.get('pre_entry_directional_30m'),
    ])
    
    extended_passed = all([
        result.checks.get('day_change_ok'),
        result.checks.get('vwap_side_ok'),
        result.checks.get('first_2h_prev_vwap'),
        result.checks.get('entry_not_crossed_6h'),
    ])
    
    overall_pass = core_passed and extended_passed
    
    print(f"\n🎯 OVERALL: {'✅ PASS' if overall_pass else '❌ FAIL'} (quality={result.quality_tier})")
    print(f"\n📊 CORE CHECKS (must ALL pass):")
    print(f"  on_approach:             {_check_symbol(result.checks.get('on_approach'))}")
    print(f"  staircase_structure:     {_check_symbol(result.checks.get('staircase_structure'))}")
    print(f"  volume_normal:           {_check_symbol(result.checks.get('volume_normal'))}")
    print(f"  not_choppy:              {_check_symbol(result.checks.get('not_choppy'))}")
    print(f"  balanced_momo_2h:        {_check_symbol(result.checks.get('balanced_momo_2h'))}")
    print(f"  parallel_30smma_2h:      {_check_symbol(result.checks.get('parallel_30smma_2h'))}")
    print(f"  smma_spread_increasing:  {_check_symbol(result.checks.get('smma_spread_increasing_2h'))}")
    print(f"  pre_entry_directional_30m: {_check_symbol(result.checks.get('pre_entry_directional_30m'))} 🆕")
    
    print(f"\n📊 EXTENDED CHECKS:")
    print(f"  day_change_ok:           {_check_symbol(result.checks.get('day_change_ok'))}")
    print(f"  vwap_side_ok:            {_check_symbol(result.checks.get('vwap_side_ok'))}")
    print(f"  first_2h_prev_vwap:      {_check_symbol(result.checks.get('first_2h_prev_vwap'))}")
    print(f"  entry_not_crossed_6h:    {_check_symbol(result.checks.get('entry_not_crossed_6h'))}")
    
    if show_metrics:
        print(f"\n📈 KEY METRICS:")
        print(f"  Day change: {result.metrics.get('day_change_pct', 0):.2f}%")
        print(f"  Retracements (2h): {result.metrics.get('retracements_found', 0)}")
        print(f"  Entry crosses (6h): {result.metrics.get('entry_cross_count_6h', 0)}")
        
        # Pre-entry 30m metrics
        print(f"\n🔍 PRE-ENTRY 30m WINDOW:")
        print(f"  Directional move: {result.metrics.get('pre_entry_move_30m_pct', 0):.2f}% (need ≥0.40%)")
        print(f"  Bar ratio: {result.metrics.get('pre_entry_dir_bar_ratio_30m', 0):.2f} (need ≥0.40)")
        print(f"  Efficiency: {result.metrics.get('pre_entry_efficiency_30m', 0):.3f} (need ≥0.20)")
        
        # Grind quality (2h)
        print(f"\n📊 GRIND QUALITY (2h):")
        print(f"  8m grind windows: {result.metrics.get('grind_windows_count', 0)}")
        print(f"  Avg impulse: {result.metrics.get('avg_grind_impulse_pct', 0):.2f}%")
        print(f"  Impulse std: {result.metrics.get('grind_impulse_std', 0):.4f}")
        print(f"  Bar participation: {result.metrics.get('grind_bar_participation', 0):.2f}")
    
    return overall_pass, result

def backtest_from_csv(csv_path: str, limit: int = None, days: int = 7, show_metrics: bool = False):
    """Re-evaluate entries from a CSV log (only last N days for chart compatibility)."""
    print(f"\n🔄 Backtesting entries from: {csv_path}")
    print(f"{'='*80}\n")
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Filter to last N days (default 7 for 10k bar chart compatibility)
    now = datetime.now(EST)
    cutoff = now - timedelta(days=days)
    rows = [r for r in rows if parse_timestamp(r['run_timestamp_est']) >= cutoff]
    
    # Filter to entries WITHOUT pre_entry_30m check (these were before we added the filter)
    old_entries = [r for r in rows if 'pre_entry_30m=' not in r.get('checks', '')]
    new_entries = [r for r in rows if 'pre_entry_30m=' in r.get('checks', '')]
    
    print(f"Found {len(rows)} entries in last {days} days")
    print(f"  ├─ {len(old_entries)} before pre_entry_30m filter was added")
    print(f"  └─ {len(new_entries)} after pre_entry_30m filter was added")
    print(f"\nRe-evaluating {len(old_entries)} OLD entries with NEW filter...\n")
    
    if limit and len(old_entries) > limit:
        old_entries = old_entries[-limit:]
    
    results = []
    for i, row in enumerate(old_entries, 1):
        symbol = row['symbol']
        direction = row['direction']
        timestamp = parse_timestamp(row['run_timestamp_est'])
        
        print(f"\n[{i}/{len(old_entries)}] ", end='')
        passed, result = evaluate_current(symbol, direction, timestamp, show_metrics)
        
        
        results.append({
            'timestamp': row['run_timestamp_est'],
            'symbol': symbol,
            'direction': direction,
            'old_pass': True,  # It was in the CSV, so it passed before
            'new_pass': passed,
            'quality': result.quality_tier,
            'pre_entry_check': result.checks.get('pre_entry_directional_30m'),
        })
    
    # Summary
    print(f"\n\n{'='*80}")
    print(f"📊 BACKTEST SUMMARY")
    print(f"{'='*80}")
    
    still_passing = sum(1 for r in results if r['new_pass'])
    now_filtered = sum(1 for r in results if not r['new_pass'])
    pre_entry_fails = sum(1 for r in results if not r['pre_entry_check'])
    
    print(f"\nTotal tested: {len(results)}")
    if len(results) > 0:
        print(f"Still passing: {still_passing} ({still_passing/len(results)*100:.1f}%)")
        print(f"Now filtered out: {now_filtered} ({now_filtered/len(results)*100:.1f}%)")
        print(f"  └─ Failed pre_entry_directional_30m: {pre_entry_fails}")
    
        # Show which symbols are now filtered
        if now_filtered > 0:
            print(f"\n🚫 NEWLY FILTERED SYMBOLS:")
            for r in results:
                if not r['new_pass']:
                    reason = "pre_entry_30m" if not r['pre_entry_check'] else "other"
                    print(f"  {r['timestamp']} | {r['symbol']:12} | {r['direction']:5} | Reason: {reason}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Backtest momentum scanner on CSV log entries')
    parser.add_argument('--from-csv', required=True, help='CSV log file to backtest')
    parser.add_argument('--limit', type=int, help='Limit number of entries to test (most recent N)')
    parser.add_argument('--days', type=int, default=7, help='Only test entries from last N days (default: 7 for 10k bar charts)')
    parser.add_argument('--show-metrics', action='store_true', help='Show detailed metrics for each evaluation')
    
    args = parser.parse_args()
    
    backtest_from_csv(args.from_csv, args.limit, args.days, args.show_metrics)

if __name__ == '__main__':
    main()


if __name__ == '__main__':
    main()
