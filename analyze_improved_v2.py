#!/usr/bin/env python3
"""Analyze the improved v2 scan results."""

import pandas as pd

df = pd.read_csv('binance_30min_7d_improved_v2_full.csv')
print('=== IMPROVED V2 30-MINUTE SCAN RESULTS ===')
print(f'Total rows: {len(df)}')
print(f'Unique symbols: {df["symbol"].nunique()}')
print(f'Date range: {df["timestamp_utc"].min()} to {df["timestamp_utc"].max()}')
print()

# Count passes with grind ON and OFF
grind_on_passes = df['overall_pass_grind_on'].sum()
grind_off_passes = df['overall_pass_grind_off'].sum()
print(f'Setups passing ALL checks (grind ON):  {grind_on_passes} ({grind_on_passes/len(df)*100:.3f}%)')
print(f'Setups passing ALL checks (grind OFF): {grind_off_passes} ({grind_off_passes/len(df)*100:.3f}%)')
print()

# Check failure analysis
print("=== CHECK FAILURE ANALYSIS ===\n")
checks = [
    'slow_grind_approach',
    'left_side_staircase', 
    'volume_not_decreasing',
    'not_choppy',
    'balanced_momo_2h_grind_on',
    'balanced_momo_2h_grind_off',
]

for check in checks:
    if check in df.columns:
        pass_count = df[check].sum()
        pct = pass_count / len(df) * 100
        print(f"{check:30s}: {pass_count:6d} passes ({pct:6.2f}%)")

# Show top symbols by pass count (grind ON)
print()
if grind_on_passes > 0:
    top_symbols = df[df['overall_pass_grind_on']]['tv_symbol'].value_counts().head(10)
    print('Top 10 symbols by pass count (grind ON):')
    for sym, count in top_symbols.items():
        print(f'  {sym}: {count} passes')

print()
print('=== COMPARISON ===')
print('Original 7-day hourly (grind ON):   78 passes / 25,060 rows = 0.311%')
print(f'Improved V1 30-min (grind ON):       0 passes / 48,900 rows = 0.000% (TOO STRICT)')
print(f'Improved V2 30-min (grind ON):      {grind_on_passes:2d} passes / {len(df)} rows = {grind_on_passes/len(df)*100:.3f}%')
