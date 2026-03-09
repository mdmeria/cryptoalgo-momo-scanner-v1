#!/usr/bin/env python3
"""Analyze the improved 30-minute scan results."""

import pandas as pd

df = pd.read_csv('binance_30min_7d_improved_full.csv')
print('=== IMPROVED 30-MINUTE SCAN RESULTS ===')
print(f'Total rows: {len(df)}')
print(f'Unique symbols: {df["symbol"].nunique()}')
print(f'Date range: {df["timestamp_utc"].min()} to {df["timestamp_utc"].max()}')
print()

# Count passes with grind ON and OFF
grind_on_passes = df['overall_pass_grind_on'].sum()
grind_off_passes = df['overall_pass_grind_off'].sum()
print(f'Setups passing ALL checks (grind ON):  {grind_on_passes} ({grind_on_passes/len(df)*100:.2f}%)')
print(f'Setups passing ALL checks (grind OFF): {grind_off_passes} ({grind_off_passes/len(df)*100:.2f}%)')
print()

# Show top symbols by pass count (grind ON)
if grind_on_passes > 0:
    top_symbols = df[df['overall_pass_grind_on']]['tv_symbol'].value_counts().head(10)
    print('Top 10 symbols by pass count (grind ON):')
    for sym, count in top_symbols.items():
        print(f'  {sym}: {count} passes')
else:
    print('No passing setups in this scan.')

print()
print('=== COMPARISON WITH HOURLY SCAN ===')
print('Original 7-day hourly (grind ON): 78 passes / 25,060 rows = 0.31%')
print(f'Improved 30-min (grind ON):       {grind_on_passes} passes / {len(df)} rows = {grind_on_passes/len(df)*100:.2f}%')
print()
print('Original 7-day hourly (grind OFF): 124 passes / 25,060 rows = 0.49%')
print(f'Improved 30-min (grind OFF):       {grind_off_passes} passes / {len(df)} rows = {grind_off_passes/len(df)*100:.2f}%')

