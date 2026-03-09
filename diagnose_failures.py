#!/usr/bin/env python3
"""Analyze what's preventing setups from passing."""

import pandas as pd

df = pd.read_csv('binance_30min_7d_improved_full.csv')

# Examine the individual check columns
print("=== CHECK FAILURE ANALYSIS ===\n")
checks = [
    'slow_grind_approach',
    'left_side_staircase', 
    'volume_not_decreasing',
    'not_choppy',
    'balanced_momo_2h_grind_on',
    'balanced_momo_2h_grind_off',
    'day_change_ok',
    'vwap_side_ok'
]

for check in checks:
    if check in df.columns:
        pass_count = df[check].sum()
        pct = pass_count / len(df) * 100
        print(f"{check:30s}: {pass_count:6d} passes ({pct:6.2f}%)")

print("\n=== EFFICIENCY DISTRIBUTION (Grind ON passes) ===")
grind_on = df[df['balanced_momo_2h_grind_on']]
if len(grind_on) > 0:
    print(f"Min efficiency: {grind_on['efficiency_2h'].min():.4f}")
    print(f"Max efficiency: {grind_on['efficiency_2h'].max():.4f}")
    print(f"Mean efficiency: {grind_on['efficiency_2h'].mean():.4f}")
else:
    print("No grind_on passes to analyze")

# Check regime metrics if available
print("\n=== CHECKING FOR REGIME METRICS ===")
import os
if os.path.exists('binance_30min_7d_improved_samples.csv'):
    samples = pd.read_csv('binance_30min_7d_improved_samples.csv')
    print(f"Samples CSV has {len(samples.columns)} columns")
    print("Sample columns:", list(samples.columns)[:20])
