#!/usr/bin/env python3
"""Debug what specific balanced_momo_2h subchecks are failing."""

import pandas as pd

# Since we don't have detailed subcheck columns in the output,
# let's examine the efficiency and directional move ranges for balanced_momo_2h passes

df = pd.read_csv('binance_30min_7d_improved_v2_full.csv')

# Look at the 30 rows that passed balanced_momo_2h_grind_on
balanced_passes = df[df['balanced_momo_2h_grind_on']]

print("=== BALANCED_MOMO_2H_GRIND_ON PASSES (30 total) ===\n")
if len(balanced_passes) > 0:
    print(f"Efficiency 2h range: {balanced_passes['efficiency_2h'].min():.4f} - {balanced_passes['efficiency_2h'].max():.4f}")
    print(f"Dir move 2h%: {balanced_passes['dir_move_2h_pct'].min():.2f}% - {balanced_passes['dir_move_2h_pct'].max():.2f}%")
    print()
    print("Sample passing rows:")
    print(balanced_passes[['tv_symbol', 'direction', 'efficiency_2h', 'dir_move_2h_pct']].head(10))
else:
    print("No balanced_momo_2h_grind_on passes to analyze")

print("\n" + "="*60 + "\n")

# Now look at what ALMOST passed balanced_momo_2h_grind_on
# These are rows that passed everything except balanced_momo_2h_grind_on
almost_passed = df[
    df['slow_grind_approach'] &
    df['volume_not_decreasing'] & 
    df['not_choppy'] &
    ~df['balanced_momo_2h_grind_on']  # Failed balanced check
]

print(f"=== ALMOST PASSED (failed balanced_momo_2h only): {len(almost_passed)} ===\n")
if len(almost_passed) > 0:
    print(f"Efficiency 2h range: {almost_passed['efficiency_2h'].min():.4f} - {almost_passed['efficiency_2h'].max():.4f}")
    print(f"Efficiency 2h mean: {almost_passed['efficiency_2h'].mean():.4f}")
    print(f"Dir move 2h%: {almost_passed['dir_move_2h_pct'].min():.2f}% - {almost_passed['dir_move_2h_pct'].max():.2f}%")
    print(f"Dir move 2h% mean: {almost_passed['dir_move_2h_pct'].mean():.2f}%")
    
    # Distribution of efficiency
    print("\nEfficiency distribution:")
    print(f"  <0.18: {(almost_passed['efficiency_2h'] < 0.18).sum()}")
    print(f"  0.18-0.75: {((almost_passed['efficiency_2h'] >= 0.18) & (almost_passed['efficiency_2h'] <= 0.75)).sum()}")
    print(f"  >0.75: {(almost_passed['efficiency_2h'] > 0.75).sum()}")
