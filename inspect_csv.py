#!/usr/bin/env python3
"""Investigate the CSV structure."""

import pandas as pd

df = pd.read_csv('binance_30min_7d_improved_full.csv')
print("Columns in CSV:")
for i, col in enumerate(df.columns):
    print(f"  {i}: {col}")
print()
print("Data types:")
print(df.dtypes)
print()
print("First row sample:")
print(df.iloc[0])
print()
print(f"Passed column (first 10): {df['passed'].head(10).tolist()}")
