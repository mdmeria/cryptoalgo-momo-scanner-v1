import pandas as pd

# Load the structural backtest results
df = pd.read_csv('momo_with_backtest_structural.csv')

print("First few rows of key columns:")
print(df[['timestamp_est', 'symbol', 'direction', 'sl_hit_4h', 'tp_hit_4h', 'time_to_exit_1h_bars', 'time_to_exit_2h_bars', 'time_to_exit_4h_bars', 'trade_outcome']].head(10))
print()

# Check unique values in outcome column
print(f"Unique trade outcomes: {df['trade_outcome'].unique()}")
print()

# Check how many completed trades
completed = df[(df['sl_hit_4h'] == 'True') | (df['tp_hit_4h'] == 'True')]
print(f"Completed trades (using string 'True'): {len(completed)}")

completed2 = df[(df['sl_hit_4h'] == True) | (df['tp_hit_4h'] == True)]
print(f"Completed trades (using boolean True): {len(completed2)}")
print()

# Show sample of completed trades
if len(completed) > 0:
    print("Sample completed trades:")
    print(completed[['timestamp_est', 'symbol', 'sl_hit_4h', 'tp_hit_4h', 'trade_outcome']].head(5))
