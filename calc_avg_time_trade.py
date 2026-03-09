import pandas as pd

# Load the structural backtest results
df = pd.read_csv('momo_with_backtest_structural.csv')

# Filter for completed trades (either TP or SL hit in 4h window)
# Since the outcome column shows TP_HIT, SL_HIT, OPEN_4H, use that instead
completed = df[df['trade_outcome'].isin(['TP_HIT', 'SL_HIT'])].copy()

# Calculate time to exit for each completed trade
times_hours = []
for idx, row in completed.iterrows():
    if row['sl_hit_1h'] or row['tp_hit_1h']:
        times_hours.append(int(row['time_to_exit_1h_bars']))
    elif row['sl_hit_2h'] or row['tp_hit_2h']:
        times_hours.append(int(row['time_to_exit_2h_bars']))
    else:  # Must be 4h
        times_hours.append(int(row['time_to_exit_4h_bars']))

# Calculate statistics
avg_bars = sum(times_hours) / len(times_hours)
min_bars = min(times_hours)
max_bars = max(times_hours)

print("=" * 60)
print("AVERAGE TIME IN TRADE (Completed Trades)")
print("=" * 60)
print(f"\nTotal trades evaluated:     {len(df)}")
print(f"Completed trades (TP/SL):   {len(completed)}")
print(f"Open after 4h:              {len(df) - len(completed)}")
print(f"\nTIME TO EXIT:")
print(f"  Average: {avg_bars:.1f} hour-bars (~{avg_bars * 60:.0f} minutes)")
print(f"  Range:   {min_bars} to {max_bars} hour-bars")
print()

# Breakdown by exit window
exit_1h = len(df[(df['sl_hit_1h']) | (df['tp_hit_1h'])])
exit_2h = len(df[(df['sl_hit_2h']) | (df['tp_hit_2h'])])
exit_4h = len(df[(df['sl_hit_4h']) | (df['tp_hit_4h'])])

print("EXIT TIMING:")
print(f"  By 1h:  {exit_1h}/27 ({exit_1h/27*100:.1f}%)")
print(f"  By 2h:  {exit_2h}/27 ({exit_2h/27*100:.1f}%)")
print(f"  By 4h:  {exit_4h}/27 ({exit_4h/27*100:.1f}%)")
print()

# Breakdown by outcome
tp_hits = len(completed[completed['trade_outcome'] == 'TP_HIT'])
sl_hits = len(completed[completed['trade_outcome'] == 'SL_HIT'])

print("OUTCOME BREAKDOWN:")
print(f"  TP Hit: {tp_hits}/{len(completed)} ({tp_hits/len(completed)*100:.1f}%)")
print(f"  SL Hit: {sl_hits}/{len(completed)} ({sl_hits/len(completed)*100:.1f}%)")
print("=" * 60)
