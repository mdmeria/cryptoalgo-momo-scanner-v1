"""
Analyze HOMEUSDT price action at specific time periods to understand grindy staircase criteria
"""
import requests
from datetime import datetime, timezone
import pandas as pd

def fetch_klines(symbol, interval, start_time, end_time):
    """Fetch klines from Binance"""
    url = "https://api.binance.com/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': int(start_time.timestamp() * 1000),
        'endTime': int(end_time.timestamp() * 1000),
        'limit': 1000
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    
    df = pd.DataFrame(data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    return df

# EST to UTC conversion (EST = UTC-5)
# March 7, 2026 - HOMEUSDT was flagged as passing at 23:34:55 UTC (18:34 EST)
# User is analyzing the 2-hour lookback period before that
from datetime import timedelta

# Specific times from user's observation (EST -> UTC)
period1_start = datetime(2026, 3, 7, 21, 10, tzinfo=timezone.utc)  # 16:10 EST - GOOD GRIND
period1_end = datetime(2026, 3, 7, 21, 38, tzinfo=timezone.utc)    # 16:38 EST

period2_start = datetime(2026, 3, 7, 21, 38, tzinfo=timezone.utc)  # 16:38 EST - LITTLE CHOP (acceptable)
period2_end = datetime(2026, 3, 7, 22, 0, tzinfo=timezone.utc)     # 17:00 EST

period3_start = datetime(2026, 3, 7, 22, 21, tzinfo=timezone.utc)  # 17:21 EST - RETRACEMENT (should FAIL)
period3_end = datetime(2026, 3, 7, 22, 41, tzinfo=timezone.utc)    # 17:41 EST

# Fetch broader context - from 15:00 EST to when scanner ran at 18:34 EST
fetch_start = datetime(2026, 3, 7, 20, 0, tzinfo=timezone.utc)   # 15:00 EST
fetch_end = datetime(2026, 3, 7, 23, 35, tzinfo=timezone.utc)     # 18:35 EST (just after scanner ran)

print("Fetching HOMEUSDT 1m klines...")
df = fetch_klines('HOMEUSDT', '1m', fetch_start, fetch_end)

print(f"\nTotal candles fetched: {len(df)}")
print(f"Time range: {df['open_time'].iloc[0]} to {df['open_time'].iloc[-1]}\n")

# Analyze Period 1: 16:10-16:38 EST (Good grind)
print("=" * 80)
print("PERIOD 1: 16:10-16:38 EST (21:10-21:38 UTC) - User calls this a GRIND")
print("=" * 80)
p1 = df[(df['open_time'] >= period1_start) & (df['open_time'] <= period1_end)]
if len(p1) > 0:
    print(f"Duration: {len(p1)} minutes")
    print(f"Price: {p1['close'].iloc[0]:.6f} -> {p1['close'].iloc[-1]:.6f}")
    print(f"Change: {((p1['close'].iloc[-1] / p1['close'].iloc[0]) - 1) * 100:.2f}%")
    print(f"High: {p1['high'].max():.6f}, Low: {p1['low'].min():.6f}")
    print(f"Range: {((p1['high'].max() / p1['low'].min()) - 1) * 100:.2f}%")
    
    # Count directional moves
    up_bars = 0
    down_bars = 0
    for i in range(len(p1)):
        if p1['close'].iloc[i] > p1['open'].iloc[i]:
            up_bars += 1
        elif p1['close'].iloc[i] < p1['open'].iloc[i]:
            down_bars += 1
    
    print(f"Up candles: {up_bars}, Down candles: {down_bars}, Doji: {len(p1) - up_bars - down_bars}")
    print(f"\nPrice action (first 10 and last 10 bars):")
    print(p1[['open_time', 'open', 'high', 'low', 'close', 'volume']].head(10).to_string(index=False))
    print("...")
    print(p1[['open_time', 'open', 'high', 'low', 'close', 'volume']].tail(10).to_string(index=False))
else:
    print("No data found for this period")

# Analyze Period 2: 16:38-17:00 EST (Little chop but acceptable)
print("\n" + "=" * 80)
print("PERIOD 2: 16:38-17:00 EST (21:38-22:00 UTC) - Little chop but ACCEPTABLE")
print("=" * 80)
p2 = df[(df['open_time'] >= period2_start) & (df['open_time'] <= period2_end)]
if len(p2) > 0:
    print(f"Duration: {len(p2)} minutes")
    print(f"Price: {p2['close'].iloc[0]:.6f} -> {p2['close'].iloc[-1]:.6f}")
    print(f"Change: {((p2['close'].iloc[-1] / p2['close'].iloc[0]) - 1) * 100:.2f}%")
    print(f"High: {p2['high'].max():.6f}, Low: {p2['low'].min():.6f}")
    print(f"Range: {((p2['high'].max() / p2['low'].min()) - 1) * 100:.2f}%")
    
    up_bars = sum(1 for i in range(len(p2)) if p2['close'].iloc[i] > p2['open'].iloc[i])
    down_bars = sum(1 for i in range(len(p2)) if p2['close'].iloc[i] < p2['open'].iloc[i])
    
    print(f"Up candles: {up_bars}, Down candles: {down_bars}, Doji: {len(p2) - up_bars - down_bars}")
    print(f"\nPrice action (all bars):")
    print(p2[['open_time', 'open', 'high', 'low', 'close', 'volume']].to_string(index=False))
else:
    print("No data found for this period")

# Analyze Period 3: 17:21-17:41 EST (Retracement - should FAIL)
print("\n" + "=" * 80)
print("PERIOD 3: 17:21-17:41 EST (22:21-22:41 UTC) - Price retracted - should FAIL")
print("=" * 80)
p3 = df[(df['open_time'] >= period3_start) & (df['open_time'] <= period3_end)]
if len(p3) > 0:
    print(f"Duration: {len(p3)} minutes")
    print(f"Price: {p3['close'].iloc[0]:.6f} -> {p3['close'].iloc[-1]:.6f}")
    print(f"Change: {((p3['close'].iloc[-1] / p3['close'].iloc[0]) - 1) * 100:.2f}%")
    print(f"High: {p3['high'].max():.6f}, Low: {p3['low'].min():.6f}")
    print(f"Range: {((p3['high'].max() / p3['low'].min()) - 1) * 100:.2f}%")
    
    up_bars = sum(1 for i in range(len(p3)) if p3['close'].iloc[i] > p3['open'].iloc[i])
    down_bars = sum(1 for i in range(len(p3)) if p3['close'].iloc[i] < p3['open'].iloc[i])
    
    print(f"Up candles: {up_bars}, Down candles: {down_bars}, Doji: {len(p3) - up_bars - down_bars}")
    print(f"\nPrice action (all bars):")
    print(p3[['open_time', 'open', 'high', 'low', 'close', 'volume']].to_string(index=False))
else:
    print("No data found for this period")

# Show broader context
print("\n" + "=" * 80)
print("SUMMARY - What makes a good grind vs should fail:")
print("=" * 80)
if len(p1) > 0 and len(p3) > 0:
    print(f"\nGood grind (Period 1):")
    print(f"  - Consistent direction with minimal retracement")
    print(f"  - Moved {((p1['close'].iloc[-1] / p1['close'].iloc[0]) - 1) * 100:.2f}% over {len(p1)} minutes")
    
    print(f"\nFail criteria (Period 3):")
    print(f"  - Noticeable retracement even if short duration")
    print(f"  - Moved {((p3['close'].iloc[-1] / p3['close'].iloc[0]) - 1) * 100:.2f}% over {len(p3)} minutes")
    print(f"  - This breaks the 'grindy staircase' pattern")
