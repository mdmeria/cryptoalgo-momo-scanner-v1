"""Simple test to fetch HOMEUSDT recent data"""
import requests
from datetime import datetime, timezone, timedelta

# Try fetching last 500 bars of 1m data
url = "https://api.binance.com/api/v3/klines"
params = {
    'symbol': 'HOMEUSDT',
    'interval': '1m',
    'limit': 500
}

print("Fetching last 500 1m candles for HOMEUSDT...")
resp = requests.get(url, params=params, timeout=10)
print(f"Status: {resp.status_code}")

if resp.status_code == 200:
    data = resp.json()
    print(f"Received {len(data)} candles")
    
    # Show first and last candle times
    if data:
        first_time = datetime.fromtimestamp(data[0][0]/1000, tz=timezone.utc)
        last_time = datetime.fromtimestamp(data[-1][0]/1000, tz=timezone.utc)
        print(f"First candle: {first_time} UTC")
        print(f"Last candle: {last_time} UTC")
        
        # Convert to EST (UTC-5)
        first_est = first_time - timedelta(hours=5)
        last_est = last_time - timedelta(hours=5)
        print(f"First candle EST: {first_est}")
        print(f"Last candle EST: {last_est}")
        
        # Show some sample data
        print("\nLast 10 candles:")
        for i in range(max(0, len(data)-10), len(data)):
            candle = data[i]
            t = datetime.fromtimestamp(candle[0]/1000, tz=timezone.utc)
            t_est = t - timedelta(hours=5)
            print(f"{t_est.strftime('%H:%M EST')}: O={float(candle[1]):.6f} H={float(candle[2]):.6f} L={float(candle[3]):.6f} C={float(candle[4]):.6f}")
else:
    print(f"Error: {resp.text}")
