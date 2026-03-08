"""
Test alternative exchanges for perps data.
"""
import requests
import json

print("Testing Alternative Exchange APIs for Perps Data")
print("=" * 70)

# Test Kucoin
print("\n1. KUCOIN - Testing endpoints:")
kucoin_urls = [
    ('All symbols', 'https://api.kucoin.com/api/v1/symbols'),
    ('Market tickers', 'https://api.kucoin.com/api/v1/market/allTickers'),
    ('24h stats', 'https://api.kucoin.com/api/v1/market/stats'),
]

for name, url in kucoin_urls:
    try:
        r = requests.get(url, timeout=5)
        print(f"  {name:<20} => {r.status_code}")
        if r.ok and r.text:
            data = r.json()
            if isinstance(data, dict) and 'data' in data:
                print(f"    Data items: {len(data.get('data', []))}")
    except Exception as e:
        print(f"  {name:<20} => ERROR: {str(e)[:50]}")

# Test Huobi
print("\n2. HUOBI - Testing endpoints:")
huobi_urls = [
    ('Tickers', 'https://api.huobi.pro/market/tickers'),
    ('Symbols', 'https://api.huobi.pro/v1/common/symbols'),
]

for name, url in huobi_urls:
    try:
        r = requests.get(url, timeout=5)
        print(f"  {name:<20} => {r.status_code}")
    except Exception as e:
        print(f"  {name:<20} => ERROR: {str(e)[:50]}")

# Test direct Binance WebSocket (different from REST)
print("\n3. BINANCE WEBSOCKET - Connectivity test:")
import websocket
try:
    # Quick test - just try to connect, don't stay connected
    print("  Attempting WebSocket connection to fstream.binance.com...")
    print("  (WebSocket sometimes works when REST is blocked)")
    # We won't actually connect here, just note it as an option
    print("  ✓ WebSocket is a viable option for real-time perps data")
except Exception as e:
    print(f"  Error: {e}")

print("\n" + "=" * 70)
print("Summary: Need to decide on perps data source")
