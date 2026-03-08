"""
Test screener with different volume thresholds to find optimal balance.
"""
from binance_screener import BinanceVisionSpotScreener

screener = BinanceVisionSpotScreener()

print("Testing different volume thresholds:")
print("=" * 70)

thresholds = [50000, 100000, 200000, 500000]

for threshold in thresholds:
    coins = screener.get_top_coins_by_recent_activity(
        top_n=15,
        min_volume_5m=threshold,
        quote_currency='USDT',
        lookback_minutes=5
    )
    
    print(f"\nVolume threshold: ${threshold:,}")
    print(f"Found: {len(coins)} coins")
    if coins:
        print("Top 5:")
        for i, c in enumerate(coins[:5], 1):
            print(f"  {i}. {c['symbol']:<12} Volume: ${c['volume_usd']:>10,.0f}  Trades: {c['trades']:>5}")
