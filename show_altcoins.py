"""
Display screener results with $100k volume threshold.
"""
from binance_screener import BinanceVisionSpotScreener

screener = BinanceVisionSpotScreener()
coins = screener.get_top_coins_by_recent_activity(
    top_n=20,
    min_volume_5m=100000,
    quote_currency='USDT',
    lookback_minutes=5
)

print(f'\nFound {len(coins)} active altcoins (last 5 min, $100k+):\n')
print(f'{"#":<3} {"Symbol":<12} {"Volume":<18} {"Trades":<8} {"Volatility"}')
print('-' * 65)

for i, c in enumerate(coins, 1):
    print(f'{i:<3} {c["symbol"]:<12} ${c["volume_usd"]:>15,.0f}  {c["trades"]:>6}  {c["volatility"]:>7.2f}%')

print(f'\nNow using $100k threshold by default in config.py')
print(f'You can adjust MIN_VOLUME_5M in .env for different discovery levels')
