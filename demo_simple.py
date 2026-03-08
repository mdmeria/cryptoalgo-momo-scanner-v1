"""
Simple demo to test market condition logic without heavy dependencies.
This shows the structure and flow without requiring pandas/ta libraries.
"""

class SimpleMarketState:
    """Simplified market states"""
    MOMENTUM_LONG = "momentum_long"
    MOMENTUM_SHORT = "momentum_short"
    MEAN_REVERSION = "mean_reversion"
    SPIKE = "spike"
    RANGE = "range"
    AVOID = "avoid"


def simple_market_evaluation():
    """
    Demonstrate the market condition evaluation flow.
    """
    print("="*60)
    print("Market Condition Evaluator - Demo (No Dependencies)")
    print("="*60)
    print()
    
    # Simulated market data (in real version, this comes from Binance/TradingView)
    print("📊 Step 1: Fetching Market Data...")
    print("   - Timeframe: 1 minute")
    print("   - Lookback: 100 candles")
    print("   - Source: Binance/TradingView (simulated)")
    print()
    
    # Simulate indicator calculations
    print("🔍 Step 2: Calculating Technical Indicators...")
    adx = 32.5  # Simulated ADX (> 25 = trending)
    atr_ratio = 1.8  # Simulated ATR ratio (> 1.5 = high volatility)
    volume_ratio = 1.2  # Simulated volume ratio
    ema_20 = 105.0
    ema_50 = 103.0
    close = 106.5
    
    print(f"   - ADX: {adx:.1f} (threshold: 25)")
    print(f"   - ATR Ratio: {atr_ratio:.2f} (threshold: 1.5)")
    print(f"   - Volume Ratio: {volume_ratio:.2f}")
    print(f"   - EMA-20: ${ema_20:.2f}")
    print(f"   - EMA-50: ${ema_50:.2f}")
    print(f"   - Close: ${close:.2f}")
    print()
    
    # Market classification logic
    print("🎯 Step 3: Classifying Market Condition...")
    
    # Check for spikes
    is_spike = volume_ratio > 2.0
    if is_spike:
        state = SimpleMarketState.SPIKE
        print(f"   ✅ SPIKE detected (volume {volume_ratio:.1f}x average)")
    
    # Check for trending
    elif adx > 25:
        if close > ema_20 > ema_50:
            state = SimpleMarketState.MOMENTUM_LONG
            print(f"   ✅ MOMENTUM_LONG (ADX={adx:.1f}, bullish EMA alignment)")
        elif close < ema_20 < ema_50:
            state = SimpleMarketState.MOMENTUM_SHORT
            print(f"   ✅ MOMENTUM_SHORT (ADX={adx:.1f}, bearish EMA alignment)")
        else:
            state = SimpleMarketState.RANGE
            print(f"   ⚠️ RANGE (ADX high but EMAs not aligned)")
    
    # Check for mean reversion
    elif atr_ratio > 1.2:
        state = SimpleMarketState.MEAN_REVERSION
        print(f"   ✅ MEAN_REVERSION (low ADX, high volatility)")
    
    else:
        state = SimpleMarketState.AVOID
        print(f"   ❌ AVOID (conditions unclear)")
    
    print()
    print(f"📌 Market State: {state.upper()}")
    print()
    
    return state


def simple_screener_demo():
    """
    Demonstrate the coin screener flow.
    """
    print("="*60)
    print("Orion Screener - Demo (No Dependencies)")
    print("="*60)
    print()
    
    print("🔎 Fetching top active coins from Orion...")
    print()
    
    # Simulated screener results
    top_coins = [
        {'symbol': 'BTCUSDT', 'volume_24h': 25_000_000_000, 'change_24h': 2.5},
        {'symbol': 'ETHUSDT', 'volume_24h': 15_000_000_000, 'change_24h': 3.1},
        {'symbol': 'SOLUSDT', 'volume_24h': 3_000_000_000, 'change_24h': 5.2},
        {'symbol': 'ADAUSDT', 'volume_24h': 2_500_000_000, 'change_24h': 1.8},
        {'symbol': 'AVAXUSDT', 'volume_24h': 2_000_000_000, 'change_24h': 4.5},
    ]
    
    print("Top 5 Coins by Volume:")
    for i, coin in enumerate(top_coins, 1):
        print(f"  {i}. {coin['symbol']:12} "
              f"Volume: ${coin['volume_24h']:>15,} "
              f"Change: {coin['change_24h']:>+6.2f}%")
    
    print()
    print(f"🎯 Selected for analysis: {top_coins[0]['symbol']}")
    print()
    
    return top_coins[0]['symbol']


def trading_decision_demo(market_state, symbol):
    """
    Demonstrate the trading decision flow.
    """
    print("="*60)
    print(f"Trading Decision for {symbol}")
    print("="*60)
    print()
    
    tradeable_states = [
        SimpleMarketState.MOMENTUM_LONG,
        SimpleMarketState.MOMENTUM_SHORT,
        SimpleMarketState.MEAN_REVERSION,
        SimpleMarketState.SPIKE
    ]
    
    if market_state in tradeable_states:
        print(f"✅ Market is TRADEABLE ({market_state})")
        print()
        print("Next steps:")
        print("  1. Apply specific entry rules based on market state")
        print("  2. Calculate position size (1R risk)")
        print("  3. Set stop loss and take profit levels")
        print("  4. Send order to BitUnix exchange")
        print("  5. Monitor position and apply early cut rules")
    else:
        print(f"❌ Market is NOT TRADEABLE ({market_state})")
        print()
        print("Action: Standing aside, waiting for better conditions")
    
    print()


def main():
    """Run the complete demo."""
    print()
    print("🤖 CRYPTO TRADING BOT - DEMO MODE")
    print()
    
    # Step 1: Get top coin from screener
    symbol = simple_screener_demo()
    
    # Step 2: Evaluate market conditions
    market_state = simple_market_evaluation()
    
    # Step 3: Make trading decision
    trading_decision_demo(market_state, symbol)
    
    print("="*60)
    print("Demo Complete!")
    print("="*60)
    print()
    print("📝 To run the full version:")
    print("   1. Install dependencies: pip install -r requirements.txt")
    print("   2. Configure .env file with API keys")
    print("   3. Run: python main.py")
    print()


if __name__ == "__main__":
    main()
