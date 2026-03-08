"""
Main entry point for the crypto trading bot.
Demonstrates market condition evaluation and coin screening.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import time
import logging
import requests

from config import Config
from market_condition import MarketConditionEvaluator, MarketState
from binance_screener import BinanceScreener, BinanceSpotScreener, BinanceUSSpotScreener, BinanceVisionSpotScreener
from orion_screener import OrionTerminalScreener
from momentum_quality import evaluate_momentum_setup

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_klines_binance(symbol: str, interval: str = '1m', limit: int = 100) -> pd.DataFrame:
    """
    Fetch real OHLCV data from Binance Vision API.
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        interval: Kline interval (1m, 5m, 1h, etc.)
        limit: Number of candles to fetch
    
    Returns:
        DataFrame with OHLCV data
    """
    try:
        url = 'https://data-api.binance.vision/api/v3/klines'
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        klines = response.json()
        
        # Parse klines into DataFrame
        # Kline format: [time, open, high, low, close, volume, close_time, quote_volume, trades, ...]
        data = []
        for k in klines:
            data.append({
                'timestamp': pd.to_datetime(k[0], unit='ms'),
                'open': float(k[1]),
                'high': float(k[2]),
                'low': float(k[3]),
                'close': float(k[4]),
                'volume': float(k[5])
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        logger.debug(f"Fetched {len(df)} klines for {symbol}")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching klines for {symbol}: {e}")
        # Return sample data as fallback
        return generate_sample_ohlcv(periods=limit)


def generate_sample_ohlcv(periods: int = 100) -> pd.DataFrame:
    """
    Generate sample OHLCV data for testing.
    Replace this with actual data from Binance/TradingView.
    """
    dates = pd.date_range(start=datetime.now(), periods=periods, freq='1min')
    
    # Generate more realistic price data with trend
    base_price = 100
    trend = np.linspace(0, 5, periods)  # Upward trend
    noise = np.random.randn(periods) * 0.5
    close = base_price + trend + noise.cumsum()
    
    df = pd.DataFrame({
        'open': close + np.random.randn(periods) * 0.2,
        'high': close + abs(np.random.randn(periods) * 0.5),
        'low': close - abs(np.random.randn(periods) * 0.5),
        'close': close,
        'volume': np.random.randint(1000, 5000, periods)
    }, index=dates)
    
    # Ensure high is highest and low is lowest
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    return df


def main():
    """Main trading loop demonstration."""
    
    logger.info("="*60)
    logger.info("Crypto Trading Bot - Market Condition & Screener Demo")
    logger.info("="*60)
    
    # Initialize components
    logger.info("\n1. Initializing Market Condition Evaluator...")
    evaluator = MarketConditionEvaluator(
        adx_threshold=Config.ADX_TREND_THRESHOLD,
        atr_threshold=Config.ATR_VOLATILITY_THRESHOLD,
        volume_spike_threshold=Config.VOLUME_SPIKE_THRESHOLD
    )
    logger.info("✓ Market evaluator initialized")
    
    logger.info("\n2. Initializing Screener...")
    if Config.SCREENER_SOURCE.lower() == 'orion':
        screener = OrionTerminalScreener(base_url=Config.ORION_API_URL)
        logger.info("✓ Using Orion screener for coin discovery")
    else:
        # Use Binance for real-time screening (no API key needed)
        if Config.BINANCE_MODE == 'binance_vision':
            screener = BinanceVisionSpotScreener()
            logger.info("✓ Using Binance Vision-first Spot screener")
        elif Config.BINANCE_MODE == 'binance_us':
            screener = BinanceUSSpotScreener()
            logger.info("✓ Using Binance.US-first Spot screener")
        elif Config.BINANCE_MODE == 'spot':
            screener = BinanceSpotScreener()
            logger.info("✓ Using Binance Spot screener")
        else:
            screener = BinanceScreener()
            logger.info("✓ Using Binance Futures screener")
    
    # Main evaluation loop
    logger.info("\n" + "="*60)
    logger.info("Starting Market Analysis Loop")
    logger.info("="*60)
    
    for iteration in range(3):  # Run 3 iterations for demo
        logger.info(f"\n--- Iteration {iteration + 1} ---")
        
        # Step 1: Get top coins from screener
        logger.info(
            f"\nFetching top active {Config.SCREENER_QUOTE_CURRENCY} coins "
            f"(last {Config.SCREENER_LOOKBACK_MINUTES}min)..."
        )
        top_coins = screener.get_top_coins_by_recent_activity(
            top_n=Config.TOP_N_COINS,
            min_volume_5m=Config.MIN_VOLUME_5M,
            quote_currency=Config.SCREENER_QUOTE_CURRENCY,
            lookback_minutes=Config.SCREENER_LOOKBACK_MINUTES
        )
        
        if not top_coins:
            logger.warning("No coins returned from screener. Skipping iteration.")
            time.sleep(5)
            continue
        
        logger.info(f"Top {len(top_coins)} coins by recent activity:")
        for i, coin in enumerate(top_coins[:5], 1):  # Show top 5
            logger.info(f"  {i}. {coin['symbol']}: "
                       f"Volume=${coin['volume_usd']:,.0f}, "
                       f"Trades={coin['trades']}, "
                       f"Volatility={coin['volatility']:.2f}%")
        
        # Step 2: Evaluate market condition for top coin (using REAL market data)
        top_symbol = top_coins[0]['symbol']
        logger.info(f"\nEvaluating market conditions for {top_symbol}...")
        
        # Fetch real 1-minute klines from Binance
        df = fetch_klines_binance(
            symbol=top_symbol,
            interval='1m',
            limit=Config.LOOKBACK_PERIODS
        )
        
        # Classify market state
        market_state = evaluator.evaluate(df)
        metrics = evaluator.get_market_metrics(df)
        
        logger.info(f"\n{'🔥' if market_state != MarketState.AVOID else '⚠️'} "
                   f"Market State: {market_state.value.upper()}")
        logger.info("Market Metrics:")
        logger.info(f"  ADX: {metrics['adx']:.2f}")
        logger.info(f"  ATR Ratio: {metrics['atr_ratio']:.2f}")
        logger.info(f"  Volume Ratio: {metrics['volume_ratio']:.2f}")
        logger.info(f"  ROC: {metrics['roc']:.2f}%")
        logger.info(f"  Close: ${metrics['close']:.2f}")
        
        # Step 3: Trading decision logic
        logger.info("\nTrading Decision:")
        if market_state in [MarketState.MOMENTUM_LONG, MarketState.MOMENTUM_SHORT,
                           MarketState.MEAN_REVERSION, MarketState.SPIKE_UP,
                           MarketState.SPIKE_DOWN]:
            logger.info(f"✅ Market condition is TRADEABLE ({market_state.value})")

            if market_state in [MarketState.MOMENTUM_LONG, MarketState.MOMENTUM_SHORT]:
                momentum_direction = "long" if market_state == MarketState.MOMENTUM_LONG else "short"
                quality = evaluate_momentum_setup(df, direction=momentum_direction)
                logger.info(
                    "   Momentum quality: "
                    f"score={quality.score:.2f} "
                    f"(approach={quality.checks['slow_grind_approach']}, "
                    f"staircase={quality.checks['left_side_staircase']}, "
                    f"volume={quality.checks['volume_not_decreasing']}, "
                    f"not_choppy={quality.checks['not_choppy']})"
                )

                if quality.passed:
                    logger.info(f"   → MOMENTUM quality PASSED, entry search allowed on {top_symbol}")
                else:
                    logger.info(f"   → MOMENTUM quality FAILED, skipping trade on {top_symbol}")
            else:
                logger.info(f"   → Would look for entry signals on {top_symbol}")
            
            # Here you would:
            # - Apply your specific entry rules
            # - Calculate position size (1R risk)
            # - Set stop loss and take profit
            # - Send order to BitUnix
            
        else:
            logger.info(f"❌ Market condition is NOT TRADEABLE ({market_state.value})")
            logger.info("   → Standing aside, waiting for better conditions")
        
        # Wait before next iteration (in real bot, this would be continuous)
        if iteration < 2:
            logger.info(f"\nWaiting {Config.SCREENER_REFRESH_SECONDS} seconds before next check...")
            time.sleep(Config.SCREENER_REFRESH_SECONDS)
    
    logger.info("\n" + "="*60)
    logger.info("Demo completed!")
    logger.info("="*60)
    logger.info("\nNext steps:")
    logger.info("1. Configure .env file with real API keys")
    logger.info("2. Implement Binance/TradingView data fetching")
    logger.info("3. Add BitUnix order execution module")
    logger.info("4. Define specific entry/exit rules")
    logger.info("5. Implement position management and early cut logic")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nBot stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
