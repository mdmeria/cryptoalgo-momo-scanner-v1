#!/usr/bin/env python3
"""
OKX Perpetuals Trading Bot - Main Loop
Screens active OKX perpetuals and evaluates market conditions
"""

import logging
import time
import pandas as pd
from okx_screener_fast import OKXPerpScreenerFast
from market_condition import MarketConditionEvaluator, MarketState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main bot loop - screen OKX perps → analyze conditions → generate signals."""
    
    logger.info('=' * 70)
    logger.info('OKX Perpetuals Trading Bot - Active Screener + Market Analysis')
    logger.info('=' * 70)
    logger.info('')
    
    # Initialize components
    logger.info('1. Initializing Market Condition Evaluator...')
    evaluator = MarketConditionEvaluator()
    logger.info('✓ Market evaluator initialized')
    logger.info('')
    
    logger.info('2. Initializing OKX Perpetuals Screener (Fast Mode)...')
    screener = OKXPerpScreenerFast()
    logger.info('✓ OKX fast screener initialized')
    logger.info('')
    
    logger.info('=' * 70)
    logger.info('Starting OKX Perpetuals Market Analysis Loop')
    logger.info('=' * 70)
    logger.info('')
    
    iteration = 1
    
    while True:
        try:
            logger.info(f'--- Iteration {iteration} ---')
            logger.info('')
            
            # Step 1: Screen for active OKX perpetuals
            logger.info('Fetching top active OKX perpetuals (last 5min)...')
            coins = screener.get_active_perps(
                quote='USDT',
                limit=10,
                min_volume_5m=100000  # $100k 5-min volume threshold
            )
            
            if not coins:
                logger.warning('No active coins found')
                iteration += 1
                time.sleep(60)
                continue
            
            # Display top coins
            logger.info(f'Top 5 active OKX perpetuals:')
            for i, c in enumerate(coins[:5], 1):
                logger.info(
                    f'  {i}. {c["symbol"]:<18} Volume=${c["volume_usd"]:>12,.0f}, '
                    f'Trades={c["trades"]:>5}, Volatility={c["volatility"]:>6.2f}%'
                )
            logger.info('')
            
            # Step 2: Pick top coin and analyze it
            top_symbol = coins[0]['symbol']
            logger.info(f'Evaluating market conditions for {top_symbol}...')
            
            # Fetch OHLCV data
            df = screener.get_ohlcv(top_symbol, '1m', limit=100)
            
            if df is None or len(df) == 0:
                logger.error(f'Failed to fetch OHLCV for {top_symbol}')
                iteration += 1
                time.sleep(60)
                continue
            
            logger.info('')
            
            # Step 3: Evaluate market conditions
            try:
                market_state = evaluator.evaluate(df)
                
                # Display market state and metrics
                logger.info(f'🔥 Market State: {market_state}')
                
                metrics = evaluator.get_market_metrics()
                logger.info('Market Metrics:')
                logger.info(f'  ADX: {metrics.get("adx", 0):.2f}')
                logger.info(f'  ATR Ratio: {metrics.get("atr_ratio", 0):.2f}')
                logger.info(f'  Volume Ratio: {metrics.get("volume_ratio", 0):.2f}')
                logger.info(f'  ROC: {metrics.get("roc", 0):.2f}%')
                logger.info(f'  Close: ${metrics.get("close", 0):.2f}')
                logger.info('')
                
                # Step 4: Trading decision
                logger.info('Trading Decision:')
                
                if market_state == MarketState.AVOID:
                    logger.info('❌ AVOID state - do not trade')
                    logger.info('   → Market volatility or conditions unfavorable')
                
                elif market_state == MarketState.RANGE:
                    logger.info('❌ Market condition is NOT TRADEABLE (range)')
                    logger.info('   → Standing aside, waiting for better conditions')
                
                elif market_state in [MarketState.MOMENTUM_LONG, MarketState.MOMENTUM_SHORT]:
                    direction = '📈 LONG' if market_state == MarketState.MOMENTUM_LONG else '📉 SHORT'
                    logger.info(f'✅ MOMENTUM SETUP - {direction}')
                    logger.info(f'   → Entry signal available on {top_symbol}')
                    logger.info(f'   → Position size: 1R with {metrics.get("stop_distance", 0):.2f}% SL')
                
                elif market_state == MarketState.MEAN_REVERSION:
                    logger.info('✅ MEAN REVERSION SETUP')
                    logger.info(f'   → Pair trade on extremes expected on {top_symbol}')
                    logger.info(f'   → Wait for rejection at BB bands')
                
                elif market_state in [MarketState.SPIKE_UP, MarketState.SPIKE_DOWN]:
                    direction = '⬆️ UP' if market_state == MarketState.SPIKE_UP else '⬇️ DOWN'
                    logger.info(f'✅ SPIKE DETECTED - {direction}')
                    logger.info(f'   → Opportunity for momentum scalp on {top_symbol}')
                    logger.info(f'   → Tight SL at recent swing')
                
            except Exception as e:
                logger.error(f'Error evaluating market: {e}')
                logger.debug(f'DataFrame shape: {df.shape if df is not None else None}')
            
            logger.info('')
            logger.info(f'Waiting 60 seconds before next check...')
            logger.info('')
            
            iteration += 1
            time.sleep(60)
        
        except KeyboardInterrupt:
            logger.info('Bot stopped by user')
            break
        except Exception as e:
            logger.error(f'Unexpected error in main loop: {e}', exc_info=True)
            time.sleep(60)


if __name__ == '__main__':
    main()
