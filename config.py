"""
Configuration module for the crypto trading bot.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Main configuration class"""
    
    # API Keys
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
    BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '')
    
    ORION_API_KEY = os.getenv('ORION_API_KEY', '')
    ORION_API_URL = os.getenv('ORION_API_URL', 'https://screener.orionterminal.com/api/screener')
    
    BITUNIX_API_KEY = os.getenv('BITUNIX_API_KEY', '')
    BITUNIX_API_SECRET = os.getenv('BITUNIX_API_SECRET', '')
    
    # Screener settings (coin discovery source)
    SCREENER_SOURCE = os.getenv('SCREENER_SOURCE', 'orion')  # 'orion' or 'binance'

    # Binance screener settings (no API key needed for market data)
    BINANCE_MODE = os.getenv('BINANCE_MODE', 'binance_vision')  # 'futures', 'spot', 'binance_us', or 'binance_vision'
    
    # Trading settings
    TRADING_MODE = os.getenv('TRADING_MODE', 'paper')
    RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', '0.01'))
    MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', '3'))
    
    # Market condition thresholds
    ADX_TREND_THRESHOLD = 25  # ADX > 25 indicates trending
    ATR_VOLATILITY_THRESHOLD = 1.5  # Relative ATR threshold
    VOLUME_SPIKE_THRESHOLD = 2.0  # 2x average volume
    
    # Timeframe
    TIMEFRAME = '1m'
    LOOKBACK_PERIODS = 100  # Number of candles to analyze
    
    # Top coins selection
    TOP_N_COINS = 10  # Number of top coins to monitor
    SCREENER_REFRESH_SECONDS = 60  # How often to refresh screener data
    SCREENER_LOOKBACK_MINUTES = 5  # Time window for recent activity
    MIN_VOLUME_5M = float(os.getenv('MIN_VOLUME_5M', '100000'))  # Minimum volume in 5min window (USD)
    SCREENER_QUOTE_CURRENCY = os.getenv('SCREENER_QUOTE_CURRENCY', 'USDT')
