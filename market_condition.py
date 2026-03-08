"""
Market Condition Evaluator
Classifies market state as: MOMENTUM, MEAN_REVERSION, SPIKE, RANGE, or AVOID
"""
import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, Tuple
import pandas_ta as ta


class MarketState(Enum):
    """Possible market states"""
    MOMENTUM_LONG = "momentum_long"
    MOMENTUM_SHORT = "momentum_short"
    MEAN_REVERSION = "mean_reversion"
    SPIKE_UP = "spike_up"
    SPIKE_DOWN = "spike_down"
    RANGE = "range"
    AVOID = "avoid"


class MarketConditionEvaluator:
    """
    Evaluates market conditions using technical indicators.
    Determines if the current market is suitable for momentum, mean-reversion, or spike trading.
    """
    
    def __init__(self, 
                 adx_threshold: float = 25,
                 atr_threshold: float = 1.5,
                 volume_spike_threshold: float = 2.0):
        """
        Initialize the evaluator with threshold parameters.
        
        Args:
            adx_threshold: ADX value above which market is considered trending
            atr_threshold: Relative ATR multiplier for volatility detection
            volume_spike_threshold: Volume multiplier for spike detection
        """
        self.adx_threshold = adx_threshold
        self.atr_threshold = atr_threshold
        self.volume_spike_threshold = volume_spike_threshold
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators needed for market classification.
        
        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
        
        Returns:
            DataFrame with added indicator columns
        """
        df = df.copy()
        
        # Trend indicators
        df['ema_20'] = ta.ema(df['close'], length=20)
        df['ema_50'] = ta.ema(df['close'], length=50)
        df['adx'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
        
        # Volatility
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atr_ma'] = df['atr'].rolling(window=20).mean()
        df['atr_ratio'] = df['atr'] / df['atr_ma']
        
        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Price momentum
        df['roc'] = ta.roc(df['close'], length=5)  # 5-period rate of change
        df['returns'] = df['close'].pct_change()
        df['returns_std'] = df['returns'].rolling(window=20).std()
        
        # Bollinger Bands for mean reversion
        bbands = ta.bbands(df['close'], length=20, std=2)
        df['bb_upper'] = bbands['BBU_20_2.0_2.0']
        df['bb_middle'] = bbands['BBM_20_2.0_2.0']
        df['bb_lower'] = bbands['BBL_20_2.0_2.0']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        return df
    
    def detect_spike(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Detect sudden price or volume spikes.
        
        Returns:
            (is_spike, direction) tuple
        """
        latest = df.iloc[-1]
        
        # Volume spike
        volume_spike = latest['volume_ratio'] > self.volume_spike_threshold
        
        # Price spike (z-score of returns)
        if len(df) >= 20:
            recent_returns = df['returns'].iloc[-20:]
            zscore = (latest['returns'] - recent_returns.mean()) / recent_returns.std()
            
            if volume_spike and zscore > 3:
                return True, 'up'
            elif volume_spike and zscore < -3:
                return True, 'down'
        
        return False, 'none'
    
    def classify_trend(self, df: pd.DataFrame) -> str:
        """
        Classify if market is trending up, down, or ranging.
        
        Returns:
            'up', 'down', or 'range'
        """
        latest = df.iloc[-1]
        
        # Check ADX for trend strength
        if pd.isna(latest['adx']) or latest['adx'] < self.adx_threshold:
            return 'range'
        
        # Check EMA alignment
        ema_20 = latest['ema_20']
        ema_50 = latest['ema_50']
        close = latest['close']
        
        if close > ema_20 > ema_50:
            return 'up'
        elif close < ema_20 < ema_50:
            return 'down'
        else:
            return 'range'
    
    def check_mean_reversion_conditions(self, df: pd.DataFrame) -> bool:
        """
        Check if conditions are suitable for mean reversion strategy.
        
        Returns:
            True if suitable for mean reversion
        """
        latest = df.iloc[-1]
        
        # Low ADX (not trending)
        low_adx = latest['adx'] < self.adx_threshold
        
        # High volatility (price oscillating)
        high_volatility = latest['atr_ratio'] > 1.2
        
        # Price near bollinger band extremes
        close = latest['close']
        near_bands = (close <= latest['bb_lower'] * 1.01) or (close >= latest['bb_upper'] * 0.99)
        
        return low_adx and high_volatility and near_bands
    
    def evaluate(self, df: pd.DataFrame) -> MarketState:
        """
        Main evaluation function - determines current market state.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            MarketState enum value
        """
        # Calculate all indicators
        df = self.calculate_indicators(df)
        
        if len(df) < 50:
            return MarketState.AVOID
        
        # Check for spikes first (highest priority)
        is_spike, spike_direction = self.detect_spike(df)
        if is_spike:
            return MarketState.SPIKE_UP if spike_direction == 'up' else MarketState.SPIKE_DOWN
        
        # Check for mean reversion conditions
        if self.check_mean_reversion_conditions(df):
            return MarketState.MEAN_REVERSION
        
        # Check for momentum conditions
        trend = self.classify_trend(df)
        latest = df.iloc[-1]
        
        if trend == 'up' and latest['atr_ratio'] > 1.0:
            return MarketState.MOMENTUM_LONG
        elif trend == 'down' and latest['atr_ratio'] > 1.0:
            return MarketState.MOMENTUM_SHORT
        elif trend == 'range':
            return MarketState.RANGE
        
        return MarketState.AVOID
    
    def get_market_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Get current market metrics for logging/debugging.
        
        Returns:
            Dictionary of key metrics
        """
        df = self.calculate_indicators(df)
        latest = df.iloc[-1]
        
        return {
            'adx': latest['adx'],
            'atr_ratio': latest['atr_ratio'],
            'volume_ratio': latest['volume_ratio'],
            'roc': latest['roc'],
            'ema_20': latest['ema_20'],
            'ema_50': latest['ema_50'],
            'close': latest['close'],
            'bb_width': latest['bb_width']
        }


if __name__ == "__main__":
    # Example usage
    print("Market Condition Evaluator initialized")
    print("Use this module to classify market states")
    
    # Demo with random data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
    demo_df = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    evaluator = MarketConditionEvaluator()
    state = evaluator.evaluate(demo_df)
    metrics = evaluator.get_market_metrics(demo_df)
    
    print(f"\nDemo Market State: {state.value}")
    print(f"Demo Metrics: {metrics}")
