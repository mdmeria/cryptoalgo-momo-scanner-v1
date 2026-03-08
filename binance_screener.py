"""
Binance Screener - Real-time screening for most active altcoins
Finds coins with highest trade count and volume in recent time windows
"""
import requests
import logging
from typing import List, Dict, Optional
from datetime import datetime
import time


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BinanceScreener:
    """
    Screen Binance perpetual futures for most active coins.
    Uses real Binance API - no API key required for market data.
    """
    
    def __init__(self, base_url: str = "https://fapi.binance.com"):
        """
        Initialize Binance screener.
        
        Args:
            base_url: Binance API base URL (fapi for futures, api for spot)
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
              'Content-Type': 'application/json',
              'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
              'Accept': 'application/json'
        })
        self._last_fetch = None
        self._cache = []
        self._cache_duration = 10  # Cache for 10 seconds (5min data changes fast)
    
    def get_top_coins_by_recent_activity(self,
                                         top_n: int = 10,
                                         min_volume_5m: float = 500_000,
                                         quote_currency: str = 'USDT',
                                         lookback_minutes: int = 5) -> List[Dict]:
        """
        Get top coins by recent trading activity (trades + volume in last N minutes).
        
        Args:
            top_n: Number of top coins to return
            min_volume_5m: Minimum volume in recent window (USD)
            quote_currency: Quote currency filter (USDT, BUSD, etc.)
            lookback_minutes: Time window to analyze (default 5 minutes)
        
        Returns:
            List of coin dictionaries sorted by activity
        """
        # Return cached if valid
        if self._is_cache_valid():
            logger.info(f"Returning cached screener data ({len(self._cache)} coins)")
            return self._cache[:top_n]
        
        try:
            # Step 1: Get all active symbols from 24h ticker (for initial filtering)
            logger.info("Fetching active symbols from Binance...")
            symbols = self._get_active_symbols(quote_currency)
            logger.info(f"Found {len(symbols)} {quote_currency} pairs")
            
            # Step 2: Get recent kline data for each symbol (last N minutes)
            logger.info(f"Analyzing last {lookback_minutes} minutes of data...")
            candidates = []
            
            for symbol in symbols:
                recent_stats = self._get_recent_stats(symbol, lookback_minutes)
                
                if recent_stats and recent_stats['volume_usd'] >= min_volume_5m:
                    candidates.append(recent_stats)
                
                # Small delay to avoid rate limits (1200 weight/min limit)
                time.sleep(0.05)  # ~20 requests/sec
            
            # Step 3: Sort by activity score (trades + volume weighted)
            candidates.sort(key=lambda x: x['activity_score'], reverse=True)
            
            # Update cache
            self._cache = candidates
            self._last_fetch = datetime.now()
            
            logger.info(f"Found {len(candidates)} coins meeting criteria")
            return candidates[:top_n]
            
        except Exception as e:
            logger.error(f"Error in screener: {e}")
            if self._cache:
                logger.warning("Returning stale cache due to error")
                return self._cache[:top_n]
            return []
    
    def _get_active_symbols(self, quote_currency: str) -> List[str]:
        """
        Get list of active trading symbols for a quote currency.
        Uses 24h ticker to find symbols with recent activity.
        """
        try:
            endpoint = f"{self.base_url}/fapi/v1/ticker/24hr"
            response = self.session.get(endpoint, timeout=10)
            response.raise_for_status()
            
            tickers = response.json()
            
            # Filter by quote currency and minimum activity
            symbols = []
            for ticker in tickers:
                symbol = ticker['symbol']
                if symbol.endswith(quote_currency):
                    # Only include if decent 24h volume (> $1M)
                    volume_usd = float(ticker.get('quoteVolume', 0))
                    if volume_usd > 1_000_000:
                        symbols.append(symbol)
            
            return symbols
            
        except Exception as e:
            logger.error(f"Error fetching active symbols: {e}")
            return []
    
    def _get_recent_stats(self, symbol: str, minutes: int = 5) -> Optional[Dict]:
        """
        Get trading statistics for recent time window by analyzing klines.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            minutes: Lookback period in minutes
        
        Returns:
            Dictionary with recent stats or None if error
        """
        try:
            # Fetch 1-minute klines for last N minutes
            endpoint = f"{self.base_url}/fapi/v1/klines"
            params = {
                'symbol': symbol,
                'interval': '1m',
                'limit': minutes  # Last N candles
            }
            
            response = self.session.get(endpoint, params=params, timeout=5)
            response.raise_for_status()
            
            klines = response.json()
            
            if not klines:
                return None
            
            # Aggregate data across the klines
            total_volume = 0
            total_trades = 0
            high_price = 0
            low_price = float('inf')
            latest_close = 0
            
            for kline in klines:
                # Kline format: [time, open, high, low, close, volume, close_time, 
                #                quote_volume, trades, taker_buy_base, taker_buy_quote, ignore]
                volume = float(kline[5])  # Base asset volume
                quote_volume = float(kline[7])  # Quote asset volume (USDT)
                trades = int(kline[8])  # Number of trades
                high = float(kline[2])
                low = float(kline[3])
                close = float(kline[4])
                
                total_volume += quote_volume
                total_trades += trades
                high_price = max(high_price, high)
                low_price = min(low_price, low)
                latest_close = close
            
            # Calculate volatility
            if low_price > 0:
                volatility = ((high_price - low_price) / low_price) * 100
            else:
                volatility = 0
            
            # Activity score: weighted combination of trades and volume
            # Normalize: trades per minute + $100k volume units
            trades_per_min = total_trades / minutes
            volume_100k_units = total_volume / 100_000
            activity_score = trades_per_min + volume_100k_units
            
            return {
                'symbol': symbol,
                'base': symbol[:-4] if symbol.endswith('USDT') else symbol,
                'quote': 'USDT',
                'volume_usd': total_volume,
                'trades': total_trades,
                'trades_per_min': trades_per_min,
                'volatility': volatility,
                'price': latest_close,
                'high': high_price,
                'low': low_price,
                'activity_score': activity_score,
                'lookback_minutes': minutes,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.debug(f"Error fetching stats for {symbol}: {e}")
            return None
    
    def get_top_symbols(self, top_n: int = 10, **kwargs) -> List[str]:
        """
        Get just the symbol names as a list.
        
        Args:
            top_n: Number of symbols to return
            **kwargs: Additional parameters for get_top_coins_by_recent_activity
        
        Returns:
            List of symbol strings
        """
        coins = self.get_top_coins_by_recent_activity(top_n=top_n, **kwargs)
        return [coin['symbol'] for coin in coins]
    
    def _is_cache_valid(self) -> bool:
        """Check if cached data is still valid."""
        if not self._last_fetch or not self._cache:
            return False
        
        age = (datetime.now() - self._last_fetch).total_seconds()
        return age < self._cache_duration
    
    def clear_cache(self):
        """Clear the cache."""
        self._cache = []
        self._last_fetch = None


class BinanceSpotScreener(BinanceScreener):
    """
    Screener for Binance Spot market instead of Futures.
    """
    
    def __init__(self, base_urls: Optional[List[str]] = None):
        # Try regional endpoint first, then fall back to global.
        resolved_urls = base_urls or ["https://api.binance.us", "https://api.binance.com"]
        super().__init__(base_url=resolved_urls[0])
        self.base_urls = [url.rstrip('/') for url in resolved_urls]

    def _request_spot_json(self, path: str, params: Optional[Dict] = None, timeout: int = 10):
        """Try configured spot endpoints in order and return the first successful JSON response."""
        last_error = None
        for base_url in self.base_urls:
            endpoint = f"{base_url}{path}"
            try:
                response = self.session.get(endpoint, params=params, timeout=timeout)
                response.raise_for_status()
                # Store working endpoint for visibility.
                self.base_url = base_url
                return response.json()
            except requests.exceptions.RequestException as exc:
                last_error = exc
                status_code = getattr(getattr(exc, "response", None), "status_code", "unknown")
                logger.warning(f"Spot endpoint failed ({base_url}, status={status_code}): {exc}")
                continue

        if last_error:
            raise last_error
        raise RuntimeError("No spot endpoint configured")
    
    def _get_active_symbols(self, quote_currency: str) -> List[str]:
        """Get active spot symbols."""
        try:
            # Spot uses /api/v3 endpoints.
            tickers = self._request_spot_json("/api/v3/ticker/24hr", timeout=10)
            
            symbols = []
            for ticker in tickers:
                symbol = ticker['symbol']
                if symbol.endswith(quote_currency):
                    volume_usd = float(ticker.get('quoteVolume', 0))
                    if volume_usd > 1_000_000:
                        symbols.append(symbol)
            
            return symbols
            
        except Exception as e:
            logger.error(f"Error fetching spot symbols: {e}")
            return []
    
    def _get_recent_stats(self, symbol: str, minutes: int = 5) -> Optional[Dict]:
        """Get spot market stats (spot klines endpoint)."""
        try:
            params = {
                'symbol': symbol,
                'interval': '1m',
                'limit': minutes
            }

            klines = self._request_spot_json("/api/v3/klines", params=params, timeout=5)
            
            if not klines:
                return None
            
            # Same aggregation logic as futures
            total_volume = 0
            total_trades = 0
            high_price = 0
            low_price = float('inf')
            latest_close = 0
            
            for kline in klines:
                quote_volume = float(kline[7])
                trades = int(kline[8])
                high = float(kline[2])
                low = float(kline[3])
                close = float(kline[4])
                
                total_volume += quote_volume
                total_trades += trades
                high_price = max(high_price, high)
                low_price = min(low_price, low)
                latest_close = close
            
            if low_price > 0:
                volatility = ((high_price - low_price) / low_price) * 100
            else:
                volatility = 0
            
            trades_per_min = total_trades / minutes
            volume_100k_units = total_volume / 100_000
            activity_score = trades_per_min + volume_100k_units
            
            return {
                'symbol': symbol,
                'base': symbol[:-4] if symbol.endswith('USDT') else symbol,
                'quote': 'USDT',
                'volume_usd': total_volume,
                'trades': total_trades,
                'trades_per_min': trades_per_min,
                'volatility': volatility,
                'price': latest_close,
                'high': high_price,
                'low': low_price,
                'activity_score': activity_score,
                'lookback_minutes': minutes,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.debug(f"Error fetching spot stats for {symbol}: {e}")
            return None


class BinanceUSSpotScreener(BinanceSpotScreener):
    """Spot screener pinned to Binance US first."""

    def __init__(self):
        super().__init__(base_urls=["https://api.binance.us", "https://api.binance.com"])


class BinanceVisionSpotScreener(BinanceSpotScreener):
    """Spot screener pinned to Binance public market-data endpoint."""

    def __init__(self):
        super().__init__(base_urls=["https://data-api.binance.vision", "https://api.binance.com"])


if __name__ == "__main__":
    # Demo
    print("="*70)
    print("Binance Screener - Finding Most Active Altcoins (Last 5 Minutes)")
    print("="*70)
    print()
    
    
    # Use Binance public market-data endpoint first (works in more regions).
    screener = BinanceVisionSpotScreener()
    print("Using Binance Vision-first spot screener")
    
    print("Scanning Binance spot markets for most active coins...")
    print("(This will take ~30-60 seconds to analyze all pairs)")
    print()
    
    top_coins = screener.get_top_coins_by_recent_activity(
        top_n=10,
        min_volume_5m=500_000,  # $500k minimum volume
        lookback_minutes=5
    )
    
    if top_coins:
        print(f"Top {len(top_coins)} Most Active Coins (Last 5 Minutes):")
        print("-" * 70)
        print(f"{'#':<3} {'Symbol':<12} {'Volume':<15} {'Trades':<8} {'Vol%':<8} {'Activity'}")
        print("-" * 70)
        
        for i, coin in enumerate(top_coins, 1):
            print(f"{i:<3} {coin['symbol']:<12} "
                  f"${coin['volume_usd']:>12,.0f}  "
                  f"{coin['trades']:>6}  "
                  f"{coin['volatility']:>5.2f}%  "
                  f"{coin['activity_score']:>8.1f}")
        
        print()
        print("Legend:")
        print("  Volume: Total USD volume in last 5 minutes")
        print("  Trades: Number of trades executed")
        print("  Vol%: Price volatility (high-low range)")
        print("  Activity: Combined score (higher = more active)")
    else:
        print("No coins found meeting criteria")
    
    print()
    print("="*70)
