#!/usr/bin/env python3
"""OKX Perpetuals Screener - Fast version that only analyzes top 50 instruments."""

import requests
import pandas as pd
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OKXPerpScreenerFast:
    """Fast OKX screener - analyzes only top instruments by 24h volume."""
    
    BASE_URL = 'https://www.okx.com/api/v5'
    
    def __init__(self):
        self.session = requests.Session()
        self.session.timeout = 10
    
    def get_active_perps(self, quote='USDT', limit: int = 10, min_volume_5m: float = 100000) -> List[Dict]:
        """
        Find most active OKX perpetuals in the last 5 minutes (fast mode).
        Only analyzes top 50 instruments by 24h volume to save time.
        
        Args:
            quote: Quote currency filter (default 'USDT')
            limit: Return top N coins
            min_volume_5m: Minimum 5-minute volume threshold
        
        Returns:
            List of dicts with symbol, volume_usd, trades, volatility, activity_score
        """
        logger.info(f'Screening OKX perpetuals (FAST MODE)...')
        
        # Get instruments sorted by volume
        instruments = self._get_instruments_by_volume(quote)
        
        if not instruments:
            logger.warning('No instruments found')
            return []
        
        # Only keep top 50 for performance
        top_instruments = instruments[:50]
        logger.info(f'Analyzing top 50 instruments by 24h volume...')
        
        # Analyze recent activity
        active_coins = self._analyze_activity(top_instruments, lookback_minutes=5)
        
        # Filter and sort
        filtered = [c for c in active_coins if c['volume_usd'] >= min_volume_5m]
        logger.info(f'Found {len(filtered)} coins with ${min_volume_5m:,.0f}+ 5-min volume')
        
        sorted_coins = sorted(filtered, key=lambda x: x['volume_usd'], reverse=True)
        return sorted_coins[:limit]
    
    def _get_instruments_by_volume(self, quote: str = 'USDT') -> List[str]:
        """Get instruments sorted by 24h volume."""
        try:
            url = f'{self.BASE_URL}/public/instruments'
            params = {'instType': 'SWAP'}
            r = self.session.get(url, params=params, timeout=10)
            
            if r.status_code != 200:
                logger.error(f'Failed to fetch instruments: {r.status_code}')
                return []
            
            data = r.json()
            if data.get('code') != '0':
                logger.error(f'API error: {data.get("msg")}')
                return []
            
            # Get all USDT instruments with volume data
            instruments_with_vol = []
            for d in data.get('data', []):
                if quote in d.get('instId', ''):
                    vol = float(d.get('volCcy24h', 0))
                    instruments_with_vol.append((d['instId'], vol))
            
            # Sort by volume descending
            instruments_with_vol.sort(key=lambda x: x[1], reverse=True)
            instruments = [inst for inst, vol in instruments_with_vol]
            
            logger.info(f'Found {len(instruments)} {quote} perpetuals')
            return instruments
        
        except Exception as e:
            logger.error(f'Error fetching instruments: {e}')
            return []
    
    def _analyze_activity(self, instruments: List[str], lookback_minutes: int = 5) -> List[Dict]:
        """Analyze trading activity for instruments."""
        results = []
        
        for inst_id in instruments:
            try:
                candles = self._get_candles(inst_id, '1m', limit=lookback_minutes)
                
                if not candles or len(candles) < 2:
                    continue
                
                df = pd.DataFrame(candles, columns=['ts', 'o', 'h', 'l', 'c', 'vol', 'volCcy', 'volQuote', 'confirm'])
                df['vol'] = pd.to_numeric(df['vol'], errors='coerce')
                df['c'] = pd.to_numeric(df['c'], errors='coerce')
                
                volume_usd = float(df['vol'].sum()) * float(df['c'].iloc[-1])
                latest_price = float(df['c'].iloc[-1])
                high = float(df['h'].max())
                low = float(df['l'].min())
                volatility = ((high - low) / low * 100) if low > 0 else 0
                
                results.append({
                    'symbol': inst_id,
                    'volume_usd': volume_usd,
                    'volatility': volatility,
                    'activity_score': volume_usd * len(df),
                    'trades': len(df),
                    'close': latest_price
                })
            
            except Exception as e:
                logger.debug(f'Error analyzing {inst_id}: {e}')
                continue
        
        return results
    
    def _get_candles(self, inst_id: str, bar: str = '1m', limit: int = 100) -> Optional[List]:
        """Fetch OHLCV candles."""
        try:
            url = f'{self.BASE_URL}/market/candles'
            params = {'instId': inst_id, 'bar': bar, 'limit': limit}
            r = self.session.get(url, params=params, timeout=10)
            
            if r.status_code != 200 or r.json().get('code') != '0':
                return None
            
            return list(reversed(r.json().get('data', [])))
        except:
            return None
    
    def get_ohlcv(self, symbol: str, interval: str = '1m', limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data for analysis."""
        try:
            candles = self._get_candles(symbol, interval, limit)
            
            if not candles:
                logger.error(f'No candles for {symbol}')
                return None
            
            df = pd.DataFrame(candles, columns=['ts', 'o', 'h', 'l', 'c', 'vol', 'volCcy', 'volQuote', 'confirm'])
            
            df['timestamp'] = pd.to_numeric(df['ts'], errors='coerce')
            df['open'] = pd.to_numeric(df['o'], errors='coerce')
            df['high'] = pd.to_numeric(df['h'], errors='coerce')
            df['low'] = pd.to_numeric(df['l'], errors='coerce')
            df['close'] = pd.to_numeric(df['c'], errors='coerce')
            df['volume'] = pd.to_numeric(df['vol'], errors='coerce')
            
            result = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].dropna()
            
            return result if len(result) > 0 else None
        
        except Exception as e:
            logger.error(f'Error fetching OHLCV for {symbol}: {e}')
            return None


if __name__ == '__main__':
    screener = OKXPerpScreenerFast()
    
    print('='*70)
    print('OKX Perpetuals Screener - FAST MODE')
    print('='*70)
    print()
    
    coins = screener.get_active_perps(quote='USDT', limit=10, min_volume_5m=100000)
    
    if coins:
        print(f'Found {len(coins)} active coins (5min, $100k+ volume):\n')
        print(f'{"#":<3} {"Symbol":<18} {"Volume":<15} {"Trades":<8} {"Volatility"}')
        print('-'*70)
        
        for i, c in enumerate(coins, 1):
            print(f'{i:<3} {c["symbol"]:<18} ${c["volume_usd"]:>12,.0f}  {c["trades"]:>6}  {c["volatility"]:>6.2f}%')
    else:
        print('No active coins found')
