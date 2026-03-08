#!/usr/bin/env python3
"""OKX Perpetuals Screener - Find active trading opportunities on OKX SWAP contracts."""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OKXPerpScreener:
    """Screen OKX perpetuals (SWAP contracts) for trading opportunities."""
    
    BASE_URL = 'https://www.okx.com/api/v5'
    
    def __init__(self):
        self.session = requests.Session()
        self.session.timeout = 10
        self._inst_cache = None
        self._inst_cache_time = None
    
    def get_active_perps(self, quote='USDT', limit: int = 100, min_volume_5m: float = 100000) -> List[Dict]:
        """
        Find most active OKX perpetuals in the last 5 minutes.
        
        Args:
            quote: Quote currency filter (default 'USDT')
            limit: Return top N coins
            min_volume_5m: Minimum 5-minute volume in USD (default $100k)
        
        Returns:
            List of dicts with symbol, volume_usd, trades, volatility, activity_score
        """
        logger.info(f'Screening OKX perpetuals for {quote} pairs...')
        
        # Get all available instruments
        instruments = self._get_instruments(quote)
        logger.info(f'Found {len(instruments)} {quote} perpetuals')
        
        if not instruments:
            logger.warning(f'No {quote} perpetuals found')
            return []
        
        # Analyze recent activity (last 5 minutes)
        logger.info(f'Analyzing last 5 minutes of trading activity...')
        active_coins = self._analyze_activity(instruments, lookback_minutes=5)
        
        # Filter by minimum volume
        filtered = [c for c in active_coins if c['volume_usd'] >= min_volume_5m]
        logger.info(f'Found {len(filtered)} coins with ${min_volume_5m:,.0f}+ 5-min volume')
        
        # Sort by activity score and return top N
        sorted_coins = sorted(filtered, key=lambda x: x['activity_score'], reverse=True)
        return sorted_coins[:limit]
    
    def _get_instruments(self, quote: str = 'USDT') -> List[str]:
        """Get all available USDT perpetual instruments."""
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
            
            # Filter to USDT pairs
            instruments = [
                d['instId'] for d in data.get('data', [])
                if quote in d.get('instId', '')
            ]
            
            return instruments
        except Exception as e:
            logger.error(f'Error fetching instruments: {e}')
            return []
    
    def _analyze_activity(self, instruments: List[str], lookback_minutes: int = 5) -> List[Dict]:
        """Analyze trading activity over lookback period."""
        results = []
        
        # For each instrument, fetch candles and calculate metrics
        for inst_id in instruments:
            try:
                # Fetch last N candles (1 minute each)
                candles = self._get_candles(inst_id, '1m', limit=lookback_minutes)
                
                if not candles or len(candles) < 2:
                    continue
                
                # Calculate metrics
                df = pd.DataFrame(candles, columns=['timestamp', 'o', 'h', 'l', 'c', 'vol', 'volCcy', 'volCcyQuote', 'confirm'])
                df['vol'] = pd.to_numeric(df['vol'])
                df['c'] = pd.to_numeric(df['c'])
                
                volume_usd = float(df['vol'].sum()) * float(df['c'].iloc[-1])
                volatility = (float(df['h'].max()) - float(df['l'].min())) / float(df['l'].min()) * 100
                
                # Activity score = volume * trade frequency
                activity_score = volume_usd * len(df)
                
                results.append({
                    'symbol': inst_id,
                    'volume_usd': volume_usd,
                    'volatility': volatility,
                    'activity_score': activity_score,
                    'trades': len(df),
                    'close': float(df['c'].iloc[-1])
                })
            
            except Exception as e:
                logger.debug(f'Error analyzing {inst_id}: {e}')
                continue
        
        return results
    
    def _get_candles(self, inst_id: str, bar: str = '1m', limit: int = 100) -> Optional[List]:
        """Fetch OHLCV candles from OKX."""
        try:
            url = f'{self.BASE_URL}/market/candles'
            params = {
                'instId': inst_id,
                'bar': bar,
                'limit': limit
            }
            r = self.session.get(url, params=params, timeout=10)
            
            if r.status_code != 200:
                return None
            
            data = r.json()
            if data.get('code') != '0':
                return None
            
            # OKX returns candles newest first, reverse to chronological order
            return list(reversed(data.get('data', [])))
        except Exception as e:
            logger.debug(f'Error fetching candles for {inst_id}: {e}')
            return None
    
    def get_ohlcv(self, symbol: str, interval: str = '1m', limit: int = 100) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a symbol suitable for technical analysis.
        
        Args:
            symbol: OKX instrument ID (e.g., 'BTC-USDT-SWAP')
            interval: Candle interval ('1m', '5m', '15m', etc.)
            limit: Number of candles to fetch
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            candles = self._get_candles(symbol, interval, limit)
            
            if not candles or len(candles) == 0:
                logger.error(f'No candles found for {symbol}')
                return None
            
            df = pd.DataFrame(candles, columns=['timestamp', 'o', 'h', 'l', 'c', 'vol', 'volCcy', 'volCcyQuote', 'confirm'])
            
            # Convert to numeric and create standard OHLCV format
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            df['open'] = pd.to_numeric(df['o'], errors='coerce')
            df['high'] = pd.to_numeric(df['h'], errors='coerce')
            df['low'] = pd.to_numeric(df['l'], errors='coerce')
            df['close'] = pd.to_numeric(df['c'], errors='coerce')
            df['volume'] = pd.to_numeric(df['vol'], errors='coerce')
            
            # Return clean DataFrame
            result = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            result = result.dropna()
            
            if len(result) == 0:
                logger.error(f'No valid data for {symbol}')
                return None
            
            return result
        
        except Exception as e:
            logger.error(f'Error fetching OHLCV for {symbol}: {e}')
            return None


if __name__ == '__main__':
    # Test the screener
    screener = OKXPerpScreener()
    
    print('='*70)
    print('OKX Perpetuals Screener - Active Coins Discovery')
    print('='*70)
    print()
    
    coins = screener.get_active_perps(quote='USDT', limit=15, min_volume_5m=100000)
    
    if coins:
        print(f'Found {len(coins)} active coins (5min, $100k+ volume):\n')
        print(f'{"#":<3} {"Symbol":<18} {"Volume":<15} {"Trades":<8} {"Activity Score"}')
        print('-'*70)
        
        for i, c in enumerate(coins, 1):
            print(f'{i:<3} {c["symbol"]:<18} ${c["volume_usd"]:>12,.0f}  {c["trades"]:>6}  {c["activity_score"]:>14,.0f}')
        
        # Test OHLCV fetch for top coin
        print()
        print('='*70)
        print(f'Sample: {coins[0]["symbol"]} Last 5 Candles')
        print('='*70)
        df = screener.get_ohlcv(coins[0]['symbol'], '1m', 100)
        if df is not None:
            print()
            print(f'{"Time":<20} {"Open":>10} {"High":>10} {"Low":>10} {"Close":>10}')
            print('-'*70)
            for idx, row in df.tail(5).iterrows():
                ts = str(int(row['timestamp']))[-10:]
                print(f'{ts:<20} {row["open"]:>10.2f} {row["high"]:>10.2f} {row["low"]:>10.2f} {row["close"]:>10.2f}')
    else:
        print('No active coins found matching criteria')
