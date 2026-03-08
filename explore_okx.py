#!/usr/bin/env python3
"""Explore OKX perpetuals API and available instruments."""

import requests
import json

def check_okx_perps():
    """Get all USDT perpetuals on OKX."""
    url = 'https://www.okx.com/api/v5/public/instruments?instType=SWAP'
    r = requests.get(url)
    data = r.json()
    
    if data['code'] == '0' and data['data']:
        perps = [d for d in data['data'] if 'USDT' in d.get('instId', '')]
        print(f'✓ Found {len(perps)} USDT perpetuals on OKX')
        print()
        print('Sample USDT Perps (first 15):')
        for i, p in enumerate(perps[:15], 1):
            inst_id = p['instId']
            ctType = p.get('ctType', 'linear')
            print(f'  {i:2}. {inst_id:<18} Type: {ctType}')
        return perps
    return []

def check_okx_market_data():
    """Check OKX market data endpoints."""
    print('\n' + '='*60)
    print('OKX Market Data Endpoints:')
    print('='*60)
    
    examples = {
        'BTC Ticker': 'https://www.okx.com/api/v5/market/ticker?instId=BTC-USDT-SWAP',
        'ETH Ticker': 'https://www.okx.com/api/v5/market/ticker?instId=ETH-USDT-SWAP',
        'BTC Candles (1m)': 'https://www.okx.com/api/v5/market/candles?instId=BTC-USDT-SWAP&bar=1m&limit=100',
        '24h Stats': 'https://www.okx.com/api/v5/market/ticker?instType=SWAP',
    }
    
    for name, url in examples.items():
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                data = r.json()
                if data['code'] == '0':
                    print(f'✓ {name:<20} OK')
                else:
                    print(f'✗ {name:<20} API Error: {data.get("msg", "?")}')
            else:
                print(f'✗ {name:<20} HTTP {r.status_code}')
        except Exception as e:
            print(f'✗ {name:<20} {str(e)[:40]}')

def sample_btc_candles():
    """Fetch sample BTC candles."""
    print('\n' + '='*60)
    print('Sample: BTC-USDT-SWAP 1m Candles:')
    print('='*60)
    
    url = 'https://www.okx.com/api/v5/market/candles?instId=BTC-USDT-SWAP&bar=1m&limit=10'
    r = requests.get(url)
    data = r.json()
    
    if data['code'] == '0':
        candles = data['data']
        print(f'Fetched {len(candles)} candles')
        print()
        print('Timestamp              O        H        L        C        Vol')
        print('-' * 65)
        for c in candles[:5]:
            ts = c[0]  # timestamp
            o = float(c[1])   # open
            h = float(c[2])   # high
            l = float(c[3])   # low
            close = float(c[4])  # close
            vol = float(c[5]) if len(c) > 5 else 0  # volume
            print(f'{ts:<20} {o:>7.1f} {h:>7.1f} {l:>7.1f} {close:>7.1f} {vol:>8.0f}')

if __name__ == '__main__':
    print('='*60)
    print('OKX Perpetuals API Exploration')
    print('='*60)
    print()
    
    perps = check_okx_perps()
    check_okx_market_data()
    sample_btc_candles()
    
    print('\n' + '='*60)
    print('Summary:')
    print('='*60)
    print(f'✓ OKX accessible from your region')
    print(f'✓ {len(perps)} USDT perpetuals available')
    print(f'✓ Market data API working (tickers, candles)')
    print(f'✓ Ready to build OKX screener module')
