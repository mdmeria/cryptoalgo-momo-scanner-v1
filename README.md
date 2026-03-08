# Crypto Trading Bot - Market Condition & Screener

A Python-based algorithmic trading system for crypto perpetuals that trades based on market conditions (Momentum, Mean Reversion, Spikes) on 1-minute timeframes.

## 🎯 Strategy Overview

The bot follows a systematic approach:

1. **Market Condition Evaluation** - Classifies current market as:
   - Momentum (Long/Short)
   - Mean Reversion
   - Spike (Up/Down)
   - Range
   - Avoid

2. **Top Coin Selection** - Uses Orion screener to identify most active altcoins

3. **Trade Execution** - Only trades when market conditions align with strategy

4. **Risk Management** - 1:1 Risk/Reward ratio with early cut rules

## 📁 Project Structure

```
CryptoAlgo/
├── config.py              # Configuration and settings
├── market_condition.py    # Market state evaluation logic
├── orion_screener.py      # Orion API integration for coin screening
├── main.py                # Main entry point and demo
├── requirements.txt       # Python dependencies
├── .env.example          # Example environment variables
└── README.md             # This file
```

## 🚀 Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy `.env.example` to `.env` and fill in your API credentials:

```bash
cp .env.example .env
```

Edit `.env`:
```
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here
ORION_API_KEY=your_orion_key
BITUNIX_API_KEY=your_bitunix_key
BITUNIX_API_SECRET=your_bitunix_secret
```

### 3. Run the Demo

```bash
python main.py
```

## 🔧 Components

### Market Condition Evaluator (`market_condition.py`)

Analyzes OHLCV data using technical indicators:
- **ADX** - Trend strength detection
- **ATR** - Volatility measurement
- **Volume ratios** - Spike detection
- **EMA crossovers** - Trend direction
- **Bollinger Bands** - Mean reversion zones

### Orion Screener (`orion_screener.py`)

Fetches top active coins based on:
- 24h volume
- Price volatility
- Number of trades
- Price change percentage

Includes a `MockOrionScreener` for testing without API access.

### Configuration (`config.py`)

Centralized settings for:
- API credentials
- Trading parameters (risk %, max positions)
- Technical indicator thresholds
- Timeframe and lookback periods

## 📊 Market States

| State | Description | Trading Action |
|-------|-------------|----------------|
| MOMENTUM_LONG | Strong uptrend with high ADX | Look for long entries |
| MOMENTUM_SHORT | Strong downtrend with high ADX | Look for short entries |
| MEAN_REVERSION | Choppy market near BB extremes | Trade reversals |
| SPIKE_UP | Sudden price/volume surge up | Momentum or fade |
| SPIKE_DOWN | Sudden price/volume drop | Momentum or fade |
| RANGE | Low ADX, sideways movement | Avoid or range trade |
| AVOID | Unclear/unfavorable conditions | No trades |

## 🔄 Next Steps

- [ ] Implement Binance/TradingView data fetching
- [ ] Add BitUnix order execution module
- [ ] Define specific entry/exit rules for each market state
- [ ] Implement position management and trailing stops
- [ ] Add early cut logic
- [ ] Build backtesting framework
- [ ] Add paper trading mode
- [ ] Create monitoring dashboard

## ⚠️ Disclaimer

This is a trading bot framework. Trading cryptocurrencies carries significant risk. Always test thoroughly in paper trading mode before using real funds. Past performance does not guarantee future results.

## 📝 License

MIT License - Use at your own risk
