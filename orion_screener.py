"""Orion Terminal screener integration for coin discovery only."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List

import requests

logger = logging.getLogger(__name__)


class OrionTerminalScreener:
    """Fetch top active coins from Orion Terminal screener API."""

    def __init__(self, base_url: str = "https://screener.orionterminal.com/api/screener"):
        self.base_url = base_url
        self.session = requests.Session()
        self._cache: List[Dict] = []
        self._last_fetch: datetime | None = None
        self._cache_seconds = 15

    def get_top_coins_by_recent_activity(
        self,
        top_n: int = 10,
        min_volume_5m: float = 100000,
        quote_currency: str = "USDT",
        lookback_minutes: int = 5,
        force_refresh: bool = False,
    ) -> List[Dict]:
        """Return top coins ranked by recent activity from Orion's Binance-based feed."""
        if not force_refresh and self._is_cache_valid():
            return self._cache[:top_n]

        payload = self._fetch_payload()
        tickers = payload.get("tickers", [])

        if not tickers:
            logger.warning("No tickers returned from Orion screener")
            return []

        quote_currency = quote_currency.upper()
        tf_key = self._resolve_tf_key(lookback_minutes)

        ranked: List[Dict] = []
        for t in tickers:
            symbol = str(t.get("symbol", "")).upper()
            if not symbol.endswith(quote_currency):
                continue

            tf = t.get(tf_key) or {}
            volume = float(tf.get("volume", 0) or 0)
            trades = int(tf.get("trades", 0) or 0)
            volatility = float(tf.get("volatility", 0) or 0)

            if volume < min_volume_5m:
                continue

            oi_change = float(tf.get("oiChange", 0) or 0)
            vdelta = float(tf.get("vdelta", 0) or 0)

            # Score prioritizes flow (volume/trades), then volatility and derivatives context.
            activity_score = (volume * max(trades, 1)) * (1.0 + abs(volatility)) * (1.0 + abs(oi_change))

            ranked.append(
                {
                    "symbol": symbol,
                    "volume_usd": volume,
                    "trades": trades,
                    "volatility": volatility,
                    "activity_score": activity_score,
                    "source": "orion_binance",
                    "oi_change": oi_change,
                    "vdelta": vdelta,
                }
            )

        ranked.sort(key=lambda x: x["activity_score"], reverse=True)
        self._cache = ranked
        self._last_fetch = datetime.now()

        logger.info("Orion screener returned %s coins after filtering", len(ranked))
        return ranked[:top_n]

    def _fetch_payload(self) -> Dict:
        try:
            response = self.session.get(self.base_url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            logger.error("Failed to fetch Orion screener data: %s", exc)
            return {}

    @staticmethod
    def _resolve_tf_key(lookback_minutes: int) -> str:
        if lookback_minutes <= 5:
            return "tf5m"
        if lookback_minutes <= 15:
            return "tf15m"
        return "tf1h"

    def _is_cache_valid(self) -> bool:
        if not self._last_fetch or not self._cache:
            return False
        return (datetime.now() - self._last_fetch) < timedelta(seconds=self._cache_seconds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    screener = OrionTerminalScreener()
    coins = screener.get_top_coins_by_recent_activity(
        top_n=10,
        min_volume_5m=100000,
        quote_currency="USDT",
        lookback_minutes=5,
    )

    print("Top Orion coins (discovery only):")
    for idx, c in enumerate(coins, 1):
        print(
            f"{idx:2}. {c['symbol']:<12} vol=${c['volume_usd']:>12,.0f} "
            f"trades={c['trades']:>6} volat={c['volatility']:.4f}"
        )
