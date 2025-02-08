import logging
import os
from typing import Dict

import requests

from .plugin import Plugin


# Author: https://github.com/zchk0
class CoinMarketCap(Plugin):
    """
    A plugin to fetch the current rate of various cryptocurrencies
    """
    def get_source_name(self) -> str:
        return "CoinMarketCap by zchk0"

    def get_spec(self) -> [Dict]:
        return [{
            "name": "get_crypto_rate",
            "description": "Get the current rate of various cryptocurrencies from coinmarketcap",
            "parameters": {
                "type": "object",
                "properties": {
                    "asset": {"type": "string", "description": "Ticker of the cryptocurrency in uppercase (e.g., BTC, ETH, XRP)"},
                    "currency": { "type": "string", "description": "Currency to convert (e.g. USD, RUB, etc.). Default is USD"}
                },
                "required": ["asset"],
            },
        }]

    def get_crypto_price(self, asset, currency="USD"):
        headers = {
            'X-CMC_PRO_API_KEY': os.environ.get('COINMARKETCAP_KEY', '')
        }
        asset = asset.upper()
        currency = currency.upper()
        params = {
            'amount': 1,
            'symbol': asset,
            'convert': currency
        }
        try:
            response = requests.get(
                "https://pro-api.coinmarketcap.com/v2/tools/price-conversion",
                headers=headers,
                params=params
            )
            response.raise_for_status()
            data = response.json()
            if "data" in data:
                conversion_data = data["data"]
                if isinstance(conversion_data, list):
                    for item in conversion_data:
                        if "quote" in item and currency in item["quote"]:
                            price = item["quote"][currency].get("price")
                            if price is not None:
                                return price
            return "Not found"
        except requests.exceptions.RequestException as e:
            logging.info(f"An error occurred: {e}")
            return None

    async def execute(self, function_name, helper, **kwargs) -> dict:
        asset = kwargs.get('asset', '')
        currency = kwargs.get('currency', 'USD')
        rate = self.get_crypto_price(asset, currency)
        return {"asset": asset, "currency": currency, "rate": rate}
