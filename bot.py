import asyncio
import json
import os
import pandas as pd
import requests
from datetime import datetime
from dotenv import load_dotenv
import numpy as np  # for safety, though not heavily used

from solana.rpc.async_api import AsyncClient
from anchorpy import Provider, Wallet
from solders.keypair import Keypair
from driftpy.drift_client import DriftClient
from driftpy.constants import configs
from driftpy.drift_user import DriftUser
from driftpy.types import PositionDirection

load_dotenv()

# Config from .env
PRIVATE_KEY_JSON = json.loads(os.getenv("PRIVATE_KEY_JSON"))
RPC_URL = os.getenv("RPC_URL")
BIRDEYE_API_KEY = os.getenv("BIRDEYE_API_KEY")
MARKET_INDEX = int(os.getenv("MARKET_INDEX", 0))  # SOL-PERP
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", 0.005))
LEVERAGE = int(os.getenv("LEVERAGE", 8))
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", 60))

SOL_ADDRESS = "So11111111111111111111111111111111111111112"

async def get_candles(limit=200):
    url = "https://public-api.birdeye.so/defi/v3/ohlcv"
    params = {
        "address": SOL_ADDRESS,
        "type": "5m",
        "currency": "usd",
        "mode": "count",
        "count": limit
    }
    headers = {
        "accept": "application/json",
        "x-chain": "solana",
        "x-api-key": BIRDEYE_API_KEY
    }
    resp = requests.get(url, params=params, headers=headers)
    if resp.status_code != 200:
        raise Exception(f"Birdeye API error: {resp.text}")
    data = resp.json()["data"]["items"]
    df = pd.DataFrame(data)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df["timestamp"] = pd.to_datetime(df["unixTime"], unit="s")
    return df[["open", "high", "low", "close", "volume"]].astype(float)

def calculate_indicators(df):
    # EMA
    df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()

    # RSI(9)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=9).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=9).mean()
    rs = gain / loss
    df['rsi9'] = 100 - (100 / (1 + rs))

    # MACD(12,26,9)
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd_line'] = ema12 - ema26
    df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd_line'] - df['macd_signal']

    return df

async def main():
    keypair = Keypair.from_secret_key(bytes(PRIVATE_KEY_JSON))
    wallet = Wallet(keypair)
    connection = AsyncClient(RPC_URL)
    provider = Provider(connection, wallet)
    config = configs["mainnet"]
    drift_client = DriftClient.from_config(config, provider, perp_market_indexes=[MARKET_INDEX])
    await drift_client.subscribe()

    drift_user = DriftUser(drift_client)
    print(f"🚀 Bot started | Collateral: {await drift_user.get_total_collateral():.2f} USDC")

    in_position = False
    position_side = None

    while True:
        try:
            df = await get_candles()
            df = calculate_indicators(df)
            latest = df.iloc[-1]
            prev = df.iloc[-2]

            # Long signal: price above EMAs + RSI cross up from <25 + MACD hist cross up from negative
            long_signal = (
                latest["close"] > latest["ema9"] > latest["ema21"] and
                prev["rsi9"] < 25 <= latest["rsi9"] and
                prev["macd_hist"] < 0 <= latest["macd_hist"]
            )

            # Short signal: mirror
            short_signal = (
                latest["close"] < latest["ema9"] < latest["ema21"] and
                prev["rsi9"] > 75 >= latest["rsi9"] and
                prev["macd_hist"] > 0 >= latest["macd_hist"]
            )

            user_positions = await drift_user.get_user_positions()
            has_position = any(p.market_index == MARKET_INDEX and abs(p.base_asset_amount) > 0 for p in user_positions)

            if not has_position:
                collateral = await drift_user.get_total_collateral()
                size_usd = collateral * LEVERAGE * RISK_PER_TRADE * 2  # conservative
                size_base = int(size_usd / latest["close"] * 1e9)  # Drift precision

                if long_signal:
                    await drift_client.open_position(PositionDirection.LONG(), size_base, MARKET_INDEX)
                    print(f"✅ LONG opened @ ~${latest['close']:.2f} | Size ~${size_usd:.0f}")
                    in_position = True
                    position_side = "LONG"

                elif short_signal:
                    await drift_client.open_position(PositionDirection.SHORT(), size_base, MARKET_INDEX)
                    print(f"✅ SHORT opened @ ~${latest['close']:.2f} | Size ~${size_usd:.0f}")
                    in_position = True
                    position_side = "SHORT"

            elif has_position and ((position_side == "LONG" and short_signal) or (position_side == "SHORT" and long_signal)):
                await drift_client.close_position(MARKET_INDEX)
                print(f"🔄 {position_side} CLOSED on opposite signal")
                in_position = False

            await asyncio.sleep(CHECK_INTERVAL)

        except Exception as e:
            print(f"Error: {e}")
            await asyncio.sleep(30)

if __name__ == "__main__":
    asyncio.run(main())
