import asyncio
import json
import os
import pandas as pd
import requests
from datetime import datetime
from dotenv import load_dotenv
import numpy as np

from solana.rpc.async_api import AsyncClient
from anchorpy import Provider, Wallet
from solders.keypair import Keypair
from driftpy.drift_client import DriftClient
from driftpy.drift_user import DriftUser
from driftpy.types import PositionDirection

load_dotenv()

# Config from env vars
private_key_str = os.getenv("PRIVATE_KEY_JSON")
if private_key_str is None:
    raise ValueError("PRIVATE_KEY_JSON env var is not set! Check Railway Variables tab.")
PRIVATE_KEY_JSON = json.loads(private_key_str)
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
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()["data"]["items"]
        df = pd.DataFrame(data)
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        df["timestamp"] = pd.to_datetime(df["unixTime"], unit="s")
        return df[["open", "high", "low", "close", "volume"]].astype(float)
    except Exception as e:
        print(f"Failed to fetch candles: {e}")
        return None

def calculate_indicators(df):
    if df is None or len(df) < 50:
        return None

    df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=9).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=9).mean()
    rs = gain / loss
    df['rsi9'] = 100 - (100 / (1 + rs))

    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd_line'] = ema12 - ema26
    df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd_line'] - df['macd_signal']

    return df

async def main():
    print("Starting bot initialization...")

    keypair = Keypair.from_bytes(bytes(PRIVATE_KEY_JSON))
    wallet = Wallet(keypair)
    connection = AsyncClient(RPC_URL)
    provider = Provider(connection, wallet)

    print("Creating DriftClient...")

    drift_client = DriftClient(
        connection=connection,
        wallet=wallet,
        env="mainnet-beta",
        perp_market_indexes=[MARKET_INDEX]
    )

    try:
        await drift_client.subscribe()
        print("DriftClient subscribed OK")
    except Exception as e:
        print(f"Subscribe failed: {e}")
        return

    drift_user = DriftUser(drift_client)
    collateral = await drift_user.get_total_collateral()
    print(f"🚀 Bot started | Collateral: {collateral:.2f} USDC")

    in_position = False
    position_side = None

    while True:
        try:
            df = await get_candles()
            if df is None:
                await asyncio.sleep(CHECK_INTERVAL)
                continue

            df = calculate_indicators(df)
            if df is None:
                await asyncio.sleep(CHECK_INTERVAL)
                continue

            latest = df.iloc[-1]
            prev = df.iloc[-2]

            long_signal = (
                latest["close"] > latest["ema9"] > latest["ema21"] and
                prev["rsi9"] < 25 <= latest["rsi9"] and
                prev["macd_hist"] < 0 <= latest["macd_hist"]
            )

            short_signal = (
                latest["close"] < latest["ema9"] < latest["ema21"] and
                prev["rsi9"] > 75 >= latest["rsi9"] and
                prev["macd_hist"] > 0 >= latest["macd_hist"]
            )

            user_positions = await drift_user.get_user_positions()
            has_position = any(
                p.market_index == MARKET_INDEX and abs(p.base_asset_amount) > 0
                for p in user_positions
            )

            if not has_position:
                collateral = await drift_user.get_total_collateral()
                if collateral <= 0:
                    print("No collateral - skipping")
                    await asyncio.sleep(CHECK_INTERVAL)
                    continue

                size_usd = collateral * LEVERAGE * RISK_PER_TRADE * 2
                size_base = int(size_usd / latest["close"] * 1e9)

                if long_signal:
                    await drift_client.open_position(PositionDirection.LONG(), size_base, MARKET_INDEX)
                    print(f"✅ LONG @ ~${latest['close']:.2f} | Size ~${size_usd:.0f}")
                    in_position = True
                    position_side = "LONG"

                elif short_signal:
                    await drift_client.open_position(PositionDirection.SHORT(), size_base, MARKET_INDEX)
                    print(f"✅ SHORT @ ~${latest['close']:.2f} | Size ~${size_usd:.0f}")
                    in_position = True
                    position_side = "SHORT"

            elif has_position and ((position_side == "LONG" and short_signal) or (position_side == "SHORT" and long_signal)):
                await drift_client.close_position(MARKET_INDEX)
                print(f"🔄 {position_side} CLOSED")
                in_position = False
                position_side = None

            await asyncio.sleep(CHECK_INTERVAL)

        except Exception as e:
            print(f"Loop error: {e}")
            await asyncio.sleep(30)

if __name__ == "__main__":
    asyncio.run(main())
