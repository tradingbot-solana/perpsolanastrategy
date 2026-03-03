import asyncio
import json
import os
import pandas as pd
import requests
from datetime import datetime
from dotenv import load_dotenv

from solana.rpc.async_api import AsyncClient
from anchorpy import Provider, Wallet
from solders.keypair import Keypair
from driftpy.client import DriftClient
from driftpy.drift_user import DriftUser
from driftpy.types import PositionDirection

load_dotenv()

# ────────────────────────────────────────────────
# Environment variables from Railway
# ────────────────────────────────────────────────
PRIVATE_KEY_JSON_STR = os.getenv("PRIVATE_KEY_JSON")
if PRIVATE_KEY_JSON_STR is None:
    raise ValueError("PRIVATE_KEY_JSON is not set in Railway variables!")

try:
    PRIVATE_KEY_JSON = json.loads(PRIVATE_KEY_JSON_STR)
except json.JSONDecodeError as e:
    raise ValueError(f"PRIVATE_KEY_JSON is not valid JSON: {e}")

RPC_URL = os.getenv("RPC_URL")
if RPC_URL is None:
    raise ValueError("RPC_URL is not set!")

BIRDEYE_API_KEY = os.getenv("BIRDEYE_API_KEY")
if BIRDEYE_API_KEY is None:
    raise ValueError("BIRDEYE_API_KEY is not set!")

MARKET_INDEX = int(os.getenv("MARKET_INDEX", "0"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.005"))
LEVERAGE = int(os.getenv("LEVERAGE", "8"))
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "60"))

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
        print(f"Failed to fetch candles from Birdeye: {e}")
        return None

def calculate_indicators(df):
    if df is None or len(df) < 50:
        print("Not enough candle data for indicators")
        return None

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
    print("=== Bot starting ===")

    # 1. Load wallet
    try:
        keypair = Keypair.from_bytes(bytes(PRIVATE_KEY_JSON))
        wallet = Wallet(keypair)
        print(f"Keypair loaded | Public key: {keypair.pubkey()}")
    except Exception as e:
        print(f"Failed to load keypair: {e}")
        return

    # 2. Connection & Provider
    connection = AsyncClient(RPC_URL)
    provider = Provider(connection, wallet)

    # 3. Drift Client
    print("Creating DriftClient...")
    drift_client = DriftClient(
        connection=connection,
        wallet=wallet,
        env="mainnet-beta",
        perp_market_indexes=[MARKET_INDEX]
    )

    try:
        await drift_client.subscribe()
        print("DriftClient subscribed successfully")
    except Exception as e:
        print(f"Drift subscribe failed: {e}")
        return

    # 4. Drift User & Collateral check
    drift_user = DriftUser(drift_client)
    try:
        collateral = await drift_user.get_total_collateral()
        print(f"Bot ready | Collateral: ${collateral:.2f}")
    except Exception as e:
        print(f"Failed to get collateral: {e}")
        return

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

            positions = await drift_user.get_user_positions()
            has_position = any(
                p.market_index == MARKET_INDEX and abs(p.base_asset_amount) > 0
                for p in positions
            )

            if not has_position:
                collateral = await drift_user.get_total_collateral()
                if collateral <= 0:
                    print("No collateral — skipping this cycle")
                    await asyncio.sleep(CHECK_INTERVAL)
                    continue

                size_usd = collateral * LEVERAGE * RISK_PER_TRADE * 2
                size_base = int(size_usd / latest["close"] * 1e9)

                if long_signal:
                    await drift_client.open_position(PositionDirection.LONG(), size_base, MARKET_INDEX)
                    print(f"LONG opened @ ~${latest['close']:.2f} | Size ${size_usd:.0f}")
                    in_position = True
                    position_side = "LONG"

                elif short_signal:
                    await drift_client.open_position(PositionDirection.SHORT(), size_base, MARKET_INDEX)
                    print(f"SHORT opened @ ~${latest['close']:.2f} | Size ${size_usd:.0f}")
                    in_position = True
                    position_side = "SHORT"

            elif has_position and ((position_side == "LONG" and short_signal) or (position_side == "SHORT" and long_signal)):
                await drift_client.close_position(MARKET_INDEX)
                print(f"{position_side} closed on opposite signal")
                in_position = False
                position_side = None

            await asyncio.sleep(CHECK_INTERVAL)

        except Exception as e:
            print(f"Main loop error: {e}")
            await asyncio.sleep(30)

if __name__ == "__main__":
    asyncio.run(main())
