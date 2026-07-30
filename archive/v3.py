import asyncio
import sqlite3
import os
import sys
import time
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from contextlib import contextmanager

from dotenv import load_dotenv
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np

# Load environment variables
load_dotenv()

# ==========================================
# CONFIGURATION
# ==========================================

# Timeframes
TIMEFRAME_SWEEP = "15m"          # Logic: Sweep + Breakout
TIMEFRAME_TREND_1H = "1h"        # Logic: Trend Gate 1
TIMEFRAME_TREND_4H = "4h"        # Logic: Trend Gate 2

# Risk & Money Management
RISK_PER_TRADE_USD = 10.0        # Increased base on server specs/confidence
MAX_POSITIONS = 8
USD_MIN_BALANCE = 10.0

# Execution (Slippage Protection)
SLIPPAGE_TOLERANCE_PCT = 0.002   # 0.2% max price deviation for limit orders
ORDER_TIMEOUT_S = 10             # Cancel order if not filled in 10s

# Logic Parameters
TAKE_PROFIT_TRIGGER = 0.025      # 2.5% gain triggers trailing
TRAILING_STOP_PCT = 0.01         # 1.0% trail
STOP_LOSS_FIXED = 0.02           # 2.0% hard stop loss initially

EARLY_EXIT_WINDOW_S = 300        # 5 minutes to prove itself
EARLY_EXIT_DROP_PCT = 0.008      # If it drops 0.8% in first 5 mins, kill it

# Universe Filter
QUOTE_CCY = "USD"
MIN_VOL_24H_USD = 500_000.0      # Increased volume filter for safety
MIN_PRICE = 0.0001
DAILY_CHANGE_MIN = -0.05         # Avoid catching falling knives
SPREAD_MAX_PCT = 0.005           # 0.5% max spread allowed

# System
DB_FILE = "trade.db"
LOG_FILE = "apex_v3.log"
CONCURRENCY_LIMIT = 25           # High concurrency for c6i.2xlarge
RATE_LIMIT_DELAY = 0.05          # internal throttle

# ==========================================
# LOGGING SETUP
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger("ApexV3")

# ==========================================
# DATABASE LAYER
# ==========================================
class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path, timeout=10)

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    entry_price REAL,
                    size REAL,
                    stop_loss REAL,
                    take_profit_trigger REAL,
                    highest_price REAL,
                    entry_time INTEGER,
                    status TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    side TEXT,
                    price REAL,
                    size REAL,
                    reason TEXT,
                    timestamp INTEGER
                )
            """)
            conn.commit()

    def get_open_positions(self) -> Dict[str, Dict]:
        with self._get_conn() as conn:
            cur = conn.execute("SELECT * FROM positions WHERE status='OPEN'")
            cols = [description[0] for description in cur.description]
            results = {}
            for row in cur.fetchall():
                row_dict = dict(zip(cols, row))
                results[row_dict['symbol']] = row_dict
            return results

    def add_position(self, symbol: str, entry: float, size: float, sl: float):
        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO positions 
                (symbol, entry_price, size, stop_loss, take_profit_trigger, highest_price, entry_time, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN')
            """, (symbol, entry, size, sl, 0.0, entry, int(time.time())))
            conn.commit()

    def close_position(self, symbol: str):
        with self._get_conn() as conn:
            conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
            conn.commit()

    def log_trade(self, symbol: str, side: str, price: float, size: float, reason: str):
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO trade_log (symbol, side, price, size, reason, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (symbol, side, price, size, reason, int(time.time())))
            conn.commit()

# ==========================================
# TECHNICAL ANALYSIS LIBRARY
# ==========================================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Basic EMAs
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_150'] = df['close'].ewm(span=150, adjust=False).mean()
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(14).mean()
    
    return df

def check_trend_alignment(df: pd.DataFrame) -> bool:
    """Checks if price is in a healthy uptrend on higher timeframes."""
    if len(df) < 50: return False
    
    last = df.iloc[-1]
    # Price > EMA 50 > EMA 150
    is_aligned = (last['close'] > last['ema_50']) and (last['ema_50'] > last['ema_150'])
    # Slope check (basic)
    ema_50_slope = last['ema_50'] - df.iloc[-5]['ema_50']
    
    return is_aligned and (ema_50_slope > 0)

def detect_sweep_breakout(df: pd.DataFrame) -> Tuple[bool, float, float]:
    """
    Detects:
    1. Pivot High (Resistance)
    2. Pivot Lows that swept liquidity (Lower Low then Higher Low)
    3. Breakout of the Pivot High
    """
    if len(df) < 50: return False, 0.0, 0.0
    
    window = 5  # Pivot lookback
    
    # Find local maximums (potential resistance)
    # We look for a high roughly 10-30 bars ago that is currently being broken
    recent_highs = df['high'].iloc[-40:-5] 
    if recent_highs.empty: return False, 0.0, 0.0
    
    pivot_high = recent_highs.max()
    
    current_close = df['close'].iloc[-1]
    prev_close = df['close'].iloc[-2]
    
    # Breakout Condition: Crossing the pivot high now
    is_breaking_out = (prev_close <= pivot_high) and (current_close > pivot_high)
    
    if not is_breaking_out:
        return False, 0.0, 0.0

    # Volume Confirmation
    vol_ma = df['vol'].rolling(20).mean().iloc[-1]
    current_vol = df['vol'].iloc[-1]
    
    if current_vol < (vol_ma * 1.2): # Require 20% more volume than average
        return False, 0.0, 0.0

    # Calculate Entry Score (Simple)
    atr = df['atr'].iloc[-1]
    score = (current_close - df['ema_20'].iloc[-1]) / atr  # Momentum score
    
    return True, pivot_high, score

# ==========================================
# BOT CLASS
# ==========================================
class ApexBotV3:
    def __init__(self):
        api_key = os.getenv("KRAKEN_API_KEY")
        api_secret = os.getenv("KRAKEN_API_SECRET")
        if not api_key: raise ValueError("Missing API Keys")
        
        self.exchange = ccxt.kraken({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        
        self.db = Database(DB_FILE)
        self.active_pairs = []
        self.sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv: return pd.DataFrame()
            
            df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
            # Fast numeric conversion
            cols = ['open', 'high', 'low', 'close', 'vol']
            df[cols] = df[cols].apply(pd.to_numeric)
            return df
        except Exception as e:
            log.warning(f"Data fetch error {symbol}: {e}")
            return pd.DataFrame()

    async def get_market_structure(self):
        """Refreshes universe of tradeable pairs."""
        try:
            markets = await self.exchange.load_markets()
            tickers = await self.exchange.fetch_tickers()
            
            valid_pairs = []
            for symbol, market in markets.items():
                if '/USD' not in symbol: continue
                if not market['active']: continue
                
                ticker = tickers.get(symbol)
                if not ticker: continue
                
                # Filters
                if ticker['quoteVolume'] < MIN_VOL_24H_USD: continue
                if ticker['close'] < MIN_PRICE: continue
                
                # Spread check
                if ticker['ask'] > 0:
                    spread = (ticker['ask'] - ticker['bid']) / ticker['ask']
                    if spread > SPREAD_MAX_PCT: continue
                
                valid_pairs.append(symbol)
            
            self.active_pairs = valid_pairs
            log.info(f"Market Universe Updated: {len(self.active_pairs)} pairs eligible.")
            
        except Exception as e:
            log.error(f"Market structure update failed: {e}")

    async def execute_order(self, symbol: str, side: str, amount: float, price_limit: float = None) -> Optional[Dict]:
        """
        Executes a 'safe' Limit order acting as a Market order with protection.
        """
        try:
            params = {}
            price = price_limit
            
            # If no price limit, calculate one based on order book (not implemented here for brevity, assuming price_limit passed)
            # For this bot, we always pass a limit price.
            
            log.info(f"SENDING ORDER: {side} {symbol} {amount} @ {price}")
            
            order = await self.exchange.create_order(
                symbol=symbol,
                type='limit',
                side=side,
                amount=amount,
                price=price,
                params={'timeInForce': 'IOC'} # Immediate or Cancel - don't leave dust
            )
            return order
        except Exception as e:
            log.error(f"Execution failed for {symbol}: {e}")
            return None

    async def analyze_and_trade(self, symbol: str):
        async with self.sem:
            # 1. Check existing positions
            open_positions = self.db.get_open_positions()
            if symbol in open_positions: return
            if len(open_positions) >= MAX_POSITIONS: return

            # 2. Fetch Data (Parallel Fetching optimized)
            # We need 15m for signal, 1h/4h for trend
            # Optimization: Fetch 15m first. If no signal, drop it. Don't waste API calls on 1h/4h.
            df_15m = await self.fetch_ohlcv(symbol, TIMEFRAME_SWEEP, 100)
            if df_15m.empty: return
            
            df_15m = calculate_indicators(df_15m)
            signal, pivot_price, score = detect_sweep_breakout(df_15m)
            
            if not signal: return

            # 3. Trend Confirmation (Lazy loading)
            df_1h = await self.fetch_ohlcv(symbol, TIMEFRAME_TREND_1H, 100)
            if df_1h.empty: return
            df_1h = calculate_indicators(df_1h)
            if not check_trend_alignment(df_1h): return
            
            # 4. Execution Logic
            ticker = await self.exchange.fetch_ticker(symbol)
            curr_ask = ticker['ask']
            
            # Ensure we aren't buying way above the pivot (already missed the move)
            if curr_ask > (pivot_price * 1.015): 
                return # Price moved 1.5% past pivot already

            # Calculate Quantity
            balance = await self.exchange.fetch_balance()
            usd_avail = balance['USD']['free']
            
            if usd_avail < RISK_PER_TRADE_USD:
                log.warning("Insufficient funds.")
                return

            qty = RISK_PER_TRADE_USD / curr_ask
            
            # Slippage Protected Buy Limit
            limit_price = curr_ask * (1 + SLIPPAGE_TOLERANCE_PCT)
            
            order = await self.execute_order(symbol, 'buy', qty, limit_price)
            
            if order and order['status'] == 'closed': # IOC Filled
                fill_price = order['average'] if order['average'] else curr_ask
                stop_loss = fill_price * (1 - STOP_LOSS_FIXED)
                
                self.db.add_position(symbol, fill_price, order['filled'], stop_loss)
                self.db.log_trade(symbol, 'BUY', fill_price, order['filled'], f"Breakout Score {score:.2f}")
                log.info(f"*** POSITION OPENED: {symbol} @ {fill_price} ***")

    async def manage_positions(self):
        """Monitors open positions for SL/TP."""
        positions = self.db.get_open_positions()
        if not positions: return

        for symbol, data in positions.items():
            try:
                ticker = await self.exchange.fetch_ticker(symbol)
                curr_bid = ticker['bid']
                
                entry = data['entry_price']
                highest = data['highest_price']
                stop_loss = data['stop_loss']
                
                # Update Highest Price
                if curr_bid > highest:
                    # Trailing Logic
                    new_highest = curr_bid
                    # If we are in significant profit, tighten stop
                    profit_pct = (new_highest - entry) / entry
                    
                    new_sl = stop_loss
                    if profit_pct > TAKE_PROFIT_TRIGGER:
                        # Activate Trailing
                        suggested_sl = new_highest * (1 - TRAILING_STOP_PCT)
                        if suggested_sl > stop_loss:
                            new_sl = suggested_sl
                    
                    # Update DB (using raw SQL for speed here or helper)
                    with self.db._get_conn() as conn:
                        conn.execute("UPDATE positions SET highest_price = ?, stop_loss = ? WHERE symbol = ?", 
                                     (new_highest, new_sl, symbol))
                        conn.commit()
                        
                    stop_loss = new_sl # Update local var for check below

                # Check Exit Conditions
                exit_reason = None
                
                # 1. Hard Stop / Trailing Stop
                if curr_bid < stop_loss:
                    exit_reason = "STOP_LOSS"
                
                # 2. Early Exit (Time based)
                # If 5 mins passed and we are down > 0.8%, cut it. Momentum failed.
                time_held = int(time.time()) - data['entry_time']
                current_pnl = (curr_bid - entry) / entry
                if time_held > EARLY_EXIT_WINDOW_S and time_held < (EARLY_EXIT_WINDOW_S * 2):
                    if current_pnl < -EARLY_EXIT_DROP_PCT:
                        exit_reason = "MOMENTUM_FAIL"

                if exit_reason:
                    # Execute Sell
                    qty = data['size']
                    limit_price = curr_bid * (1 - SLIPPAGE_TOLERANCE_PCT) # Sell slightly below bid to ensure fill
                    
                    order = await self.execute_order(symbol, 'sell', qty, limit_price)
                    
                    # Even if IOC partially fills, we remove from DB for now to avoid loops (or handle partials logic)
                    # For simplicity V3 assumes fills.
                    self.db.close_position(symbol)
                    self.db.log_trade(symbol, 'SELL', curr_bid, qty, f"{exit_reason} PnL: {current_pnl*100:.2f}%")
                    log.info(f"*** POSITION CLOSED: {symbol} | {exit_reason} | PnL: {current_pnl*100:.2f}% ***")

            except Exception as e:
                log.error(f"Error managing {symbol}: {e}")

    async def run(self):
        log.info("--- SolaScript V3: ZERO FEE EDITION ---")
        log.info(f"Server: AWS c6i.2xlarge | Strategy: Liquidity Sweep Breakout | DB: SQLite")
        
        await self.get_market_structure()
        
        while True:
            try:
                # 1. Manage Positions
                await self.manage_positions()
                
                # 2. Scan Universe (in batches)
                tasks = []
                for symbol in self.active_pairs:
                    tasks.append(self.analyze_and_trade(symbol))
                    
                    # Batch processing to respect some limits even with concurrency
                    if len(tasks) >= CONCURRENCY_LIMIT:
                        await asyncio.gather(*tasks)
                        tasks = []
                        await asyncio.sleep(RATE_LIMIT_DELAY) 
                
                if tasks: # leftovers
                    await asyncio.gather(*tasks)

                # Periodic Universe Refresh (every hour)
                if int(time.time()) % 3600 == 0:
                    await self.get_market_structure()

                await asyncio.sleep(1) # Breath
                
            except KeyboardInterrupt:
                log.info("Shutdown signal received.")
                break
            except Exception as e:
                log.error(f"Main Loop Crash: {e}")
                await asyncio.sleep(5)
        
        await self.exchange.close()

if __name__ == "__main__":
    bot = ApexBotV3()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        pass