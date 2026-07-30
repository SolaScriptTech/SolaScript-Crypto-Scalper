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
TIMEFRAME_TREND_4H = "4h"        # Logic: Trend Gate 2 (Now Enforced)

# Risk & Money Management
RISK_PER_TRADE_USD = 10.0        # Risk per trade
MAX_POSITIONS = 8
USD_MIN_BALANCE = 10.0           # Minimum free USD required to attempt trades

# Execution (Slippage Protection)
SLIPPAGE_TOLERANCE_PCT = 0.002   # 0.2% max price deviation for limit orders
ORDER_TIMEOUT_S = 10             # (Unused for IOC, kept for future expansion)

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
UNIVERSE_REFRESH_INTERVAL = 3600 # Refresh universe every hour

# System
DB_FILE = "trade.db"
LOG_FILE = "apex_v3.log"
CONCURRENCY_LIMIT = 25           
RATE_LIMIT_DELAY = 0.05          

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

    def update_position_size(self, symbol: str, new_size: float):
        with self._get_conn() as conn:
            conn.execute("UPDATE positions SET size = ? WHERE symbol = ?", (new_size, symbol))
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
    if len(df) < 50: return False, 0.0, 0.0
    
    # Find local maximums (potential resistance)
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
        self.last_universe_refresh = 0
        self.sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv: return pd.DataFrame()
            
            df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
            cols = ['open', 'high', 'low', 'close', 'vol']
            df[cols] = df[cols].apply(pd.to_numeric)
            return df
        except Exception as e:
            log.warning(f"Data fetch error {symbol}: {e}")
            return pd.DataFrame()

    async def get_market_structure(self):
        """Refreshes universe of tradeable pairs."""
        try:
            await self.exchange.load_markets()
            tickers = await self.exchange.fetch_tickers()
            
            valid_pairs = []
            for symbol, market in self.exchange.markets.items():
                if '/USD' not in symbol: continue
                if not market['active']: continue
                if not market.get('spot', True): continue
                
                ticker = tickers.get(symbol)
                if not ticker: continue
                
                # Filters
                if ticker['quoteVolume'] < MIN_VOL_24H_USD: continue
                if ticker['close'] < MIN_PRICE: continue
                
                # Filter falling knives
                if ticker['percentage'] is not None and (float(ticker['percentage']) / 100) < DAILY_CHANGE_MIN:
                    continue

                # Spread check
                if ticker['ask'] > 0:
                    spread = (ticker['ask'] - ticker['bid']) / ticker['ask']
                    if spread > SPREAD_MAX_PCT: continue
                
                valid_pairs.append(symbol)
            
            self.active_pairs = valid_pairs
            self.last_universe_refresh = int(time.time())
            log.info(f"Market Universe Updated: {len(self.active_pairs)} pairs eligible.")
            
        except Exception as e:
            log.error(f"Market structure update failed: {e}")

    async def execute_order_safe(self, symbol: str, side: str, amount: float, price_limit: float = None) -> Tuple[Optional[Dict], float]:
        """
        Executes a precision-checked Limit order.
        Returns: (order_object, filled_qty)
        """
        try:
            # 1. Precision Enforcement
            amount = self.exchange.amount_to_precision(symbol, amount)
            price = self.exchange.price_to_precision(symbol, price_limit)
            
            # 2. Min Limits Check
            market = self.exchange.market(symbol)
            min_amount = market['limits']['amount']['min']
            min_cost = market['limits']['cost']['min']
            
            if float(amount) < min_amount:
                log.warning(f"Order skipped {symbol}: Amount {amount} < Min {min_amount}")
                return None, 0.0
                
            if (float(amount) * float(price)) < min_cost:
                log.warning(f"Order skipped {symbol}: Cost < Min Cost {min_cost}")
                return None, 0.0

            log.info(f"SENDING ORDER: {side} {symbol} {amount} @ {price}")
            
            # 3. Create Order
            order = await self.exchange.create_order(
                symbol=symbol,
                type='limit',
                side=side,
                amount=amount,
                price=price,
                params={'timeInForce': 'IOC'} 
            )
            
            # 4. Confirm Status (Don't trust response blindly)
            if order.get('status') == 'closed':
                 # IOC fully filled usually returns closed immediately
                 return order, float(order['filled'])
            
            # Fetch to be sure for partials
            try:
                confirmed_order = await self.exchange.fetch_order(order['id'], symbol)
                return confirmed_order, float(confirmed_order['filled'])
            except Exception:
                # If fetch fails, fall back to initial response
                return order, float(order.get('filled', 0.0))

        except Exception as e:
            log.error(f"Execution failed for {symbol}: {e}")
            return None, 0.0

    async def analyze_and_trade(self, symbol: str):
        async with self.sem:
            # 1. Check existing positions
            open_positions = self.db.get_open_positions()
            if symbol in open_positions: return
            if len(open_positions) >= MAX_POSITIONS: return

            # 2. Fetch Data (15m)
            df_15m = await self.fetch_ohlcv(symbol, TIMEFRAME_SWEEP, 100)
            if df_15m.empty: return
            
            df_15m = calculate_indicators(df_15m)
            signal, pivot_price, score = detect_sweep_breakout(df_15m)
            
            if not signal: return

            # 3. Trend Confirmation 1H
            df_1h = await self.fetch_ohlcv(symbol, TIMEFRAME_TREND_1H, 100)
            if df_1h.empty: return
            df_1h = calculate_indicators(df_1h)
            if not check_trend_alignment(df_1h): return

            # 4. Trend Confirmation 4H (ADDED)
            df_4h = await self.fetch_ohlcv(symbol, TIMEFRAME_TREND_4H, 100)
            if df_4h.empty: return
            df_4h = calculate_indicators(df_4h)
            if not check_trend_alignment(df_4h): return
            
            # 5. Execution Logic
            ticker = await self.exchange.fetch_ticker(symbol)
            curr_ask = ticker['ask']
            
            if curr_ask > (pivot_price * 1.015): 
                return 

            # Balance Check
            balance = await self.exchange.fetch_balance()
            usd_avail = balance[QUOTE_CCY]['free']
            
            if usd_avail < USD_MIN_BALANCE: # Enforce min balance
                return
            
            if usd_avail < RISK_PER_TRADE_USD:
                return

            qty = RISK_PER_TRADE_USD / curr_ask
            limit_price = curr_ask * (1 + SLIPPAGE_TOLERANCE_PCT)
            
            order, filled = await self.execute_order_safe(symbol, 'buy', qty, limit_price)
            
            if filled > 0:
                fill_price = order['average'] if order['average'] else curr_ask
                stop_loss = fill_price * (1 - STOP_LOSS_FIXED)
                
                self.db.add_position(symbol, fill_price, filled, stop_loss)
                self.db.log_trade(symbol, 'BUY', fill_price, filled, f"Breakout Score {score:.2f}")
                log.info(f"*** POSITION OPENED: {symbol} @ {fill_price} ***")

    async def manage_positions(self):
        positions = self.db.get_open_positions()
        if not positions: return

        for symbol, data in positions.items():
            try:
                ticker = await self.exchange.fetch_ticker(symbol)
                curr_bid = ticker['bid']
                
                entry = data['entry_price']
                highest = data['highest_price']
                stop_loss = data['stop_loss']
                current_size = data['size']
                
                # Update Highest Price
                if curr_bid > highest:
                    new_highest = curr_bid
                    profit_pct = (new_highest - entry) / entry
                    new_sl = stop_loss
                    if profit_pct > TAKE_PROFIT_TRIGGER:
                        suggested_sl = new_highest * (1 - TRAILING_STOP_PCT)
                        if suggested_sl > stop_loss:
                            new_sl = suggested_sl
                    
                    with self.db._get_conn() as conn:
                        conn.execute("UPDATE positions SET highest_price = ?, stop_loss = ? WHERE symbol = ?", 
                                     (new_highest, new_sl, symbol))
                        conn.commit()     
                    stop_loss = new_sl 

                # Check Exit Conditions
                exit_reason = None
                if curr_bid < stop_loss:
                    exit_reason = "STOP_LOSS"
                
                time_held = int(time.time()) - data['entry_time']
                current_pnl = (curr_bid - entry) / entry
                if time_held > EARLY_EXIT_WINDOW_S and time_held < (EARLY_EXIT_WINDOW_S * 2):
                    if current_pnl < -EARLY_EXIT_DROP_PCT:
                        exit_reason = "MOMENTUM_FAIL"

                if exit_reason:
                    qty = current_size
                    limit_price = curr_bid * (1 - SLIPPAGE_TOLERANCE_PCT)
                    
                    order, filled = await self.execute_order_safe(symbol, 'sell', qty, limit_price)
                    
                    # Update DB based on what ACTUALLY happened
                    if filled >= qty * 0.99: # Almost full fill
                        self.db.close_position(symbol)
                        self.db.log_trade(symbol, 'SELL', curr_bid, filled, f"{exit_reason} PnL: {current_pnl*100:.2f}%")
                        log.info(f"*** POSITION CLOSED: {symbol} | {exit_reason} ***")
                    elif filled > 0:
                        # Partial Fill - Update remaining size
                        remaining = qty - filled
                        # Check if remaining is dust
                        market = self.exchange.market(symbol)
                        if remaining < market['limits']['amount']['min']:
                             self.db.close_position(symbol) # Close it out effectively (abandon dust)
                        else:
                             self.db.update_position_size(symbol, remaining)
                             log.info(f"*** PARTIAL SELL: {symbol} | Filled: {filled} | Rem: {remaining} ***")

            except Exception as e:
                log.error(f"Error managing {symbol}: {e}")

    async def run(self):
        log.info("--- SolaScript V3: REFACTORED EDITION ---")
        log.info(f"Server: AWS c6i.2xlarge | Strategy: 4H+1H Trend + Sweep | DB: SQLite")
        
        await self.get_market_structure()
        
        while True:
            try:
                # 1. Manage Positions
                await self.manage_positions()
                
                # 2. Scan Universe (in batches)
                tasks = []
                for symbol in self.active_pairs:
                    tasks.append(self.analyze_and_trade(symbol))
                    
                    if len(tasks) >= CONCURRENCY_LIMIT:
                        await asyncio.gather(*tasks)
                        tasks = []
                        await asyncio.sleep(RATE_LIMIT_DELAY) 
                
                if tasks: 
                    await asyncio.gather(*tasks)
                
                # Heartbeat Log
                print(f"[{time.strftime('%H:%M:%S')}] Scanned {len(self.active_pairs)} pairs... Active Positions: {len(self.db.get_open_positions())}")

                # Periodic Universe Refresh (Corrected Logic)
                if (int(time.time()) - self.last_universe_refresh) > UNIVERSE_REFRESH_INTERVAL:
                    await self.get_market_structure()

                await asyncio.sleep(1) 
                
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