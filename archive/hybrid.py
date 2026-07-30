"""
SOLASCRIPT V4: THE UNICORN HUNTER
---------------------------------
Server: AWS c6i.2xlarge ((US-East-1)(Virginia))
Target: Kraken Exchange (closest proximity VM)
Logic: Time-Based Dual Strategy (Unicorn Momentum vs. Deep Research)

PHASE 1 (03:00 - 12:00 PST): UNICORN MODE
- Aggressive scans for outlier volume (RVOL).
- Enters on Bollinger Band Breakouts.
- Exits fast on momentum loss (Sliding Stop).

PHASE 2 (12:00 - 03:00 PST): RESEARCH MODE
- Conservative scanning for RSI dips in uptrends.
- Enters on pullbacks (buying the red).
- Exits on technical targets.
"""

import asyncio
import sqlite3
import os
import sys
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, Tuple, List
import math

# Third-party libraries (Install: pip install ccxt pandas numpy python-dotenv)
from dotenv import load_dotenv
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
import pandas_ta as ta  # Ensure pandas_ta is installed

"""
-python3 -m venv venv
	-source venv/bin/activate
		-pip install ccxt pandas python-dotenv
"""

# Load environment variables
load_dotenv()

"""
This installs the system tools (pip) and all the specific Python libraries your super_sleuth.py needs (ccxt, pandas, pandas_ta, aiohttp).
-------------------------------------------------------------------------------------------------------------------------------------------
# 1. Update Ubuntu and install Pip (the Python installer)
sudo apt update && sudo apt install -y python3-pip python3-venv sqlite3

# 2. Install the Trading Libraries (using the break-system flag for Ubuntu 24.04)
pip install ccxt pandas pandas_ta aiohttp numpy --break-system-packages

# 3. Verify everything loaded correctly
python3 -c "import ccxt; import pandas_ta; print('SUCCESS: All libraries installed.')"
"""

# ==========================================
# ⚙️ GLOBAL CONFIGURATION
# ==========================================

# --- Time Settings (PST) ---
TIMEZONE_OFFSET = -8  # PST is UTC-8 (Adjust for Daylight Savings if needed)
UNICORN_START_HOUR = 3
UNICORN_END_HOUR = 12

# --- Budget & Risk ---
# Phase 1: Unicorns (Aggressive, small sizes, wide net)
CONFIG_UNICORN = {
    "MAX_POSITIONS": 8,
    "TRADE_SIZE_USD": 7.0,
    "TIMEFRAME": "5m",            # Fast signal detection
    "MIN_VOL_24H": 100_000.0,     # Lower volume filter to catch new pumps
    "SLIPPAGE_TOLERANCE": 0.02,   # 2% slippage allowed (these move fast)
    "STOP_LOSS_FIXED": 0.05,      # 5% initial hard stop
    "TRAILING_START": 0.03,       # Start trailing after 3% profit
    "TRAILING_STEP": 0.015,       # 1.5% trail distance
    "TIME_LIMIT_S": 900           # 15 mins: If it doesn't pump, cut it.
}

# Phase 2: Research (Conservative, larger sizes, strict quality)
CONFIG_RESEARCH = {
    "MAX_POSITIONS": 6,
    "TRADE_SIZE_USD": 10.0,
    "TIMEFRAME": "15m",           # Slower, more reliable signals
    "MIN_VOL_24H": 500_000.0,     # High liquidity only
    "SLIPPAGE_TOLERANCE": 0.005,  # 0.5% slippage strict
    "STOP_LOSS_FIXED": 0.03,      # 3% hard stop
    "TAKE_PROFIT": 0.04,          # 4% target
    "TIME_LIMIT_S": 14400         # 4 hours hold max
}

# --- System ---
DB_FILE = "apex_unicorn.db"
LOG_FILE = "unicorn_engine.log"
CONCURRENCY = 50                  # High concurrency for c6i.2xlarge
RATE_LIMIT_DELAY = 0.1            # Kraken API pacing

# ==========================================
# 📝 LOGGING SETUP
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger("UnicornBot")

# ==========================================
# 💾 DATABASE ENGINE (Auto-Healing)
# ==========================================
class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path, timeout=30)

    def _init_db(self):
        with self._get_conn() as conn:
            # Positions Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    entry_price REAL,
                    size REAL,
                    stop_loss REAL,
                    highest_price REAL,
                    entry_time INTEGER,
                    mode TEXT,
                    status TEXT
                )
            """)
            # Trade Log
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    side TEXT,
                    price REAL,
                    size REAL,
                    reason TEXT,
                    profit_usd REAL,
                    timestamp INTEGER
                )
            """)
            conn.commit()

    def get_active_positions(self) -> Dict[str, Dict]:
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute("SELECT * FROM positions WHERE status='OPEN'")
            return {row['symbol']: dict(row) for row in cur.fetchall()}

    def open_position(self, symbol, entry, size, sl, mode):
        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO positions 
                (symbol, entry_price, size, stop_loss, highest_price, entry_time, mode, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN')
            """, (symbol, entry, size, sl, entry, int(time.time()), mode))
            conn.commit()

    def update_position(self, symbol, highest_price, stop_loss):
        with self._get_conn() as conn:
            conn.execute("""
                UPDATE positions SET highest_price = ?, stop_loss = ? WHERE symbol = ?
            """, (highest_price, stop_loss, symbol))
            conn.commit()

    def close_position(self, symbol):
        with self._get_conn() as conn:
            conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
            conn.commit()

    def log_trade(self, symbol, side, price, size, reason, profit=0.0):
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO trade_log (symbol, side, price, size, reason, profit_usd, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (symbol, side, price, size, reason, profit, int(time.time())))
            conn.commit()

# ==========================================
# 🧠 INTELLIGENCE LAYER
# ==========================================
class Analyzer:
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Applies technical indicators used by both modes."""
        # 1. Trend: EMAs
        df['ema_9'] = df.ta.ema(length=9)
        df['ema_20'] = df.ta.ema(length=20)
        df['ema_50'] = df.ta.ema(length=50)
        df['ema_200'] = df.ta.ema(length=200)

        # 2. Momentum: RSI & MACD
        df['rsi'] = df.ta.rsi(length=14)
        macd = df.ta.macd(fast=12, slow=26, signal=9)
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        df['macd_hist'] = macd['MACDh_12_26_9']

        # 3. Volatility: Bollinger Bands & ATR
        bb = df.ta.bbands(length=20, std=2.0)
        df['bb_upper'] = bb['BBU_20_2.0']
        df['bb_lower'] = bb['BBL_20_2.0']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close'] # Squeeze metric
        df['atr'] = df.ta.atr(length=14)

        # 4. Volume: VWAP & Relative Volume (RVOL)
        df['vwap'] = df.ta.vwap()
        # Simple RVOL: Current Vol / SMA(Vol, 20)
        df['vol_ma'] = df['volume'].rolling(20).mean()
        df['rvol'] = df['volume'] / df['vol_ma']
        
        return df

    @staticmethod
    def detect_unicorn(df: pd.DataFrame) -> Tuple[bool, str]:
        """
        UNICORN LOGIC (3AM - 12PM):
        - High relative volume (pumping)
        - Price breaking out of BB (Volatility expansion)
        - Price above VWAP (Institutional support)
        """
        if len(df) < 50: return False, ""
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # 1. Volume Spike: Is the crowd here?
        has_volume = last['rvol'] > 3.0  # 3x normal volume
        
        # 2. Breakout: Is price exploding out of bands?
        # We want the Candle Close to be ABOVE the Upper Band
        is_breakout = last['close'] > last['bb_upper']
        
        # 3. Trend: Is it a valid pump, not a chop?
        # Price > VWAP and MACD Histogram is Green and increasing
        is_trending = (last['close'] > last['vwap']) and (last['macd_hist'] > prev['macd_hist']) and (last['macd_hist'] > 0)
        
        if has_volume and is_breakout and is_trending:
            score = last['rvol']
            return True, f"UNICORN_PUMP (RVOL: {score:.1f})"
            
        return False, ""

    @staticmethod
    def detect_research_dip(df: pd.DataFrame) -> Tuple[bool, str]:
        """
        RESEARCH LOGIC (12PM - 3AM):
        - Long Term Trend (4H/1H implied via EMA alignment) is UP.
        - Short Term (15m) is OVERSOLD (RSI Dip).
        - "Buy the Red"
        """
        if len(df) < 200: return False, ""
        
        last = df.iloc[-1]
        
        # 1. Macro Trend Check (EMA 50 > EMA 200)
        # We want to buy dips in an uptrend, not catch falling knives.
        uptrend = last['ema_50'] > last['ema_200']
        
        # 2. The Dip (RSI < 35)
        # Buying when everyone else is selling
        oversold = last['rsi'] < 35
        
        # 3. Reversal Sign (Price holding above 200 EMA support?)
        # Ensure we haven't crashed through the floor
        valid_support = last['close'] > last['ema_200']
        
        if uptrend and oversold and valid_support:
            return True, f"RESEARCH_DIP (RSI: {last['rsi']:.1f})"
            
        return False, ""

# ==========================================
# 🤖 TRADING BOT ENGINE
# ==========================================
class UnicornEngine:
    def __init__(self):
        # API Keys
        api_key = os.getenv("KRAKEN_API_KEY")
        api_secret = os.getenv("KRAKEN_API_SECRET")
        if not api_key: raise ValueError("Missing KRAKEN_API_KEY or KRAKEN_API_SECRET in .env")
        
        # Exchange Setup
        self.exchange = ccxt.kraken({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        
        self.db = Database(DB_FILE)
        self.sem = asyncio.Semaphore(CONCURRENCY)
        self.active_pairs = []
        self.last_universe_refresh = 0

    def get_current_mode(self) -> str:
        """Determines if we are in UNICORN or RESEARCH mode based on PST."""
        utc_now = datetime.now(timezone.utc)
        pst_now = utc_now + timedelta(hours=TIMEZONE_OFFSET)
        hour = pst_now.hour
        
        if UNICORN_START_HOUR <= hour < UNICORN_END_HOUR:
            return "UNICORN"
        return "RESEARCH"

    async def get_config(self):
        mode = self.get_current_mode()
        return mode, (CONFIG_UNICORN if mode == "UNICORN" else CONFIG_RESEARCH)

    async def update_universe(self):
        """Fetches tradeable pairs, filtering by volume to remove dead coins."""
        try:
            mode, config = await self.get_config()
            await self.exchange.load_markets()
            tickers = await self.exchange.fetch_tickers()
            
            valid_pairs = []
            for symbol, data in tickers.items():
                # Basic Filters
                if '/USD' not in symbol: continue
                if 'quoteVolume' not in data or data['quoteVolume'] is None: continue
                
                # Volume Filter
                if data['quoteVolume'] < config['MIN_VOL_24H']: continue
                
                # Spread Filter (Don't trade if spread > 1%)
                if data['ask'] > 0 and data['bid'] > 0:
                    spread = (data['ask'] - data['bid']) / data['ask']
                    if spread > 0.01: continue
                else:
                    continue

                valid_pairs.append(symbol)
            
            self.active_pairs = valid_pairs
            log.info(f"🌌 UNIVERSE REFRESH ({mode}): {len(self.active_pairs)} pairs loaded.")
            self.last_universe_refresh = time.time()
            
        except Exception as e:
            log.error(f"Universe refresh failed: {e}")

    async def execute_buy(self, symbol, reason, config, mode):
        try:
            # 1. Check Balance
            balance = await self.exchange.fetch_balance()
            usd_free = balance['USD']['free']
            
            if usd_free < config['TRADE_SIZE_USD']:
                log.warning(f"Skipping {symbol}: Insufficient funds (${usd_free:.2f})")
                return

            # 2. Calculate Size
            ticker = await self.exchange.fetch_ticker(symbol)
            price = ticker['ask']
            amount = config['TRADE_SIZE_USD'] / price
            
            # 3. Place Limit Order (Crossing Spread)
            # We buy slightly above ask to ensure fill (aggressive) but protected cap
            limit_price = price * (1 + config['SLIPPAGE_TOLERANCE'])
            amount = self.exchange.amount_to_precision(symbol, amount)
            limit_price = self.exchange.price_to_precision(symbol, limit_price)
            
            log.info(f"🚀 BUYING {symbol} ({mode}) | {reason} | Amt: {amount} @ {limit_price}")
            
            order = await self.exchange.create_order(
                symbol, 'limit', 'buy', amount, limit_price, params={'timeInForce': 'IOC'}
            )
            
            # 4. Record Position
            if order['status'] == 'closed' or float(order['filled']) > 0:
                fill_price = float(order['average']) if order['average'] else price
                fill_qty = float(order['filled'])
                
                # Set Initial Stop Loss
                sl_price = fill_price * (1 - config['STOP_LOSS_FIXED'])
                
                self.db.open_position(symbol, fill_price, fill_qty, sl_price, mode)
                self.db.log_trade(symbol, 'BUY', fill_price, fill_qty, reason)
                log.info(f"✅ FILLED {symbol} @ {fill_price}")
            else:
                log.warning(f"❌ Order cancelled/unfilled for {symbol}")

        except Exception as e:
            log.error(f"Buy Execution Failed {symbol}: {e}")

    async def execute_sell(self, symbol, pos, reason):
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            bid = ticker['bid']
            qty = pos['size']
            
            # Sell slightly below bid to instant fill
            limit_price = bid * 0.99 
            
            qty = self.exchange.amount_to_precision(symbol, qty)
            limit_price = self.exchange.price_to_precision(symbol, limit_price)
            
            log.info(f"🔻 SELLING {symbol} | {reason} | Amt: {qty} @ {limit_price}")
            
            order = await self.exchange.create_order(
                symbol, 'limit', 'sell', qty, limit_price, params={'timeInForce': 'IOC'}
            )
            
            # Log Profit
            entry = pos['entry_price']
            profit_pct = (bid - entry) / entry
            profit_usd = (bid - entry) * float(qty)
            
            self.db.close_position(symbol)
            self.db.log_trade(symbol, 'SELL', bid, qty, reason, profit_usd)
            log.info(f"💰 CLOSED {symbol}: {profit_pct*100:.2f}% (${profit_usd:.2f})")
            
        except Exception as e:
            log.error(f"Sell Execution Failed {symbol}: {e}")

    async def scan_market(self):
        """The Hunter: Scans the universe for signals."""
        mode, config = await self.get_config()
        active_pos = self.db.get_active_positions()
        
        # Max Position Check
        if len(active_pos) >= config['MAX_POSITIONS']:
            return

        async with self.sem:
            for symbol in self.active_pairs:
                if symbol in active_pos: continue # Skip if owned
                
                try:
                    # Fetch Candles
                    ohlcv = await self.exchange.fetch_ohlcv(symbol, config['TIMEFRAME'], limit=100)
                    if not ohlcv: continue
                    
                    df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
                    df = Analyzer.calculate_indicators(df)
                    
                    signal = False
                    reason = ""
                    
                    if mode == "UNICORN":
                        signal, reason = Analyzer.detect_unicorn(df)
                    else:
                        signal, reason = Analyzer.detect_research_dip(df)
                    
                    if signal:
                        await self.execute_buy(symbol, reason, config, mode)
                        # Quick sleep to let DB update before next scan
                        await asyncio.sleep(0.5)
                        
                        # Re-check max positions to avoid over-buying in a loop
                        if len(self.db.get_active_positions()) >= config['MAX_POSITIONS']:
                            break
                            
                except Exception as e:
                    # Silent fail on data errors to keep loop fast
                    continue
                
                # Tiny throttle per symbol to respect rate limits inside semaphore
                await asyncio.sleep(RATE_LIMIT_DELAY)

    async def manage_positions(self):
        """The Manager: Handles Sliding Stops and Time Limits."""
        active_pos = self.db.get_active_positions()
        if not active_pos: return
        
        mode, config = await self.get_config()
        
        for symbol, pos in active_pos.items():
            try:
                ticker = await self.exchange.fetch_ticker(symbol)
                curr_price = ticker['bid']
                entry_price = pos['entry_price']
                highest_price = pos['highest_price']
                stop_loss = pos['stop_loss']
                
                # 1. Update High Water Mark
                if curr_price > highest_price:
                    highest_price = curr_price
                    # Sliding Scale Logic for Unicorns
                    # If profit > 3%, trail by 1.5%. If profit > 10%, trail by 0.5% (Capture the parabolic run)
                    profit_pct = (highest_price - entry_price) / entry_price
                    
                    new_sl = stop_loss
                    
                    if mode == "UNICORN":  # unicorn I came up with that
                        if profit_pct > 0.10: # Mega Pump
                            new_sl = highest_price * 0.99 # Tight 1% trail
                        elif profit_pct > config['TRAILING_START']:
                            new_sl = highest_price * (1 - config['TRAILING_STEP'])
                    else:
                        # Research Mode: Standard Take Profit or Trail
                        if profit_pct > config['TAKE_PROFIT']:
                            await self.execute_sell(symbol, pos, "TAKE_PROFIT_HIT")
                            continue

                    if new_sl > stop_loss:
                        stop_loss = new_sl
                        self.db.update_position(symbol, highest_price, stop_loss)

                # 2. Check Stop Loss
                if curr_price < stop_loss:
                    await self.execute_sell(symbol, pos, "STOP_LOSS_HIT")
                    continue
                
                # 3. Momentum Check (Unicorn Special)
                # If we bought a unicorn, and it prints 2 Red Candles below entry immediately, kill it.
                if mode == "UNICORN":
                    # (Simplified check: Current price < Entry for X time)
                    time_held = int(time.time()) - pos['entry_time']
                    if time_held > 180 and curr_price < entry_price: # 3 mins and red?
                         await self.execute_sell(symbol, pos, "MOMENTUM_FAIL_EARLY")
                         continue

                # 4. Time Limit Expiry
                time_held = int(time.time()) - pos['entry_time']
                if time_held > config['TIME_LIMIT_S']:
                    pnl_pct = (curr_price - entry_price) / entry_price
                    if pnl_pct < 0:
                        await self.execute_sell(symbol, pos, "TIME_LIMIT_EXPIRED_CUT")
                    elif pnl_pct > 0:
                        await self.execute_sell(symbol, pos, "TIME_LIMIT_EXPIRED_PROFIT")

            except Exception as e:
                log.error(f"Error managing {symbol}: {e}")

    async def run(self):
        log.info("🦄 SolaScript V4: UNICORN ENGINE STARTED")
        log.info(f"Server: AWS c6i.2xlarge | Mode: DYNAMIC | Concurrency: {CONCURRENCY}")
        
        await self.update_universe()
        
        while True:
            try:
                # 1. Determine Mode
                current_mode = self.get_current_mode()
                
                # 2. Periodic Universe Refresh (Every Hour)
                if time.time() - self.last_universe_refresh > 3600:
                    await self.update_universe()

                # 3. Parallel Tasks: Scan & Manage
                scan_task = asyncio.create_task(self.scan_market())
                manage_task = asyncio.create_task(self.manage_positions())
                
                await asyncio.gather(scan_task, manage_task)
                
                # Heartbeat that continues to keep the user informed on what its doing
                active = len(self.db.get_active_positions())
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Mode: {current_mode} | Active Trades: {active}")
                
                await asyncio.sleep(1) # Short breath
                
            except KeyboardInterrupt:
                log.info("Manual Shutdown.")
                break
            except Exception as e:
                log.error(f"CRITICAL LOOP FAILURE: {e}")
                await asyncio.sleep(5)
        
        await self.exchange.close()

if __name__ == "__main__":
    bot = UnicornEngine()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        pass