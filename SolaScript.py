import asyncio
import sqlite3
import os
import sys
import time
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Dict, Any, Tuple, List

# Third-party
from dotenv import load_dotenv
import ccxt.async_support as ccxt
import pandas as pd
import pandas_ta as ta

load_dotenv()

# ==========================================
# ⚙️ GLOBAL SETTINGS
# ==========================================
CONCURRENCY = 5
RATE_LIMIT_SLEEP = 1.0
DB_FILE = "apex_v5.db"
LOG_FILE = "apex_v5.log"

TZ_PST = ZoneInfo("America/Los_Angeles")
UNICORN_START = 3
UNICORN_END = 12

CONFIG_UNICORN = {
    "NAME": "🦄 UNICORN_HUNTER",
    "MAX_POS": 8,
    "SIZE": 7.0,
    "TIMEFRAME": "5m",
    "SLIPPAGE": 0.02,
    "STOP_LOSS": 0.05,
    "TAKE_PROFIT": 0.15,
    "FETCH_LIMIT": 120,
    "MIN_VOL_24H": 50_000.0,
    "TRAIL_AFTER_PROFIT": 0.10,
    "TRAIL_OFFSET": 0.02,
    "STAGNANT_S": 900,
    "STAGNANT_MIN_GAIN": 0.01,
}

CONFIG_RESEARCH = {
    "NAME": "🔬 DEEP_RESEARCH",
    "MAX_POS": 6,
    "SIZE": 10.0,
    "TIMEFRAME": "15m",
    "SLIPPAGE": 0.01,
    "STOP_LOSS": 0.03,
    "TAKE_PROFIT": 0.08,
    "FETCH_LIMIT": 300,
    "MIN_VOL_24H": 500_000.0,
}

# ==========================================
# 📝 LOGGING & DB
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("ApexBot")


class Database:
    def __init__(self, path: str):
        self.path = path
        self._init_db()

    def _conn(self):
        # FIX: Open a fresh connection for every operation to avoid threading issues
        return sqlite3.connect(self.path, timeout=30)

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    entry_price REAL,
                    size REAL,
                    stop_loss REAL,
                    highest_price REAL,
                    entry_time INTEGER,
                    mode TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    side TEXT,
                    price REAL,
                    size REAL,
                    profit REAL,
                    reason TEXT,
                    timestamp INTEGER
                )
            """)
            conn.commit()

    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        with self._conn() as conn:
            cur = conn.execute("SELECT * FROM positions")
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
            return {row[0]: dict(zip(cols, row)) for row in rows}

    def add_position(self, sym: str, price: float, size: float, sl: float, mode: str):
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO positions VALUES (?,?,?,?,?,?,?)",
                (sym, price, size, sl, price, int(time.time()), mode),
            )
            conn.commit()

    def update_position(self, sym: str, highest: float, sl: float):
        with self._conn() as conn:
            conn.execute(
                "UPDATE positions SET highest_price = ?, stop_loss = ? WHERE symbol = ?",
                (highest, sl, sym),
            )
            conn.commit()

    def delete_position(self, sym: str):
        with self._conn() as conn:
            conn.execute("DELETE FROM positions WHERE symbol = ?", (sym,))
            conn.commit()

    def log_trade(self, sym: str, side: str, price: float, size: float, profit: float, reason: str):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO trades (symbol, side, price, size, profit, reason, timestamp) VALUES (?,?,?,?,?,?,?)",
                (sym, side, price, size, profit, reason, int(time.time())),
            )
            conn.commit()


# ==========================================
# 🧠 LOGIC CORE
# ==========================================
class Logic:
    @staticmethod
    def is_unicorn_pump(df: pd.DataFrame) -> bool:
        try:
            if len(df) < 50:
                return False

            vol_avg = df["volume"].rolling(20).mean().iloc[-1]
            current_vol = df["volume"].iloc[-1]
            rvol = current_vol / (vol_avg + 1e-9)

            df.ta.bbands(length=20, std=2.0, append=True)
            df.ta.vwap(append=True)

            last = df.iloc[-1]
            bb_upper = last["BBU_20_2.0"]
            vwap = last["VWAP_D"]

            return (rvol > 3.0) and (last["close"] > bb_upper) and (last["close"] > vwap)
        except Exception:
            return False

    @staticmethod
    def is_research_dip(df: pd.DataFrame) -> bool:
        try:
            if len(df) < 220:
                return False

            df.ta.ema(length=50, append=True)
            df.ta.ema(length=200, append=True)
            df.ta.rsi(length=14, append=True)

            last = df.iloc[-1]
            # Safety check if EMA failed to calculate (not enough data)
            if pd.isna(last.get("EMA_200")):
                return False

            uptrend = last["EMA_50"] > last["EMA_200"]
            oversold = last["RSI_14"] < 35
            return uptrend and oversold
        except Exception:
            return False


# ==========================================
# 🤖 BOT ENGINE
# ==========================================
class Bot:
    def __init__(self):
        self.db = Database(DB_FILE)
        # FIX: Locks to prevent race conditions and DB collisions
        self.db_lock = asyncio.Lock()
        self.buy_lock = asyncio.Lock()

        api_key = os.getenv("KRAKEN_API_KEY")
        api_secret = os.getenv("KRAKEN_API_SECRET")
        if not api_key or not api_secret:
            raise ValueError("Missing KRAKEN_API_KEY or KRAKEN_API_SECRET in .env")

        self.exchange = ccxt.kraken(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            }
        )
        self.sem = asyncio.Semaphore(CONCURRENCY)

    async def get_config(self) -> Tuple[str, Dict[str, Any]]:
        now_pst = datetime.now(timezone.utc).astimezone(TZ_PST)
        if UNICORN_START <= now_pst.hour < UNICORN_END:
            return "UNICORN", CONFIG_UNICORN
        return "RESEARCH", CONFIG_RESEARCH

    def _clean_precision(self, symbol: str, amount: float, price: float) -> Tuple[float, float]:
        try:
            a = float(self.exchange.amount_to_precision(symbol, amount))
            p = float(self.exchange.price_to_precision(symbol, price))
            return a, p
        except Exception as e:
            log.error(f"Precision Error {symbol}: {e}")
            return amount, price

    # --- THREAD-SAFE DB WRAPPERS ---
    async def _get_positions(self) -> Dict[str, Dict[str, Any]]:
        async with self.db_lock:
            return self.db.get_positions()

    async def _add_position(self, sym: str, price: float, size: float, sl: float, mode: str):
        async with self.db_lock:
            self.db.add_position(sym, price, size, sl, mode)

    async def _update_position(self, sym: str, highest: float, sl: float):
        async with self.db_lock:
            self.db.update_position(sym, highest, sl)

    async def _close_position(self, sym: str, price: float, size: float, profit: float, reason: str):
        async with self.db_lock:
            self.db.delete_position(sym)
            self.db.log_trade(sym, "SELL", price, size, profit, reason)

    async def execute_buy(self, symbol: str, config: Dict[str, Any]):
        try:
            bal = await self.exchange.fetch_balance()
            usd_free = float(bal.get("USD", {}).get("free", 0.0))
            if usd_free < float(config["SIZE"]):
                return

            ticker = await self.exchange.fetch_ticker(symbol)
            ask = float(ticker.get("ask") or 0.0)
            if ask <= 0:
                return

            amount = float(config["SIZE"]) / ask
            limit_price = ask * (1 + float(config["SLIPPAGE"]))
            
            # Precision cleanup before sending to Kraken
            amount, limit_price = self._clean_precision(symbol, amount, limit_price)

            log.info(f"🚀 BUY {symbol} ({config['NAME']}) | amt {amount} @ {limit_price}")

            order = await self.exchange.create_order(
                symbol, "limit", "buy", amount, limit_price, params={"timeInForce": "IOC"}
            )

            filled = float(order.get("filled") or 0.0)
            if filled <= 0:
                return

            fill_price = float(order.get("average") or ask)
            sl_price = fill_price * (1 - float(config["STOP_LOSS"]))

            await self._add_position(symbol, fill_price, filled, sl_price, config["NAME"])

        except Exception as e:
            log.error(f"Buy Error {symbol}: {e}")

    async def sell(self, sym: str, pos: Dict[str, Any], price: float, reason: str):
        try:
            amount = float(pos.get("size") or 0.0)
            if amount <= 0:
                # Cleanup ghost positions if size is invalid
                await self._close_position(sym, price, 0.0, 0.0, "BAD_SIZE_CLEANUP")
                return

            # Ensure sell amount matches precision reqs
            amount, _ = self._clean_precision(sym, amount, price)

            log.info(f"🔻 SELL {sym} | {reason} | amt {amount}")

            await self.exchange.create_order(sym, "market", "sell", amount)

            entry = float(pos.get("entry_price") or price)
            profit = (price - entry) * amount

            await self._close_position(sym, price, amount, profit, reason)
            log.info(f"✅ CLOSED {sym} | {reason} | PnL ${profit:.2f}")

        except Exception as e:
            log.error(f"Sell Error {sym}: {e}")

    async def manage_positions(self):
        positions = await self._get_positions()
        if not positions:
            return

        for sym, pos in positions.items():
            try:
                mode = pos.get("mode") or ""
                # Select Strategy Config
                strat_cfg = CONFIG_UNICORN if "UNICORN" in mode else CONFIG_RESEARCH

                ticker = await self.exchange.fetch_ticker(sym)
                bid = float(ticker.get("bid") or 0.0)
                if bid <= 0:
                    continue

                entry = float(pos.get("entry_price") or bid)
                highest = float(pos.get("highest_price") or entry)
                
                # Default SL to entry-based if missing
                sl = float(pos.get("stop_loss") or (entry * (1 - float(strat_cfg["STOP_LOSS"]))))
                entry_time = int(pos.get("entry_time") or int(time.time()))

                # 1. Update High Water Mark
                if bid > highest:
                    highest = bid

                    # Trailing Stop only for Unicorn
                    if "UNICORN" in mode:
                        profit_pct = (highest - entry) / entry if entry > 0 else 0.0
                        if profit_pct >= float(strat_cfg.get("TRAIL_AFTER_PROFIT", 0.10)):
                            new_sl = highest * (1 - float(strat_cfg.get("TRAIL_OFFSET", 0.02)))
                            if new_sl > sl:
                                sl = new_sl

                    await self._update_position(sym, highest, sl)

                # 2. Stop Loss
                if bid < sl:
                    await self.sell(sym, pos, bid, "STOP_LOSS")
                    continue

                # 3. Take Profit
                tp = float(strat_cfg.get("TAKE_PROFIT", 0.0))
                if tp > 0 and bid >= entry * (1 + tp):
                    await self.sell(sym, pos, bid, "TAKE_PROFIT")
                    continue

                # 4. Stagnancy Check (Unicorn only)
                if "UNICORN" in mode:
                    dur = int(time.time()) - entry_time
                    if dur >= int(strat_cfg.get("STAGNANT_S", 900)):
                        if bid < entry * (1 + float(strat_cfg.get("STAGNANT_MIN_GAIN", 0.01))):
                            await self.sell(sym, pos, bid, "STAGNANT")
                            continue

            except Exception as e:
                log.error(f"Manage Error {sym}: {e}")

    async def scan_symbol(self, sym: str, mode_name: str, config: Dict[str, Any]):
        async with self.sem:
            try:
                limit = int(config.get("FETCH_LIMIT", 100))
                ohlcv = await self.exchange.fetch_ohlcv(sym, config["TIMEFRAME"], limit=limit)
                if not ohlcv:
                    return

                df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])

                if mode_name == "UNICORN":
                    buy = Logic.is_unicorn_pump(df)
                else:
                    buy = Logic.is_research_dip(df)

                if not buy:
                    await asyncio.sleep(RATE_LIMIT_SLEEP)
                    return

                # FIX: Buy Lock prevents race condition where multiple threads exceed MAX_POS
                async with self.buy_lock:
                    positions = await self._get_positions()
                    if sym in positions:
                        return
                    if len(positions) >= int(config["MAX_POS"]):
                        return
                    await self.execute_buy(sym, config)

                await asyncio.sleep(RATE_LIMIT_SLEEP)

            except Exception:
                return

    async def scan(self):
        mode_name, config = await self.get_config()

        positions = await self._get_positions()
        if len(positions) >= int(config["MAX_POS"]):
            return

        if not self.exchange.markets:
            await self.exchange.load_markets()

        tickers = await self.exchange.fetch_tickers()
        # FIX: Universe filter uses config variable, not hardcoded 50k
        min_vol = float(config.get("MIN_VOL_24H", 50_000.0))

        valid_pairs: List[str] = []
        for s, t in tickers.items():
            if "/USD" not in s:
                continue
            if s in positions:
                continue
            qv = t.get("quoteVolume")
            if qv is None or float(qv) < min_vol:
                continue
            valid_pairs.append(s)

        # Max 30 candidates
        tasks = [self.scan_symbol(sym, mode_name, config) for sym in valid_pairs[:30]]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def run(self):
        log.info("🔥 SolaScript V5.2 STARTED (ENTERPRISE)")
        await self.exchange.load_markets()

        while True:
            try:
                await self.scan()
                await self.manage_positions()
                await asyncio.sleep(2)
            except KeyboardInterrupt:
                break
            except Exception as e:
                log.error(f"Loop Error: {e}")
                await asyncio.sleep(5)

        await self.exchange.close()


if __name__ == "__main__":
    try:
        asyncio.run(Bot().run())
    except KeyboardInterrupt:
        pass