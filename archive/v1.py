import asyncio
import json
import os
import sys
import time
import logging
from typing import Dict, Any, Optional, Tuple, List

from dotenv import load_dotenv
import ccxt.async_support as ccxt
import pandas as pd


load_dotenv()


TIMEFRAME_ENTRY = "5m"
TIMEFRAME_MOMENTUM = "1m"

RISK_PER_TRADE_USD = 10.0
MAX_POSITIONS = 6

STATE_FILE = "trade_state.json"
LOG_FILE = "apex_engine.log"

LOOP_SLEEP_S = 2.0

BB_PERIOD = 20
BB_STD = 2.0
VOLUME_SPIKE_MULT = 1.5

TAKE_PROFIT_TRIGGER = 0.015
TRAILING_STOP_PCT = 0.008
TIGHT_STOP_PCT = 0.003

EARLY_STOP_PCT = 0.006
EARLY_WINDOW_S = 180
EARLY_RED_CANDLES = 2

COOLDOWN_AFTER_SELL_S = 45 * 60
COOLDOWN_AFTER_BUY_S = 10 * 60

DAILY_CHANGE_MIN = -0.03
ENTRY_SCORE_MIN = 7.0

HEARTBEAT_EVERY_S = 30
MONITOR_LOG_EVERY_S = 30

QUOTE_CCY = "USD"

UNIVERSE_REFRESH_S = 60 * 60
UNIVERSE_MIN_USD_VOLUME_24H = 250000.0
UNIVERSE_MIN_PRICE = 0.0000005

TICKER_REFRESH_S = 20
TOP_TICKER_CANDIDATES = 50
DEEP_SCAN_BATCH_SIZE = 20
DEEP_SCAN_CONCURRENCY = 4

CANDIDATE_CACHE_TTL_S = 45

SCAN_THROTTLE_S = 0.10


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _now_s() -> int:
    return int(time.time())


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, 1e-12))
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _slope(series: pd.Series, lookback: int = 10) -> float:
    if series is None or len(series) < lookback + 1:
        return 0.0
    a = float(series.iloc[-1])
    b = float(series.iloc[-1 - lookback])
    return (a - b) / max(abs(b), 1e-12)


class ApexBot:
    def __init__(self) -> None:
        api_key = (os.getenv("KRAKEN_API_KEY") or os.getenv("EXCHANGE_KEY") or "").strip()
        api_secret = (os.getenv("KRAKEN_API_SECRET") or os.getenv("EXCHANGE_SECRET") or "").strip()

        if not api_key or not api_secret:
            raise RuntimeError("Missing KRAKEN_API_KEY/KRAKEN_API_SECRET (or EXCHANGE_KEY/EXCHANGE_SECRET) in environment.")

        self.exchange = ccxt.kraken({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })

        self.positions: Dict[str, Dict[str, Any]] = {}
        self.usd_balance: float = 0.0
        self.markets_loaded: bool = False

        self.last_action_ts_s: Dict[str, int] = {}
        self.last_heartbeat_s: int = 0

        self.monitor_only: bool = False
        self.last_monitor_log_s: int = 0

        self.universe: List[str] = []
        self.universe_index: int = 0
        self.last_universe_refresh_s: int = 0

        self.tickers: Dict[str, Any] = {}
        self.last_ticker_refresh_s: int = 0

        self.best_cache: Optional[Tuple[str, float, float]] = None
        self.best_cache_ts_s: int = 0

        self._deep_sem = asyncio.Semaphore(DEEP_SCAN_CONCURRENCY)

    async def load_markets(self) -> None:
        if self.markets_loaded:
            return
        await self.exchange.load_markets()
        self.markets_loaded = True

    async def fetch_balance(self) -> None:
        try:
            bal = await self.exchange.fetch_balance()
            usd = bal.get(QUOTE_CCY, {})
            self.usd_balance = float(usd.get("free", 0.0) or 0.0)
        except Exception as e:
            logging.error(f"Balance check failed: {e}")

    def save_state(self) -> None:
        try:
            data = {
                "ts_ms": _now_ms(),
                "positions": self.positions,
                "last_action_ts_s": self.last_action_ts_s,
                "universe_index": self.universe_index,
            }
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True)
        except Exception as e:
            logging.error(f"State save failed: {e}")

    def load_state(self) -> None:
        if not os.path.exists(STATE_FILE):
            return
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            positions = data.get("positions", {})
            if isinstance(positions, dict):
                self.positions = positions

            last_action = data.get("last_action_ts_s", {})
            if isinstance(last_action, dict):
                self.last_action_ts_s = {k: int(v) for k, v in last_action.items()}

            ui = data.get("universe_index", 0)
            try:
                self.universe_index = int(ui)
            except Exception:
                self.universe_index = 0

            if self.positions:
                logging.info(f"Loaded {len(self.positions)} open position(s) from {STATE_FILE}")
        except Exception as e:
            logging.error(f"State load failed: {e}")

    async def _fetch_ohlcv_df(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv or len(ohlcv) < 30:
                return None
            df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "vol"])
            df["open"] = df["open"].astype(float)
            df["high"] = df["high"].astype(float)
            df["low"] = df["low"].astype(float)
            df["close"] = df["close"].astype(float)
            df["vol"] = df["vol"].astype(float)
            return df
        except Exception as e:
            logging.error(f"OHLCV error {symbol} {timeframe}: {e}")
            return None

    async def _fetch_recent_ohlcv(self, symbol: str, timeframe: str, limit: int) -> Optional[List[List[float]]]:
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv or len(ohlcv) < limit:
                return None
            return ohlcv
        except Exception:
            return None

    async def _daily_change_ok(self, symbol: str) -> bool:
        try:
            t = await self.exchange.fetch_ticker(symbol)
            pct = t.get("percentage", None)
            if pct is None:
                return True
            pct = float(pct) / 100.0
            return pct >= DAILY_CHANGE_MIN
        except Exception:
            return True

    def _min_amount_ok(self, symbol: str, amount: float) -> bool:
        try:
            market = self.exchange.market(symbol)
            min_amt = (market.get("limits", {}).get("amount", {}) or {}).get("min", None)
            if min_amt is None:
                return True
            return float(amount) >= float(min_amt)
        except Exception:
            return True

    async def _get_base_free(self, symbol: str) -> float:
        try:
            base = self.exchange.market(symbol)["base"]
            bal = await self.exchange.fetch_balance()
            free = bal.get(base, {}).get("free", 0.0)
            return float(free or 0.0)
        except Exception:
            return 0.0

    def _cooldown_ok(self, symbol: str) -> bool:
        now = _now_s()
        last = int(self.last_action_ts_s.get(symbol, 0))
        if last <= 0:
            return True
        if symbol in self.positions:
            return (now - last) >= COOLDOWN_AFTER_BUY_S
        return (now - last) >= COOLDOWN_AFTER_SELL_S

    def _entry_breakout_ok(self, df: pd.DataFrame) -> Tuple[bool, float]:
        if df is None or len(df) < BB_PERIOD + 5:
            return False, 0.0

        close = df["close"]
        vol = df["vol"]

        sma = close.rolling(window=BB_PERIOD).mean()
        std = close.rolling(window=BB_PERIOD).std()
        upper = sma + (BB_STD * std)

        vol_ma = vol.rolling(window=BB_PERIOD).mean()

        price = float(close.iloc[-1])
        upper_v = float(upper.iloc[-1]) if pd.notna(upper.iloc[-1]) else 0.0
        vol_v = float(vol.iloc[-1])
        vol_ma_v = float(vol_ma.iloc[-1]) if pd.notna(vol_ma.iloc[-1]) else 0.0

        if upper_v <= 0 or vol_ma_v <= 0:
            return False, price

        breakout = price > upper_v
        spike = vol_v > (vol_ma_v * VOLUME_SPIKE_MULT)

        return (breakout and spike), price

    def _trend_score(self, df: pd.DataFrame) -> float:
        close = df["close"]
        vol = df["vol"]

        ema20 = _ema(close, 20)
        ema50 = _ema(close, 50)

        rsi14 = _rsi(close, 14)
        atr14 = _atr(df, 14)

        score = 0.0

        if float(close.iloc[-1]) > float(ema20.iloc[-1]):
            score += 1.5
        if float(ema20.iloc[-1]) > float(ema50.iloc[-1]):
            score += 1.5
        if _slope(ema20, 10) > 0:
            score += 1.0
        if _slope(ema50, 10) > 0:
            score += 1.0

        r = float(rsi14.iloc[-1]) if pd.notna(rsi14.iloc[-1]) else 50.0
        if 52.0 <= r <= 70.0:
            score += 1.0
        elif r > 70.0:
            score += 0.3

        vol_ma = vol.rolling(window=20).mean()
        vol_ma_v = float(vol_ma.iloc[-1]) if pd.notna(vol_ma.iloc[-1]) else 0.0
        if vol_ma_v > 0 and float(vol.iloc[-1]) > vol_ma_v:
            score += 0.7

        atr_v = float(atr14.iloc[-1]) if pd.notna(atr14.iloc[-1]) else 0.0
        if atr_v > 0:
            score += 0.3

        return score

    def _trend_gate(self, df: pd.DataFrame) -> bool:
        close = df["close"]
        ema20 = _ema(close, 20)
        ema50 = _ema(close, 50)
        if len(close) < 60:
            return False
        if float(close.iloc[-1]) <= float(ema50.iloc[-1]):
            return False
        if float(ema20.iloc[-1]) <= float(ema50.iloc[-1]):
            return False
        if _slope(ema50, 12) <= 0:
            return False
        return True

    async def refresh_universe(self, force: bool = False) -> None:
        now = _now_s()
        if not force and self.universe and (now - self.last_universe_refresh_s) < UNIVERSE_REFRESH_S:
            return

        try:
            await self.load_markets()
            markets = self.exchange.markets or {}

            eligible: List[str] = []
            for sym, m in markets.items():
                try:
                    if not m:
                        continue
                    if not bool(m.get("active", True)):
                        continue
                    if not bool(m.get("spot", False)):
                        continue
                    quote = (m.get("quote") or "").upper()
                    if quote != QUOTE_CCY:
                        continue
                    if "/USD" not in sym:
                        continue
                    eligible.append(sym)
                except Exception:
                    continue

            eligible = sorted(list(set(eligible)))
            if not eligible:
                logging.warning("Universe refresh produced empty list, keeping prior universe.")
                return

            prev = len(self.universe)
            self.universe = eligible
            self.last_universe_refresh_s = now

            if self.universe_index >= len(self.universe):
                self.universe_index = 0

            logging.info(f"Universe ready | USD spot pairs: {len(self.universe)} | Prev: {prev}")

        except Exception as e:
            logging.error(f"Universe refresh failed: {e}")

    async def refresh_tickers(self) -> None:
        now = _now_s()
        if (now - self.last_ticker_refresh_s) < TICKER_REFRESH_S and self.tickers:
            return
        try:
            t = await self.exchange.fetch_tickers()
            if isinstance(t, dict) and t:
                self.tickers = t
                self.last_ticker_refresh_s = now
        except Exception as e:
            logging.warning(f"Ticker refresh failed: {e}")

    def _stage_a_top_symbols(self) -> List[str]:
        if not self.universe:
            return []
        if not self.tickers:
            return []

        scored: List[Tuple[str, float]] = []

        for sym in self.universe:
            t = self.tickers.get(sym)
            if not t:
                continue

            last = t.get("last", None)
            qv = t.get("quoteVolume", None)
            pct = t.get("percentage", None)
            bid = t.get("bid", None)
            ask = t.get("ask", None)

            try:
                if last is None or qv is None:
                    continue
                last_f = float(last)
                if last_f <= UNIVERSE_MIN_PRICE:
                    continue

                qv_f = float(qv)
                if qv_f < UNIVERSE_MIN_USD_VOLUME_24H:
                    continue

                pct_f = 0.0
                if pct is not None:
                    pct_f = float(pct) / 100.0
                    if pct_f < DAILY_CHANGE_MIN:
                        continue

                spread_penalty = 0.0
                if bid is not None and ask is not None:
                    bid_f = float(bid)
                    ask_f = float(ask)
                    if bid_f > 0 and ask_f > 0 and ask_f >= bid_f:
                        spread = (ask_f - bid_f) / ask_f
                        if spread > 0.01:
                            continue
                        spread_penalty = spread * 500.0

                rank = (qv_f / 1_000_000.0) + (pct_f * 20.0) - spread_penalty
                scored.append((sym, rank))
            except Exception:
                continue

        scored.sort(key=lambda x: x[1], reverse=True)
        top = [s for s, _ in scored[:TOP_TICKER_CANDIDATES]]
        return top

    def _next_deep_batch(self, candidates: List[str]) -> List[str]:
        if not candidates:
            return []
        n = len(candidates)
        if n <= 0:
            return []

        batch: List[str] = []
        count = 0
        idx = self.universe_index % n

        while count < min(DEEP_SCAN_BATCH_SIZE, n):
            batch.append(candidates[idx])
            idx = (idx + 1) % n
            count += 1

        self.universe_index = idx
        return batch

    async def _analyze_symbol(self, symbol: str) -> Optional[Tuple[str, float, float]]:
        async with self._deep_sem:
            try:
                if symbol in self.positions:
                    return None
                if not self._cooldown_ok(symbol):
                    return None

                df_entry = await self._fetch_ohlcv_df(symbol, TIMEFRAME_ENTRY, 160)
                if df_entry is None:
                    return None

                breakout_ok, price = self._entry_breakout_ok(df_entry)
                if not breakout_ok:
                    return None

                df1h = await self._fetch_ohlcv_df(symbol, "1h", 220)
                df4h = await self._fetch_ohlcv_df(symbol, "4h", 220)
                if df1h is None or df4h is None:
                    return None

                if not self._trend_gate(df1h):
                    return None
                if not self._trend_gate(df4h):
                    return None

                df15m = await self._fetch_ohlcv_df(symbol, "15m", 220)
                if df15m is None:
                    return None

                s15 = self._trend_score(df15m)
                s1h = self._trend_score(df1h)
                s4h = self._trend_score(df4h)

                total = s15 + (1.2 * s1h) + (1.6 * s4h)

                if total < ENTRY_SCORE_MIN:
                    return None

                return (symbol, float(price), float(total))
            except Exception as e:
                logging.error(f"Analyze error {symbol}: {e}")
                return None
            finally:
                await asyncio.sleep(SCAN_THROTTLE_S)

    async def choose_best_candidate(self) -> Optional[Tuple[str, float, float]]:
        now = _now_s()
        if self.best_cache and (now - self.best_cache_ts_s) <= CANDIDATE_CACHE_TTL_S:
            sym, _, _ = self.best_cache
            if sym not in self.positions and self._cooldown_ok(sym):
                return self.best_cache

        await self.refresh_universe()
        await self.refresh_tickers()

        stage_a = self._stage_a_top_symbols()
        if not stage_a:
            return None

        batch = self._next_deep_batch(stage_a)
        if not batch:
            return None

        tasks = [asyncio.create_task(self._analyze_symbol(sym)) for sym in batch]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        scored: List[Tuple[str, float, float]] = [r for r in results if r is not None]
        if not scored:
            return None

        scored.sort(key=lambda x: x[2], reverse=True)
        best = scored[0]
        self.best_cache = best
        self.best_cache_ts_s = _now_s()
        return best

    async def execute_buy(self, symbol: str, price: float, score: float) -> None:
        if self.usd_balance < RISK_PER_TRADE_USD:
            self.monitor_only = True
            logging.warning(f"Skipping {symbol}: insufficient USD (${self.usd_balance:.2f})")
            return

        try:
            amount = RISK_PER_TRADE_USD / price
            amount = float(self.exchange.amount_to_precision(symbol, amount))

            if amount <= 0 or not self._min_amount_ok(symbol, amount):
                logging.warning(f"Skipping {symbol}: amount too small after precision ({amount})")
                return

            logging.info(f"EXECUTE BUY: {symbol} @ {price:.6f} | Qty: {amount} | Score: {score:.2f}")

            order = await self.exchange.create_market_buy_order(symbol, amount)

            filled = float(order.get("filled") or 0.0)
            avg = float(order.get("average") or price)
            qty_effective = filled if filled > 0 else amount

            self.positions[symbol] = {
                "entry": avg,
                "highest": avg,
                "qty": qty_effective,
                "buy_order_id": order.get("id"),
                "opened_ts_ms": _now_ms(),
                "early_red_count": 0,
                "last_checked_candle_ts": 0,
                "entry_score": score,
            }

            self.last_action_ts_s[symbol] = _now_s()
            self.save_state()
            await self.fetch_balance()

        except Exception as e:
            logging.error(f"BUY failed for {symbol}: {e}")

    async def execute_sell(self, symbol: str, price: float, reason: str) -> None:
        try:
            pos = self.positions.get(symbol)
            if not pos:
                return

            qty = float(pos.get("qty") or 0.0)
            base_free = await self._get_base_free(symbol)

            if base_free > 0:
                qty = min(qty if qty > 0 else base_free, base_free)

            qty = float(self.exchange.amount_to_precision(symbol, qty))

            if qty <= 0 or not self._min_amount_ok(symbol, qty):
                logging.error(f"SELL blocked for {symbol}: qty too small ({qty})")
                return

            entry = float(pos.get("entry") or 0.0)
            pnl_pct = ((price - entry) / entry * 100.0) if entry > 0 else 0.0

            logging.info(f"SELL {symbol} @ {price:.6f} | Qty: {qty} | PnL: {pnl_pct:.2f}% | Reason: {reason}")

            await self.exchange.create_market_sell_order(symbol, qty)

            if symbol in self.positions:
                del self.positions[symbol]

            self.last_action_ts_s[symbol] = _now_s()
            self.save_state()
            await self.fetch_balance()

        except Exception as e:
            logging.error(f"SELL failed for {symbol}: {e}")

    async def _update_early_momentum_counter(self, symbol: str, pos: Dict[str, Any], entry: float) -> None:
        ohlcv = await self._fetch_recent_ohlcv(symbol, TIMEFRAME_MOMENTUM, 3)
        if not ohlcv or len(ohlcv) < 3:
            return

        prev_close = float(ohlcv[-3][4])
        last_close = float(ohlcv[-2][4])

        last_closed_ts = int(ohlcv[-2][0])
        candle_key = int(last_closed_ts // 60000)
        last_checked = int(pos.get("last_checked_candle_ts") or 0)

        if candle_key == last_checked:
            return

        pos["last_checked_candle_ts"] = candle_key

        if last_close < entry and prev_close < entry:
            pos["early_red_count"] = int(pos.get("early_red_count") or 0) + 1
        else:
            pos["early_red_count"] = 0

        self.save_state()

    async def manage_positions(self) -> None:
        if not self.positions:
            return

        for symbol in list(self.positions.keys()):
            try:
                ticker = await self.exchange.fetch_ticker(symbol)
                last = ticker.get("last", None)
                if last is None:
                    continue

                curr_price = float(last)
                pos = self.positions[symbol]

                highest = float(pos.get("highest") or pos.get("entry") or curr_price)
                entry = float(pos.get("entry") or curr_price)
                opened_ts_ms = int(pos.get("opened_ts_ms") or 0)

                if curr_price > highest:
                    pos["highest"] = curr_price
                    self.save_state()
                    highest = curr_price

                now_ms = _now_ms()
                age_s = (now_ms - opened_ts_ms) / 1000.0 if opened_ts_ms > 0 else 999999.0

                if age_s <= EARLY_WINDOW_S and entry > 0:
                    early_stop_price = entry * (1.0 - EARLY_STOP_PCT)
                    if curr_price <= early_stop_price:
                        await self.execute_sell(symbol, curr_price, f"Early stop: {EARLY_STOP_PCT*100:.2f}% below entry")
                        continue

                    await self._update_early_momentum_counter(symbol, pos, entry)

                    if int(pos.get("early_red_count") or 0) >= EARLY_RED_CANDLES:
                        await self.execute_sell(symbol, curr_price, f"Early exit: {EARLY_RED_CANDLES} consecutive 1m closes under entry")
                        continue

                profit_pct = (highest - entry) / entry if entry > 0 else 0.0

                if profit_pct > TAKE_PROFIT_TRIGGER:
                    active_stop_pct = TIGHT_STOP_PCT
                    status = "LOCKED"
                else:
                    active_stop_pct = TRAILING_STOP_PCT
                    status = "GUARDING"

                stop_price = highest * (1.0 - active_stop_pct)

                if curr_price < stop_price:
                    await self.execute_sell(symbol, curr_price, f"Stop hit ({status})")

            except Exception as e:
                logging.error(f"Manage error {symbol}: {e}")

    async def heartbeat(self) -> None:
        now = _now_s()
        if (now - self.last_heartbeat_s) < HEARTBEAT_EVERY_S:
            return
        self.last_heartbeat_s = now
        held = ", ".join([f"{k}" for k in self.positions.keys()]) if self.positions else "none"
        u = len(self.universe)
        mode = "MONITOR" if self.monitor_only else "TRADE"
        logging.info(
            f"Heartbeat | Mode: {mode} | USD: {self.usd_balance:.2f} | Positions: {len(self.positions)} | Holding: {held} | Universe: {u} | Index: {self.universe_index}"
        )

    async def _monitor_log(self) -> None:
        now = _now_s()
        if (now - self.last_monitor_log_s) < MONITOR_LOG_EVERY_S:
            return
        self.last_monitor_log_s = now
        logging.warning(f"Monitoring only: insufficient USD for new entries (${self.usd_balance:.2f} < ${RISK_PER_TRADE_USD:.2f})")

    async def run(self) -> None:
        logging.info("APEX ENGINE STARTED")
        logging.info("Strategy: stage A tickers filter + stage B multi timeframe trend gate + breakout entry")
        await self.load_markets()

        self.load_state()
        await self.fetch_balance()
        await self.refresh_universe(force=True)
        await self.refresh_tickers()

        for _, pos in self.positions.items():
            if "early_red_count" not in pos:
                pos["early_red_count"] = 0
            if "last_checked_candle_ts" not in pos:
                pos["last_checked_candle_ts"] = 0
            if "entry_score" not in pos:
                pos["entry_score"] = 0.0
        if self.positions:
            self.save_state()

        while True:
            await self.fetch_balance()

            if self.usd_balance < RISK_PER_TRADE_USD:
                self.monitor_only = True
            else:
                if self.monitor_only and len(self.positions) < MAX_POSITIONS:
                    self.monitor_only = False

            await self.manage_positions()
            await self.heartbeat()

            if self.monitor_only:
                await self._monitor_log()
                await asyncio.sleep(LOOP_SLEEP_S)
                continue

            if len(self.positions) < MAX_POSITIONS:
                best = await self.choose_best_candidate()
                if best is not None:
                    symbol, price, score = best
                    if symbol not in self.positions and self._cooldown_ok(symbol):
                        await self.execute_buy(symbol, price, score)

            await asyncio.sleep(LOOP_SLEEP_S)


async def main() -> None:
    bot = ApexBot()
    try:
        await bot.run()
    except KeyboardInterrupt:
        logging.info("MANUAL STOP")
    except Exception as e:
        logging.critical(f"CRITICAL CRASH: {e}")
    finally:
        if bot.exchange:
            logging.info("Closing API session...")
            await bot.exchange.close()
            logging.info("Session closed.")


if __name__ == "__main__":
    asyncio.run(main())
