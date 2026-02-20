#!/usr/bin/env python3
"""NSE Trading Bot v2 - Multi-indicator scanner for Nifty 50 with improved
strategy, ATR-based risk management, trailing stops, and portfolio guards."""

import argparse
import json
import math
import os
import sys
from datetime import datetime, date, timedelta

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import AverageTrueRange

# ─── Configuration Constants ─────────────────────────────────────────────────

# Indicators
RSI_PERIOD = 14
RSI_OVERSOLD = 35
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
EMA_SHORT = 20
EMA_LONG = 200
VOLUME_AVG_PERIOD = 20
SUPPORT_LOOKBACK = 50

# ATR & Stops
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
ATR_TRAIL_MULTIPLIER = 2.5

# Position Sizing & Risk
INITIAL_CAPITAL = 1_000_000  # 10 lakh INR
RISK_PER_TRADE_PCT = 1.5     # % of total portfolio risked per trade
MAX_POSITIONS = 8
MAX_EXPOSURE_PCT = 60         # max % of capital deployed
MAX_PER_SECTOR = 2

# Exit Rules
PROFIT_TARGET_1_R = 1.5       # first partial exit at 1.5R
PARTIAL_EXIT_PCT = 50          # sell 50% at first target
TIME_EXIT_DAYS = 30            # exit flat positions after N days
SUPPORT_PROXIMITY_PCT = 3      # within 3% of 50-day low

# Misc
COOLDOWN_DAYS = 10
DATA_PERIOD = "1y"
MIN_BUY_SCORE = 3

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_trades.json")

# ─── Nifty 50 Stock List ─────────────────────────────────────────────────────

NIFTY_50 = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "KOTAKBANK", "ITC",
    "LT", "AXISBANK", "BAJFINANCE", "ASIANPAINT", "MARUTI",
    "HCLTECH", "SUNPHARMA", "TRENT", "TITAN", "WIPRO",
    "ULTRACEMCO", "NTPC", "NESTLEIND", "POWERGRID", "TECHM",
    "BAJAJFINSV", "ONGC", "TATASTEEL", "JSWSTEEL", "M&M",
    "ADANIENT", "ADANIPORTS", "HDFCLIFE", "SBILIFE", "DIVISLAB",
    "BPCL", "GRASIM", "CIPLA", "TATACONSUM", "EICHERMOT",
    "DRREDDY", "APOLLOHOSP", "COALINDIA", "HEROMOTOCO", "INDUSINDBK",
    "BAJAJ-AUTO", "BRITANNIA", "HINDALCO", "LTIM", "SHRIRAMFIN",
]

# ─── Sector Mapping ──────────────────────────────────────────────────────────

SECTOR_MAP = {
    # IT
    "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "WIPRO": "IT",
    "TECHM": "IT", "LTIM": "IT",
    # Banking
    "HDFCBANK": "Banking", "ICICIBANK": "Banking", "SBIN": "Banking",
    "KOTAKBANK": "Banking", "AXISBANK": "Banking", "INDUSINDBK": "Banking",
    # Finance
    "BAJFINANCE": "Finance", "BAJAJFINSV": "Finance", "HDFCLIFE": "Finance",
    "SBILIFE": "Finance", "SHRIRAMFIN": "Finance",
    # FMCG
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG",
    "TATACONSUM": "FMCG", "BRITANNIA": "FMCG",
    # Pharma
    "SUNPHARMA": "Pharma", "DIVISLAB": "Pharma", "CIPLA": "Pharma",
    "DRREDDY": "Pharma", "APOLLOHOSP": "Pharma",
    # Auto
    "MARUTI": "Auto", "EICHERMOT": "Auto", "HEROMOTOCO": "Auto",
    "BAJAJ-AUTO": "Auto", "M&M": "Auto", "TRENT": "Auto",
    # Energy / Oil & Gas
    "RELIANCE": "Energy", "ONGC": "Energy", "BPCL": "Energy",
    "NTPC": "Energy", "POWERGRID": "Energy", "COALINDIA": "Energy",
    # Metals & Mining
    "TATASTEEL": "Metals", "JSWSTEEL": "Metals", "HINDALCO": "Metals",
    # Telecom
    "BHARTIARTL": "Telecom",
    # Infrastructure
    "LT": "Infra", "ULTRACEMCO": "Infra", "GRASIM": "Infra",
    "ADANIENT": "Infra", "ADANIPORTS": "Infra",
    # Consumer
    "ASIANPAINT": "Consumer", "TITAN": "Consumer",
}

# ─── Data Functions ───────────────────────────────────────────────────────────


def fetch_data(symbol: str, period: str = DATA_PERIOD) -> pd.DataFrame | None:
    """Fetch daily OHLCV data from Yahoo Finance."""
    ticker = f"{symbol}.NS"
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        print(f"  [!] Error fetching {symbol}: {e}")
        return None


def fetch_index_data(symbol: str = "^NSEI", period: str = DATA_PERIOD) -> pd.DataFrame | None:
    """Fetch index data (e.g. Nifty 50) from Yahoo Finance."""
    try:
        df = yf.download(symbol, period=period, progress=False, auto_adjust=True)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        print(f"  [!] Error fetching index {symbol}: {e}")
        return None


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, MACD, EMAs, ATR, and volume average columns."""
    df = df.copy()

    # RSI
    df["RSI"] = RSIIndicator(close=df["Close"], window=RSI_PERIOD).rsi()

    # MACD
    macd_ind = MACD(close=df["Close"], window_slow=MACD_SLOW,
                    window_fast=MACD_FAST, window_sign=MACD_SIGNAL)
    df["MACD"] = macd_ind.macd()
    df["MACD_Signal"] = macd_ind.macd_signal()
    df["MACD_Hist"] = macd_ind.macd_diff()

    # EMAs
    df["EMA20"] = EMAIndicator(close=df["Close"], window=EMA_SHORT).ema_indicator()
    df["EMA200"] = EMAIndicator(close=df["Close"], window=EMA_LONG).ema_indicator()

    # ATR
    df["ATR"] = AverageTrueRange(high=df["High"], low=df["Low"],
                                  close=df["Close"], window=ATR_PERIOD).average_true_range()

    # Volume average
    df["Vol_Avg"] = df["Volume"].rolling(window=VOLUME_AVG_PERIOD).mean()

    # 50-day low (support proxy)
    df["Low_50"] = df["Low"].rolling(window=SUPPORT_LOOKBACK).min()

    return df


# ─── Market Regime ────────────────────────────────────────────────────────────


def check_market_regime() -> dict:
    """Check if Nifty 50 is above/below 200 EMA to determine market regime."""
    df = fetch_index_data("^NSEI")
    if df is None or len(df) < EMA_LONG + 5:
        return {"bullish": True, "nifty_close": None, "ema200": None, "status": "unknown (insufficient data)"}

    ema200 = EMAIndicator(close=df["Close"], window=EMA_LONG).ema_indicator()
    nifty_close = float(df["Close"].iloc[-1])
    ema200_val = float(ema200.iloc[-1])
    is_bullish = nifty_close > ema200_val

    return {
        "bullish": is_bullish,
        "nifty_close": round(nifty_close, 2),
        "ema200": round(ema200_val, 2),
        "status": "BULLISH" if is_bullish else "BEARISH",
    }


# ─── Buy Signal Scoring ──────────────────────────────────────────────────────


def score_buy_signal(df: pd.DataFrame) -> dict:
    """Score a stock 0-5 based on multiple indicators.

    Returns dict with total score and breakdown of each criterion.
    """
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last
    breakdown = {}
    score = 0

    # 1. RSI oversold
    rsi_val = last["RSI"]
    if pd.notna(rsi_val) and rsi_val <= RSI_OVERSOLD:
        score += 1
        breakdown["RSI"] = f"YES ({rsi_val:.1f} <= {RSI_OVERSOLD})"
    else:
        rsi_display = f"{rsi_val:.1f}" if pd.notna(rsi_val) else "N/A"
        breakdown["RSI"] = f"NO ({rsi_display})"

    # 2. MACD histogram rising (momentum turning)
    macd_hist = last["MACD_Hist"]
    macd_hist_prev = prev["MACD_Hist"]
    if pd.notna(macd_hist) and pd.notna(macd_hist_prev) and macd_hist > macd_hist_prev:
        score += 1
        breakdown["MACD_Rising"] = f"YES ({macd_hist_prev:.3f} -> {macd_hist:.3f})"
    else:
        breakdown["MACD_Rising"] = "NO"

    # 3. Volume above 20-day average
    vol = last["Volume"]
    vol_avg = last["Vol_Avg"]
    if pd.notna(vol) and pd.notna(vol_avg) and vol_avg > 0 and vol > vol_avg:
        ratio = vol / vol_avg
        score += 1
        breakdown["Volume"] = f"YES ({ratio:.1f}x avg)"
    else:
        breakdown["Volume"] = "NO"

    # 4. Price above 200-day EMA (trend filter)
    close = last["Close"]
    ema200 = last["EMA200"]
    if pd.notna(close) and pd.notna(ema200) and close > ema200:
        score += 1
        breakdown["Above_EMA200"] = f"YES ({close:.2f} > {ema200:.2f})"
    else:
        ema_display = f"{ema200:.2f}" if pd.notna(ema200) else "N/A"
        breakdown["Above_EMA200"] = f"NO ({close:.2f} vs {ema_display})"

    # 5. Price within 3% of 50-day low (near support)
    low_50 = last["Low_50"]
    if pd.notna(close) and pd.notna(low_50) and low_50 > 0:
        pct_from_low = ((close - low_50) / low_50) * 100
        if pct_from_low <= SUPPORT_PROXIMITY_PCT:
            score += 1
            breakdown["Near_Support"] = f"YES ({pct_from_low:.1f}% from 50d low)"
        else:
            breakdown["Near_Support"] = f"NO ({pct_from_low:.1f}% from 50d low)"
    else:
        breakdown["Near_Support"] = "NO (insufficient data)"

    return {"score": score, "breakdown": breakdown}


# ─── State Management ─────────────────────────────────────────────────────────


def load_state() -> dict:
    """Load paper trading state from JSON, or initialize fresh."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
        state = migrate_state(state)
        return state
    return {
        "capital": INITIAL_CAPITAL,
        "positions": {},
        "trade_history": [],
        "cooldowns": {},
    }


def migrate_state(state: dict) -> dict:
    """Add new fields to existing state so old paper_trades.json isn't lost."""
    # Ensure cooldowns dict exists
    if "cooldowns" not in state:
        state["cooldowns"] = {}

    # Migrate each position to include new fields
    for symbol, pos in state.get("positions", {}).items():
        if "trailing_stop" not in pos:
            # Use existing stop_loss as starting trailing stop
            pos["trailing_stop"] = pos.get("stop_loss", 0)
        if "highest_since_entry" not in pos:
            pos["highest_since_entry"] = pos.get("entry_price", 0)
        if "initial_risk_per_share" not in pos:
            # Estimate: difference between entry and stop
            entry = pos.get("entry_price", 0)
            stop = pos.get("stop_loss", 0)
            pos["initial_risk_per_share"] = round(entry - stop, 2) if entry > stop else round(entry * 0.03, 2)
        if "partial_exit_done" not in pos:
            pos["partial_exit_done"] = False
        if "sector" not in pos:
            pos["sector"] = SECTOR_MAP.get(symbol, "Unknown")
        if "atr_at_entry" not in pos:
            pos["atr_at_entry"] = pos.get("initial_risk_per_share", 0) / ATR_STOP_MULTIPLIER if pos.get("initial_risk_per_share", 0) > 0 else 0
        if "original_shares" not in pos:
            pos["original_shares"] = pos.get("shares", 0)

    # Migrate trade history entries
    for trade in state.get("trade_history", []):
        if "exit_reason" not in trade:
            trade["exit_reason"] = "Unknown"

    return state


def save_state(state: dict) -> None:
    """Persist paper trading state to JSON."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ─── Portfolio Risk Guards ────────────────────────────────────────────────────


def count_sector_positions(state: dict, sector: str) -> int:
    """Count how many open positions are in a given sector."""
    count = 0
    for pos in state["positions"].values():
        if pos.get("sector") == sector:
            count += 1
    return count


def get_total_exposure(state: dict) -> float:
    """Calculate total capital currently deployed in open positions."""
    total = 0
    for pos in state["positions"].values():
        total += pos["entry_price"] * pos["shares"]
    return total


def get_portfolio_value(state: dict) -> float:
    """Total portfolio value = capital + invested amount at cost."""
    return state["capital"] + get_total_exposure(state)


def is_in_cooldown(state: dict, symbol: str) -> bool:
    """Check if symbol is in cooldown period after a stop-loss exit."""
    cooldowns = state.get("cooldowns", {})
    if symbol not in cooldowns:
        return False
    cooldown_until = datetime.strptime(cooldowns[symbol], "%Y-%m-%d").date()
    return date.today() <= cooldown_until


def check_risk_limits(state: dict, symbol: str) -> str | None:
    """Check all portfolio risk guards. Returns rejection reason or None if OK."""
    # Max positions
    if len(state["positions"]) >= MAX_POSITIONS:
        return f"Max positions ({MAX_POSITIONS}) reached"

    # Max exposure
    portfolio_val = get_portfolio_value(state)
    exposure = get_total_exposure(state)
    if portfolio_val > 0 and (exposure / portfolio_val * 100) >= MAX_EXPOSURE_PCT:
        return f"Max exposure ({MAX_EXPOSURE_PCT}%) reached"

    # Sector limit
    sector = SECTOR_MAP.get(symbol, "Unknown")
    if count_sector_positions(state, sector) >= MAX_PER_SECTOR:
        return f"Max {MAX_PER_SECTOR} positions in {sector} sector"

    # Cooldown check
    if is_in_cooldown(state, symbol):
        cooldown_date = state["cooldowns"][symbol]
        return f"In cooldown until {cooldown_date}"

    return None


# ─── Trade Execution ──────────────────────────────────────────────────────────


def execute_buy(symbol: str, df: pd.DataFrame, state: dict, score_info: dict) -> dict | None:
    """Paper buy with ATR-based position sizing and stop loss."""
    last = df.iloc[-1]
    price = float(last["Close"])
    atr = float(last["ATR"])

    if pd.isna(atr) or atr <= 0:
        return None

    # ATR-based stop loss
    stop_loss = price - (ATR_STOP_MULTIPLIER * atr)
    risk_per_share = price - stop_loss

    if risk_per_share <= 0:
        return None

    # Risk-based position sizing: risk 1.5% of portfolio
    portfolio_val = get_portfolio_value(state)
    risk_amount = portfolio_val * (RISK_PER_TRADE_PCT / 100)
    shares = int(risk_amount / risk_per_share)

    if shares <= 0:
        return None

    cost = shares * price

    # Check if we have enough capital
    if cost > state["capital"]:
        # Reduce shares to fit available capital
        shares = int(state["capital"] / price)
        if shares <= 0:
            return None
        cost = shares * price

    state["capital"] -= cost
    entry_date = str(df.index[-1].date())
    sector = SECTOR_MAP.get(symbol, "Unknown")

    state["positions"][symbol] = {
        "entry_price": round(price, 2),
        "shares": shares,
        "original_shares": shares,
        "stop_loss": round(stop_loss, 2),
        "trailing_stop": round(stop_loss, 2),
        "highest_since_entry": round(price, 2),
        "initial_risk_per_share": round(risk_per_share, 2),
        "atr_at_entry": round(atr, 2),
        "partial_exit_done": False,
        "entry_date": entry_date,
        "sector": sector,
        "buy_score": score_info["score"],
    }

    return {
        "symbol": symbol,
        "price": round(price, 2),
        "shares": shares,
        "stop_loss": round(stop_loss, 2),
        "atr": round(atr, 2),
        "risk_per_share": round(risk_per_share, 2),
        "sector": sector,
        "score": score_info["score"],
        "breakdown": score_info["breakdown"],
    }


def execute_sell(symbol: str, exit_price: float, shares_to_sell: int,
                 state: dict, reason: str, exit_date: str) -> dict | None:
    """Paper sell: return proceeds, record in history. Supports partial exits."""
    if symbol not in state["positions"]:
        return None

    pos = state["positions"][symbol]

    if shares_to_sell <= 0 or shares_to_sell > pos["shares"]:
        shares_to_sell = pos["shares"]

    proceeds = shares_to_sell * exit_price
    pnl = (exit_price - pos["entry_price"]) * shares_to_sell

    state["capital"] += proceeds

    trade_record = {
        "symbol": symbol,
        "entry_price": pos["entry_price"],
        "exit_price": round(exit_price, 2),
        "shares": shares_to_sell,
        "pnl": round(pnl, 2),
        "entry_date": pos["entry_date"],
        "exit_date": exit_date,
        "exit_reason": reason,
        "sector": pos.get("sector", "Unknown"),
    }
    state["trade_history"].append(trade_record)

    # Update or remove position
    remaining = pos["shares"] - shares_to_sell
    if remaining <= 0:
        # Full exit — set cooldown if stop loss
        if "Stop" in reason:
            cooldown_date = (date.today() + timedelta(days=COOLDOWN_DAYS)).strftime("%Y-%m-%d")
            state["cooldowns"][symbol] = cooldown_date
        del state["positions"][symbol]
    else:
        pos["shares"] = remaining

    return trade_record


def update_trailing_stop(pos: dict, current_high: float) -> None:
    """Ratchet trailing stop upward based on highest price since entry."""
    if current_high > pos["highest_since_entry"]:
        pos["highest_since_entry"] = round(current_high, 2)

    atr_at_entry = pos.get("atr_at_entry", 0)
    if atr_at_entry <= 0:
        return

    new_trail = pos["highest_since_entry"] - (ATR_TRAIL_MULTIPLIER * atr_at_entry)
    if new_trail > pos["trailing_stop"]:
        pos["trailing_stop"] = round(new_trail, 2)


# ─── Exit Logic ───────────────────────────────────────────────────────────────


def check_exits(symbol: str, df: pd.DataFrame, state: dict) -> list[dict]:
    """Check all exit conditions for an open position. Returns list of exit trades."""
    if symbol not in state["positions"]:
        return []

    pos = state["positions"][symbol]
    last = df.iloc[-1]
    current_price = float(last["Close"])
    current_high = float(last["High"])
    current_low = float(last["Low"])
    exit_date = str(df.index[-1].date())
    exits = []

    # Update trailing stop first
    update_trailing_stop(pos, current_high)

    # Priority 1: Trailing stop loss
    if current_low <= pos["trailing_stop"]:
        exit_price = pos["trailing_stop"]  # assume stopped at trail level
        shares = pos["shares"]
        trade = execute_sell(symbol, exit_price, shares, state, "Trailing Stop", exit_date)
        if trade:
            exits.append(trade)
        return exits

    # Priority 2: Partial profit at 1.5R
    if not pos["partial_exit_done"]:
        risk_per_share = pos.get("initial_risk_per_share", 0)
        if risk_per_share > 0:
            target_price = pos["entry_price"] + (PROFIT_TARGET_1_R * risk_per_share)
            if current_price >= target_price:
                shares_to_sell = max(1, int(pos["shares"] * PARTIAL_EXIT_PCT / 100))
                trade = execute_sell(symbol, current_price, shares_to_sell, state,
                                     f"Partial {PARTIAL_EXIT_PCT}% at {PROFIT_TARGET_1_R}R", exit_date)
                if trade:
                    exits.append(trade)
                    # Mark partial exit done (position may still exist with remaining shares)
                    if symbol in state["positions"]:
                        state["positions"][symbol]["partial_exit_done"] = True
                    return exits

    # Priority 3: Time-based exit (flat after N days)
    entry_date = datetime.strptime(pos["entry_date"], "%Y-%m-%d").date()
    days_held = (date.today() - entry_date).days
    if days_held >= TIME_EXIT_DAYS:
        move_pct = ((current_price - pos["entry_price"]) / pos["entry_price"]) * 100
        if abs(move_pct) < 2.0:
            trade = execute_sell(symbol, current_price, pos["shares"], state,
                                 f"Time Exit ({days_held}d, {move_pct:+.1f}%)", exit_date)
            if trade:
                exits.append(trade)
            return exits

    # Priority 4: Technical exit — close below EMA(20) AND MACD histogram negative
    ema20 = last.get("EMA20")
    macd_hist = last.get("MACD_Hist")
    if (pd.notna(ema20) and pd.notna(macd_hist) and
            current_price < ema20 and macd_hist < 0):
        # Only trigger if position is in profit (avoid selling losers on weak signals)
        if current_price > pos["entry_price"]:
            trade = execute_sell(symbol, current_price, pos["shares"], state,
                                 "Technical (EMA20+MACD)", exit_date)
            if trade:
                exits.append(trade)
            return exits

    return exits


# ─── Display Functions ────────────────────────────────────────────────────────


def print_header():
    """Print scan header with timestamp."""
    print("\n" + "=" * 78)
    print("  NSE TRADING BOT v2 — Nifty 50 Multi-Indicator Scanner")
    print(f"  Scan Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 78)


def print_market_regime(regime: dict):
    """Print market regime status."""
    print(f"\n{'─' * 78}")
    print("  MARKET REGIME")
    print(f"{'─' * 78}")
    if regime["nifty_close"] is not None:
        status_marker = "^" if regime["bullish"] else "v"
        print(f"  Nifty 50:   {regime['nifty_close']:>10.2f}")
        print(f"  200 EMA:    {regime['ema200']:>10.2f}")
        print(f"  Regime:     {regime['status']} {status_marker}")
        if not regime["bullish"]:
            print("  >> New buys SUPPRESSED (bearish regime)")
    else:
        print(f"  Status: {regime['status']}")


def print_signals(signals: list[dict]):
    """Print table of new buy signals with score breakdown."""
    print(f"\n{'─' * 78}")
    print("  NEW BUY SIGNALS")
    print(f"{'─' * 78}")
    if not signals:
        print("  No new buy signals found.")
        return

    for s in signals:
        print(f"\n  {s['symbol']} ({s['sector']})  —  Score: {s['score']}/5")
        print(f"    Entry: {s['price']:.2f}  |  Stop: {s['stop_loss']:.2f}  |  "
              f"ATR: {s['atr']:.2f}  |  Risk/sh: {s['risk_per_share']:.2f}  |  "
              f"Shares: {s['shares']}")
        # Score breakdown
        for indicator, result in s["breakdown"].items():
            marker = "+" if result.startswith("YES") else "-"
            print(f"    [{marker}] {indicator}: {result}")


def print_skipped_signals(skipped: list[dict]):
    """Print signals that were valid but rejected by risk limits."""
    print(f"\n{'─' * 78}")
    print("  SKIPPED SIGNALS (rejected by risk limits)")
    print(f"{'─' * 78}")
    if not skipped:
        print("  None.")
        return

    for s in skipped:
        print(f"  {s['symbol']:<15} Score: {s['score']}/5  —  {s['reason']}")


def print_sells(sells: list[dict]):
    """Print table of sell/stop-loss triggers."""
    print(f"\n{'─' * 78}")
    print("  EXITS TRIGGERED")
    print(f"{'─' * 78}")
    if not sells:
        print("  No exits triggered.")
        return

    header = f"  {'Symbol':<12}{'Reason':<28}{'Entry':>10}{'Exit':>10}{'Shares':>8}{'P&L':>12}"
    print(header)
    print(f"  {'-' * 78}")
    for t in sells:
        reason_display = t["exit_reason"][:26]
        pnl_str = f"{t['pnl']:>+12.2f}"
        print(
            f"  {t['symbol']:<12}{reason_display:<28}"
            f"{t['entry_price']:>10.2f}{t['exit_price']:>10.2f}"
            f"{t['shares']:>8}{pnl_str}"
        )


def print_portfolio(state: dict):
    """Print current open positions with trailing stops and R-multiples."""
    print(f"\n{'─' * 78}")
    print("  OPEN POSITIONS")
    print(f"{'─' * 78}")

    positions = state["positions"]
    if not positions:
        print("  No open positions.")
    else:
        header = (
            f"  {'Symbol':<12}{'Sector':<10}{'Entry':>9}{'Curr':>9}"
            f"{'Trail Stp':>10}{'Shares':>7}{'R-Mult':>8}{'Unrl P&L':>12}"
        )
        print(header)
        print(f"  {'-' * 75}")

        total_invested = 0
        total_current = 0

        for symbol, pos in positions.items():
            current = fetch_current_price(symbol)
            if current is None:
                current = pos["entry_price"]

            invested = pos["entry_price"] * pos["shares"]
            cur_val = current * pos["shares"]
            unrealized = cur_val - invested
            total_invested += invested
            total_current += cur_val

            # R-multiple
            risk_per_share = pos.get("initial_risk_per_share", 0)
            if risk_per_share > 0:
                r_mult = (current - pos["entry_price"]) / risk_per_share
                r_str = f"{r_mult:>+7.1f}R"
            else:
                r_str = f"{'N/A':>8}"

            sector = pos.get("sector", "?")[:9]
            partial_marker = "*" if pos.get("partial_exit_done") else " "

            print(
                f"  {symbol:<12}{sector:<10}{pos['entry_price']:>9.2f}{current:>9.2f}"
                f"{pos['trailing_stop']:>10.2f}{pos['shares']:>6}{partial_marker}"
                f"{r_str}{unrealized:>+12.2f}"
            )

        print(f"  {'-' * 75}")
        print(f"  {'* = partial exit taken'}")
        print(f"  {'Total Invested:':<40}{total_invested:>12.2f}")
        print(f"  {'Total Current Value:':<40}{total_current:>12.2f}")
        print(f"  {'Total Unrealized P&L:':<40}{total_current - total_invested:>+12.2f}")

    # Summary
    capital = state["capital"]
    invested_val = sum(
        pos["entry_price"] * pos["shares"] for pos in positions.values()
    )
    realized_pnl = sum(t["pnl"] for t in state["trade_history"])
    portfolio_val = capital + invested_val
    exposure_pct = (invested_val / portfolio_val * 100) if portfolio_val > 0 else 0

    print(f"\n{'─' * 78}")
    print("  ACCOUNT SUMMARY")
    print(f"{'─' * 78}")
    print(f"  {'Available Capital:':<40}{capital:>12.2f}")
    print(f"  {'Invested (at cost):':<40}{invested_val:>12.2f}")
    print(f"  {'Total (capital + invested):':<40}{portfolio_val:>12.2f}")
    print(f"  {'Realized P&L:':<40}{realized_pnl:>+12.2f}")
    overall_pnl_pct = ((portfolio_val) / INITIAL_CAPITAL - 1) * 100
    print(f"  {'Overall P&L %:':<40}{overall_pnl_pct:>+11.2f}%")
    print(f"  {'Exposure:':<40}{exposure_pct:>10.1f}%")
    print(f"  {'Open Positions:':<40}{len(positions):>10}/{MAX_POSITIONS}")

    # Sector breakdown
    if positions:
        sectors = {}
        for pos in positions.values():
            s = pos.get("sector", "Unknown")
            sectors[s] = sectors.get(s, 0) + 1
        sector_str = ", ".join(f"{s}:{c}" for s, c in sorted(sectors.items()))
        print(f"  {'Sector Breakdown:':<40}{sector_str}")

    print()


def fetch_current_price(symbol: str) -> float | None:
    """Quick fetch of latest close price for portfolio display."""
    try:
        ticker = f"{symbol}.NS"
        df = yf.download(ticker, period="5d", progress=False, auto_adjust=True)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return float(df["Close"].iloc[-1])
    except Exception:
        return None


def print_trade_history(state: dict):
    """Print past closed trades."""
    print(f"\n{'─' * 78}")
    print("  TRADE HISTORY")
    print(f"{'─' * 78}")

    history = state["trade_history"]
    if not history:
        print("  No completed trades yet.")
        print()
        return

    header = (
        f"  {'Symbol':<10}{'Entry Dt':<12}{'Exit Dt':<12}"
        f"{'Entry':>9}{'Exit':>9}{'Shares':>7}{'P&L':>11}  {'Reason'}"
    )
    print(header)
    print(f"  {'-' * 80}")

    total_pnl = 0
    for t in history:
        total_pnl += t["pnl"]
        reason = t.get("exit_reason", "Unknown")[:25]
        print(
            f"  {t['symbol']:<10}{t['entry_date']:<12}{t['exit_date']:<12}"
            f"{t['entry_price']:>9.2f}{t['exit_price']:>9.2f}"
            f"{t['shares']:>7}{t['pnl']:>+11.2f}  {reason}"
        )

    print(f"  {'-' * 80}")
    print(f"  {'Total Realized P&L:':<52}{total_pnl:>+11.2f}")
    wins = sum(1 for t in history if t["pnl"] > 0)
    losses = sum(1 for t in history if t["pnl"] <= 0)
    print(f"  Wins: {wins}  |  Losses: {losses}  |  Total Trades: {len(history)}")
    print()


def print_stats(state: dict):
    """Print detailed trading statistics."""
    print(f"\n{'─' * 78}")
    print("  TRADING STATISTICS")
    print(f"{'─' * 78}")

    history = state["trade_history"]
    if not history:
        print("  No completed trades to analyze.")
        print()
        return

    total_trades = len(history)
    wins = [t for t in history if t["pnl"] > 0]
    losses = [t for t in history if t["pnl"] <= 0]
    win_count = len(wins)
    loss_count = len(losses)

    # Win rate
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

    # Average win / loss
    avg_win = sum(t["pnl"] for t in wins) / win_count if win_count > 0 else 0
    avg_loss = sum(t["pnl"] for t in losses) / loss_count if loss_count > 0 else 0

    # Profit factor
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Expectancy
    expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)

    # Max drawdown (equity curve)
    equity = INITIAL_CAPITAL
    peak = equity
    max_dd = 0
    max_dd_pct = 0
    for t in history:
        equity += t["pnl"]
        if equity > peak:
            peak = equity
        dd = peak - equity
        dd_pct = (dd / peak * 100) if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd_pct

    # Total P&L
    total_pnl = sum(t["pnl"] for t in history)
    total_pnl_pct = (total_pnl / INITIAL_CAPITAL) * 100

    # Exit reason breakdown
    reasons = {}
    for t in history:
        reason = t.get("exit_reason", "Unknown")
        # Normalize partial exit reasons
        if reason.startswith("Partial"):
            reason = "Partial Profit"
        elif reason.startswith("Time Exit"):
            reason = "Time Exit"
        reasons[reason] = reasons.get(reason, 0) + 1

    # Average holding period
    hold_days = []
    for t in history:
        try:
            entry = datetime.strptime(t["entry_date"], "%Y-%m-%d").date()
            exit_d = datetime.strptime(t["exit_date"], "%Y-%m-%d").date()
            hold_days.append((exit_d - entry).days)
        except (ValueError, KeyError):
            pass
    avg_hold = sum(hold_days) / len(hold_days) if hold_days else 0

    # Print
    print(f"  {'Total Trades:':<35}{total_trades:>10}")
    print(f"  {'Wins:':<35}{win_count:>10}")
    print(f"  {'Losses:':<35}{loss_count:>10}")
    print(f"  {'Win Rate:':<35}{win_rate:>9.1f}%")
    print(f"  {'Average Win:':<35}{avg_win:>+10.2f}")
    print(f"  {'Average Loss:':<35}{avg_loss:>+10.2f}")
    print(f"  {'Profit Factor:':<35}{profit_factor:>10.2f}")
    print(f"  {'Expectancy (per trade):':<35}{expectancy:>+10.2f}")
    print(f"  {'Total P&L:':<35}{total_pnl:>+10.2f}")
    print(f"  {'Total P&L %:':<35}{total_pnl_pct:>+9.2f}%")
    print(f"  {'Max Drawdown:':<35}{max_dd:>+10.2f}")
    print(f"  {'Max Drawdown %:':<35}{max_dd_pct:>9.2f}%")
    print(f"  {'Avg Holding Period:':<35}{avg_hold:>8.1f} days")

    print(f"\n  Exit Reason Breakdown:")
    print(f"  {'-' * 40}")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        pct = count / total_trades * 100
        print(f"    {reason:<28}{count:>5} ({pct:.1f}%)")

    # Sector performance
    sector_pnl = {}
    for t in history:
        s = t.get("sector", "Unknown")
        if s not in sector_pnl:
            sector_pnl[s] = {"pnl": 0, "trades": 0}
        sector_pnl[s]["pnl"] += t["pnl"]
        sector_pnl[s]["trades"] += 1

    if sector_pnl:
        print(f"\n  Sector Performance:")
        print(f"  {'-' * 40}")
        for s, data in sorted(sector_pnl.items(), key=lambda x: -x[1]["pnl"]):
            print(f"    {s:<20}{data['pnl']:>+10.2f}  ({data['trades']} trades)")

    print()


# ─── Core Scan Logic ──────────────────────────────────────────────────────────


def scan_and_trade(dry_run: bool = False, min_score: int = MIN_BUY_SCORE,
                   no_regime_filter: bool = False):
    """Core scan loop: check exits, check market regime, score buy signals."""
    state = load_state()
    print_header()

    # Market regime
    regime = check_market_regime()
    print_market_regime(regime)

    regime_allows_buys = regime["bullish"] or no_regime_filter
    if no_regime_filter and not regime["bullish"]:
        print("  >> Regime filter OVERRIDDEN by --no-regime-filter")

    print(f"\n  Scanning {len(NIFTY_50)} Nifty 50 stocks "
          f"(min score: {min_score}/5, dry-run: {'ON' if dry_run else 'OFF'})...")

    buy_signals = []
    sell_triggers = []
    skipped_signals = []
    min_bars = max(EMA_LONG, SUPPORT_LOOKBACK, RSI_PERIOD, MACD_SLOW) + 10

    for i, symbol in enumerate(NIFTY_50, 1):
        progress = f"  [{i:>2}/{len(NIFTY_50)}] {symbol:<15}"
        print(progress, end="", flush=True)

        df = fetch_data(symbol)
        if df is None or len(df) < min_bars:
            print("— insufficient data, skipped")
            continue

        df = calculate_indicators(df)

        # Check exits for open positions first
        if symbol in state["positions"]:
            if dry_run:
                pos = state["positions"][symbol]
                current = float(df["Close"].iloc[-1])
                risk = pos.get("initial_risk_per_share", 1)
                r_mult = (current - pos["entry_price"]) / risk if risk > 0 else 0
                print(f"— holding (R={r_mult:+.1f}, trail={pos['trailing_stop']:.2f})")
            else:
                exits = check_exits(symbol, df, state)
                if exits:
                    sell_triggers.extend(exits)
                    reasons = ", ".join(t["exit_reason"] for t in exits)
                    print(f"— EXIT: {reasons}")
                else:
                    # Update trailing stop even when no exit
                    pos = state["positions"][symbol]
                    current_high = float(df["High"].iloc[-1])
                    update_trailing_stop(pos, current_high)
                    current = float(df["Close"].iloc[-1])
                    risk = pos.get("initial_risk_per_share", 1)
                    r_mult = (current - pos["entry_price"]) / risk if risk > 0 else 0
                    print(f"— holding (R={r_mult:+.1f}, trail={pos['trailing_stop']:.2f})")
        else:
            # Score for potential buy
            score_info = score_buy_signal(df)
            score = score_info["score"]
            rsi_val = df["RSI"].iloc[-1]
            rsi_str = f"RSI={rsi_val:.1f}" if pd.notna(rsi_val) else "RSI=N/A"

            if score >= min_score:
                if not regime_allows_buys:
                    skipped_signals.append({
                        "symbol": symbol, "score": score,
                        "reason": "Bearish market regime"
                    })
                    print(f"— score {score}/5 ({rsi_str}) — blocked by regime filter")
                    continue

                # Check risk limits
                rejection = check_risk_limits(state, symbol)
                if rejection:
                    skipped_signals.append({
                        "symbol": symbol, "score": score,
                        "reason": rejection
                    })
                    print(f"— score {score}/5 ({rsi_str}) — {rejection}")
                    continue

                if dry_run:
                    # Show signal without executing
                    buy_signals.append({
                        "symbol": symbol,
                        "price": round(float(df["Close"].iloc[-1]), 2),
                        "shares": 0,
                        "stop_loss": round(float(df["Close"].iloc[-1]) - ATR_STOP_MULTIPLIER * float(df["ATR"].iloc[-1]), 2),
                        "atr": round(float(df["ATR"].iloc[-1]), 2),
                        "risk_per_share": round(ATR_STOP_MULTIPLIER * float(df["ATR"].iloc[-1]), 2),
                        "sector": SECTOR_MAP.get(symbol, "Unknown"),
                        "score": score,
                        "breakdown": score_info["breakdown"],
                    })
                    print(f"— BUY SIGNAL score {score}/5 ({rsi_str}) [DRY RUN]")
                else:
                    trade = execute_buy(symbol, df, state, score_info)
                    if trade:
                        buy_signals.append(trade)
                        print(f"— BUY score {score}/5 ({rsi_str}, {trade['shares']} shares)")
                    else:
                        print(f"— score {score}/5 but insufficient capital")
            else:
                print(f"— score {score}/5 ({rsi_str})")

    if not dry_run:
        save_state(state)

    # Print results
    print_signals(buy_signals)
    print_skipped_signals(skipped_signals)
    print_sells(sell_triggers)
    print_portfolio(state)


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="NSE Trading Bot v2 — Nifty 50 Multi-Indicator Scanner with Paper Trading"
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        default=True,
        help="Scan stocks and auto-execute paper trades (default)",
    )
    parser.add_argument(
        "--portfolio",
        action="store_true",
        help="Show current portfolio only",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Show trade history only",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset paper trading state",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show signals without executing trades",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=MIN_BUY_SCORE,
        metavar="N",
        help=f"Minimum buy score (0-5, default {MIN_BUY_SCORE})",
    )
    parser.add_argument(
        "--no-regime-filter",
        action="store_true",
        help="Allow buys even in bearish market regime",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show detailed trading statistics",
    )

    args = parser.parse_args()

    if args.reset:
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
        print("  Paper trading state has been reset.")
        print(f"  Starting capital: {INITIAL_CAPITAL:,.2f}")
        return

    if args.stats:
        state = load_state()
        print_header()
        print_stats(state)
        return

    if args.portfolio:
        state = load_state()
        print_header()
        print_portfolio(state)
        return

    if args.history:
        state = load_state()
        print_header()
        print_trade_history(state)
        return

    # Default: scan and trade
    scan_and_trade(
        dry_run=args.dry_run,
        min_score=args.min_score,
        no_regime_filter=args.no_regime_filter,
    )


if __name__ == "__main__":
    main()
