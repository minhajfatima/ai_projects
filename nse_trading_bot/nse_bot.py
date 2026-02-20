#!/usr/bin/env python3
"""NSE Trading Bot - RSI-based scanner for Nifty 50 stocks with paper trading."""

import argparse
import json
import os
import sys
from datetime import datetime, date

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator

# ─── Configuration ────────────────────────────────────────────────────────────

RSI_PERIOD = 14
RSI_THRESHOLD = 40
SMA_PERIOD = 20
INITIAL_CAPITAL = 1_000_000  # 10 lakh INR
POSITION_SIZE_PCT = 5        # % of capital per trade
DATA_PERIOD = "6mo"          # how far back to fetch

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

# ─── Data Functions ───────────────────────────────────────────────────────────


def fetch_data(symbol: str) -> pd.DataFrame | None:
    """Fetch daily OHLCV data from Yahoo Finance."""
    ticker = f"{symbol}.NS"
    try:
        df = yf.download(ticker, period=DATA_PERIOD, progress=False, auto_adjust=True)
        if df.empty:
            return None
        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        print(f"  [!] Error fetching {symbol}: {e}")
        return None


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI(14) and SMA(20) columns to the DataFrame."""
    df = df.copy()
    df["RSI"] = RSIIndicator(close=df["Close"], window=RSI_PERIOD).rsi()
    df["SMA20"] = SMAIndicator(close=df["Close"], window=SMA_PERIOD).sma_indicator()
    return df


# ─── Signal Functions ─────────────────────────────────────────────────────────


def check_buy_signal(df: pd.DataFrame) -> bool:
    """Return True if latest RSI <= threshold."""
    if df.empty or pd.isna(df["RSI"].iloc[-1]):
        return False
    return df["RSI"].iloc[-1] <= RSI_THRESHOLD


def check_sell_signal(df: pd.DataFrame, position: dict) -> bool:
    """Return True if close >= SMA(20)."""
    if df.empty or pd.isna(df["SMA20"].iloc[-1]):
        return False
    return df["Close"].iloc[-1] >= df["SMA20"].iloc[-1]


def check_stop_loss(df: pd.DataFrame, position: dict) -> bool:
    """Return True if the candle low breaches the stop-loss level."""
    if df.empty:
        return False
    return df["Low"].iloc[-1] <= position["stop_loss"]


# ─── State Management ─────────────────────────────────────────────────────────


def load_state() -> dict:
    """Load paper trading state from JSON, or initialize fresh."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {
        "capital": INITIAL_CAPITAL,
        "positions": {},
        "trade_history": [],
    }


def save_state(state: dict) -> None:
    """Persist paper trading state to JSON."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ─── Trade Execution ──────────────────────────────────────────────────────────


def execute_buy(symbol: str, df: pd.DataFrame, state: dict) -> dict | None:
    """Paper buy: allocate capital, record position with stop loss = buy candle low."""
    price = float(df["Close"].iloc[-1])
    stop_loss = float(df["Low"].iloc[-1])
    allocation = state["capital"] * (POSITION_SIZE_PCT / 100)

    if allocation < price:
        return None  # not enough capital for even 1 share

    shares = int(allocation // price)
    if shares == 0:
        return None

    cost = shares * price
    state["capital"] -= cost
    entry_date = str(df.index[-1].date())

    state["positions"][symbol] = {
        "entry_price": round(price, 2),
        "shares": shares,
        "stop_loss": round(stop_loss, 2),
        "entry_date": entry_date,
    }

    return {
        "symbol": symbol,
        "price": round(price, 2),
        "shares": shares,
        "stop_loss": round(stop_loss, 2),
        "rsi": round(float(df["RSI"].iloc[-1]), 2),
    }


def execute_sell(symbol: str, df: pd.DataFrame, state: dict, reason: str) -> dict | None:
    """Paper sell: return proceeds, record in history."""
    if symbol not in state["positions"]:
        return None

    pos = state["positions"][symbol]
    exit_price = float(df["Close"].iloc[-1])

    # If stopped out, use stop-loss price as exit
    if reason == "Stop Loss" and df["Low"].iloc[-1] <= pos["stop_loss"]:
        exit_price = pos["stop_loss"]

    proceeds = pos["shares"] * exit_price
    pnl = (exit_price - pos["entry_price"]) * pos["shares"]

    state["capital"] += proceeds

    trade_record = {
        "symbol": symbol,
        "entry_price": pos["entry_price"],
        "exit_price": round(exit_price, 2),
        "shares": pos["shares"],
        "pnl": round(pnl, 2),
        "entry_date": pos["entry_date"],
        "exit_date": str(df.index[-1].date()),
        "exit_reason": reason,
    }
    state["trade_history"].append(trade_record)
    del state["positions"][symbol]

    return trade_record


# ─── Display Functions ────────────────────────────────────────────────────────


def print_header():
    """Print scan header with timestamp."""
    print("\n" + "=" * 70)
    print("  NSE TRADING BOT — Nifty 50 RSI Scanner")
    print(f"  Scan Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


def print_signals(signals: list[dict]):
    """Print table of new buy signals."""
    print(f"\n{'─' * 70}")
    print("  NEW BUY SIGNALS")
    print(f"{'─' * 70}")
    if not signals:
        print("  No new buy signals found.")
        return

    header = f"  {'Symbol':<15}{'RSI':>8}{'Entry':>12}{'Stop Loss':>12}{'Shares':>8}"
    print(header)
    print(f"  {'-' * 55}")
    for s in signals:
        print(
            f"  {s['symbol']:<15}{s['rsi']:>8.2f}{s['price']:>12.2f}"
            f"{s['stop_loss']:>12.2f}{s['shares']:>8}"
        )


def print_sells(sells: list[dict]):
    """Print table of sell/stop-loss triggers."""
    print(f"\n{'─' * 70}")
    print("  SELL / STOP-LOSS TRIGGERS")
    print(f"{'─' * 70}")
    if not sells:
        print("  No exits triggered.")
        return

    header = f"  {'Symbol':<12}{'Reason':<15}{'Entry':>10}{'Exit':>10}{'P&L':>12}"
    print(header)
    print(f"  {'-' * 55}")
    for t in sells:
        pnl_str = f"{t['pnl']:>+12.2f}"
        print(
            f"  {t['symbol']:<12}{t['exit_reason']:<15}"
            f"{t['entry_price']:>10.2f}{t['exit_price']:>10.2f}{pnl_str}"
        )


def print_portfolio(state: dict):
    """Print current open positions and capital summary."""
    print(f"\n{'─' * 70}")
    print("  OPEN POSITIONS")
    print(f"{'─' * 70}")

    positions = state["positions"]
    if not positions:
        print("  No open positions.")
    else:
        header = (
            f"  {'Symbol':<12}{'Entry':>10}{'Current':>10}"
            f"{'Stop Loss':>12}{'Shares':>8}{'Unrl P&L':>12}"
        )
        print(header)
        print(f"  {'-' * 64}")

        total_invested = 0
        total_current = 0

        for symbol, pos in positions.items():
            # Fetch current price
            current = fetch_current_price(symbol)
            if current is None:
                current = pos["entry_price"]

            invested = pos["entry_price"] * pos["shares"]
            cur_val = current * pos["shares"]
            unrealized = cur_val - invested
            total_invested += invested
            total_current += cur_val

            print(
                f"  {symbol:<12}{pos['entry_price']:>10.2f}{current:>10.2f}"
                f"{pos['stop_loss']:>12.2f}{pos['shares']:>8}{unrealized:>+12.2f}"
            )

        print(f"  {'-' * 64}")
        print(f"  {'Total Invested:':<34}{total_invested:>12.2f}")
        print(f"  {'Total Current Value:':<34}{total_current:>12.2f}")
        print(f"  {'Total Unrealized P&L:':<34}{total_current - total_invested:>+12.2f}")

    # Summary
    capital = state["capital"]
    invested_val = sum(
        pos["entry_price"] * pos["shares"] for pos in positions.values()
    )
    realized_pnl = sum(t["pnl"] for t in state["trade_history"])

    print(f"\n{'─' * 70}")
    print("  ACCOUNT SUMMARY")
    print(f"{'─' * 70}")
    print(f"  {'Available Capital:':<34}{capital:>12.2f}")
    print(f"  {'Invested (at cost):':<34}{invested_val:>12.2f}")
    print(f"  {'Total (capital + invested):':<34}{capital + invested_val:>12.2f}")
    print(f"  {'Realized P&L:':<34}{realized_pnl:>+12.2f}")
    overall_pnl_pct = ((capital + invested_val) / INITIAL_CAPITAL - 1) * 100
    print(f"  {'Overall P&L %:':<34}{overall_pnl_pct:>+11.2f}%")
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
    print(f"\n{'─' * 70}")
    print("  TRADE HISTORY")
    print(f"{'─' * 70}")

    history = state["trade_history"]
    if not history:
        print("  No completed trades yet.")
        print()
        return

    header = (
        f"  {'Symbol':<10}{'Entry Dt':<12}{'Exit Dt':<12}"
        f"{'Entry':>9}{'Exit':>9}{'Shares':>7}{'P&L':>11}{'Reason':<15}"
    )
    print(header)
    print(f"  {'-' * 75}")

    total_pnl = 0
    for t in history:
        total_pnl += t["pnl"]
        print(
            f"  {t['symbol']:<10}{t['entry_date']:<12}{t['exit_date']:<12}"
            f"{t['entry_price']:>9.2f}{t['exit_price']:>9.2f}"
            f"{t['shares']:>7}{t['pnl']:>+11.2f}  {t['exit_reason']:<15}"
        )

    print(f"  {'-' * 75}")
    print(f"  {'Total Realized P&L:':<52}{total_pnl:>+11.2f}")
    wins = sum(1 for t in history if t["pnl"] > 0)
    losses = sum(1 for t in history if t["pnl"] <= 0)
    print(f"  Wins: {wins}  |  Losses: {losses}  |  Total Trades: {len(history)}")
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────


def scan_and_trade():
    """Core scan loop: check exits for open positions, then check for new buys."""
    state = load_state()
    print_header()
    print(f"\n  Scanning {len(NIFTY_50)} Nifty 50 stocks...")

    buy_signals = []
    sell_triggers = []

    for i, symbol in enumerate(NIFTY_50, 1):
        progress = f"  [{i:>2}/{len(NIFTY_50)}] {symbol:<15}"
        print(progress, end="", flush=True)

        df = fetch_data(symbol)
        if df is None or len(df) < max(RSI_PERIOD, SMA_PERIOD) + 5:
            print("— insufficient data, skipped")
            continue

        df = calculate_indicators(df)

        # Check exits for open positions
        if symbol in state["positions"]:
            pos = state["positions"][symbol]
            if check_stop_loss(df, pos):
                trade = execute_sell(symbol, df, state, "Stop Loss")
                if trade:
                    sell_triggers.append(trade)
                    print(f"— STOP LOSS triggered")
                    continue
            elif check_sell_signal(df, pos):
                trade = execute_sell(symbol, df, state, "SMA20 Touch")
                if trade:
                    sell_triggers.append(trade)
                    print(f"— SELL signal (SMA20)")
                    continue
            print(f"— holding position")
        else:
            # Check for new buy signal
            if check_buy_signal(df):
                trade = execute_buy(symbol, df, state)
                if trade:
                    buy_signals.append(trade)
                    print(f"— BUY signal (RSI={trade['rsi']:.1f})")
                else:
                    print(f"— buy signal but insufficient capital")
            else:
                rsi_val = df["RSI"].iloc[-1]
                if pd.notna(rsi_val):
                    print(f"— RSI={rsi_val:.1f}")
                else:
                    print(f"— no signal")

    save_state(state)

    # Print results
    print_signals(buy_signals)
    print_sells(sell_triggers)
    print_portfolio(state)


def main():
    parser = argparse.ArgumentParser(
        description="NSE Trading Bot — Nifty 50 RSI Scanner with Paper Trading"
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

    args = parser.parse_args()

    if args.reset:
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
        print("  Paper trading state has been reset.")
        print(f"  Starting capital: {INITIAL_CAPITAL:,.2f}")
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
    scan_and_trade()


if __name__ == "__main__":
    main()
