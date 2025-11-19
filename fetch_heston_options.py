#!/usr/bin/env python3
"""Utility script to download option chains for the Heston module."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

os.environ.setdefault("YF_USE_EXTERNAL_FASTLIBS", "0")
for var in ("YF_IMPERSONATE", "YF_SCRAPER_IMPERSONATE"):
    os.environ.pop(var, None)

try:
    import yfinance as yf
except Exception as exc:  # pragma: no cover - safety net
    print(f"Unable to import yfinance: {exc}", file=sys.stderr)
    sys.exit(1)

try:  # pragma: no cover - defensive
    yf.set_config(proxy=None)
except Exception:
    pass


def select_monthly_expirations(expirations, years_ahead: float = 2.5) -> list[str]:
    today = pd.Timestamp.utcnow().date()
    limit_date = today + pd.Timedelta(days=365 * years_ahead)
    monthly: Dict[Tuple[int, int], Tuple[pd.Timestamp, str]] = {}
    for exp in expirations:
        try:
            exp_ts = pd.Timestamp(exp)
        except Exception:
            continue
        exp_date = exp_ts.date()
        if not (today < exp_date <= limit_date):
            continue
        key = (exp_date.year, exp_date.month)
        if key not in monthly or exp_ts < monthly[key][0]:
            monthly[key] = (exp_ts, exp)
    return [item[1] for item in sorted(monthly.values(), key=lambda x: x[0])]


def get_spot(symbol: str) -> float | None:
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        return None
    return None


def collect_options(symbol: str, option_type: str, years_ahead: float, fallback_spot: float | None) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    spot = get_spot(symbol)
    if spot is None:
        if fallback_spot is None:
            raise RuntimeError("Unable to determine spot price and no fallback provided.")
        spot = fallback_spot

    expirations = ticker.options or []
    if not expirations:
        raise RuntimeError(f"No option expirations found for {symbol}.")
    selected = select_monthly_expirations(expirations, years_ahead)

    rows: list[dict] = []
    now = pd.Timestamp.utcnow().tz_localize(None)
    price_col = "C_mkt" if option_type == "call" else "P_mkt"
    for expiry in selected:
        try:
            expiry_dt = pd.Timestamp(expiry)
        except Exception:
            continue
        T = max((expiry_dt - now).total_seconds() / (365.0 * 24 * 3600.0), 0.0)
        try:
            chain = ticker.option_chain(expiry)
        except Exception:
            continue
        data = chain.calls if option_type == "call" else chain.puts
        for _, row in data.iterrows():
            rows.append(
                {
                    "S0": spot,
                    "K": float(row["strike"]),
                    "T": T,
                    price_col: float(row["lastPrice"]),
                    "iv_market": float(row.get("impliedVolatility", float("nan"))),
                }
            )

    if not rows:
        raise RuntimeError(f"No option rows retrieved for {symbol} ({option_type}).")
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download option chains for Heston module.")
    parser.add_argument("--ticker", required=True, help="Ticker symbol (yfinance)")
    parser.add_argument("--type", choices=["call", "put"], required=True, help="Option type to download")
    parser.add_argument("--years", type=float, default=2.5, help="Look-ahead horizon in years")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--fallback-spot", type=float, default=None, help="Fallback spot if download fails")
    args = parser.parse_args()

    try:
        df = collect_options(args.ticker.upper(), args.type, args.years, args.fallback_spot)
    except Exception as exc:
        print(f"Failed to download options: {exc}", file=sys.stderr)
        return 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
