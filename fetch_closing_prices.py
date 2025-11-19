#!/usr/bin/env python3
"""
Récupère les prix de clôture quotidiens via yfinance pour une liste de tickers
et écrit un CSV (par défaut data/closing_prices.csv).

Exemple :
    python fetch_closing_prices.py --output data/closing_prices.csv
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import yfinance as yf


def fetch_closing_prices(tickers: list[str], period: str = "1mo", interval: str = "1d") -> pd.DataFrame:
    """
    Télécharge les prix de clôture ajustés pour les tickers spécifiés.
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    # Neutralise d’éventuelles options d’impersonation qui posent problème
    os.environ.pop("YF_IMPERSONATE", None)
    os.environ.pop("YF_SCRAPER_IMPERSONATE", None)
    try:
        yf.set_config(proxy=None)
    except Exception:
        pass

    data = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if data.empty:
        raise RuntimeError(f"Aucune donnée récupérée pour {tickers} sur {period}.")

    # yfinance renvoie un MultiIndex quand il y a plusieurs tickers
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.levels[0]:
            prices = data["Adj Close"]
        else:
            prices = data["Close"]
    else:
        if "Adj Close" in data.columns:
            prices = data[["Adj Close"]].copy()
        elif "Close" in data.columns:
            prices = data[["Close"]].copy()
        else:
            raise RuntimeError("Colonnes de prix introuvables dans les données yfinance.")
        prices.columns = tickers

    prices.reset_index(inplace=True)
    return prices


def main() -> None:
    parser = argparse.ArgumentParser(description="Télécharge les prix de clôture via yfinance et écrit un CSV.")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["SPY", "AAPL", "MSFT"],  # SP500 ETF + Apple + Microsoft
        help="Liste des tickers Yahoo Finance (par défaut: SPY AAPL MSFT).",
    )
    parser.add_argument(
        "--period",
        default="1mo",
        help="Fenêtre temporelle yfinance (ex: 1mo, 3mo, 1y).",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="Intervalle yfinance (ex: 1d, 1h).",
    )
    parser.add_argument(
        "--output",
        default="data/closing_prices.csv",
        help="Chemin du CSV de sortie (défaut: data/closing_prices.csv).",
    )

    args = parser.parse_args()

    df = fetch_closing_prices(args.tickers, period=args.period, interval=args.interval)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"CSV écrit: {out_path} (shape={df.shape})")


if __name__ == "__main__":
    main()

