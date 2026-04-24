from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests


FINNHUB_COMPANY_NEWS_URL = "https://finnhub.io/api/v1/company-news"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Finnhub headlines and Yahoo Finance prices.")
    parser.add_argument("--symbols", nargs="+", default=["SPY"], help="Ticker symbols, e.g. SPY AAPL MSFT")
    parser.add_argument("--start", required=True, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end", required=True, help="End date in YYYY-MM-DD format")
    parser.add_argument("--headlines-out", default="data/headlines.csv")
    parser.add_argument("--prices-out", default="data/prices.csv")
    parser.add_argument(
        "--skip-news",
        action="store_true",
        help="Download Yahoo prices only.",
    )
    parser.add_argument(
        "--skip-prices",
        action="store_true",
        help="Download Finnhub headlines only.",
    )
    return parser.parse_args()


def require_date(value: str) -> str:
    datetime.strptime(value, "%Y-%m-%d")
    return value


def get_env_value(name: str, env_path: str = ".env") -> str | None:
    value = os.environ.get(name)
    if value:
        return value

    path = Path(env_path)
    if not path.exists():
        return None

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        if key.strip() == name:
            return raw_value.strip().strip('"').strip("'")
    return None


def exclusive_yahoo_end(end: str) -> str:
    # yfinance treats end as exclusive, so include the requested end date.
    date = datetime.strptime(end, "%Y-%m-%d").date() + timedelta(days=1)
    return date.isoformat()


def fetch_finnhub_headlines(symbols: list[str], start: str, end: str, api_key: str) -> pd.DataFrame:
    
    rows = []
    for symbol in symbols:
        response = requests.get(
            FINNHUB_COMPANY_NEWS_URL,
            params={"symbol": symbol, "from": start, "to": end, "token": api_key},
            timeout=30,
        )
        response.raise_for_status()
        for item in response.json():
            headline = item.get("headline")
            timestamp = item.get("datetime")
            if not headline or not timestamp:
                continue
            date = datetime.fromtimestamp(int(timestamp), tz=timezone.utc).date().isoformat()
            rows.append({"date": date, "symbol": symbol.upper(), "headline": headline})

    frame = pd.DataFrame(rows, columns=["date", "symbol", "headline"])
    if frame.empty:
        return frame
    return frame.drop_duplicates().sort_values(["symbol", "date", "headline"]).reset_index(drop=True)


def fetch_yahoo_prices(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    rows: list[pd.DataFrame] = []
    for symbol in symbols:
        data = yf.download(
            symbol,
            start=start,
            end=exclusive_yahoo_end(end),
            auto_adjust=False,
            progress=False,
            group_by="column",
        )
        if data.empty:
            print(f"No Yahoo Finance rows returned for {symbol}.")
            continue

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        close_column = "Adj Close" if "Adj Close" in data.columns else "Close"
        symbol_prices = (
            data.reset_index()
            .rename(columns={"Date": "date", close_column: "close", "Volume": "volume"})
            [["date", "close", "volume"]]
        )
        symbol_prices["date"] = pd.to_datetime(symbol_prices["date"]).dt.date.astype(str)
        symbol_prices["symbol"] = symbol.upper()
        rows.append(symbol_prices[["date", "symbol", "close", "volume"]])

    if not rows:
        return pd.DataFrame(columns=["date", "symbol", "close", "volume"])
    return pd.concat(rows, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)


def write_to_csv(df: pd.DataFrame, path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Wrote {len(df):,} rows to {output}")


def download() -> None:
    args = parse_args()
    start = require_date(args.start)
    end = require_date(args.end)
    symbols = [symbol.upper() for symbol in args.symbols]

    if not args.skip_news:
        api_key = get_env_value("FINNHUB_API_KEY")
        headlines = fetch_finnhub_headlines(symbols, start, end, api_key)
        write_to_csv(headlines, args.headlines_out)

    if not args.skip_prices:
        prices = fetch_yahoo_prices(symbols, start, end)
        write_to_csv(prices, args.prices_out)


if __name__ == "__main__":
    download()
