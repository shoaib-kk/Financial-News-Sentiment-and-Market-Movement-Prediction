from __future__ import annotations

import numpy as np
import pandas as pd


LABEL_ORDER = ["down", "neutral", "up"]

POSITIVE_WORDS = {
    "beat",
    "boosts",
    "eases",
    "gain",
    "gains",
    "growth",
    "higher",
    "high",
    "improves",
    "jump",
    "jumps",
    "lifts",
    "optimism",
    "positive",
    "rallies",
    "rally",
    "rebound",
    "resilient",
    "strong",
    "surge",
    "upbeat",
}

NEGATIVE_WORDS = {
    "caution",
    "cautious",
    "concerns",
    "decline",
    "disappoints",
    "drops",
    "fall",
    "falls",
    "fears",
    "hot",
    "inflation",
    "lower",
    "pressure",
    "retreats",
    "selloff",
    "slips",
    "weakness",
    "weighs",
    "warn",
    "worries",
}


def label_return(value: float, threshold: float) -> str:
    if value > threshold:
        return "up"
    if value < -threshold:
        return "down"
    return "neutral"


def sentiment_score(text: str) -> float:
    tokens = {token.strip(".,:;!?()[]'\"").lower() for token in str(text).split()}
    if not tokens:
        return 0.0
    return (len(tokens & POSITIVE_WORDS) - len(tokens & NEGATIVE_WORDS)) / max(len(tokens), 1)


def make_supervised_frame(
    headlines: pd.DataFrame,
    prices: pd.DataFrame,
    horizon: int = 1,
    threshold: float = 0.005,
) -> pd.DataFrame:
    required_headlines = {"date", "symbol", "headline"}
    required_prices = {"date", "symbol", "close", "volume"}
    if not required_headlines.issubset(headlines.columns):
        raise ValueError(f"headlines must include columns: {sorted(required_headlines)}")
    if not required_prices.issubset(prices.columns):
        raise ValueError(f"prices must include columns: {sorted(required_prices)}")

    headlines = headlines.copy()
    prices = prices.copy()
    headlines["date"] = pd.to_datetime(headlines["date"])
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["symbol", "date"]).reset_index(drop=True)

    grouped = prices.groupby("symbol", group_keys=False)
    prices["return_1d"] = grouped["close"].pct_change()
    prices["return_3d_past"] = grouped["close"].pct_change(3)
    prices["volatility_5d"] = grouped["return_1d"].rolling(5).std().reset_index(level=0, drop=True)
    prices["volume_z_20d"] = grouped["volume"].transform(
        lambda s: (s - s.rolling(20, min_periods=3).mean()) / s.rolling(20, min_periods=3).std()
    )
    prices["future_close"] = grouped["close"].shift(-horizon)
    prices["future_return"] = (prices["future_close"] - prices["close"]) / prices["close"]
    prices["target"] = prices["future_return"].apply(lambda value: label_return(value, threshold))

    merged = headlines.merge(prices, on=["date", "symbol"], how="inner")
    merged["sentiment_score"] = merged["headline"].apply(sentiment_score)
    merged = merged.dropna(subset=["future_return", "target"]).reset_index(drop=True)
    numeric_columns = ["sentiment_score", "return_1d", "return_3d_past", "volatility_5d", "volume_z_20d"]
    merged[numeric_columns] = merged[numeric_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return merged
