# This module exists to resolve SRZ bounds per trade date without lookahead.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


def _normalize_date(value: str) -> str:
    return pd.to_datetime(value, errors="coerce").strftime("%Y-%m-%d")


def _find_date_col(df: pd.DataFrame) -> str | None:
    for name in ("date", "Date", "dt", "ymd"):
        if name in df.columns:
            return name
    return None


def _compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean()


def compute_zones_asof(
    code: str,
    asof_date: str,
    src_data_dir: str,
    window: int = 120,
    hvn_share: float = 0.05,
) -> tuple[float, float] | None:
    path = Path(src_data_dir) / f"{code}_daily.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    date_col = _find_date_col(df)
    if not date_col:
        return None
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            return None
    df = df.copy()
    df["_date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["_date"])
    df = df.sort_values("_date")
    cut = pd.to_datetime(asof_date, errors="coerce")
    if pd.isna(cut):
        return None
    df = df[df["_date"] <= cut]
    if df.empty:
        return None
    df = df.tail(window)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["high"] = pd.to_numeric(df["high"], errors="coerce")
    df["low"] = pd.to_numeric(df["low"], errors="coerce")
    df = df.dropna(subset=["close", "volume", "high", "low"])
    if df.empty:
        return None
    atr = _compute_atr(df)
    atr_last = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else None
    close_last = float(df["close"].iloc[-1])
    bucket_width = max((atr_last or 0.0) * 0.5, close_last * 0.003)
    if bucket_width <= 0:
        return None
    vol_by_price: dict[float, float] = {}
    for _, row in df.iterrows():
        low = float(row["low"])
        high = float(row["high"])
        volume = float(row["volume"])
        if high <= low:
            mid = float(row["close"])
            vol_by_price[mid] = vol_by_price.get(mid, 0.0) + volume
            continue
        start = np.floor(low / bucket_width) * bucket_width + bucket_width / 2
        end = np.ceil(high / bucket_width) * bucket_width + bucket_width / 2
        bins = np.arange(start, end + bucket_width / 2, bucket_width)
        if bins.size == 0:
            continue
        vol_add = volume / float(bins.size)
        for price in bins:
            vol_by_price[price] = vol_by_price.get(price, 0.0) + vol_add
    if not vol_by_price:
        return None
    total_vol = sum(vol_by_price.values())
    threshold = total_vol * hvn_share
    hvn = [p for p, v in vol_by_price.items() if v >= threshold]
    if not hvn:
        hvn = sorted(vol_by_price, key=vol_by_price.get, reverse=True)[:3]
    node = min(hvn, key=lambda p: abs(p - close_last))
    zone_half = max(bucket_width, (atr_last or bucket_width) * 0.25)
    return node - zone_half, node + zone_half


@dataclass
class ZonesResolver:
    src_data_dir: str
    window: int = 120
    hvn_share: float = 0.05
    cache: dict[tuple[str, str], tuple[float, float] | None] = field(default_factory=dict)

    def resolve(self, code: str, trade_date: str, cut_mode: str = "trade_date") -> tuple[float, float] | None:
        cut_date = self._resolve_cut_date(code, trade_date, cut_mode)
        if not cut_date:
            return None
        key = (code, cut_date)
        if key in self.cache:
            return self.cache[key]
        bounds = compute_zones_asof(
            code=code,
            asof_date=cut_date,
            src_data_dir=self.src_data_dir,
            window=self.window,
            hvn_share=self.hvn_share,
        )
        self.cache[key] = bounds
        return bounds

    def _resolve_cut_date(self, code: str, trade_date: str, cut_mode: str) -> str | None:
        date_key = _normalize_date(trade_date)
        if cut_mode == "trade_date":
            return date_key
        if cut_mode != "prev_bar":
            return date_key
        path = Path(self.src_data_dir) / f"{code}_daily.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path)
        date_col = _find_date_col(df)
        if not date_col:
            return None
        df["_date"] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=["_date"]).sort_values("_date")
        target = pd.to_datetime(trade_date, errors="coerce")
        if pd.isna(target):
            return None
        idx = df["_date"].searchsorted(target)
        prev_idx = max(idx - 1, 0)
        return df["_date"].iloc[prev_idx].strftime("%Y-%m-%d")


def build_bounds_map(
    resolver: ZonesResolver,
    codes: list[str],
    trade_dates: list[str],
    cut_mode: str = "trade_date",
) -> dict[tuple[str, str], tuple[float, float]]:
    bounds_map: dict[tuple[str, str], tuple[float, float]] = {}
    for date in trade_dates:
        date_key = _normalize_date(date)
        for code in codes:
            bounds = resolver.resolve(code, date, cut_mode=cut_mode)
            if bounds is not None:
                bounds_map[(code, date_key)] = bounds
    return bounds_map
