# This module exists to validate SRZ gate behavior and per-date zone resolution.

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mn_trading.gates.llm_gate import apply_llm_gate_to_signals
from mn_trading.zones.zones_resolver import ZonesResolver


def _make_daily_df(code: str, date: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "code": code,
                "__date_key__": date,
                "bb50_position": "BELOW",
                "bb20_cross": "ABOVE",
                "wave_state": "ACTIVE",
            }
        ]
    )


def test_resolver_uses_trade_date_cache_keys(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    path = data_dir / "000001_daily.csv"
    df = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"],
            "open": [1, 2, 3, 4, 5],
            "high": [2, 3, 4, 5, 6],
            "low": [0.5, 1.5, 2.5, 3.5, 4.5],
            "close": [1.5, 2.5, 3.5, 4.5, 5.5],
            "volume": [100, 110, 120, 130, 140],
        }
    )
    df.to_csv(path, index=False)
    resolver = ZonesResolver(src_data_dir=str(data_dir))
    resolver.resolve("000001", "2020-01-04")
    resolver.resolve("000001", "2020-01-05")
    keys = {key[1] for key in resolver.cache.keys()}
    assert "2020-01-04" in keys
    assert "2020-01-05" in keys


def test_srz_missing_policy_warn_block_fail() -> None:
    rule_df = pd.DataFrame(
        [
            {"date": "2020-01-05", "code": "000001", "action": "BUY_1", "price": 10.0, "entry_stage": 0}
        ]
    )
    daily_df = _make_daily_df("000001", "2020-01-05")
    df_warn, meta_warn = apply_llm_gate_to_signals(
        rule_df,
        seed=0,
        model="gpt-5",
        srz_bounds_map={},
        srz_missing="warn",
        daily_df=daily_df,
    )
    assert meta_warn["srz_rows_missing"] == 1
    assert meta_warn["srz_rows_allow"] == 1
    assert df_warn.loc[0, "action"].startswith("BUY")

    df_block, meta_block = apply_llm_gate_to_signals(
        rule_df,
        seed=0,
        model="gpt-5",
        srz_bounds_map={},
        srz_missing="block",
        daily_df=daily_df,
    )
    assert meta_block["srz_rows_block"] == 1
    assert df_block.loc[0, "action"] == "HOLD"

    with pytest.raises(ValueError):
        apply_llm_gate_to_signals(
            rule_df,
            seed=0,
            model="gpt-5",
            srz_bounds_map={},
            srz_missing="fail",
            daily_df=daily_df,
        )


def test_price_missing_counts_as_missing() -> None:
    rule_df = pd.DataFrame(
        [{"date": "2020-01-05", "code": "000001", "action": "BUY_1", "entry_stage": 0}]
    )
    daily_df = _make_daily_df("000001", "2020-01-05")
    bounds_map = {("000001", "2020-01-05"): (1.0, 20.0)}
    df_warn, meta_warn = apply_llm_gate_to_signals(
        rule_df,
        seed=0,
        model="gpt-5",
        srz_bounds_map=bounds_map,
        srz_missing="warn",
        daily_df=daily_df,
    )
    assert meta_warn["srz_rows_price_missing"] == 1
    assert meta_warn["srz_rows_allow"] == 1
    assert df_warn.loc[0, "action"].startswith("BUY")
