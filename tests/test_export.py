import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from gapmodel.cli import main
from gapmodel.dashboard import OilReading
from gapmodel.export import build_snapshot, dumps, summarise
from gapmodel.markets import MARKETS_BY_SYMBOL
from gapmodel.predict import Forecast


def forecast(symbol, probability, shocked=None, session="2026-08-04"):
    meta = MARKETS_BY_SYMBOL[symbol]
    contributions = pd.Series(
        {"ind_cl_f_shock": 0.21, "mkt_gspc_return": -0.08, "ind_vix_level": 0.05}
    )
    return Forecast(
        symbol=symbol,
        name=meta.name,
        region=meta.region,
        session=pd.Timestamp(session),
        probability_up=probability,
        backtest={"auc": 0.68, "brier_skill": 0.12, "accuracy": 0.66, "base_rate": 0.55},
        contributions=contributions,
        shocked_probability=shocked,
    )


def oil(symbol="CL=F", name="WTI crude", ret_1d=0.013):
    return OilReading(
        symbol=symbol,
        name=name,
        as_of=pd.Timestamp("2026-08-04"),
        close=84.67,
        return_1d=ret_1d,
        return_5d=-0.053,
        volatility_20d=0.039,
        shock=0.3,
    )


def test_snapshot_has_the_documented_shape():
    snap = build_snapshot(
        [forecast("^GSPC", 0.54)],
        [oil()],
        generated_at=datetime(2026, 8, 4, 6, 30, tzinfo=timezone.utc),
    )
    assert snap["generated_at"] == "2026-08-04T06:30:00Z"
    entry = snap["markets"][0]
    assert entry["market"] == "S&P 500"
    assert entry["symbol"] == "^GSPC"
    assert entry["p_open_up"] == 0.54
    assert entry["oos_auc"] == 0.68
    assert entry["drivers"][0] == {"name": "ind_cl_f_shock", "log_odds": 0.21}
    assert snap["crude"][0]["symbol"] == "CL=F"
    assert snap["crude"][0]["is_shock"] is False


def test_session_open_utc_is_the_auction_time():
    # S&P 500 opens 13:30 UTC on the session date.
    snap = build_snapshot([forecast("^GSPC", 0.54)], [])
    assert snap["markets"][0]["session_open_utc"] == "2026-08-04T13:30:00Z"


def test_sydney_session_opens_the_previous_calendar_day():
    # ASX 200 open_utc is -1.0 (23:00 the day before the session date).
    snap = build_snapshot([forecast("^AXJO", 0.4)], [])
    assert snap["markets"][0]["session_open_utc"] == "2026-08-03T23:00:00Z"


def test_shock_fields_appear_only_under_a_shock():
    plain = build_snapshot([forecast("^GSPC", 0.54)], [])["markets"][0]
    assert "p_shocked" not in plain
    shocked = build_snapshot([forecast("^GSPC", 0.54, shocked=0.61)], [])["markets"][0]
    assert shocked["p_shocked"] == 0.61
    assert shocked["p_change"] == pytest.approx(0.07)


def test_summary_prefers_the_headline_indices():
    line = summarise(
        [forecast("^GSPC", 0.54), forecast("^IXIC", 0.61), forecast("^N225", 0.8)],
        [oil(ret_1d=0.013), oil("BZ=F", "Brent crude", ret_1d=0.012)],
    )
    assert "S&P 500 54%" in line and "Nasdaq Composite 61%" in line
    assert "WTI +1.3%" in line and "Brent +1.2%" in line


def test_summary_falls_back_to_top_probabilities():
    line = summarise([forecast("^N225", 0.8), forecast("^HSI", 0.3)], [])
    assert line.startswith("Nikkei 225 80%")


def test_a_session_past_a_calendar_says_so_in_the_snapshot():
    """An empty ``caveats`` must not read as "nothing scheduled" to the app."""
    covered = build_snapshot([forecast("^GSPC", 0.54)], [])["markets"][0]
    assert "unchecked_releases" not in covered

    entry = build_snapshot([forecast("^GSPC", 0.54, session="2027-09-15")], [])["markets"][0]
    assert entry["unchecked_releases"] == [
        {"series": "US payrolls", "table_ends": "2026-12-31"},
        {"series": "US CPI", "table_ends": "2026-12-31"},
        {"series": "US PCE inflation", "table_ends": "2026-12-31"},
    ]


def test_snapshot_serialises_to_valid_json():
    snap = build_snapshot([forecast("^GSPC", 0.54)], [oil()])
    assert json.loads(dumps(snap)) == snap


def test_export_command_writes_a_file(tmp_path, monkeypatch):
    out = tmp_path / "snapshot.json"
    monkeypatch.setattr("gapmodel.cli._panel", lambda _args: {})
    monkeypatch.setattr("gapmodel.cli.oil_readings", lambda _panel: [oil()])
    monkeypatch.setattr(
        "gapmodel.cli.forecast_all",
        lambda _panel, **_kwargs: [forecast("^GSPC", 0.54)],
    )
    main(["export", "--out", str(out)])
    written = json.loads(out.read_text())
    assert written["markets"][0]["symbol"] == "^GSPC"
    assert "summary" in written
