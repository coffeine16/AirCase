import pandas as pd
import pytest

from shared.grid import city_cells
from ingestion.preprocessing.historical_panel import build_historical_panel


def test_build_historical_panel_shape(tmp_path):
    city = "bengaluru"
    hist = tmp_path / "historical" / city
    hist.mkdir(parents=True)
    raw = tmp_path / "raw" / city
    raw.mkdir(parents=True)

    real_cell = city_cells()[0]
    hours = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")

    pd.DataFrame({
        "station_id": ["S1"] * 48, "ts": hours, "lat": [12.97] * 48,
        "lon": [77.59] * 48, "pm25": [50.0] * 48, "cell": [real_cell] * 48,
    }).to_parquet(hist / "stations.parquet")
    pd.DataFrame({
        "ts": hours, "wind_from_deg": [180.0] * 48, "wind_ms": [2.0] * 48,
        "blh_m": [500.0] * 48, "temp_c": [28.0] * 48,
    }).to_parquet(hist / "weather.parquet")
    pd.DataFrame(columns=["ts", "lat", "lon", "frp", "confidence"]).to_parquet(hist / "fires.parquet")
    pd.DataFrame({
        "name": ["test_site"], "kind": ["industrial"], "tag": ["test"],
        "lat": [12.97], "lon": [77.59]
    }).to_parquet(raw / "osm.parquet")

    panel = build_historical_panel(city, hist_dir=hist, raw_dir=raw)

    assert len(panel) == len(city_cells()) * 48
    assert panel[panel.cell == real_cell].pm25_station.notna().all()
    assert (panel.city == city).all()
    assert (hist / "panel.parquet").exists()


def test_build_historical_panel_raises_on_no_overlap(tmp_path):
    city = "bengaluru"
    hist = tmp_path / "historical" / city
    hist.mkdir(parents=True)
    raw = tmp_path / "raw" / city
    raw.mkdir(parents=True)

    pd.DataFrame({
        "station_id": ["S1"], "ts": [pd.Timestamp("2024-01-01", tz="UTC")],
        "lat": [12.97], "lon": [77.59], "pm25": [50.0], "cell": [city_cells()[0]],
    }).to_parquet(hist / "stations.parquet")
    pd.DataFrame({
        "ts": [pd.Timestamp("2025-01-01", tz="UTC")], "wind_from_deg": [180.0],
        "wind_ms": [2.0], "blh_m": [500.0], "temp_c": [28.0],
    }).to_parquet(hist / "weather.parquet")
    pd.DataFrame(columns=["ts", "lat", "lon", "frp", "confidence"]).to_parquet(hist / "fires.parquet")
    pd.DataFrame(columns=["name", "kind", "tag", "lat", "lon"]).to_parquet(raw / "osm.parquet")

    with pytest.raises(ValueError, match="no overlapping hours"):
        build_historical_panel(city, hist_dir=hist, raw_dir=raw)
