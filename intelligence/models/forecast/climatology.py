"""Multi-scale robust historical climatology (spec section 3.2). Station
cells get their own; non-station cells fall back to ward- then city-level.
Median throughout — this is a temporal aggregate, principle 6 applies here
without exception (unlike spatial.py's composite, which does not)."""
import numpy as np
import pandas as pd

SCALES = ("cell", "ward", "city")


def _how(ts: pd.Series) -> pd.Series:
    return ts.dt.dayofweek * 24 + ts.dt.hour


def build_climatology(panel: pd.DataFrame) -> dict[str, pd.Series]:
    p = panel[panel.pm25_station.notna()].copy()
    p["how"] = _how(p.ts)
    p["month"] = p.ts.dt.month
    return {
        "cell_dow_hour": p.groupby(["cell", "how"]).pm25_station.median(),
        "cell_month": p.groupby(["cell", "month"]).pm25_station.median(),
        "ward_dow_hour": p.groupby(["ward_id", "how"]).pm25_station.median(),
        "ward_month": p.groupby(["ward_id", "month"]).pm25_station.median(),
        "city_dow_hour": p.groupby(["city", "how"]).pm25_station.median(),
        "city_month": p.groupby(["city", "month"]).pm25_station.median(),
    }


def lookup_climatology(tables: dict[str, pd.Series], cell: str, ward_id: str,
                        city: str, ts: pd.Timestamp, scale: str = "dow_hour") -> float:
    """cell -> ward -> city fallback, in that order. Returns NaN (not a
    guess) if none of the three has a matching row — LightGBM treats NaN
    as a native missing value."""
    key = (ts.dayofweek * 24 + ts.hour) if scale == "dow_hour" else ts.month
    for scope, ident in (("cell", cell), ("ward", ward_id), ("city", city)):
        table = tables[f"{scope}_{scale}"]
        if (ident, key) in table.index:
            return float(table.loc[(ident, key)])
    return float("nan")


if __name__ == "__main__":
    hours = pd.date_range("2024-01-01", periods=72, freq="h", tz="UTC")
    demo = pd.DataFrame({"cell": ["A"] * 72, "ward_id": ["W1"] * 72,
                          "city": ["bengaluru"] * 72, "ts": hours,
                          "pm25_station": [40.0 + (i % 24) for i in range(72)]})
    t = build_climatology(demo)
    print("cell climatology @ hour 5:",
          lookup_climatology(t, "A", "W1", "bengaluru", pd.Timestamp("2024-01-02T05:00:00", tz="UTC")))
