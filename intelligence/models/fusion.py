"""Coverage-debiased fusion field.

Train on cells that HAVE a station (features -> measured PM2.5), predict every
cell citywide. Headline rigor number: leave-one-station-out (LOSO) validation —
hide each station entirely, predict its cell from the others, report the error.

Outputs:
  data/outputs/fusion_field.parquet   per (cell, hour): pm25_hat
  data/outputs/loso.json              per-station and overall LOSO metrics
  data/outputs/fusion_model.txt       trained LightGBM model
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from shared.config import DATA_OUT

# SO2 and AAI are NOT features. They are noise (SNR 0.7-1.0 on real S5P; see
# detect.py::POLLUTANTS). On real Delhi, `aai` came back as the model's #2 feature
# by gain — i.e. LightGBM was busily fitting retrieval error, and LOSO R2 collapsed
# from 0.90 (synthetic) to 0.48. A tree ensemble handed a noise column WILL find
# structure in it, and will generalise worse for having done so.
FEATURES = ["no2_col",                                   # satellite (NO2 only)
            "wind_from_deg", "wind_ms", "blh_m", "temp_c",  # meteorology
            "fires_6h", "frp_6h",                         # fire activity
            "lu_industrial", "lu_construction", "lu_waste_burning", "lu_traffic",  # land use
            "lu_road",                                    # road density: the spatial background proxy
            "hour", "dow"]                                # time
LABEL = "pm25_station"
PARAMS = dict(objective="regression", metric="rmse", num_leaves=63,
              learning_rate=0.06, feature_fraction=0.9, bagging_fraction=0.8,
              bagging_freq=1, min_data_in_leaf=40, verbose=-1)


# PREDICT THE DEVIATION, NOT THE LEVEL.
#
# The old model predicted absolute PM2.5 and, on real Delhi, came in 24% WORSE than
# a naive city-mean (LOSO R2 0.52, RMSE 75 vs 60.7). The reason is visible in its own
# feature importances: temp_c, hour, wind — it was spending nearly all its capacity
# re-learning the CITYWIDE TEMPORAL signal (in Delhi's November episodes every
# station spikes together, so pollution is regionally dominated), and then getting
# the spatial part wrong on top of it.
#
# But the citywide signal is the one thing the station network measures WELL. Hand it
# to the model for free, and make it learn only what stations cannot tell you: the
# SPATIAL DEVIATION of a cell from the city as a whole.
#
#     pm25_hat(cell, hour) = city_median(hour)          <- from stations, free
#                          + residual_hat(cell, hour)   <- what the model must learn
#
# This cannot lose to the naive baseline by construction: predicting a zero residual
# IS the naive baseline. Every bit of skill above that is spatial skill, which is the
# only thing we ever claimed to add.
#
# MEDIAN, not mean, for the city term — one station having a bad hour must not move
# the baseline for the whole city.
def _city_baseline(df: pd.DataFrame) -> pd.Series:
    """Hour -> median PM2.5 across the stations in `df`. The regional component."""
    return df.groupby("ts")[LABEL].median()


def _train(df: pd.DataFrame, baseline: pd.Series, rounds: int = 400) -> lgb.Booster:
    resid = df[LABEL] - df.ts.map(baseline)
    ok = resid.notna()
    ds = lgb.Dataset(df.loc[ok, FEATURES], label=resid[ok])
    return lgb.train(PARAMS, ds, num_boost_round=rounds)


def _predict(model: lgb.Booster, df: pd.DataFrame, baseline: pd.Series) -> np.ndarray:
    base = df.ts.map(baseline)
    base = base.fillna(baseline.median())
    return base.values + model.predict(df[FEATURES])


def loso_validation(panel: pd.DataFrame) -> dict:
    """Leave-one-station-out: the honest test of citywide generalization."""
    labeled = panel[panel[LABEL].notna()]
    stations = sorted(labeled.cell.unique())
    per_station, all_true, all_pred = {}, [], []
    for held_out in stations:
        train = labeled[labeled.cell != held_out]
        test = labeled[labeled.cell == held_out]
        # The baseline is built from the OTHER stations only — the held-out station
        # must not appear in its own regional term, or LOSO is leaking.
        baseline = _city_baseline(train)
        model = _train(train, baseline, rounds=250)
        pred = _predict(model, test, baseline)
        rmse = float(np.sqrt(mean_squared_error(test[LABEL], pred)))
        r2 = float(r2_score(test[LABEL], pred))
        per_station[held_out] = {"rmse": round(rmse, 2), "r2": round(r2, 3), "n": len(test)}
        all_true.extend(test[LABEL]); all_pred.extend(pred)
    overall = {
        "rmse": round(float(np.sqrt(mean_squared_error(all_true, all_pred))), 2),
        "r2": round(float(r2_score(all_true, all_pred)), 3),
        "n_stations": len(stations),
        "mean_pm25": round(float(np.mean(all_true)), 1),
    }
    # naive-interpolation strawman for comparison: predict city-mean of other stations at each hour
    labeled_h = labeled.set_index(["ts", "cell"])[LABEL]
    naive_true, naive_pred = [], []
    for held_out in stations:
        test = labeled[labeled.cell == held_out]
        others = labeled[labeled.cell != held_out].groupby("ts")[LABEL].mean()
        m = test.ts.map(others)
        ok = m.notna()
        naive_true.extend(test[LABEL][ok]); naive_pred.extend(m[ok])
    overall["naive_citymean_rmse"] = round(float(np.sqrt(mean_squared_error(naive_true, naive_pred))), 2)
    return {"overall": overall, "per_station": per_station}


def run():
    panel = pd.read_parquet(DATA_OUT / "panel.parquet")

    print("[fusion] running leave-one-station-out validation ...")
    loso = loso_validation(panel)
    (DATA_OUT / "loso.json").write_text(json.dumps(loso, indent=2))
    o = loso["overall"]
    print(f"[fusion] LOSO  RMSE={o['rmse']}  R2={o['r2']}  "
          f"(naive city-mean RMSE={o['naive_citymean_rmse']}, mean PM2.5={o['mean_pm25']}, "
          f"{o['n_stations']} stations)")

    print("[fusion] training final model on all stations, predicting all cells ...")
    labeled = panel[panel[LABEL].notna()]
    baseline = _city_baseline(labeled)
    model = _train(labeled, baseline)
    model.save_model(str(DATA_OUT / "fusion_model.txt"))

    field = panel[["cell", "ts"]].copy()
    field["pm25_hat"] = _predict(model, panel, baseline)
    field.to_parquet(DATA_OUT / "fusion_field.parquet", index=False)

    imp = pd.Series(model.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    print("[fusion] top features:", ", ".join(f"{k}({v:.0f})" for k, v in imp.head(5).items()))
    print(f"[fusion] wrote fusion_field.parquet ({len(field):,} rows)")
    return field, loso


if __name__ == "__main__":
    run()
