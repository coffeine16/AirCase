"""Quantile LightGBM forecaster: pooled across cities, `city` as a
categorical feature with an explicitly-trained 'unknown' fallback (spec
section 4.2). A linear ceiling-check baseline trains alongside for
comparison, never as the served model (spec section 4.1)."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import QuantileRegressor

QUANTILES = (0.1, 0.5, 0.9)
UNKNOWN_CITY = "unknown"
PARAMS = dict(objective="quantile", metric="quantile", num_leaves=63,
              learning_rate=0.05, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, min_data_in_leaf=60, verbose=-1)


def mask_unknown_city(frame: pd.DataFrame, frac: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """Returns a COPY with `frac` of rows' city relabelled 'unknown', so the
    model has real supervision for the fallback category instead of relying
    on LightGBM's implicit unseen-category handling (spec 4.2)."""
    out = frame.copy()
    if UNKNOWN_CITY not in out["city"].astype(str).unique():
        out["city"] = out["city"].astype(str)
        rng = np.random.default_rng(seed)
        mask = rng.random(len(out)) < frac
        out.loc[mask, "city"] = UNKNOWN_CITY
        out["city"] = out["city"].astype("category")
    return out


def train_quantile_models(train: pd.DataFrame, feature_cols: list[str],
                           label_col: str = "y", num_boost_round: int = 2000,
                           valid: pd.DataFrame | None = None,
                           early_stopping_rounds: int = 50,
                           num_threads: int | None = None) -> dict[float, lgb.Booster]:
    """One booster per quantile in QUANTILES. `city` must already be a
    'category' dtype column in `train`/`valid` (see mask_unknown_city).

    `num_threads`: PARAMS never set this, so LightGBM auto-detects available
    cores and uses all of them for a single call -- fine for one fold at a
    time, but spatial_loso's parallel path (validation.py) runs several
    folds concurrently in separate processes, and if EACH one still tried
    to grab every core, they'd oversubscribe the machine and likely run
    SLOWER than sequential. Pass an explicit cap there; every other caller
    leaves this None and keeps today's auto-detect behavior unchanged."""
    cat_cols = [c for c in feature_cols if c == "city"]
    models = {}
    for q in QUANTILES:
        params = {**PARAMS, "alpha": q}
        if num_threads is not None:
            params["num_threads"] = num_threads
        ds = lgb.Dataset(train[feature_cols], label=train[label_col],
                          categorical_feature=cat_cols)
        callbacks, valid_sets = [], None
        if valid is not None:
            vds = lgb.Dataset(valid[feature_cols], label=valid[label_col],
                               reference=ds, categorical_feature=cat_cols)
            valid_sets = [vds]
            callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)]
        models[q] = lgb.train(params, ds, num_boost_round=num_boost_round,
                               valid_sets=valid_sets, callbacks=callbacks)
    return models


def predict_quantiles(models: dict[float, lgb.Booster], frame: pd.DataFrame,
                       feature_cols: list[str]) -> pd.DataFrame:
    out = {f"pm25_p{int(q * 100)}": models[q].predict(frame[feature_cols]) for q in QUANTILES}
    df = pd.DataFrame(out, index=frame.index)
    # a quantile crossing (p10 > p50, rare but possible with independently
    # trained boosters) is sorted away rather than reported nonsensically
    return pd.DataFrame(np.sort(df.values, axis=1), columns=df.columns, index=df.index)


def train_ceiling_baseline(train: pd.DataFrame, feature_cols: list[str],
                            label_col: str = "y") -> QuantileRegressor:
    """Linear quantile regression (median) — the 'is tree complexity
    earning its keep' sanity check (spec 4.1). Reported in eval only,
    never served."""
    X = train[feature_cols].select_dtypes(include=[np.number]).fillna(0.0)
    reg = QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs")
    reg.fit(X, train[label_col])
    return reg


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 300
    frame = pd.DataFrame({
        "lag_0": rng.uniform(20, 100, n), "wind_ms": rng.uniform(0, 10, n),
        "city": pd.Categorical(rng.choice(["bengaluru", "delhi"], n)),
        "y": rng.uniform(20, 100, n),
    })
    frame = mask_unknown_city(frame)
    models = train_quantile_models(frame, ["lag_0", "wind_ms", "city"], num_boost_round=20)
    print(predict_quantiles(models, frame.iloc[:5], ["lag_0", "wind_ms", "city"]))
