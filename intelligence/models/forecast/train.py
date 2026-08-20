# intelligence/models/forecast/train.py
"""Top-level training orchestration + promotion gate (spec sections 5.4, 7).
A freshly trained model must match or beat the currently-served model on
walk-forward skill, spatial-LOSO, and city-LOSO before it's allowed to
replace it — code-enforced, not eyeballed. quantile_coverage is computed
and recorded in every manifest but deliberately NOT gated: unlike RMSE
(lower always better) or skill (higher always better), coverage has no
single "better" direction — both over- and under-coverage relative to the
~0.80 target are miscalibration. Gating it needs a distance-from-target
comparison this version doesn't implement; stated here as a real scope
limit, not a silent gap."""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from intelligence.models.forecast.features import build_features, downcast_panel, station_cells_only
from intelligence.models.forecast.model import mask_unknown_city
from intelligence.models.forecast.validation import (
    walk_forward_folds, event_weights, spatial_loso, run_city_loso, run_walk_forward,
)
from intelligence.models.forecast.eval import (
    skill_vs_baseline, quantile_coverage, quiet_vs_event_breakdown,
    interval_scale_for_coverage, event_by_outcome,
)

# A worker's peak is its pickled payload plus the fold slice attach_climatology
# copies plus LightGBM's Dataset -- budget for the peak, not the handoff.
_WORKER_PEAK_MULTIPLE = 2.5
# Spend only half of what's left after the parent's own resident copy.
_MEMORY_SAFETY_FRACTION = 0.5
# Ceiling on fold concurrency. Memory governs independently and binds first
# on every flavor we actually use (11 workers for 8 cities on cpu-performance,
# 5 on cpu-xl), so this only stops a very large box from spawning more
# processes than the fold loops can keep busy.
_MAX_WORKERS = 16


def _available_cpus() -> int:
    """CPUs this CONTAINER may actually use.

    NOT os.cpu_count(): that reports the HOST node's cores and ignores the
    container's cgroup quota entirely. On HF Jobs it returned 64 on a
    flavor with far fewer, so the worker count below was derived from
    hardware the job could never use. HF Jobs sets CPU_CORES explicitly;
    the cgroup v2 quota and sched_getaffinity are the portable fallbacks.
    """
    env = os.environ.get("CPU_CORES")
    if env:
        try:
            return max(1, int(float(env)))
        except ValueError:
            pass
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            return max(1, int(quota) // int(period))
    except (OSError, ValueError):
        pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:      # not Linux
        return os.cpu_count() or 4


def _available_memory_bytes() -> int | None:
    """Memory this container may use, or None if it can't be determined."""
    env = os.environ.get("MEMORY")   # HF Jobs sets e.g. "32Gi"
    if env:
        text = env.strip()
        for suffix, mult in (("Gi", 1 << 30), ("Mi", 1 << 20), ("G", 10**9), ("M", 10**6)):
            if text.endswith(suffix):
                try:
                    return int(float(text[:-len(suffix)]) * mult)
                except ValueError:
                    break
        try:
            return int(text)
        except ValueError:
            pass
    try:
        text = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if text != "max":
            return int(text)
    except (OSError, ValueError):
        pass
    return None


def _resolve_workers(threads_per_fold: int, payload_bytes: int) -> int:
    """How many folds to run concurrently.

    Capped by MEMORY as well as by CPU, because each ProcessPoolExecutor
    worker receives its own PICKLED COPY of the frame passed through
    `initargs` -- concurrency here costs RAM linearly, it is not shared.
    Measured on a real 4-city panel the pooled frame is ~2.9GB, so the
    old cpu-only rule (64 phantom host cores // 2 = 32 workers) implied
    ~93GB of frame copies and the job died on the spot.

    The budget is deliberately conservative, because the failure mode is
    not a slow run, it is an OOM kill hours in with nothing saved:
      - the PARENT keeps its own copy for the whole run, so only
        (memory - payload) is ever available to workers at all;
      - a worker's PEAK is well above the payload it was handed --
        attach_climatology copies its fold slice and LightGBM builds a
        Dataset on top -- hence _WORKER_PEAK_MULTIPLE;
      - only half of what remains is spent, leaving room for the final
        served model's own three 500-round boosters.
    Returning 1 means "run sequentially in this process", which copies
    nothing at all and is the correct answer on a small container.
    """
    by_cpu = max(1, _available_cpus() // threads_per_fold)
    available = _available_memory_bytes()
    if available is None or payload_bytes <= 0:
        return max(1, min(by_cpu, _MAX_WORKERS))
    spendable = (available - payload_bytes) * _MEMORY_SAFETY_FRACTION
    by_mem = int(spendable // (payload_bytes * _WORKER_PEAK_MULTIPLE))
    return max(1, min(by_cpu, by_mem, _MAX_WORKERS))


def _mature_oof(oof: pd.DataFrame) -> pd.DataFrame:
    """OOF rows whose fold's `n_train` is at or above the row-weighted
    median of `n_train` across the whole OOF frame -- the half of
    walk-forward closest in training-data volume to what the final
    full-data model actually sees, not the average of every fold including
    the earliest, barely-trained ones. `oof["n_train"]` is constant within
    a fold (see validation.py's oof_frame), so this is a per-ROW median,
    not strictly a per-FOLD one -- with walk_forward_folds' now-uniform
    test_days across folds, row counts per fold are close enough that the
    two agree in practice, but a fold covering an unusually short tail
    window could weight the split slightly. One fold (or a frame with a
    single n_train value) keeps everything, since >= its own median is
    always true."""
    threshold = oof["n_train"].median()
    return oof[oof["n_train"] >= threshold]


def _regressed(new_val: float | None, prior_val: float | None,
               higher_is_better: bool, tolerance_pct: float) -> bool:
    """True if `new_val` is a regression vs `prior_val` beyond tolerance.
    Missing or non-finite `new_val`/`prior_val` means there is nothing to
    compare -- returns False (does not block). This is deliberate: a
    diagnostic like walk-forward skill can legitimately come back None on
    too little history (expected on a small panel, not a failure), and
    that must not be conflated with the primary metric (spatial-LOSO)
    genuinely breaking -- see the separate, stricter spatial_loso_ok check
    in train_and_promote below, which is where a non-finite result DOES
    block promotion unconditionally."""
    if new_val is None or prior_val is None:
        return False
    if not np.isfinite(new_val) or not np.isfinite(prior_val):
        return False
    # abs(prior), NOT prior, for the tolerance band. Walk-forward skill can
    # legitimately be NEGATIVE (persistence beating the model at 24h is the
    # documented normal case for this project), and prior * (1 - tol) then
    # moves the bar the WRONG way: with prior=-10 it demands >= -9.5, i.e. a
    # strict improvement, instead of allowing -10.5 as within tolerance.
    if higher_is_better:
        return new_val < prior_val - abs(prior_val) * tolerance_pct / 100
    return new_val > prior_val + abs(prior_val) * tolerance_pct / 100


def _version_id() -> str:
    # Wall-clock, NOT derived from the panels' own data timestamps: two
    # training runs on the same day's data (a retry after a rejected
    # promotion, a manual re-run) would otherwise produce an IDENTICAL
    # version string and silently overwrite each other's artifacts at the
    # same path. This is regular application code, not a Workflow script —
    # wall-clock time is exactly the right source here, not a hazard.
    return pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H%M%S")


def train_and_promote(panels_by_city: dict[str, pd.DataFrame], horizons: list[int],
                       feature_cols: list[str], out_dir: Path,
                       prior_manifest: dict | None = None,
                       regression_tolerance_pct: float = 5.0,
                       fires_by_city: dict[str, pd.DataFrame] | None = None,
                       walk_forward_kwargs: dict | None = None,
                       max_workers: int | None = None,
                       threads_per_fold: int = 2,
                       checkpoint_dir: str | None = None) -> dict:
    """`checkpoint_dir`: None (default) disables checkpointing, unchanged
    from before this parameter existed -- every existing caller (tests,
    the operational pipeline) is unaffected. When set, each of the three CV
    stages below checkpoints its own folds under `checkpoint_dir/<stage>/`
    and resumes from them on a later call with the SAME checkpoint_dir --
    see validation.py's three fold-runners and checkpoint.py. The final
    served-model fit is NOT checkpointed (a single LightGBM training call,
    not decomposable into independent folds the way the three CV stages
    are); a Job killed during that stage restarts it from scratch, same as
    today."""
    out_dir = Path(out_dir)
    # concat silently reverts each city's own categorical columns back to
    # plain string dtype whenever their category sets differ (every city's
    # `city` column, for one) -- re-downcast after, not just at load time.
    # See features.py::downcast_panel's docstring; this line is the exact
    # one that failed with an ArrowMemoryError on a real 3-city panel.
    #
    # station_cells_only FIRST, not the raw per-city panel: everything
    # below passes restrict_to_station_cells=True, which only ever keeps
    # station-cell rows anyway (see that flag's docstring) -- concatenating
    # each city's full grid (often 700-1700+ cells) just to immediately
    # discard every non-station cell is exactly the peak that OOM'd a real
    # 4-city training Job on cpu-upgrade's 32GB, measured at ~39GB before
    # this filter (panels_by_city + the concat + downcast_panel's own copy,
    # all at full-grid size simultaneously). Filtering to the ~20-40
    # station cells per city here cuts that peak by roughly the same
    # 30-100x that composite_grid/positional_block no longer waste on
    # cells this function was always going to throw away.
    # Timed explicitly -- on the real 8-city run this whole function reported
    # 4.3h total but its own [walk_forward]/[spatial_loso]/[run_city_loso]
    # stage timers summed to only 2.59h. The other ~1.7h (40% of the run)
    # was silently spent here and in the final model fit below, neither of
    # which had a timer. Print the true split instead of leaving a mystery
    # gap the next run's cost estimate has to guess at.
    t0 = time.perf_counter()
    full_panel = downcast_panel(pd.concat(
        [station_cells_only(p) for p in panels_by_city.values()], ignore_index=True))
    all_fires = (pd.concat(fires_by_city.values(), ignore_index=True)
                 if fires_by_city else None)
    # NOT masked here. mask_unknown_city used to run on this frame before
    # any fold existed, so every walk-forward fold's TEST slice (drawn from
    # THIS frame further down) carried ~5% of rows with their real city
    # artificially withheld -- a real serving call always knows its own
    # city, so walk-forward was measuring skill on a task strictly harder
    # than the one being shipped. run_walk_forward now masks train-only,
    # per fold, matching spatial_loso/run_city_loso's own pattern -- see
    # validation.py::_run_one_walk_forward_fold. This frame is masked once,
    # separately, right before the final served-model fit below, which is
    # the one place spec 4.2's "unknown" supervision actually needs to land.
    frame = build_features(full_panel, horizons, fires=all_fires,
                            restrict_to_station_cells=True)
    num_cols = [c for c in feature_cols if c != "city"]
    print(f"[train_and_promote] pooled feature build done in {time.perf_counter() - t0:.0f}s "
          f"({len(frame):,} rows)")

    # None (the parameter's own default) would run every fold across all
    # three CV stages below sequentially in this one process -- fine for
    # tests, much too slow for a real multi-city panel. Measured on real
    # 4-city data: one spatial-LOSO fold alone is ~7 min wall-clock, and a
    # full sequential run (walk-forward + spatial-LOSO across 57 real
    # stations + city-LOSO) ran 11+ hours and still wasn't done. Every fold
    # in every one of these three stages is independent of every other
    # fold in that stage (a different expanding-window split, a different
    # held-out station, a different held-out city), so all three get the
    # same treatment: auto-pick a concurrency level from whatever this
    # container actually has -- CPU **and** memory, since each worker gets
    # its own pickled copy of the frame -- capping each fold's own
    # LightGBM threads so N concurrent folds don't oversubscribe the CPU.
    resolved_max_workers = max_workers
    if resolved_max_workers is None:
        payload_bytes = int(frame.memory_usage(deep=True).sum()
                            + full_panel.memory_usage(deep=True).sum())
        resolved_max_workers = _resolve_workers(threads_per_fold, payload_bytes)
        mem = _available_memory_bytes()
        print(f"[train_and_promote] container: cpus={_available_cpus()} "
              f"memory={'unknown' if mem is None else f'{mem / 1e9:.1f}GB'} "
              f"per-worker payload={payload_bytes / 1e9:.2f}GB "
              f"(budgeted peak {payload_bytes * _WORKER_PEAK_MULTIPLE / 1e9:.2f}GB each)")
    print(f"[train_and_promote] concurrency: {resolved_max_workers} workers x "
          f"{threads_per_fold} threads/fold")

    folds = walk_forward_folds(frame, **(walk_forward_kwargs or {}))
    # Derived from the folds themselves, not from walk_forward_kwargs --
    # robust to that dict being {} and the geometry coming from
    # walk_forward_folds' own defaults, which is exactly what changed
    # between this run and a prior one trained before its defaults moved
    # from 21/21 to 42/42. Recorded below so the promotion gate can tell a
    # genuine walk_forward_skill_median regression apart from an artifact
    # of comparing two runs that sliced the calendar differently.
    wf_geometry = ({"test_days": (folds[0][2] - folds[0][1]).days,
                    "step_days": (folds[1][0] - folds[0][0]).days if len(folds) > 1 else None}
                   if folds else None)
    print(f"[train_and_promote] starting walk-forward ({len(folds)} fold(s))")
    wf_result = run_walk_forward(full_panel, frame, feature_cols, num_cols, folds,
                                  max_workers=resolved_max_workers,
                                  threads_per_fold=threads_per_fold,
                                  checkpoint_dir=checkpoint_dir)
    fold_skills, o = wf_result["fold_skills"], wf_result["oof"]

    print("[train_and_promote] starting spatial-LOSO")
    loso_result = spatial_loso(full_panel, horizons, feature_cols, fires=all_fires,
                                max_workers=resolved_max_workers,
                                threads_per_fold=threads_per_fold,
                                checkpoint_dir=checkpoint_dir)
    print("[train_and_promote] starting city-LOSO")
    city_result = run_city_loso(panels_by_city, horizons, feature_cols,
                                 fires_by_city=fires_by_city,
                                 max_workers=resolved_max_workers,
                                 threads_per_fold=threads_per_fold,
                                 checkpoint_dir=checkpoint_dir)
    print("[train_and_promote] fitting final served model")

    # LightGBM's Dataset already built inside train_quantile_models doesn't
    # take sample weights via that simple call, so the SERVED model (which
    # needs the real-event oversampling weight) is trained directly here
    # instead — walk-forward/LOSO above stay unweighted on purpose, since
    # they measure generalisation, not the production fit. Only ONE
    # training pass for the served model, not two: an earlier draft called
    # train_quantile_models AND this weighted loop, discarding the first
    # result unused — doubling the cost of the single most expensive stage
    # in this function for nothing.
    t0 = time.perf_counter()
    # Masked HERE, not when `frame` was built above -- this is the one
    # place the "unknown" category actually needs to reach a model: the
    # served one. See the comment at `frame`'s construction for why it
    # must stay unmasked until here.
    final_train = mask_unknown_city(frame.dropna(subset=["y"]))
    import lightgbm as lgb
    from intelligence.models.forecast.model import PARAMS, QUANTILES
    final_models = {}
    for q in QUANTILES:
        ds = lgb.Dataset(final_train[feature_cols], label=final_train["y"],
                          weight=event_weights(final_train), categorical_feature=["city"])
        final_models[q] = lgb.train({**PARAMS, "alpha": q}, ds, num_boost_round=500)
    print(f"[train_and_promote] final model fit done in {time.perf_counter() - t0:.0f}s "
          f"({len(final_train):,} rows, 500 rounds x {len(QUANTILES)} quantiles, "
          f"threads=auto-detected/uncapped)")

    # quantile_coverage / ceiling_skill / quiet_vs_event are scored on the
    # walk-forward folds' OUT-OF-SAMPLE predictions, never on final_train.
    # In-sample coverage from a 500-round boosted ensemble is optimistic to
    # the point of meaninglessness, and quiet_vs_event exists (spec section 6)
    # precisely to catch "the model is worse exactly when it matters" — which
    # training data cannot reveal. They are None, not faked, when there is too
    # little history for even one fold. `o` (the concatenated OOF frame) came
    # back from run_walk_forward above -- already concatenated there.
    #
    # event_by_outcome, NOT fires_6h > 0: measured on the real 8-city run,
    # the fire-based definition flagged only 0.02% of rows, 64% of which
    # were in Chennai + Bengaluru -- the two LOWEST-RMSE cities. event_rmse
    # beating quiet_rmse under that definition was very likely a CITY-MIX
    # ARTIFACT (event-rich cities are easy cities), not a real capability.
    # A per-city-relative outcome threshold can't inherit that confound.
    is_event_oof = event_by_outcome(o["y"].values, o["city"].values) if o is not None else None
    mature = _mature_oof(o) if o is not None else None

    eval_report = {
        "walk_forward_skill_median": round(float(np.median(fold_skills)), 1) if fold_skills else None,
        "walk_forward_skill_folds": len(fold_skills),
        "walk_forward_geometry": wf_geometry,
        "spatial_loso_rmse": loso_result["overall_rmse"],
        # Persistence RMSE on the SAME held-out rows -- spatial_loso_rmse had
        # no comparator on its own (is 40.51 good? bad? nobody could say).
        "spatial_loso_baseline_rmse": loso_result["baseline_rmse"],
        "spatial_loso_n_stations": loso_result["n_stations"],
        "city_loso": city_result["per_city"],
        "eval_basis": "walk_forward_out_of_sample" if o is not None else "no_walk_forward_folds",
        "quantile_coverage": (quantile_coverage(o["y"].values, o["p10"].values, o["p90"].values)
                              if o is not None else None),
        # Calibrated on the MATURE half of the out-of-sample folds (see
        # _mature_oof), not the full pooled set quantile_coverage above
        # reports on. Walk-forward's early folds train on a fraction of the
        # final model's data (fold 1 of the first real run: 318K rows vs
        # fold 38's 12.69M) -- pooling them equally with late folds when
        # fitting the SERVE-TIME correction calibrates it against the
        # average fold's immaturity, not against what the actual final,
        # fully-trained model will do. quantile_coverage itself is left on
        # the full pool: it is a REPORTED diagnostic, not a served
        # correction, and reporting it across the whole maturity range is
        # the more honest number for that purpose. Applied only at serve
        # time -- never fed back into the metrics below, which must keep
        # reporting what the raw model actually did.
        "interval_scale": (interval_scale_for_coverage(
            mature["y"].values, mature["p10"].values, mature["p50"].values, mature["p90"].values)
            if mature is not None else None),
        "ceiling_skill_vs_linear": (skill_vs_baseline(o["y"].values, o["p50"].values, o["ceiling"].values)
                                    if o is not None else None),
        "quiet_vs_event": (quiet_vs_event_breakdown(o["y"].values, o["p50"].values, is_event_oof,
                                                     city=o["city"].values)
                           if o is not None else None),
    }

    version = _version_id()
    prior_eval = (prior_manifest or {}).get("eval", {})

    # spatial-LOSO is the primary validation metric (this plan's own
    # "headline number"). Unlike walk-forward/city-LOSO, which can
    # legitimately come back None/empty on too little history (expected on
    # a small panel, not a failure -- see _regressed's docstring), a
    # non-finite spatial-LOSO RMSE means real stations existed but scoring
    # genuinely broke, and that must never be silently promoted, WITH or
    # WITHOUT a prior to compare against. This closes finding #3 from Task
    # 10's review: `promoted` used to default True unconditionally and only
    # checked finiteness inside the `prior_manifest is not None` branch, so
    # a NaN-RMSE first run promoted silently.
    spatial_loso_ok = np.isfinite(eval_report["spatial_loso_rmse"])

    # The other two gated metrics (walk-forward skill: higher better;
    # city-LOSO: lower better, compared as the MEDIAN across whatever
    # cities are present in each run -- robust to the city set changing
    # between runs, and consistent with this project's median-not-mean
    # convention everywhere else).
    prior_city_rmses = [v["rmse"] for v in prior_eval.get("city_loso", {}).values()]
    new_city_rmses = [v["rmse"] for v in city_result["per_city"].values()]
    prior_city_median = float(np.median(prior_city_rmses)) if prior_city_rmses else None
    new_city_median = float(np.median(new_city_rmses)) if new_city_rmses else None

    # walk_forward_skill_median is only a fair comparison against the prior
    # run's own number when both were measured under the SAME fold
    # geometry (test_days/step_days). A prior manifest from before
    # walk_forward_folds' defaults changed (or trained with different
    # explicit walk_forward_kwargs) sliced the calendar differently -- its
    # median comes from a different set of folds over different windows,
    # not from a worse or better model. Comparing anyway would let a pure
    # measurement-methodology difference refuse (or wrongly pass) a
    # promotion on its own. prior_eval.get("walk_forward_geometry") is None
    # for any manifest written before this field existed, which correctly
    # falls into "unknown, don't compare" rather than a false match.
    geometry_matches = (wf_geometry is not None
                         and wf_geometry == prior_eval.get("walk_forward_geometry"))
    wf_regressed = (geometry_matches and _regressed(
        eval_report["walk_forward_skill_median"], prior_eval.get("walk_forward_skill_median"),
        higher_is_better=True, tolerance_pct=regression_tolerance_pct))
    if not geometry_matches and prior_eval.get("walk_forward_skill_median") is not None:
        print(f"[train_and_promote] walk-forward fold geometry changed since the prior "
              f"model ({wf_geometry} vs {prior_eval.get('walk_forward_geometry')}) -- "
              f"skipping the walk_forward_skill_median regression check this run, it is "
              f"not a fair comparison")

    promoted = spatial_loso_ok and not any([
        _regressed(eval_report["spatial_loso_rmse"], prior_eval.get("spatial_loso_rmse"),
                   higher_is_better=False, tolerance_pct=regression_tolerance_pct),
        wf_regressed,
        _regressed(new_city_median, prior_city_median,
                   higher_is_better=False, tolerance_pct=regression_tolerance_pct),
    ])

    manifest = {"version": version, "trained_at": pd.Timestamp.utcnow().isoformat(),
                "cities": sorted(panels_by_city), "eval": eval_report, "promoted": promoted}

    if promoted:
        version_dir = out_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
        for q, m in final_models.items():
            m.save_model(str(version_dir / f"model_p{int(q * 100)}.txt"))
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"[train] promoted {version}: walk-forward skill "
              f"{eval_report['walk_forward_skill_median']}%, spatial-LOSO RMSE "
              f"{eval_report['spatial_loso_rmse']}")
    else:
        reason = ("spatial-LOSO RMSE is non-finite (no scoreable stations)"
                  if not spatial_loso_ok else
                  f"regressed beyond the {regression_tolerance_pct}% tolerance "
                  f"on spatial-LOSO, walk-forward skill, and/or city-LOSO vs the current model")
        print(f"[train] {version} trained but NOT promoted — {reason}")
    return manifest
