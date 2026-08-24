"use client";
/**
 * WardComparison — "is my ward normal, or is it one of the bad ones?"
 *
 * The ward page already answers "how bad is it here" (an AQI number) and "why"
 * (the accountability panel). It could not answer the question a resident asks
 * immediately after those two: compared to what? An AQI of 403 is meaningless
 * in isolation — a person cannot tell whether their ward is the problem or
 * whether the whole city is like this today, and those two facts call for
 * completely different reactions.
 *
 * So this ranks every ward in the city by its own median AQI and says where
 * this one falls: a rank, a percentile, and the distance from the typical ward.
 *
 * MEDIANS, NOT MEANS — twice over (project principle 6, and it matters more
 * here than usual). A ward's AQI is the median over its cells, and the city
 * figure is the median over those ward medians. One severe ward, or a single
 * spiking cell inside a ward, would drag a mean upward and make the city look
 * worse than any resident's actual experience of it. "Typical" is what a
 * citizen wants to know, and the median is what computes it.
 *
 * Everything here derives from the fusion field the page has ALREADY fetched —
 * no new endpoint, no new contract, no extra request.
 */
import { useMemo } from "react";
import { pm25ToAqi, getAqiCategory } from "@/lib/colors";
import { icon, Activity } from "@/components/Icon";
import type { FusionCell } from "@/lib/types";

/** Percentile needs a population. Below this, "cleaner than N%" is arithmetic
 *  noise dressed as a statistic, so it is withheld and only the comparison to
 *  the typical ward is shown. */
const MIN_WARDS_FOR_PERCENTILE = 5;

/** Within this many AQI points, "worse"/"better" is over-claiming precision the
 *  underlying estimate does not have. */
const SAME_AS_TYPICAL_AQI = 3;

function median(values: number[]): number {
  if (!values.length) return NaN;
  const s = [...values].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

export default function WardComparison({
  wardId,
  cells,
  cityLabel,
}: {
  wardId: string;
  cells: FusionCell[];
  cityLabel: string;
}) {
  /** ward_id -> that ward's median AQI. `unassigned` is excluded: it is not a
   *  place, it is every cell that fell outside the municipal boundary, and
   *  letting it into the ranking would invent a competitor ward. */
  const wardAqi = useMemo(() => {
    const byWard = new Map<string, number[]>();
    for (const c of cells) {
      if (!c.ward_id || c.ward_id === "unassigned") continue;
      // pm25 is null where the forecaster has no value for a cell (see
      // FusionCell). Bind it to a local so the compiler carries the narrowing
      // into the pushes below — Number.isFinite alone does not narrow a
      // `number | null`, even though it correctly rejects null at runtime.
      const v = c.pm25;
      if (v == null || !Number.isFinite(v)) continue;
      const arr = byWard.get(c.ward_id);
      if (arr) arr.push(v);
      else byWard.set(c.ward_id, [v]);
    }
    const out = new Map<string, number>();
    for (const [w, arr] of byWard) out.set(w, pm25ToAqi(median(arr)));
    return out;
  }, [cells]);

  const stats = useMemo(() => {
    const mine = wardAqi.get(wardId);
    if (mine == null || !Number.isFinite(mine)) return null;

    const all = [...wardAqi.values()].filter((v) => Number.isFinite(v));
    if (all.length < 2) return null;

    // Rank counts wards strictly CLEANER, so ties share a rank instead of being
    // ordered arbitrarily by map iteration order.
    const cleaner = all.filter((v) => v < mine).length;
    const dirtier = all.filter((v) => v > mine).length;
    return {
      mine,
      cityTypical: median(all),
      nWards: all.length,
      rank: cleaner + 1,
      /** Share of the OTHER wards that are dirtier than this one. */
      cleanerThanPct: Math.round((dirtier / (all.length - 1)) * 100),
      lo: Math.min(...all),
      hi: Math.max(...all),
    };
  }, [wardAqi, wardId]);

  // No cells for this ward (or a one-ward city): say nothing rather than
  // render a comparison against a population of one.
  if (!stats) return null;

  const { mine, cityTypical, nWards, rank, cleanerThanPct, lo, hi } = stats;
  const cat = getAqiCategory(mine);
  const delta = Math.round(mine - cityTypical);
  const showPercentile = nWards >= MIN_WARDS_FOR_PERCENTILE;

  // Marker position along the city's own range. A degenerate range (every ward
  // identical) is centred rather than dividing by zero.
  const span = hi - lo;
  const pos = span > 0 ? ((mine - lo) / span) * 100 : 50;
  const medianPos = span > 0 ? ((cityTypical - lo) / span) * 100 : 50;

  const comparison =
    Math.abs(delta) <= SAME_AS_TYPICAL_AQI
      ? `about the same as a typical ${cityLabel} ward`
      : `${Math.abs(delta)} AQI points ${delta > 0 ? "worse" : "better"} than a typical ${cityLabel} ward`;

  return (
    <div className="card" style={{ marginBottom: "var(--space-lg)" }}>
      <h5 style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
        <Activity {...icon.sm} aria-hidden />
        How your ward compares
      </h5>
      <p style={{ fontSize: "0.75rem", color: "var(--text-tertiary)", marginBottom: 14 }}>
        Every {cityLabel} ward ranked by its own air quality right now.
      </p>

      {/* The claim in words first — the numbers below are for anyone who wants
          to check it, not the primary way of reading this card. */}
      <p style={{ fontSize: "0.95rem", lineHeight: 1.55, marginBottom: 14 }}>
        Your ward is <strong>{comparison}</strong>
        {showPercentile && (
          <>
            {" "}— cleaner than <strong>{cleanerThanPct}%</strong> of them.
          </>
        )}
        {!showPercentile && "."}
      </p>

      {/* Where this ward sits across the city's full range. */}
      <div
        role="img"
        aria-label={
          `Your ward's AQI is ${mine}. Across ${cityLabel}'s ${nWards} wards the range is ` +
          `${lo} to ${hi}, and the typical ward is ${Math.round(cityTypical)}.`
        }
      >
        <div
          style={{
            position: "relative",
            height: 8,
            borderRadius: 99,
            background:
              "linear-gradient(to right, var(--positive-soft), var(--caution-soft), var(--critical-soft))",
            border: "1px solid var(--border-subtle)",
          }}
        >
          {/* the typical ward */}
          <span
            aria-hidden
            style={{
              position: "absolute", left: `${medianPos}%`, top: -3, bottom: -3,
              width: 2, background: "var(--text-tertiary)", opacity: 0.75,
              transform: "translateX(-1px)",
            }}
          />
          {/* this ward */}
          <span
            aria-hidden
            style={{
              position: "absolute", left: `${pos}%`, top: "50%",
              width: 14, height: 14, borderRadius: "50%",
              background: cat.color,
              border: "2px solid var(--bg-primary)",
              boxShadow: "0 0 0 1px var(--border-strong)",
              transform: "translate(-50%, -50%)",
            }}
          />
        </div>
        <div
          className="mono"
          style={{
            display: "flex", justifyContent: "space-between",
            fontSize: "0.68rem", color: "var(--text-tertiary)", marginTop: 5,
          }}
        >
          {/* Word first. "36 cleanest" reads as "36th cleanest" — a rank —
              when it is the cleanest ward's AQI. Same digits, opposite meaning. */}
          <span>cleanest ward · {lo}</span>
          <span>{hi} · worst</span>
        </div>
      </div>

      <div
        style={{
          display: "flex", gap: "var(--space-lg)", flexWrap: "wrap",
          borderTop: "1px solid var(--border-subtle)", paddingTop: 12, marginTop: 14,
        }}
      >
        <div>
          <div className="section-label" style={{ marginBottom: 3 }}>Your ward</div>
          <div className="mono" style={{ fontSize: "1.05rem", fontWeight: 600 }}>
            {mine}{" "}
            <span style={{ fontSize: "0.7rem", color: "var(--text-tertiary)" }}>AQI</span>
          </div>
        </div>
        <div>
          <div className="section-label" style={{ marginBottom: 3 }}>Typical ward</div>
          <div className="mono" style={{ fontSize: "1.05rem", fontWeight: 600 }}>
            {Math.round(cityTypical)}{" "}
            <span style={{ fontSize: "0.7rem", color: "var(--text-tertiary)" }}>AQI</span>
          </div>
        </div>
        <div>
          <div className="section-label" style={{ marginBottom: 3 }}>Rank</div>
          <div className="mono" style={{ fontSize: "1.05rem", fontWeight: 600 }}>
            {rank}
            <span style={{ fontSize: "0.7rem", color: "var(--text-tertiary)" }}>
              {" "}of {nWards}
            </span>
          </div>
        </div>
      </div>

      <p style={{ fontSize: "0.68rem", color: "var(--text-tertiary)", marginTop: 10, lineHeight: 1.5 }}>
        {/* Dropped "every figure is a median, so one unusually bad street
            cannot skew a ward" — it defends our statistics to a reader who
            never doubted them. */}
        Rank 1 is the cleanest ward.
      </p>
    </div>
  );
}
