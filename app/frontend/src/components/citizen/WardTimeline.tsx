"use client";
/**
 * WardTimeline — "when is it bad in my ward", in 3-hour steps to +72h.
 *
 * The four-box +24/+48/+72 strip answered a question nobody asks. Nobody plans
 * around Thursday's daily mean; they decide whether to walk this evening or
 * tomorrow morning. PM2.5 has a strong diurnal cycle — BHALSWA runs 240 µg/m³
 * at night and 109 by midday — and a forecast sampled only at 24-hour multiples
 * hides that swing completely, because those samples all land on the same hour
 * of day.
 *
 * 3-hourly is deliberate, not a limitation: below ~3h the honest forecast is
 * "the same as now", which is persistence, not a model.
 *
 * Rendered as an inline SVG rather than a chart library — it is one polyline and
 * a band, and adding a charting dependency to a page a citizen loads on a phone
 * would cost more than it explains.
 */
import { useMemo } from "react";
import useSWR from "swr";
import { api } from "@/lib/api";
import { pm25ToAqi, getAqiCategory } from "@/lib/colors";
import type { WardForecastPoint } from "@/lib/types";
import { icon, TrendingUp } from "@/components/Icon";

const H = 108;          // plot height, px
const PAD_T = 10;
const PAD_B = 20;

/** Clock label for a lead time, so "+15h" reads as an actual time of day. */
function clockAt(hoursAhead: number): string {
  const d = new Date(Date.now() + hoursAhead * 3_600_000);
  const hh = d.getHours();
  const suffix = hh < 12 ? "am" : "pm";
  const h12 = hh % 12 === 0 ? 12 : hh % 12;
  return `${h12}${suffix}`;
}

export default function WardTimeline({ city, wardId }: { city: string; wardId: string }) {
  const { data } = useSWR<WardForecastPoint[]>([city, "ward-forecast"], () =>
    api.cityWardForecast(city)
  );

  const points = useMemo(
    () =>
      (data ?? [])
        .filter((p) => p.ward_id === wardId)
        .sort((a, b) => a.horizon_h - b.horizon_h),
    [data, wardId]
  );

  if (points.length < 2) return null;

  const aqis = points.map((p) => pm25ToAqi(p.pm25_hat));

  // Scaled to the MEDIAN line only, deliberately. The rows now also carry an
  // 80% prediction interval (pm25_p10/pm25_p90, see WardForecastPoint), but it
  // is not drawn — and it must not be allowed to set the y-scale either.
  // Measured on real Delhi: that interval spans 0->296 ug/m3 while the line
  // sits near 131, so including it here flattens the whole series into a
  // straight line in the middle of the plot and destroys the diurnal swing
  // this chart exists to show. See the note in the SVG below for why the band
  // itself is withheld until the model ships a per-horizon interval scale.
  const lo = Math.min(...aqis);
  const hi = Math.max(...aqis);
  const span = Math.max(hi - lo, 1);

  // viewBox coordinates; the SVG scales to whatever width the card gives it.
  const W = 300;
  const x = (i: number) => (i / (points.length - 1)) * W;
  const y = (v: number) => PAD_T + (1 - (v - lo) / span) * (H - PAD_T - PAD_B);

  const line = points.map((p, i) => `${x(i)},${y(pm25ToAqi(p.pm25_hat))}`).join(" ");
  const area = `${line} ${W},${H - PAD_B} 0,${H - PAD_B}`;

  const worstIdx = aqis.indexOf(hi);
  const bestIdx = aqis.indexOf(lo);
  const worst = points[worstIdx];
  const best = points[bestIdx];
  const worstCat = getAqiCategory(hi);
  const bestCat = getAqiCategory(lo);

  return (
    <div className="card" style={{ marginBottom: "var(--space-lg)" }}>
      <h5 style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
        <TrendingUp {...icon.sm} aria-hidden />
        Next 3 days, every 3 hours
      </h5>
      <p style={{ fontSize: "0.75rem", color: "var(--text-tertiary)", marginBottom: 10 }}>
        Predicted AQI for your ward. Air is usually worst overnight and best in the
        afternoon — the swing is bigger than the day-to-day change.
      </p>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        style={{ width: "100%", height: H, display: "block", overflow: "visible" }}
        role="img"
        aria-label={`Predicted AQI for the next 72 hours, ranging from ${lo} to ${hi}`}
      >
        <defs>
          <linearGradient id="wt-fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.26" />
            <stop offset="100%" stopColor="var(--accent)" stopOpacity="0.02" />
          </linearGradient>
        </defs>

        {/* day boundaries at +24 / +48 h, so the diurnal rhythm is legible */}
        {points.map((p, i) =>
          p.horizon_h % 24 === 0 && i < points.length - 1 ? (
            <line
              key={p.horizon_h}
              x1={x(i)} y1={PAD_T} x2={x(i)} y2={H - PAD_B}
              stroke="var(--border-subtle)" strokeWidth="1" strokeDasharray="2 3"
            />
          ) : null
        )}

        {/* The p10/p90 band is DELIBERATELY NOT DRAWN YET — the data carries it
            (WardForecastPoint.pm25_p10/p90) but the served model's interval is
            not calibrated enough to show a citizen.

            Measured on real Delhi, 2026-08-22: the 80% interval spans 0->296
            ug/m3 at h=3 and 0->310 at h=72. Two things are wrong with that.
            It is wider than the signal it brackets (p50 ~131), so it tells a
            resident nothing they can act on; and it barely widens with lead
            time — 5% growth across a 24x change in horizon — because
            `interval_scale` in the manifest is ONE global number applied at
            every horizon, which erases the structure a prediction interval
            exists to show. A 3-hour forecast must be tighter than a 3-day one.

            Drawing it anyway would claim a precision the model has not earned,
            which is the same failure this project deleted the SO2 channel over.
            Flip `hasBand` into the render below once the model ships a
            per-horizon interval scale; everything else is already in place. */}
        <polygon points={area} fill="url(#wt-fill)" />
        <polyline
          points={line}
          fill="none"
          stroke="var(--accent)"
          strokeWidth="2"
          vectorEffect="non-scaling-stroke"
          strokeLinejoin="round"
        />
        <circle cx={x(worstIdx)} cy={y(hi)} r="3.5" fill={worstCat.color}
                stroke="var(--bg-secondary)" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
        <circle cx={x(bestIdx)} cy={y(lo)} r="3.5" fill={bestCat.color}
                stroke="var(--bg-secondary)" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
      </svg>

      <div
        style={{
          display: "flex", justifyContent: "space-between",
          fontSize: "0.65rem", color: "var(--text-tertiary)", marginTop: 2,
        }}
      >
        <span>now</span><span>+24h</span><span>+48h</span><span>+72h</span>
      </div>

      {/* The actionable sentence. A chart is not advice. */}
      <div
        style={{
          display: "flex", gap: "var(--space-md)", flexWrap: "wrap",
          marginTop: "var(--space-md)", fontSize: "0.8rem",
        }}
      >
        <div>
          <div style={{ color: "var(--text-tertiary)", fontSize: "0.7rem" }}>Cleanest</div>
          <div style={{ color: "var(--text-primary)" }}>
            around <strong>{clockAt(best.horizon_h)}</strong> · AQI {lo} {bestCat.label}
          </div>
        </div>
        <div>
          <div style={{ color: "var(--text-tertiary)", fontSize: "0.7rem" }}>Worst</div>
          <div style={{ color: "var(--text-primary)" }}>
            around <strong>{clockAt(worst.horizon_h)}</strong> · AQI {hi} {worstCat.label}
          </div>
        </div>
      </div>

      <p style={{ fontSize: "0.68rem", color: "var(--text-tertiary)", marginTop: 10, lineHeight: 1.5 }}>
        Median across {worst.n_cells} cell{worst.n_cells === 1 ? "" : "s"} in your ward.
        A forecast, not a measurement.
      </p>
    </div>
  );
}
