"use client";
/**
 * WardAccountability — the three questions no other AQI app answers.
 *
 *   1. WHY is my air bad?      (attribution + its evidence chain)
 *   2. WHAT is being done?     (the enforcement case, its status, its legal basis)
 *   3. HOW SURE are you?       (distance to the nearest real monitor)
 *
 * Every other product in this category ships question zero — the number — and
 * stops. The data for 1–3 already exists in this pipeline and was, until now,
 * visible only to administrators.
 *
 * TWO DELIBERATE OMISSIONS, both about not overclaiming in public:
 *
 * · No named private entity. The admin console names candidates; this page names
 *   a CATEGORY and a ZONE. Detection recall is 0/4 on NO2-confounded sources, so
 *   a public page asserting that a specific named business is polluting would be
 *   publishing an accusation the evidence does not support.
 *
 * · No SO2 or aerosol-index evidence. Both were measured as noise (SNR 0.66-1.03)
 *   and dropped from detection, but they still appear in LLM-written evidence
 *   prose. Citing a channel we have publicly called noise as a reason a citizen
 *   should believe us is not an option. They are filtered here.
 */
import { useMemo } from "react";
import useSWR from "swr";
import { cellToLatLng } from "h3-js";
import { api } from "@/lib/api";
import { SOURCE_LABELS } from "@/lib/constants";
import type {
  Attribution, Action, Memo, Hotspot, SourceCategory, FusionCell,
} from "@/lib/types";
import { icon, FileSearch, FileText, Gauge } from "@/components/Icon";

/** Evidence drawn from a channel we have measured as noise. See header. */
const NOISE_EVIDENCE = /\b(aai|aerosol|so2|so₂)\b/i;

function haversineKm(a: [number, number], b: [number, number]): number {
  const R = 6371;
  const dLat = ((b[0] - a[0]) * Math.PI) / 180;
  const dLon = ((b[1] - a[1]) * Math.PI) / 180;
  const la1 = (a[0] * Math.PI) / 180;
  const la2 = (b[0] * Math.PI) / 180;
  const h =
    Math.sin(dLat / 2) ** 2 + Math.cos(la1) * Math.cos(la2) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(h));
}

function Card({
  title,
  icon: Ico,
  rail,
  children,
}: {
  title: string;
  icon: typeof Gauge;
  rail: string;
  children: React.ReactNode;
}) {
  return (
    <div
      className="card card-rail"
      style={{ marginBottom: "var(--space-lg)", ["--rail" as string]: rail }}
    >
      <h5 style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 10 }}>
        <Ico {...icon.sm} aria-hidden />
        {title}
      </h5>
      {children}
    </div>
  );
}

export default function WardAccountability({
  city,
  wardId,
  cells,
}: {
  city: string;
  wardId: string;
  cells: FusionCell[];
}) {
  const { data: attributions } = useSWR<Attribution[]>([city, "attributions"], () =>
    api.cityAttributions(city)
  );
  const { data: hotspots } = useSWR<Hotspot[]>([city, "hotspots"], () => api.cityHotspots(city));
  const { data: actions } = useSWR<Action[]>([city, "actions"], () => api.cityActions(city));
  const { data: memos } = useSWR<Memo[]>([city, "memos"], () => api.cityMemos(city));
  const { data: stations } = useSWR([city, "stations"], () => api.cityStations(city));

  const wardHotspots = useMemo(
    () => (hotspots ?? []).filter((h) => h.ward_id === wardId),
    [hotspots, wardId]
  );

  /**
   * Join everything downstream by ZONE, not by ward.
   *
   * `attribution.ward_id` / `action.ward_id` are the ZONE's ward — every cell in
   * a zone is stamped with the zone's, so they disagree with the cell's own ward
   * whenever a zone straddles a boundary (25 of Delhi's 58 attributed cells do).
   * Filtering on ward_id therefore showed a citizen in DELHI CANTT five hotspots
   * and no explanation, because their zone is registered to KAPASHERA.
   *
   * Zone is also the right unit on the merits: an inspector is dispatched to a
   * zone, so everyone living under that zone has a stake in its case.
   */
  const wardZones = useMemo(
    () => new Set(wardHotspots.map((h) => h.zone_id)),
    [wardHotspots]
  );

  /** The ward's dominant attributed source, by cell count. Ties break on the
   *  higher median confidence, so a single well-evidenced cell cannot be
   *  outvoted by a crowd of weak ones. */
  const dominant = useMemo(() => {
    const mine = (attributions ?? []).filter((a) => wardZones.has(a.zone_id));
    if (!mine.length) return null;
    const by = new Map<SourceCategory, Attribution[]>();
    for (const a of mine) by.set(a.primary_source, [...(by.get(a.primary_source) ?? []), a]);
    const ranked = [...by.entries()].sort(
      (x, y) =>
        y[1].length - x[1].length ||
        Math.max(...y[1].map((a) => a.confidence)) - Math.max(...x[1].map((a) => a.confidence))
    );
    const [source, group] = ranked[0];
    const best = [...group].sort((a, b) => b.confidence - a.confidence)[0];
    return { source, best, nCells: group.length, total: mine.length };
  }, [attributions, wardZones]);

  // Is anything here actually enforceable, or is it all diffuse background?
  const enforceable = wardHotspots.some((h) => h.attributable);
  const hasHotspots = wardHotspots.length > 0;

  const wardActions = useMemo(
    () => (actions ?? []).filter((a) => wardZones.has(a.zone_id)),
    [actions, wardZones]
  );
  const wardMemos = useMemo(
    () => (memos ?? []).filter((m) => wardZones.has(m.zone_id)),
    [memos, wardZones]
  );

  /** Distance from this ward to the nearest real monitor. The whole platform
   *  exists because that distance is usually large. */
  const nearestKm = useMemo(() => {
    const mine = cells.filter((c) => c.ward_id === wardId);
    if (!mine.length || !stations?.length) return null;
    let best = Infinity;
    for (const c of mine) {
      const [lat, lon] = cellToLatLng(c.cell);
      for (const s of stations) best = Math.min(best, haversineKm([lat, lon], [s.lat, s.lon]));
    }
    return Number.isFinite(best) ? best : null;
  }, [cells, wardId, stations]);

  const evidence = (dominant?.best.evidence_factors ?? []).filter((f) => !NOISE_EVIDENCE.test(f));

  return (
    <>
      {/* ── 1. Why ───────────────────────────────────────────────────────────── */}
      <Card title="Why your air is like this" icon={FileSearch} rail="var(--accent)">
        {!hasHotspots || !dominant ? (
          <p style={{ fontSize: "0.875rem", color: "var(--text-secondary)", lineHeight: 1.6 }}>
            No pollution source has been attributed in your ward in this window. That
            means our instruments found nothing standing out here — not that the air
            is clean. Read the AQI above for that.
          </p>
        ) : (
          <>
            <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginBottom: 8, flexWrap: "wrap" }}>
              <span style={{ fontSize: "1.05rem", fontWeight: 600 }}>
                {SOURCE_LABELS[dominant.source]}
              </span>
              <span className="mono" style={{ fontSize: "0.72rem", color: "var(--text-tertiary)" }}>
                confidence {(dominant.best.confidence * 100).toFixed(0)}%
              </span>
            </div>

            {evidence.length > 0 && (
              <ul
                style={{
                  listStyle: "none", padding: 0, margin: "0 0 10px",
                  display: "flex", flexDirection: "column", gap: 5,
                }}
              >
                {evidence.map((f) => (
                  <li
                    key={f}
                    style={{
                      fontSize: "0.8rem", color: "var(--text-secondary)",
                      display: "flex", gap: 7, alignItems: "flex-start", lineHeight: 1.5,
                    }}
                  >
                    <span aria-hidden style={{ color: "var(--accent)", marginTop: 1 }}>·</span>
                    {f}
                  </li>
                ))}
              </ul>
            )}

            <p style={{ fontSize: "0.75rem", color: "var(--text-tertiary)", lineHeight: 1.55 }}>
              Based on {dominant.nCells} of {dominant.total} flagged location
              {dominant.total === 1 ? "" : "s"} in your ward, from satellite columns,
              thermal fire detections and wind direction. We name a source category
              and an area — not a business.
            </p>
          </>
        )}
      </Card>

      {/* ── 2. What is being done ────────────────────────────────────────────── */}
      {hasHotspots && (
        <Card title="What is being done about it" icon={FileText} rail="var(--caution)">
          {!enforceable ? (
            <p style={{ fontSize: "0.875rem", color: "var(--text-secondary)", lineHeight: 1.6 }}>
              This is <strong>diffuse background pollution</strong> — spread across the
              area rather than coming from one site, so there is no single operator to
              serve a notice on. It is a policy matter (traffic, construction and
              cooking across the whole neighbourhood), not a single inspection. It is
              on the city map and feeds ward advisories.
            </p>
          ) : wardActions.length === 0 ? (
            <p style={{ fontSize: "0.875rem", color: "var(--text-secondary)", lineHeight: 1.6 }}>
              An enforceable source has been identified in your ward and is queued for
              prioritisation. No case file has been drafted yet.
            </p>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-md)" }}>
              {wardActions.map((a) => {
                const memo = wardMemos.find((m) => m.zone_id === a.zone_id);
                return (
                  <div key={a.action_id}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", marginBottom: 6 }}>
                      <span className="mono" style={{ fontSize: "0.72rem", color: "var(--text-tertiary)" }}>
                        {memo?.memo_id ?? a.action_id}
                      </span>
                      <span className="badge badge-diffuse">{a.status}</span>
                      <span style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
                        {SOURCE_LABELS[a.source]} · priority {a.eps.toFixed(0)}/100
                      </span>
                    </div>
                    {memo?.legal_basis?.length ? (
                      <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "flex", flexDirection: "column", gap: 4 }}>
                        {memo.legal_basis.map((l) => (
                          <li key={l.id} style={{ fontSize: "0.8rem", color: "var(--text-secondary)", lineHeight: 1.5 }}>
                            <strong style={{ color: "var(--text-primary)", fontWeight: 560 }}>
                              {l.statute}
                            </strong>
                            {l.provision ? `, ${l.provision}` : ""} — {l.summary}
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p style={{ fontSize: "0.8rem", color: "var(--text-tertiary)" }}>
                        No statutory rule matched this source category automatically.
                      </p>
                    )}
                  </div>
                );
              })}
              <p style={{ fontSize: "0.72rem", color: "var(--text-tertiary)", lineHeight: 1.55 }}>
                Case files are drafts generated for an authorised officer to review and
                sign. A draft is not a penalty, and citations are indicative.
              </p>
            </div>
          )}
        </Card>
      )}

      {/* ── 3. How sure ──────────────────────────────────────────────────────── */}
      <Card title="How much to trust this number" icon={Gauge} rail="var(--text-tertiary)">
        <p style={{ fontSize: "0.875rem", color: "var(--text-secondary)", lineHeight: 1.6 }}>
          {nearestKm == null ? (
            <>
              This reading is a model estimate for your ward, not a direct measurement
              at your doorstep.
            </>
          ) : nearestKm <= 2 ? (
            <>
              The nearest government monitor is{" "}
              <strong style={{ color: "var(--text-primary)" }}>{nearestKm.toFixed(1)} km</strong>{" "}
              away, so this reading is anchored to a real instrument close by.
            </>
          ) : (
            <>
              The nearest government monitor is{" "}
              <strong style={{ color: "var(--text-primary)" }}>{nearestKm.toFixed(1)} km</strong>{" "}
              away. There is no sensor in your ward, so this number is an{" "}
              <strong style={{ color: "var(--text-primary)" }}>estimate</strong> built
              from satellite data, weather and local land use — not a measurement. Most
              of this city has no monitor; showing you a number without saying so would
              be the dishonest part.
            </>
          )}
        </p>
      </Card>
    </>
  );
}
