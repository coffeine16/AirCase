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
import { SOURCE_LABELS, confidenceLabel } from "@/lib/constants";
import { plainEvidenceChain } from "@/lib/plainEvidence";
import type {
  Attribution, Action, Memo, Hotspot, SourceCategory, FusionCell,
} from "@/lib/types";
import { icon, FileSearch, FileText, Gauge } from "@/components/Icon";

/** Evidence drawn from a channel we have measured as noise. See header. */

/** Beyond this, the nearest traced source says nothing useful about the air in
 *  YOUR ward — Pune's W008 sits 12.6 km from the closest one — so the line is
 *  withheld rather than padded out with a distance nobody can act on. */
const NEAREST_SOURCE_MAX_KM = 15;

/** Within this, "it can still affect what you breathe" is a fair thing to say.
 *  Past it, stating the distance is fine but claiming influence is not. */
const NEAREST_SOURCE_RELEVANT_KM = 5;

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

  /* Rewritten for a citizen, not an inspector: no OSM ids, no cosines, no
     unexplained jargon, and the model's own scoring lines dropped. The admin
     console still renders the raw strings — see lib/plainEvidence.ts. */
  const evidence = plainEvidenceChain(dominant?.best.evidence_factors ?? []);

  /**
   * For a ward with nothing flagged in it — which is MOST wards, 250 of Delhi's
   * 266 — "we found nothing" is true but a dead end. We still know where the
   * nearest source we DID trace is, and that is the thing a resident actually
   * wants to know next. Measured as nearest cell-to-cell, not centroid-to-
   * centroid, because "how far is it from me" is the question being answered.
   */
  const nearestSource = useMemo(() => {
    if (hasHotspots) return null;
    const mine = cells.filter((c) => c.ward_id === wardId);
    const srcOf = new Map((attributions ?? []).map((a) => [a.zone_id, a]));
    const cand = (hotspots ?? []).filter((h) => srcOf.has(h.zone_id));
    if (!mine.length || !cand.length) return null;

    const pt = (cell: string): [number, number] | null => {
      try { return cellToLatLng(cell) as [number, number]; } catch { return null; }
    };
    let best: { km: number; zoneId: string; wardName: string; wardId: string } | null = null;
    for (const m of mine) {
      const mp = pt(m.cell);
      if (!mp) continue;
      for (const c of cand) {
        const cp = pt(c.cell);
        if (!cp) continue;
        const km = haversineKm(mp, cp);
        if (!best || km < best.km)
          best = { km, zoneId: c.zone_id, wardName: c.ward_name, wardId: c.ward_id };
      }
    }
    if (!best || best.km > NEAREST_SOURCE_MAX_KM) return null;
    const src = srcOf.get(best.zoneId);
    // A zone can sit in the `unassigned` ward — every cell that fell outside the
    // municipal boundary. Its ward_name is the literal string "Outside city
    // limits", which renders as "…1.6 km away, in Outside city limits." Real on
    // Pune's W001. Such a zone is still worth reporting (it is 1.6 km from you
    // and it is really there); only the place-name clause has to change.
    const outside =
      best.wardId === "unassigned" || /outside city limits/i.test(best.wardName);
    return {
      km: best.km,
      wardName: outside ? null : best.wardName,
      source: src?.primary_source as SourceCategory | undefined,
      nZones: new Set(cand.map((c) => c.zone_id)).size,
    };
  }, [hasHotspots, cells, wardId, hotspots, attributions]);

  return (
    <>
      {/* ── 1. Why ───────────────────────────────────────────────────────────── */}
      <Card title="Why your air is like this" icon={FileSearch} rail="var(--accent)">
        {!hasHotspots ? (
          /* Nothing flagged at all. The only case where "we found nothing" is
             actually true. */
          <>
            <p style={{ fontSize: "0.875rem", color: "var(--text-secondary)", lineHeight: 1.6 }}>
              {/* Two short sentences. The old copy went on to explain that this
                  "is not the same as clean air" — correct, and a paragraph a
                  worried resident will not finish. */}
              No source traced to your ward this window. That does not mean the
              air is clean — the reading above still stands.
            </p>
            {nearestSource?.source && (
              <p
                style={{
                  fontSize: "0.875rem", color: "var(--text-secondary)",
                  lineHeight: 1.6, marginTop: 10,
                  paddingTop: 10, borderTop: "1px solid var(--border-subtle)",
                }}
              >
                The nearest source we did trace is{" "}
                <strong style={{ color: "var(--text-primary)" }}>
                  {SOURCE_LABELS[nearestSource.source].toLowerCase()}
                </strong>{" "}
                about{" "}
                <strong style={{ color: "var(--text-primary)" }}>
                  {nearestSource.km < 1
                    ? `${Math.round(nearestSource.km * 1000)} m`
                    : `${nearestSource.km.toFixed(1)} km`}
                </strong>{" "}
                away
                {nearestSource.wardName
                  ? `, in ${nearestSource.wardName}`
                  : ", just beyond the city boundary"}
                .
                {/* Four words, not a sentence. A resident needs to know whether
                    it matters to them, not why air crosses ward boundaries. */}
                {nearestSource.km <= NEAREST_SOURCE_RELEVANT_KM
                  ? " Close enough to affect your air."
                  : " Probably too far to affect you."}
              </p>
            )}
          </>
        ) : !dominant && !enforceable ? (
          /* WE DID FIND SOMETHING — it just has nobody to blame.
             This branch used to print the "found nothing standing out" line above,
             which is simply false here: these wards have flagged cells, and every
             one of their zones came back `attributable: false`. That is the
             enforceable/diffuse split the whole detector is built around, and
             collapsing it into "nothing found" threw away the more useful and
             more honest answer. Measured across the eight cities, 16 wards land
             in exactly this state (Pune's only two flagged wards are both here). */
          <p style={{ fontSize: "0.875rem", color: "var(--text-secondary)", lineHeight: 1.6 }}>
            We did detect raised pollution in your ward — {wardHotspots.length} flagged
            location{wardHotspots.length === 1 ? "" : "s"} — but it is spread across the
            area rather than coming from one place. That pattern is <strong>diffuse
            urban background</strong>: mostly traffic across the whole road network,
            not a single site anyone can be served a notice for. It is real pollution
            and a policy problem, which is why no inspection is queued here.
          </p>
        ) : !dominant ? (
          /* Flagged, and at least one zone IS enforceable, but no attribution row
             resolved for it yet. Rare (2 zones in Chennai at the time of writing)
             and worth stating plainly rather than implying nothing was found. */
          <p style={{ fontSize: "0.875rem", color: "var(--text-secondary)", lineHeight: 1.6 }}>
            {wardHotspots.length} location{wardHotspots.length === 1 ? " is" : "s are"} flagged
            in your ward, but none has a confirmed source yet — the evidence so far does
            not point clearly enough at any one place to name it. Naming a source we
            cannot evidence would be a guess, so we are not making one.
          </p>
        ) : (
          <>
            <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginBottom: 8, flexWrap: "wrap" }}>
              <span style={{ fontSize: "1.05rem", fontWeight: 600 }}>
                {SOURCE_LABELS[dominant.source]}
              </span>
              <span style={{ fontSize: "0.72rem", color: "var(--text-tertiary)" }}>
                {confidenceLabel(dominant.best.confidence)}{" "}
                <span className="mono">({(dominant.best.confidence * 100).toFixed(0)}%)</span>
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

            {/* "19 of 19" reads as a bug, not a statistic — when every flagged
                location agrees, say so. The instrument list also named satellite
                twice ("satellite readings, heat from fires picked up by
                satellite"). The last sentence stays whatever else goes: naming a
                CATEGORY and an AREA rather than a business is the line between
                an evidence chain and an accusation. */}
            <p style={{ fontSize: "0.75rem", color: "var(--text-tertiary)", lineHeight: 1.55 }}>
              {dominant.nCells === dominant.total
                ? `Based on all ${dominant.total} flagged location${dominant.total === 1 ? "" : "s"}`
                : `Based on ${dominant.nCells} of ${dominant.total} flagged locations`}{" "}
              in your ward — satellite, fire heat and wind direction. We name a
              type of source and an area, never a business.
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
              {/* Says the same thing in a third of the words. The old copy
                  argued its own ethics at the reader ("showing you a number
                  without saying so would be the dishonest part") — true, and
                  not something a person checking their air will read. The
                  asterisk carries the caveat; the sentence carries the fact. */}
              No sensor in your ward — nearest is{" "}
              <strong style={{ color: "var(--text-primary)" }}>{nearestKm.toFixed(1)} km</strong>{" "}
              away, so this is an{" "}
              <strong style={{ color: "var(--text-primary)" }}>estimate*</strong>{" "}
              from satellite, weather and land use.
              <br />
              <span style={{ opacity: 0.7 }}>
                *not a direct measurement
              </span>
            </>
          )}
        </p>
      </Card>
    </>
  );
}
