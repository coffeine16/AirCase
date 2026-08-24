"use client";
/**
 * ActionQueue — EPS-ranked zone cards.
 *
 * KEY INVARIANTS (see architecture.md + eps-spec.md):
 * 1. An ACTION is a ZONE, not a cell.
 *    hotspots.json has ~74 rows but only ~5 zones. Group by zone_id.
 *    An inspector visits a place, not a 460 m hexagon.
 * 2. Only `attributable: true` zones enter the queue.
 *    Diffuse zones (urban background) stay on the map, feed ward advisories,
 *    but are EXCLUDED here — there is nobody to serve a notice on.
 * 3. EPS is zone-level: severity = max across zone's cells,
 *    attribution_conf = max over zone's cells.
 * 4. forecast_delta = 0. The forecast agent IS built and runs — 3-hourly to
 *    +72h — but its output is deliberately not yet wired into the EPS severity
 *    term (FORECAST_WEIGHT = 0.0 in prioritise.py). A zero we can explain beats
 *    a weight we have not validated.
 *    FORECAST_WEIGHT = 0.0 — a zero we can explain beats a number we invented.
 */
import { useState, useMemo } from "react";
import useSWR from "swr";
import { api } from "@/lib/api";
import type { Hotspot, Attribution, Memo } from "@/lib/types";
import { SOURCE_LABELS, PERSISTENCE_LABELS } from "@/lib/constants";
import { SOURCE_COLORS } from "@/lib/colors";
import { useCity } from "@/lib/CityContext";
import { icon, ChevronDown, FileText, Crosshair, Check, Flame, Wind, Target } from "@/components/Icon";
import MemoModal from "./MemoModal";

interface Props {
  hotspots: Hotspot[];
  loading: boolean;
  selectedCell: string | null;
  onSelectCell: (cell: string) => void;
}

/**
 * Does this zone sit outside the municipal boundary?
 *
 * The H3 fabric is built from a rectangular BBOX; a municipality is an
 * irregular polygon inside it. So a real, correctly-detected source can land in
 * the bbox but outside the city — 165 of 580 hotspot cells and 12 of 46 actions
 * do, and for Pune it is the ENTIRE queue (2 of 2). `wards.py` already assigns
 * those cells to an "Outside city limits" bucket; nothing downstream read it.
 *
 * We tag rather than drop, deliberately. A landfill 2 km beyond the PMC line
 * still pollutes Pune, so deleting it loses a true finding. But dispatching a
 * municipal inspector there sends them somewhere their authority does not
 * reach. The honest action is to refer it to the state board — which is a
 * routing decision, not a detection failure.
 */
function isOutsideJurisdiction(wardName: string): boolean {
  return wardName.toLowerCase().includes("outside city limits");
}

// ── Derived zone type (grouped from hotspot cells) ─────────────────────────────

interface Zone {
  zone_id: string;
  ward_id: string;
  ward_name: string;
  kind: Hotspot["kind"];
  attributable: boolean;
  /** Severity = max across cells (zone is as bad as its worst cell) */
  severity: number;
  /** PM2.5 median = median of cell-level pm25_med values */
  pm25_med: number;
  n_cells: number;
  cells: string[];
  /** Representative cell for attribution lookup (highest severity) */
  lead_cell: string;
  detection_basis: string;
  fires_6h: number;
}

function groupIntoZones(hotspots: Hotspot[]): Zone[] {
  const zoneMap = new Map<string, Zone>();
  for (const h of hotspots) {
    const existing = zoneMap.get(h.zone_id);
    if (!existing) {
      zoneMap.set(h.zone_id, {
        zone_id: h.zone_id,
        ward_id: h.ward_id,
        ward_name: h.ward_name,
        kind: h.kind,
        attributable: h.attributable,
        severity: h.severity,
        pm25_med: h.pm25_med,
        n_cells: 1,
        cells: [h.cell],
        lead_cell: h.cell,
        detection_basis: h.detection_basis,
        fires_6h: h.fires_6h,
      });
    } else {
      // kind: use the most persistent verdict across cells
      const kindRank: Record<Hotspot["kind"], number> = { chronic: 3, emerging: 2, acute: 1 };
      if (kindRank[h.kind] > kindRank[existing.kind]) existing.kind = h.kind;
      // severity: max
      if (h.severity > existing.severity) {
        existing.severity = h.severity;
        existing.lead_cell = h.cell;
      }
      // pm25_med: running average (will be approximately the zone median)
      existing.pm25_med = (existing.pm25_med * existing.n_cells + h.pm25_med) / (existing.n_cells + 1);
      existing.n_cells += 1;
      existing.cells.push(h.cell);
      existing.fires_6h = Math.max(existing.fires_6h, h.fires_6h);
      if (h.attributable) existing.attributable = true;
    }
  }
  return Array.from(zoneMap.values());
}

// ── Sub-components ─────────────────────────────────────────────────────────────

function ConfidenceMeter({ value }: { value: number }) {
  // Three semantic tokens, not three raw hexes. Confidence is a JUDGEMENT about
  // our own evidence, so it belongs in the state palette (positive/caution/
  // critical) rather than getting an accent of its own.
  const tint = value >= 0.7 ? "var(--positive)" : value >= 0.5 ? "var(--caution)" : "var(--critical)";
  return (
    <div
      style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 96 }}
      title={`Attribution confidence ${value.toFixed(2)}`}
    >
      <div className="meter" style={{ ["--tint" as string]: tint }}>
        <i style={{ width: `${Math.round(value * 100)}%` }} />
      </div>
      <span className="mono" style={{ fontSize: "0.72rem", color: tint, minWidth: 28 }}>
        {value.toFixed(2)}
      </span>
    </div>
  );
}

function EPSBreakdown({ components, eps }: { components: Record<string, number>; eps: number }) {
  const items = [
    { key: "severity",         label: "Severity",    weight: 0.35 },
    { key: "attribution_conf", label: "Conf.",        weight: 0.25 },
    { key: "actionability",    label: "Actionability",weight: 0.20 },
    { key: "vulnerability",    label: "Vulnerability",weight: 0.20 },
  ];
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 2 }}>
        <span className="section-label">EPS breakdown</span>
        <span className="mono" style={{ fontSize: "0.8rem", fontWeight: 600, color: "var(--text-primary)" }}>
          {eps.toFixed(1)}
        </span>
      </div>
      {items.map(({ key, label, weight }) => {
        const val: number = (components[key] as number) ?? 0;
        return (
          <div key={key} style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ fontSize: "0.68rem", color: "var(--text-tertiary)", minWidth: 74 }}>{label}</span>
            <div className="meter">
              <i style={{ width: `${Math.round(val * 100)}%` }} />
            </div>
            <span className="mono" style={{ fontSize: "0.68rem", color: "var(--text-tertiary)", minWidth: 24, textAlign: "right" }}>
              {(val * weight * 100).toFixed(0)}
            </span>
          </div>
        );
      })}
      <div style={{ fontSize: "0.65rem", color: "var(--text-tertiary)", marginTop: 2 }}>
        forecast_delta = 0 — forecast not yet fed into EPS
      </div>
    </div>
  );
}

function EvidenceChain({ attribution }: { attribution: Attribution }) {
  const sourceColor = SOURCE_COLORS[attribution.primary_source] ?? "var(--text-tertiary)";
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-sm)" }}>
      <div
        style={{
          padding: "10px 12px",
          background: "var(--bg-primary)",
          borderRadius: "var(--radius-sm)",
          fontSize: "0.82rem",
          lineHeight: 1.6,
          color: "var(--text-secondary)",
          borderLeft: `3px solid ${sourceColor}`,
        }}
      >
        {attribution.reason}
      </div>
      {attribution.evidence_factors.length > 0 && (
        <div>
          <div className="section-label" style={{ marginBottom: 6 }}>Evidence</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            {attribution.evidence_factors.map((f, i) => (
              <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 7, fontSize: "0.775rem", lineHeight: 1.5, color: "var(--text-secondary)" }}>
                <Check {...icon.sm} aria-hidden style={{ color: "var(--positive)", marginTop: 3, flexShrink: 0 }} />
                {f}
              </div>
            ))}
          </div>
        </div>
      )}
      {attribution.evidence?.meteorology && (
        <div style={{ display: "flex", gap: "var(--space-md)", fontSize: "0.73rem", color: "var(--text-tertiary)", flexWrap: "wrap" }}>
          <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
            <Wind {...icon.sm} aria-hidden />
            {attribution.evidence.meteorology.wind_from_deg}° · {attribution.evidence.meteorology.wind_ms.toFixed(1)} m/s
          </span>
          <span>
            BLH {attribution.evidence.meteorology.blh_m} m
            {attribution.evidence.meteorology.air_trapped && (
              <span style={{ color: "var(--caution)" }}> · trapped</span>
            )}
          </span>
        </div>
      )}
      <div style={{ fontSize: "0.68rem", color: "var(--text-tertiary)" }}>
        Explained by: {attribution.explained_by}
      </div>
    </div>
  );
}

function ZoneCard({ zone, rank, isSelected, onSelect }: {
  zone: Zone;
  rank: number;
  isSelected: boolean;
  onSelect: () => void;
}) {
  const { city } = useCity();
  const [expanded, setExpanded] = useState(isSelected);
  const { data: attribution, isLoading: loadingAttr } = useSWR<Attribution>(
    expanded ? [city, "attribution", zone.lead_cell] : null,
    () => api.getAttribution(zone.lead_cell)
  );
  const sourceColor = (attribution && SOURCE_COLORS[attribution.primary_source]) || "var(--text-tertiary)";

  // Memo modal — the demo climax. Fetch the precomputed enforcement memo for this
  // zone on click; render it in a modal.
  const [memoOpen, setMemoOpen] = useState(false);
  const [memo, setMemo] = useState<Memo | null>(null);
  const [memoLoading, setMemoLoading] = useState(false);
  const [memoError, setMemoError] = useState<string | null>(null);

  const openMemo = async () => {
    setMemoOpen(true);
    setMemoLoading(true);
    setMemoError(null);
    setMemo(null);
    try {
      setMemo(await api.getMemo(zone.zone_id, city));
    } catch {
      setMemoError(`No memo for zone ${zone.zone_id}. Run the pipeline to draft one.`);
    } finally {
      setMemoLoading(false);
    }
  };

  return (
    <div
      className="card animate-fade-in"
      style={{
        borderColor: isSelected ? "var(--accent-line)" : "var(--border-subtle)",
        background: isSelected ? "var(--accent-soft)" : undefined,
        transition: "border-color var(--transition-fast), background var(--transition-fast)",
      }}
    >
      {/* Header */}
      <div
        style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 8, cursor: "pointer" }}
        onClick={() => { onSelect(); setExpanded((e) => !e); }}
      >
        <div style={{ flex: 1, minWidth: 0 }}>
          {/* Rank + badges */}
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
            <span
              className="mono"
              style={{
                fontSize: "0.65rem", fontWeight: 600,
                color: "var(--text-tertiary)",
                background: "var(--bg-tertiary)",
                padding: "1px 5px",
                borderRadius: "var(--radius-sm)",
              }}
            >
              {rank}
            </span>
            <span className={`badge badge-${zone.kind}`}>{PERSISTENCE_LABELS[zone.kind]}</span>
            {isOutsideJurisdiction(zone.ward_name) && (
              <span
                className="badge"
                title="This zone falls outside the municipal boundary. The detection stands — but a city inspector has no authority here, so it routes to the state pollution control board instead."
                style={{
                  ["--tint" as string]: "var(--text-tertiary)",
                  ["--tint-soft" as string]: "color-mix(in srgb, var(--text-tertiary) 12%, transparent)",
                  ["--tint-line" as string]: "color-mix(in srgb, var(--text-tertiary) 24%, transparent)",
                }}
              >
                refer to state board
              </span>
            )}
          </div>
          {/* Zone + ward */}
          <div style={{ fontWeight: 600, fontSize: "0.9rem", color: "var(--text-primary)", marginBottom: 2 }} className="truncate">
            {zone.zone_id} · {zone.ward_name}
          </div>
          {/* Metrics */}
          <div className="mono" style={{ display: "flex", alignItems: "center", gap: 5, fontSize: "0.73rem", color: "var(--text-tertiary)" }}>
            {zone.pm25_med.toFixed(1)} µg/m³ · {zone.n_cells} cells
            {zone.fires_6h > 0 && (
              <span style={{ display: "inline-flex", alignItems: "center", gap: 3, color: "var(--persist-acute)" }}>
                · <Flame {...icon.sm} aria-hidden /> {zone.fires_6h}
              </span>
            )}
          </div>
        </div>
        {/* Severity. A solid red tile whose ALPHA encoded the value was
            unreadable at the low end (white text on near-transparent) and
            screamed at the high end. It is now a number with a proportional
            bar — the value is legible at every level. */}
        <div style={{ flexShrink: 0, marginLeft: "var(--space-sm)", width: 46, textAlign: "right" }}>
          <div className="mono" style={{ fontSize: "1.05rem", fontWeight: 600, lineHeight: 1, color: "var(--text-primary)" }}>
            {(zone.severity * 100).toFixed(0)}
          </div>
          <div className="section-label" style={{ fontSize: "0.6rem", margin: "3px 0 4px" }}>sev</div>
          <div className="meter" style={{ ["--tint" as string]: "var(--critical)" }}>
            <i style={{ width: `${Math.round(zone.severity * 100)}%` }} />
          </div>
        </div>
      </div>

      {/* Expand toggle */}
      <button
        onClick={() => setExpanded((e) => !e)}
        aria-expanded={expanded}
        style={{
          display: "flex", alignItems: "center", gap: 4,
          background: "none", border: "none", cursor: "pointer",
          fontSize: "0.73rem", color: "var(--text-tertiary)",
          fontFamily: "inherit", padding: 0,
          marginBottom: expanded ? 10 : 0,
          transition: "color var(--transition-fast)",
        }}
      >
        <ChevronDown
          {...icon.sm}
          aria-hidden
          style={{
            transform: expanded ? "rotate(180deg)" : "none",
            transition: "transform var(--transition-normal)",
          }}
        />
        {expanded ? "Collapse" : "Evidence + EPS"}
      </button>

      {expanded && (
        <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-sm)" }}>
          {/* Attribution */}
          {attribution ? (
            <>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                <span
                  className="badge"
                  style={{
                    ["--tint" as string]: sourceColor,
                    ["--tint-soft" as string]: `${sourceColor}1f`,
                    ["--tint-line" as string]: `${sourceColor}3d`,
                  }}
                >
                  {SOURCE_LABELS[attribution.primary_source] ?? attribution.primary_source}
                </span>
                <ConfidenceMeter value={attribution.confidence} />
              </div>
              <EvidenceChain attribution={attribution} />
            </>
          ) : loadingAttr ? (
            <div className="skeleton" style={{ height: 80, borderRadius: "var(--radius-sm)" }} />
          ) : (
            <div style={{ fontSize: "0.8rem", color: "var(--text-tertiary)" }}>
              No attribution available for this zone
            </div>
          )}

          {/* EPS breakdown (from actions.json if available, else stub) */}
          {attribution && (
            <EPSBreakdown
              eps={zone.severity * 100}  // approximate until actions.json lands
              components={{
                severity: zone.severity,
                attribution_conf: attribution.confidence,
                actionability: zone.kind === "chronic" ? 0.6 : zone.kind === "emerging" ? 0.8 : 1.0,
                vulnerability: 0.4,  // placeholder until vulnerability data lands
              }}
            />
          )}

          {/* Detection basis */}
          <div style={{ fontSize: "0.72rem", color: "var(--text-tertiary)", borderTop: "1px solid var(--border-subtle)", paddingTop: 8 }}>
            {zone.detection_basis}
          </div>

          {/* Cells */}
          <div style={{ fontSize: "0.65rem", color: "var(--text-tertiary)" }}>
            {zone.n_cells} cells · lead: <code style={{ fontFamily: "var(--font-mono)" }}>{zone.lead_cell.slice(0, 12)}…</code>
          </div>

          {/* Action buttons */}
          <div style={{ display: "flex", gap: "var(--space-sm)" }}>
            {/* One primary action per card. "Generate memo" is the product's
                point; showing on the map is a navigation aid, so it is quiet.
                They used to be two equally-weighted tinted buttons. */}
            <button
              className="btn btn-primary btn-sm"
              onClick={(e) => { e.stopPropagation(); openMemo(); }}
              style={{ flex: 1 }}
            >
              <FileText {...icon.sm} aria-hidden />
              Generate memo
            </button>
            <button
              className="btn btn-ghost btn-sm"
              onClick={(e) => { e.stopPropagation(); onSelect(); }}
              title="Select this zone on the map"
            >
              <Crosshair {...icon.sm} aria-hidden />
              Show on map
            </button>
          </div>
        </div>
      )}

      {memoOpen && (
        <MemoModal
          memo={memo}
          loading={memoLoading}
          error={memoError}
          onClose={() => setMemoOpen(false)}
        />
      )}
    </div>
  );
}

// ── Main component ──────────────────────────────────────────────────────────────

export default function ActionQueue({ hotspots, loading, selectedCell, onSelectCell }: Props) {
  // Step 1: group cells into zones
  const allZones = useMemo(() => groupIntoZones(hotspots), [hotspots]);

  // Step 2: only attributable zones enter the enforcement queue
  const enforceableZones = useMemo(
    () => allZones
      .filter((z) => z.attributable)
      .sort((a, b) => b.severity - a.severity),
    [allZones]
  );

  // Step 3: diffuse zones stay on the map and feed advisories — NOT in the queue
  const diffuseZones = useMemo(
    () => allZones.filter((z) => !z.attributable).sort((a, b) => b.severity - a.severity),
    [allZones]
  );

  // Find which zone contains the selected cell
  const selectedZoneId = selectedCell
    ? allZones.find((z) => z.cells.includes(selectedCell))?.zone_id ?? null
    : null;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
      {/* Panel header */}
      <div style={{ padding: "11px var(--space-md)", borderBottom: "1px solid var(--border-subtle)", flexShrink: 0 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 2 }}>
          <h4>Action queue</h4>
          <span className="mono" style={{ fontSize: "0.72rem", color: "var(--text-tertiary)" }}>
            {enforceableZones.length} zone{enforceableZones.length !== 1 ? "s" : ""}
          </span>
        </div>
        <p style={{ fontSize: "0.73rem", color: "var(--text-tertiary)" }}>
          Zone-level · sorted by severity
        </p>
      </div>

      {/* Scrollable list */}
      <div style={{ flex: 1, overflowY: "auto", padding: "var(--space-sm)" }}>
        {loading ? (
          <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-sm)" }}>
            {[1, 2, 3].map((i) => (
              <div key={i} className="skeleton" style={{ height: 88, borderRadius: "var(--radius-md)" }} />
            ))}
          </div>
        ) : enforceableZones.length === 0 && !loading ? (
          <div className="empty" style={{ border: "none", minHeight: 200 }}>
            <Target {...icon.lg} aria-hidden />
            <p>No enforceable zones detected. Run the detection and attribution agents to refresh.</p>
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-sm)" }}>
            {/* Enforceable zones */}
            <div className="section-label" style={{ padding: "4px" }}>
              Enforcement queue · {enforceableZones.length}
            </div>
            {enforceableZones.map((zone, i) => (
              <ZoneCard
                key={zone.zone_id}
                zone={zone}
                rank={i + 1}
                isSelected={selectedZoneId === zone.zone_id}
                onSelect={() => onSelectCell(zone.lead_cell)}
              />
            ))}

            {/* Diffuse zones — shown as a secondary list, NOT in the queue */}
            {diffuseZones.length > 0 && (
              <>
                <div
                  className="section-label"
                  style={{
                    padding: "12px 4px 4px", marginTop: 12,
                    borderTop: "1px solid var(--border-subtle)",
                  }}
                >
                  Policy targets · diffuse · {diffuseZones.length}
                </div>
                <div className="note">
                  Real pollution, but diffuse urban background with no single actor
                  responsible. These stay on the map and feed ward advisories —
                  dispatching an inspector would serve a notice to a road.
                </div>
                {diffuseZones.map((zone) => (
                  <div
                    key={zone.zone_id}
                    className="card card-hover"
                    style={{ cursor: "pointer", background: "transparent" }}
                    onClick={() => onSelectCell(zone.lead_cell)}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between" }}>
                      <div>
                        <div style={{ fontWeight: 500, fontSize: "0.85rem", marginBottom: 2 }}>
                          {zone.zone_id} · {zone.ward_name}
                        </div>
                        <div className="mono" style={{ fontSize: "0.73rem", color: "var(--text-tertiary)" }}>
                          {zone.pm25_med.toFixed(1)} µg/m³ · {zone.n_cells} cells
                        </div>
                      </div>
                      <span className={`badge badge-${zone.kind}`} style={{ alignSelf: "flex-start" }}>
                        {zone.kind}
                      </span>
                    </div>
                  </div>
                ))}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
