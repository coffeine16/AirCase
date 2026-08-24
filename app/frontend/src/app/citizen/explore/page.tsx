"use client";
/**
 * Citizen Explore — full-bleed map, three switchable views, tap-to-reveal detail.
 *
 * Deliberately NOT the admin console shrunk down: no side panel, no checkbox
 * layer stack, no filter bar. A mode is a big glass chip floating on the map
 * itself, and tapping the map is the only way to see anything — there is no
 * list to scroll instead; a bottom sheet reveals whatever you tapped. Three
 * views:
 *   - Air quality  — the live fusion field, extruded by PM2.5 on a tilted
 *     camera, same choropleth data as the ward page.
 *   - Forecast     — the same field at +24/+48/+72h.
 *   - Cases        — enforcement actions as flat pins, coloured by status
 *     (extrusion only applies to the hex layer; pins never get one).
 *
 * Full-bleed like the admin console: the page fills the viewport below the
 * citizen header (`calc(100vh - 52px)` — 52 must match citizen/layout.tsx's
 * header height) with `overflow: hidden`, and every control floats over the
 * map instead of pushing it down the page.
 *
 * "Cases" deliberately withholds what the admin ActionQueue shows: no EPS
 * score (an internal triage number, not a public ranking), no dispatch
 * routes (publishing an inspector's live route/order is a real operational-
 * security problem, not a UX one), no named private entity — same category +
 * zone discipline WardAccountability.tsx already applies. See
 * docs/citizen-view-improvements.md for the full reasoning.
 */
import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import dynamic from "next/dynamic";
import useSWR from "swr";
import { api } from "@/lib/api";
import { useCity } from "@/lib/CityContext";
import { pm25ToAqi, getAqiCategory, ACTION_STATUS_HEX } from "@/lib/colors";
import {
  SOURCE_LABELS, ACTION_STATUS_LABELS, ACTION_STATUS_BADGE, confidenceLabel,
} from "@/lib/constants";
import { icon, Wind, TrendingUp, Target, ArrowRight, X, Search } from "@/components/Icon";
import type { FusionResponse, WardsResponse, ForecastCell, Action, Memo, ActionStatus } from "@/lib/types";

const CitizenMap = dynamic(() => import("@/components/citizen/CitizenMap"), {
  ssr: false,
  loading: () => (
    <div className="skeleton" style={{ position: "absolute", inset: 0 }} />
  ),
});

// Must match citizen/layout.tsx's header `height: 52`.
const HEADER_H = 52;

type Mode = "air" | "forecast" | "cases";
type Horizon = 24 | 48 | 72;

const MODES: { id: Mode; label: string; icon: typeof Wind }[] = [
  { id: "air", label: "Air quality", icon: Wind },
  { id: "forecast", label: "Forecast", icon: TrendingUp },
  { id: "cases", label: "Cases", icon: Target },
];

export default function ExplorePage() {
  const { city, cityLabel } = useCity();
  const [mode, setMode] = useState<Mode>("air");
  const [horizon, setHorizon] = useState<Horizon>(24);
  const [view3D, setView3D] = useState(true);
  const [selectedWard, setSelectedWard] = useState<{ wardId: string; wardName: string; pm25: number } | null>(null);
  const [selectedActionId, setSelectedActionId] = useState<string | null>(null);
  // The ward the citizen SEARCHED for, as opposed to one they tapped. Only this
  // drives the map camera: tapping a hex must not yank the viewport around while
  // someone is exploring, but typing a ward name is an explicit "take me there".
  const [focusWard, setFocusWard] = useState<string | null>(null);
  const [query, setQuery] = useState("");

  // Switching mode invalidates whatever was selected under the old one.
  useEffect(() => { setSelectedWard(null); setSelectedActionId(null); setFocusWard(null); setQuery(""); }, [mode]);

  const { data: fusion } = useSWR<FusionResponse>([city, "fusion"], () => api.cityFusion(city));
  const { data: wardsResp } = useSWR<WardsResponse>([city, "wards"], () => api.cityWards(city));
  const { data: forecastData } = useSWR<ForecastCell[]>(
    mode === "forecast" ? [city, "forecast", horizon] : null,
    () => api.cityForecast(city, horizon)
  );
  const { data: actions } = useSWR<Action[]>(mode === "cases" ? [city, "actions"] : null, () => api.cityActions(city));
  const { data: memos } = useSWR<Memo[]>(mode === "cases" ? [city, "memos"] : null, () => api.cityMemos(city));

  const wardName = useMemo(() => {
    const m = new Map<string, string>();
    for (const c of wardsResp?.cells ?? []) m.set(c.ward_id, c.ward_name);
    return m;
  }, [wardsResp]);
  const wardOf = useMemo(() => {
    const m = new Map<string, string>();
    for (const c of wardsResp?.cells ?? []) m.set(c.cell, c.ward_id);
    return m;
  }, [wardsResp]);

  // Only one of `cells` / `pins` is ever non-empty — CitizenMap renders
  // whichever it's given and skips the other.
  const cells = useMemo(() => {
    if (mode === "forecast") {
      return (forecastData ?? []).map((f) => ({
        cell: f.cell, pm25: f.pm25_hat, ward_id: wardOf.get(f.cell) ?? "unassigned",
      }));
    }
    if (mode === "air") return fusion?.cells ?? [];
    return [];
  }, [mode, forecastData, fusion, wardOf]);

  const pins = useMemo(() => {
    if (mode !== "cases") return undefined;
    return (actions ?? []).map((a) => ({
      id: a.action_id,
      lat: a.centroid.lat,
      lon: a.centroid.lon,
      color: ACTION_STATUS_HEX[a.status],
      selected: a.action_id === selectedActionId,
    }));
  }, [mode, actions, selectedActionId]);

  const handlePickWard = (wardId: string, cell: string) => {
    setSelectedActionId(null);
    const c = cells.find((c) => c.cell === cell);
    setSelectedWard({ wardId, wardName: wardName.get(wardId) ?? wardId, pm25: c?.pm25 ?? NaN });
  };
  const handlePickPin = (id: string) => {
    setSelectedWard(null);
    setSelectedActionId(id);
  };

  /** Every ward in the city, once, sorted by name — the search corpus.
   *  Built from wards.json rather than from `cells`, so a ward with no fusion
   *  coverage in the current mode is still findable instead of silently absent. */
  const wards = useMemo(() => {
    const seen = new Map<string, string>();
    for (const c of wardsResp?.cells ?? []) {
      if (!c.ward_id || c.ward_id === "unassigned" || seen.has(c.ward_id)) continue;
      seen.set(c.ward_id, c.ward_name);
    }
    return [...seen.entries()].map(([ward_id, ward_name]) => ({ ward_id, ward_name }))
      .sort((a, b) => a.ward_name.localeCompare(b.ward_name));
  }, [wardsResp]);

  const searchHits = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return [];
    return wards
      .filter((w) => w.ward_name.toLowerCase().includes(q) || w.ward_id.toLowerCase().includes(q))
      .slice(0, 8);
  }, [wards, query]);

  /** A searched ward has no single cell behind it, so its reading is the MEDIAN
   *  over the ward's cells — the same statistic the ward page and WardComparison
   *  use, so the two surfaces cannot disagree about the same ward. */
  const wardMedianPm25 = (wardId: string): number => {
    const vals = cells
      .filter((c) => c.ward_id === wardId && Number.isFinite(c.pm25))
      .map((c) => c.pm25)
      .sort((a, b) => a - b);
    if (!vals.length) return NaN;
    const mid = Math.floor(vals.length / 2);
    return vals.length % 2 ? vals[mid] : (vals[mid - 1] + vals[mid]) / 2;
  };

  const handleSearchPick = (wardId: string, wardName: string) => {
    setSelectedActionId(null);
    setFocusWard(wardId);
    setQuery("");
    setSelectedWard({ wardId, wardName, pm25: wardMedianPm25(wardId) });
  };

  /** Clearing the search returns the camera to the whole city — CitizenMap
   *  re-fits to all cells the moment nothing is highlighted, which is exactly
   *  what "clear" should mean here. */
  const clearSearch = () => {
    setFocusWard(null);
    setQuery("");
    setSelectedWard(null);
  };

  const selectedAction = useMemo(
    () => (actions ?? []).find((a) => a.action_id === selectedActionId) ?? null,
    [actions, selectedActionId]
  );
  const selectedMemo = useMemo(
    () => (selectedAction ? (memos ?? []).find((m) => m.zone_id === selectedAction.zone_id) ?? null : null),
    [memos, selectedAction]
  );

  const wardAqi = selectedWard && Number.isFinite(selectedWard.pm25) ? pm25ToAqi(selectedWard.pm25) : null;
  const wardCat = wardAqi != null ? getAqiCategory(wardAqi) : null;
  const casesEmpty = mode === "cases" && !selectedAction && (actions?.length ?? 0) === 0;

  return (
    <div style={{ position: "relative", width: "100%", height: `calc(100vh - ${HEADER_H}px)`, overflow: "hidden" }}>
      <CitizenMap
        cells={cells}
        pins={pins}
        highlightWard={focusWard}
        onPickWard={handlePickWard}
        onPickPin={handlePickPin}
        height="100%"
        extruded={view3D}
        bleed
      />

      {/* Ward search — top-left, above the map.
          Hidden in Cases mode for the same reason the 2D/3D toggle is: that mode
          renders pins, not the hex layer, so highlighting a ward would have
          nothing to draw and the control would visibly do nothing. */}
      {mode !== "cases" && (
        <div className="explore-search">
          {focusWard ? (
            <div
              className="glass"
              style={{
                display: "flex", alignItems: "center", justifyContent: "space-between",
                gap: 8, padding: "8px 10px", borderRadius: "var(--radius-lg)",
                boxShadow: "var(--shadow-md)",
              }}
            >
              <span className="truncate" style={{ fontSize: "0.8rem", fontWeight: 550 }}>
                {selectedWard?.wardName ?? focusWard}
              </span>
              <button
                onClick={clearSearch}
                aria-label="Clear ward search and show the whole city"
                className="btn btn-ghost btn-icon btn-sm"
                style={{ flexShrink: 0 }}
              >
                <X {...icon.sm} aria-hidden />
              </button>
            </div>
          ) : (
            <div className="glass" style={{ position: "relative", borderRadius: "var(--radius-lg)", boxShadow: "var(--shadow-md)" }}>
              <Search
                {...icon.sm}
                aria-hidden
                style={{
                  position: "absolute", left: 10, top: "50%", transform: "translateY(-50%)",
                  color: "var(--text-tertiary)", pointerEvents: "none",
                }}
              />
              <input
                type="search"
                aria-label="Search your ward by name"
                placeholder="Search your ward…"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                style={{
                  width: "100%", paddingLeft: 32, fontSize: "0.82rem",
                  background: "transparent", border: "none",
                }}
              />
              {searchHits.length > 0 && (
                <div
                  role="listbox"
                  aria-label="Matching wards"
                  className="menu"
                  style={{ position: "absolute", top: "calc(100% + 4px)", left: 0, right: 0, zIndex: 1 }}
                >
                  {searchHits.map((w) => (
                    <button
                      key={w.ward_id}
                      role="option"
                      aria-selected={false}
                      className="menu-item"
                      onClick={() => handleSearchPick(w.ward_id, w.ward_name)}
                    >
                      <span className="truncate">{w.ward_name}</span>
                      <span className="menu-meta mono">{w.ward_id}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* 2D/3D toggle — only meaningful for the hex layer (air/forecast); pins
          never extrude, so hide it in Cases mode rather than offer a control
          that visibly does nothing. */}
      {mode !== "cases" && (
        <div
          className="glass explore-viewtoggle"
          style={{
            display: "flex", gap: 4, padding: 4, borderRadius: "var(--radius-lg)",
            boxShadow: "var(--shadow-md)",
          }}
        >
          <button className="chip" data-active={!view3D} onClick={() => setView3D(false)} aria-pressed={!view3D}>
            2D
          </button>
          <button className="chip" data-active={view3D} onClick={() => setView3D(true)} aria-pressed={view3D}>
            3D
          </button>
        </div>
      )}

      {/* Floating mode switcher + (forecast) horizon, stacked top-center */}
      <div
        style={{
          position: "absolute", top: 12, left: "50%", transform: "translateX(-50%)",
          zIndex: "var(--z-panel)", display: "flex", flexDirection: "column",
          gap: 6, alignItems: "center",
        }}
      >
        <div className="glass" style={{ display: "flex", gap: 4, padding: 4, borderRadius: "var(--radius-lg)", boxShadow: "var(--shadow-md)" }}>
          {MODES.map((m) => (
            <button
              key={m.id}
              className="chip"
              data-active={mode === m.id}
              onClick={() => setMode(m.id)}
              style={{ display: "flex", alignItems: "center", gap: 5 }}
            >
              <m.icon {...icon.sm} aria-hidden />
              {m.label}
            </button>
          ))}
        </div>

        {mode === "forecast" && (
          <div className="glass" style={{ display: "flex", gap: 4, padding: 4, borderRadius: "var(--radius-lg)", boxShadow: "var(--shadow-md)" }}>
            {([24, 48, 72] as Horizon[]).map((h) => (
              <button key={h} className="chip" data-active={horizon === h} onClick={() => setHorizon(h)}>
                +{h}h
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Bottom sheet — legend/hint by default, or whatever you tapped */}
      <div
        className="glass"
        style={{
          position: "absolute", left: 0, right: 0, bottom: 0, zIndex: "var(--z-panel)",
          borderTop: "1px solid var(--border-subtle)",
          borderTopLeftRadius: "var(--radius-lg)", borderTopRightRadius: "var(--radius-lg)",
          padding: "var(--space-md)", maxHeight: "42vh", overflowY: "auto",
          boxShadow: "var(--shadow-lg)",
        }}
      >
        {selectedWard ? (
          <div style={{ display: "flex", alignItems: "center", gap: "var(--space-md)" }}>
            <div style={{
              width: 50, height: 50, borderRadius: "var(--radius-md)", flexShrink: 0,
              background: wardCat?.color ?? "var(--bg-tertiary)", color: wardCat?.textColor,
              display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
            }}>
              <div className="mono" style={{ fontSize: "1.05rem", fontWeight: 600, lineHeight: 1 }}>{wardAqi ?? "—"}</div>
              <div style={{ fontSize: "0.5rem", opacity: 0.8, marginTop: 2 }}>AQI</div>
            </div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontWeight: 600, fontSize: "0.95rem" }} className="truncate">{selectedWard.wardName}</div>
              <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
                {wardCat?.label ?? "No data"} {mode === "forecast" ? `· in ${horizon}h` : "· now"}
              </div>
            </div>
            <Link href={`/citizen/${selectedWard.wardId}`} className="btn btn-primary btn-sm" style={{ textDecoration: "none", flexShrink: 0 }}>
              View ward
              <ArrowRight {...icon.sm} aria-hidden />
            </Link>
          </div>
        ) : selectedAction ? (
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 8, marginBottom: 8 }}>
              <div style={{ minWidth: 0 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4, flexWrap: "wrap" }}>
                  <span style={{ fontWeight: 600, fontSize: "0.95rem" }}>{SOURCE_LABELS[selectedAction.source]}</span>
                  <span className={`badge ${ACTION_STATUS_BADGE[selectedAction.status]}`}>
                    {ACTION_STATUS_LABELS[selectedAction.status]}
                  </span>
                </div>
                <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>{selectedAction.ward_name}</div>
              </div>
              <button
                className="btn btn-ghost btn-icon btn-sm"
                onClick={() => setSelectedActionId(null)}
                aria-label="Close case details"
              >
                <X {...icon.sm} aria-hidden />
              </button>
            </div>

            <div style={{ fontSize: "0.8rem", color: "var(--text-tertiary)", marginBottom: 10 }}>
              {confidenceLabel(selectedAction.confidence)}
            </div>

            {selectedMemo?.legal_basis?.length ? (
              <ul style={{ listStyle: "none", padding: 0, margin: "0 0 12px", display: "flex", flexDirection: "column", gap: 4 }}>
                {selectedMemo.legal_basis.map((l) => (
                  <li key={l.id} style={{ fontSize: "0.8rem", color: "var(--text-secondary)", lineHeight: 1.5 }}>
                    <strong style={{ color: "var(--text-primary)", fontWeight: 560 }}>{l.statute}</strong>
                    {l.provision ? `, ${l.provision}` : ""} — {l.summary}
                  </li>
                ))}
              </ul>
            ) : (
              <p style={{ fontSize: "0.8rem", color: "var(--text-tertiary)", marginBottom: 12 }}>
                No statutory rule matched this case yet.
              </p>
            )}

            <Link href={`/citizen/${selectedAction.ward_id}`} className="btn btn-primary btn-sm" style={{ textDecoration: "none" }}>
              View ward
              <ArrowRight {...icon.sm} aria-hidden />
            </Link>
          </div>
        ) : casesEmpty ? (
          <div style={{ display: "flex", alignItems: "center", gap: 10, color: "var(--text-tertiary)" }}>
            <Target {...icon.md} aria-hidden />
            <p style={{ fontSize: "0.85rem" }}>No enforcement cases open in {cityLabel} right now.</p>
          </div>
        ) : (
          <div>
            {mode === "cases" && (
              <div style={{ display: "flex", gap: "var(--space-md)", flexWrap: "wrap", marginBottom: 8 }}>
                {(Object.keys(ACTION_STATUS_LABELS) as ActionStatus[]).map((s) => (
                  <span key={s} style={{ display: "inline-flex", alignItems: "center", gap: 5, fontSize: "0.72rem", color: "var(--text-tertiary)" }}>
                    <span aria-hidden style={{ width: 8, height: 8, borderRadius: "50%", background: ACTION_STATUS_HEX[s], flexShrink: 0 }} />
                    {ACTION_STATUS_LABELS[s]}
                  </span>
                ))}
              </div>
            )}
            <p style={{ fontSize: "0.8rem", color: "var(--text-tertiary)" }}>
              {mode === "cases" ? "Tap a pin for case details." : "Tap any area on the map to see its ward."}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
