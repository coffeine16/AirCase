"use client";
/**
 * Admin Console — Main Dashboard Page
 * Full-bleed map (left ~70%) + collapsible Action Queue panel (right ~30%)
 */
import dynamic from "next/dynamic";
import { useState, useCallback, useMemo, useEffect, useRef } from "react";
import useSWR, { mutate } from "swr";

import { api } from "@/lib/api";
import { useCity } from "@/lib/CityContext";
import { useFilters }  from "@/hooks/useFilters";
import { filterHotspots } from "@/hooks/useHotspots";
import { useAgentRun } from "@/hooks/useAgentRun";
import { useToast } from "@/components/Toast";
import { AGENT_LABELS } from "@/lib/constants";
import { useIsMobile, useIsCompact } from "@/hooks/useMediaQuery";
import { icon, Zap, LoaderCircle, SlidersHorizontal, X, Hexagon, ChevronLeft, ChevronRight, ChevronDown, ChevronUp } from "@/components/Icon";

// All WebGL/heavy components are dynamically imported — no SSR
const MapContainer    = dynamic(() => import("@/components/map/MapContainer"),                { ssr: false, loading: () => <MapPlaceholder /> });
const LayerToggle     = dynamic(() => import("@/components/map/controls/LayerToggle"),        { ssr: false });
const TimeSlider      = dynamic(() => import("@/components/map/controls/TimeSlider"),         { ssr: false });
const FilterBar       = dynamic(() => import("@/components/filters/FilterBar"),               { ssr: false });
const ActionQueue     = dynamic(() => import("@/components/panels/ActionQueue"),              { ssr: false });
const AgentPipelinePanel = dynamic(() => import("@/components/agents/AgentPipelinePanel"),   { ssr: false });

function MapPlaceholder() {
  return (
    <div style={{
      width: "100%", height: "100%",
      background: "var(--bg-secondary)",
      display: "flex", alignItems: "center", justifyContent: "center",
      color: "var(--text-tertiary)", fontSize: "0.85rem", gap: 8,
    }}>
      <Hexagon {...icon.md} className="animate-breathe" aria-hidden />
      Loading map…
    </div>
  );
}

export default function AdminPage() {
  const isMobile = useIsMobile();
  // LAYOUT keys off compact (≤1024), not phone (≤768): a tablet cannot carry a
  // full-bleed map AND a 372px queue AND a floating control rail side by side.
  // CONTENT still keys off isMobile where a phone genuinely needs different
  // copy or sizing. See hooks/useMediaQuery.ts.
  const isCompact = useIsCompact();
  // `null` = the user has not chosen yet, so fall back to the layout default:
  // open on desktop (it is a side panel, it costs the map nothing) and closed
  // on compact (it is an overlay — landing on a map with 72% of it covered,
  // before you have looked at the map, is backwards). Resolved at RENDER time
  // rather than by an effect, because isCompact only becomes true after mount
  // and syncing it with setState would cascade a re-render on every load.
  const [panelPref, setPanelPref] = useState<boolean | null>(null);
  const [hourOffset, setHourOffset] = useState(0);
  const [selectedCell, setSelectedCell] = useState<string | null>(null);
  const [mobileControls, setMobileControls] = useState(false);
  const [pipelineOpen, setPipelineOpen] = useState(false);

  // ── Filters & layers ────────────────────────────────────────────────────────
  const {
    filters, layers,
    setWards, setSources, setPersistence, toggleLayer, setHorizon, resetFilters,
  } = useFilters();

  // ── Map data — CITY-SCOPED. Every contract keys on `city`, so switching city
  //    in the header refetches everything and the map moves with it. ────────────
  const { city } = useCity();
  const { data: rawHotspots, isLoading: hotspotsLoading } = useSWR([city, "hotspots"], () => api.cityHotspots(city));
  const { data: fusionResp }   = useSWR([city, "fusion"],    () => api.cityFusion(city));
  const { data: wardsResp }    = useSWR([city, "wards"],     () => api.cityWards(city));
  const { data: stations = [] }= useSWR([city, "stations"],  () => api.cityStations(city));
  const { data: fires = [] }   = useSWR([city, "fires"],     () => api.cityFires(city));
  const { data: audit }        = useSWR([city, "audit"],     () => api.cityAudit(city));
  const { data: satellite = [] }= useSWR([city, "satellite"],() => api.citySatellite(city));
  const { data: dispatchRoutes = [] } = useSWR([city, "dispatch"], () => api.cityDispatch(city));

  const allHotspots = useMemo(() => rawHotspots ?? [], [rawHotspots]);
  const hotspots = useMemo(() => filterHotspots(allHotspots, filters), [allHotspots, filters]);
  const wardCells = wardsResp?.cells ?? [];
  const blindSpots = audit?.blind_spots ?? [];

  // Forecast horizon: 0 = the live fusion field, else a 3-hourly lead (3..72) from
  // the forecast agent. The choropleth switches between them, so the time control
  // actually drives the map (it used to be a dead control).
  //
  // 3-hourly matters here: at 24-hour steps every sample lands on the SAME hour of
  // day, so scrubbing showed four near-identical maps and the diurnal swing — the
  // largest thing moving — was invisible between them.
  const { data: horizonForecast } = useSWR(
    hourOffset > 0 ? [city, "forecast", hourOffset] : null,
    () => api.cityForecast(city, hourOffset)
  );
  const wardOf = useMemo(() => {
    const m = new Map<string, string>();
    for (const c of wardCells) m.set(c.cell, c.ward_id);
    return m;
  }, [wardCells]);
  const fusionCells = useMemo(() => {
    if (hourOffset === 0) return fusionResp?.cells ?? [];
    return (horizonForecast ?? []).map((f) => ({
      cell: f.cell,
      ward_id: wardOf.get(f.cell) ?? "unassigned",
      pm25: f.pm25_hat,
    }));
  }, [hourOffset, fusionResp, horizonForecast, wardOf]);

  // ── Agent pipeline ──────────────────────────────────────────────────────────
  const onAgentComplete = useCallback(() => {
    // Re-read this city's contracts after a pipeline run.
    mutate([city, "hotspots"]);
    mutate([city, "fusion"]);
    mutate([city, "dispatch"]);
    mutate([city, "audit"]);
  }, [city]);
  const { agents, running, runAgent, resetAgents } = useAgentRun(onAgentComplete);

  // A pipeline run is slow, started deliberately, and finishes while you are
  // usually looking at the map with the drawer shut. `useAgentRun` has always
  // exposed the outcome; until now nothing rendered it, so a failed run was
  // silent. Report it once per run, at the moment it settles.
  const toast = useToast();
  const wasRunning = useRef(false);
  useEffect(() => {
    if (wasRunning.current && !running) {
      const failed = agents.filter((a) => a.status === "error");
      const done = agents.filter((a) => a.status === "done");
      if (failed.length) {
        toast({
          tone: "critical",
          title: `${failed.length} agent${failed.length > 1 ? "s" : ""} failed`,
          body: `${failed.map((a) => AGENT_LABELS[a.name]).join(", ")}. The map still shows the last good batch output.`,
        });
      } else if (done.length) {
        toast({
          tone: "positive",
          title: `Pipeline complete · ${done.length} agent${done.length > 1 ? "s" : ""}`,
          body: "Hotspots, fusion, dispatch and audit have been refreshed.",
        });
      }
    }
    wasRunning.current = running;
  }, [running, agents, toast]);

  const panelOpen = panelPref ?? !isCompact;
  const togglePanel = () => setPanelPref(!panelOpen);

  // On compact the sheet and the floating map controls fight for the same bottom
  // strip. The layer/filter stack steps aside when the sheet is open — it is a
  // tall panel and would stack three deep.
  const showMapControls = !(isCompact && panelOpen);

  // The forecast slider is the EXCEPTION: it drives what the map is showing, so
  // hiding it behind "collapse the queue first" made the map's own time control
  // reachable only by a scroll-and-collapse nobody discovers. It stays mounted
  // and rides above the sheet instead — sheet height when open, handle height
  // when collapsed.
  const sheetH = isCompact ? (panelOpen ? (isMobile ? "72vh" : "62vh") : "44px") : "0px";

  return (
    <div style={{ display: "flex", height: "100%", overflow: "hidden" }}>

      {/* ── Map area ─────────────────────────────────────────────────────────── */}
      <div style={{ flex: 1, position: "relative", overflow: "hidden", minWidth: 0 }}>

        {/* Top-left control stack: Pipeline trigger + layer/filter controls. The
            agent pipeline lives in a side drawer (AgentPipelinePanel), not smeared
            across the top of the map. */}
        <div style={{
          position: "absolute", top: 12, left: 12, zIndex: "var(--z-panel)",
          display: "flex", flexDirection: "column", gap: "var(--space-sm)", alignItems: "flex-start",
        }}>
          <button
            className="btn btn-primary btn-sm"
            onClick={() => setPipelineOpen(true)}
            style={{ boxShadow: "var(--shadow-md)" }}
          >
            {running ? (
              <>
                <LoaderCircle {...icon.sm} className="animate-spin" aria-hidden />
                Pipeline {agents.filter((a) => a.status === "done").length}/9
              </>
            ) : (
              <>
                <Zap {...icon.sm} aria-hidden />
                Agent pipeline
              </>
            )}
          </button>
          {isCompact && (
            <button
              className="btn btn-ghost btn-sm glass"
              onClick={() => setMobileControls((s) => !s)}
              aria-expanded={mobileControls}
            >
              {mobileControls
                ? <><X {...icon.sm} aria-hidden /> Close</>
                : <><SlidersHorizontal {...icon.sm} aria-hidden /> Layers &amp; filters</>}
            </button>
          )}
        </div>

        {/* Layer/filter controls. Desktop: always-visible left rail below the
            Pipeline button. Compact: toggled open over the map. */}
        {(!isCompact || mobileControls) && (
          <div style={{
            position: "absolute",
            top: isCompact ? 92 : 56, left: 12,
            zIndex: "var(--z-overlay)",
            display: "flex", flexDirection: "column",
            gap: "var(--space-sm)",
            maxWidth: isCompact ? "min(280px, calc(100vw - 24px))" : 188,
            maxHeight: isCompact ? "60vh" : undefined,
            overflowY: isCompact ? "auto" : undefined,
          }}>
            <LayerToggle layers={layers} onToggle={toggleLayer} />
            <FilterBar
              filters={filters}
              onWards={setWards}
              onSources={setSources}
              onPersistence={setPersistence}
              onReset={resetFilters}
            />
          </div>
        )}

        {/* Forecast horizon — always present on compact, sitting directly above
            the sheet whatever height the sheet currently has. `fixed`, not
            `absolute`, because the sheet is fixed to the viewport: positioning
            this against the map wrapper instead would put it underneath an
            opaque panel and make it untappable, which is what used to happen. */}
        <div style={{
          position: isCompact ? "fixed" : "absolute",
          bottom: isCompact ? `calc(${sheetH} + 12px)` : 24,
          left: "50%",
          transform: "translateX(-50%)",
          zIndex: "var(--z-panel)",
          width: isCompact ? "calc(100% - 24px)" : "min(480px, 80%)",
          transition: "bottom var(--transition-slow)",
          pointerEvents: "auto",
        }}>
          <TimeSlider value={hourOffset} onChange={setHourOffset} />
        </div>

        {/* The full-bleed map */}
        <MapContainer
          layers={layers}
          filters={filters}
          fusionCells={fusionCells}
          hotspots={hotspots}
          stations={stations}
          fires={fires}
          wardCells={wardCells}
          blindSpots={blindSpots}
          satellite={satellite}
          dispatchRoutes={dispatchRoutes}
          hourOffset={hourOffset}
          selectedCell={selectedCell}
          onCellClick={setSelectedCell}
          recenterKey={city}
          showOverlays={showMapControls}
        />
      </div>

      {/* ── Action Queue — right panel (desktop) / bottom sheet (compact) ────── */}
      {isCompact ? (
        // SOLID, not glass — a content panel over a busy map must not let the
        // map (or the legend) bleed through its text.
        <div style={{
          position: "fixed", left: 0, right: 0, bottom: 0,
          // Single source of truth: the slider positions itself against this.
          height: sheetH,
          transition: "height var(--transition-slow)",
          background: "var(--bg-primary)",
          borderTop: "1px solid var(--border-default)",
          borderTopLeftRadius: "var(--radius-lg)", borderTopRightRadius: "var(--radius-lg)",
          zIndex: "var(--z-panel)",
          display: "flex", flexDirection: "column",
          boxShadow: "var(--shadow-lg)",
        }}>
          {/* Drag handle / toggle */}
          <button
            onClick={togglePanel}
            aria-expanded={panelOpen}
            style={{
              height: 44, flexShrink: 0, border: "none", background: "transparent",
              display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
              gap: 4, cursor: "pointer", color: "var(--text-secondary)",
            }}
          >
            <span aria-hidden style={{ width: 32, height: 3, borderRadius: 2, background: "var(--border-strong)" }} />
            <span style={{ display: "inline-flex", alignItems: "center", gap: 4, fontSize: "0.72rem", fontWeight: 550 }}>
              {panelOpen ? (
                <><ChevronDown {...icon.sm} aria-hidden /> Hide</>
              ) : (
                <><ChevronUp {...icon.sm} aria-hidden /> Action queue</>
              )}
            </span>
          </button>
          {panelOpen && (
            <div style={{ flex: 1, overflow: "hidden" }}>
              <ActionQueue
                hotspots={hotspots}
                loading={hotspotsLoading}
                selectedCell={selectedCell}
                onSelectCell={setSelectedCell}
              />
            </div>
          )}
        </div>
      ) : (
        <div style={{
          width: panelOpen ? 372 : 0,
          minWidth: panelOpen ? 330 : 0,
          maxWidth: 430,
          overflow: "hidden",
          transition: "width var(--transition-slow), min-width var(--transition-slow)",
          background: "var(--bg-primary)",
          display: "flex", flexDirection: "column",
          borderLeft: "1px solid var(--border-subtle)",
          zIndex: "var(--z-panel)",
          position: "relative",
        }}>
          <button
            onClick={togglePanel}
            className="btn btn-ghost btn-icon"
            style={{
              position: "absolute", left: -25, top: "50%",
              transform: "translateY(-50%)",
              width: 25, height: 48, minHeight: 0, padding: 0,
              borderRadius: "var(--radius-md) 0 0 var(--radius-md)",
              background: "var(--bg-secondary)",
              borderColor: "var(--border-subtle)",
              borderRight: "none",
              zIndex: 1,
            }}
            title={panelOpen ? "Collapse panel" : "Expand panel"}
            aria-expanded={panelOpen}
            aria-label={panelOpen ? "Collapse action queue" : "Expand action queue"}
          >
            {panelOpen
              ? <ChevronRight {...icon.sm} aria-hidden />
              : <ChevronLeft {...icon.sm} aria-hidden />}
          </button>

          {panelOpen && (
            <ActionQueue
              hotspots={hotspots}
              loading={hotspotsLoading}
              selectedCell={selectedCell}
              onSelectCell={setSelectedCell}
            />
          )}
        </div>
      )}

      {/* Agent pipeline — a side drawer, opened from the Pipeline button. */}
      <AgentPipelinePanel
        open={pipelineOpen}
        onClose={() => setPipelineOpen(false)}
        agents={agents}
        running={running}
        onRun={runAgent}
        onReset={resetAgents}
      />
    </div>
  );
}
