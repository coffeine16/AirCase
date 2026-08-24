"use client";
/**
 * MapContainer — Deck.gl + MapLibre GL base map.
 * Composes all toggleable layers from layer builder functions.
 * Layers are rebuilt only when their data or visibility changes.
 */
import { useState, useCallback, useMemo, useEffect, useRef } from "react";
import Map from "react-map-gl/maplibre";
import DeckGL from "@deck.gl/react";
import { H3HexagonLayer } from "@deck.gl/geo-layers";
import { cellToLatLng } from "h3-js";
import "maplibre-gl/dist/maplibre-gl.css";

import { initialViewFor, MAP_STYLE } from "@/lib/constants";
import { pm25ToRgbaArray, SEVERITY_COLORS, hexToRgba, UNKNOWN_HEX, NO_FORECAST_RGBA } from "@/lib/colors";
import { OVERLAY_PARAMETERS } from "./layers/overlay";
import type { FusionCell, Hotspot, LayerVisibility, MapFilters, DispatchRoute, BlindSpot } from "@/lib/types";
import type { Station, FireDetection } from "@/hooks/useMapData";

import { buildStationLayer } from "./layers/StationLayer";
import { buildFireLayer }    from "./layers/FireLayer";
import { buildWardLayer }    from "./layers/WardLayer";
import { buildBlindSpotLayer } from "./layers/BlindSpotLayer";
import { buildDispatchLayers } from "./layers/DispatchLayer";
import { buildSatelliteLayer, type SatelliteCell } from "./layers/SatelliteLayer";
import LegendBar from "./controls/LegendBar";
import { useIsMobile } from "@/hooks/useMediaQuery";
import { icon, Crosshair, Flame } from "@/components/Icon";

// ── Tooltip state ─────────────────────────────────────────────────────────────

interface TooltipState {
  x: number;
  y: number;
  content: React.ReactNode;
}

// ── Props ─────────────────────────────────────────────────────────────────────

interface Props {
  layers: LayerVisibility;
  filters: MapFilters;
  fusionCells: FusionCell[];
  hotspots: Hotspot[];
  stations?: Station[];
  fires?: FireDetection[];
  wardCells?: { cell: string; ward_id: string; ward_name: string }[];
  blindSpots?: BlindSpot[];
  satellite?: SatelliteCell[];
  dispatchRoutes?: DispatchRoute[];
  hourOffset: number;
  selectedCell: string | null;
  onCellClick: (cell: string | null) => void;
  /** When this changes (e.g. the city), the map re-centres on the new data. */
  recenterKey?: string;
  /** Fly to one ward (from the FilterBar search), e.g. { wardId: "W042", token: 3 }.
   *  `token` is a bump-per-search counter, not just the wardId — searching the
   *  SAME ward twice in a row is still two "go there" commands and must still
   *  fly, even though wardId alone wouldn't change between them. */
  focusWard?: { wardId: string; token: number } | null;
  /** False while a compact-layout sheet covers the bottom of the map — the
   *  legend and recentre button hide rather than sitting under an opaque
   *  panel where they are visible-but-untappable. */
  showOverlays?: boolean;
}

export default function MapContainer({
  layers,
  filters,
  fusionCells,
  hotspots,
  stations = [],
  fires = [],
  wardCells = [],
  blindSpots = [],
  satellite = [],
  dispatchRoutes = [],
  selectedCell,
  onCellClick,
  recenterKey,
  focusWard,
  showOverlays = true,
}: Props) {
  // Open on THIS city, not on Delhi. The recentre effect below still fits to the
  // loaded data; this makes the first frame land in the right hemisphere so the
  // map is never blank while the data is on its way.
  const [viewState, setViewState] = useState<{
    longitude: number; latitude: number; zoom: number; pitch: number; bearing: number;
  }>(() => initialViewFor(recenterKey ?? "delhi"));

  /** 2D / 3D. Off by default because the console is a working surface first and
   *  a flat field is easier to read a queue against — NOT, any longer, because
   *  3D hid the evidence. It did: the extruded columns occluded stations, fires,
   *  hotspot zones, blind spots and dispatch routes, since those are drawn flat
   *  at ground level and lose the depth test against a 2,500-unit column. Every
   *  overlay now carries OVERLAY_PARAMETERS (see ./layers/overlay.ts), so 3D is
   *  a safe view rather than one that quietly costs you the layers you came for. */
  const [extruded, setExtruded] = useState(false);

  // Theme-aware base map: dark-matter on dark, positron (light) on light. Follows
  // the same data-theme attribute the ThemeToggle stamps on <html>.
  const [mapStyle, setMapStyle] = useState(MAP_STYLE);
  useEffect(() => {
    const apply = () =>
      setMapStyle(
        document.documentElement.getAttribute("data-theme") === "light"
          ? "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
          : MAP_STYLE
      );
    apply();
    const obs = new MutationObserver(apply);
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => obs.disconnect();
  }, []);
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
  const isMobile = useIsMobile();

  // Auto-center on whatever city the loaded data belongs to, and RE-center whenever
  // recenterKey (the city) changes — otherwise switching Delhi -> Chennai leaves the
  // viewport 1700 km away and the map looks empty until you pan there by hand.
  /** Centroid of whatever this city's data covers, or null if nothing loaded. */
  const dataCenter = useCallback((): { latitude: number; longitude: number } | null => {
    const cells =
      (hotspots.length && hotspots.map((h) => h.cell)) ||
      (wardCells.length && wardCells.map((w) => w.cell)) ||
      (fusionCells.length && fusionCells.map((f) => f.cell)) ||
      [];
    if (!cells.length) return null;
    let sumLat = 0, sumLon = 0, n = 0;
    for (const c of cells) {
      try {
        const [lat, lon] = cellToLatLng(c);
        sumLat += lat; sumLon += lon; n++;
      } catch { /* skip a malformed cell id */ }
    }
    return n ? { latitude: sumLat / n, longitude: sumLon / n } : null;
  }, [hotspots, wardCells, fusionCells]);

  /** Snap back to the city — the Google-Maps-style recentre button. */
  const recenter = useCallback(() => {
    const c = dataCenter();
    if (c) setViewState((vs) => ({ ...vs, ...c, zoom: 10.5, pitch: 30, bearing: 0 }));
  }, [dataCenter]);

  const fittedKey = useRef<string | null>(null);
  useEffect(() => {
    const key = recenterKey ?? "default";
    if (fittedKey.current === key) return;   // already fitted to this city's data
    const c = dataCenter();
    // No data yet: leave the viewport where it is and do NOT mark this city
    // fitted, so the fit runs as soon as the hexagons arrive. We can afford to
    // wait now because the map already OPENS on the right city (see useState
    // above) — previously it opened on a hardcoded Delhi, which is what left
    // Chennai and Bengaluru looking blank on a cold load.
    if (!c) return;
    fittedKey.current = key;
    setViewState((vs) => ({ ...vs, ...c, zoom: 10.5 }));
  }, [recenterKey, dataCenter]);

  // Ward search: fly to the searched ward's own cells, tighter than the
  // whole-city recentre above (12.5 vs 10.5 — matches the citizen map's own
  // ward-zoom level). Independent of dataCenter/fittedKey: a search result
  // must always fly, even to a ward with zero hotspots in it right now.
  const lastFocusToken = useRef<number | null>(null);
  useEffect(() => {
    if (!focusWard || lastFocusToken.current === focusWard.token) return;
    lastFocusToken.current = focusWard.token;
    const cells = wardCells.filter((w) => w.ward_id === focusWard.wardId).map((w) => w.cell);
    if (!cells.length) return;
    let sLat = 0, sLon = 0, n = 0;
    for (const c of cells) {
      try { const [lat, lon] = cellToLatLng(c); sLat += lat; sLon += lon; n++; } catch { /* skip malformed cell */ }
    }
    if (!n) return;
    setViewState((vs) => ({ ...vs, latitude: sLat / n, longitude: sLon / n, zoom: 12.5, pitch: 30, bearing: 0 }));
  }, [focusWard, wardCells]);

  const setTip = useCallback(
    (info: { x: number; y: number; content: React.ReactNode } | null) => setTooltip(info),
    []
  );

  // Flipping to 2D snaps back to top-down: "2D" has to mean flat, even if the
  // analyst had rotated the 3D view first. Watches `extruded` after mount rather
  // than only reading it at init, so it tracks the toggle rather than the
  // initial value. Same shape as CitizenMap's own re-tilt effect.
  const wasExtruded = useRef(extruded);
  useEffect(() => {
    if (wasExtruded.current === extruded) return;
    wasExtruded.current = extruded;
    setViewState((vs) => ({ ...vs, pitch: extruded ? 45 : 0, bearing: extruded ? -12 : 0 }));
  }, [extruded]);

  // ── Fusion PM2.5 choropleth ─────────────────────────────────────────────────
  const fusionLayer = useMemo(() => {
    if (!layers.fusion || !fusionCells.length) return null;
    return new H3HexagonLayer<FusionCell>({
      id: "fusion-choropleth",
      data: fusionCells,
      getHexagon: (d) => d.cell,
      // A null pm25 is "no value here at this time", not zero. Colouring it on
      // the AQI ramp would paint an absence as clean air, which is the most
      // misleading thing this map could do; dropping the cell instead makes the
      // choropleth silently shrink. So it gets its own near-transparent grey,
      // legible as a gap rather than as a reading.
      getFillColor: (d) => (d.pm25 == null ? NO_FORECAST_RGBA : pm25ToRgbaArray(d.pm25, 175)),
      // Height IS the PM2.5 value, so a column reads directly as "how much worse
      // than clean air" rather than an arbitrary scale. elevationScale 12 matches
      // CitizenMap so the same city looks the same height on both surfaces.
      extruded,
      // A no-forecast cell is FLAT, not a zero-height column of "clean air".
      // Math.max(null, 0) would quietly coerce to 0 and typecheck only because
      // null is falsy — the explicit null branch says the absence is deliberate.
      getElevation: extruded ? (d: FusionCell) => (d.pm25 == null ? 0 : Math.max(d.pm25, 0)) : 0,
      elevationScale: 12,
      wireframe: false,
      pickable: true,
      autoHighlight: true,
      highlightColor: [255, 255, 255, 40],
      onHover: (info) => {
        if (info.object) {
          const d = info.object as FusionCell;
          setTip({
            x: info.x, y: info.y,
            content: (
              <div>
                <div style={{ fontWeight: 600, marginBottom: 4 }}>
                  {d.pm25 == null ? "No forecast" : "Fusion PM2.5"}
                </div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: d.pm25 == null ? "0.8rem" : "1.1rem" }}>
                  {d.pm25 == null
                    ? "upwind of every monitor at this hour"
                    : `${d.pm25.toFixed(1)} µg/m³`}
                </div>
                <div style={{ color: "var(--text-tertiary)", fontSize: "0.75rem", marginTop: 4 }}>
                  {d.cell} · {d.ward_id}
                </div>
              </div>
            ),
          });
        } else setTip(null);
      },
      onClick: (info) => info.object && onCellClick((info.object as FusionCell).cell),
      updateTriggers: { getFillColor: [fusionCells], getElevation: [extruded] },
    });
    // `extruded` MUST be in these deps: without it the memo returns the cached
    // 2D layer and the toggle only tilts the camera over a flat field, which
    // reads as "3D is broken" rather than "3D is off".
  }, [layers.fusion, fusionCells, setTip, onCellClick, extruded]);

  // ── Hotspot zones ───────────────────────────────────────────────────────────
  const hotspotLayer = useMemo(() => {
    if (!layers.hotspots || !hotspots.length) return null;
    return new H3HexagonLayer<Hotspot>({
      id: "hotspot-zones",
      // Flat, at ground level, and therefore BEHIND the extruded fusion columns
      // as far as the depth buffer is concerned. Without this the hotspot zones
      // — the whole point of the console — disappear the moment 3D is switched
      // on. See ./layers/overlay.ts.
      parameters: OVERLAY_PARAMETERS,
      data: hotspots,
      getHexagon: (d) => d.cell,
      getFillColor: (d) => SEVERITY_COLORS[d.kind]?.fill ?? hexToRgba(UNKNOWN_HEX, 100),
      getLineColor: (d) => SEVERITY_COLORS[d.kind]?.border ?? hexToRgba(UNKNOWN_HEX, 200),
      getLineWidth: (d) => (d.kind === "chronic" ? 3 : d.kind === "emerging" ? 2 : 1.5),
      lineWidthMinPixels: 1.5,
      extruded: false,
      wireframe: true,
      pickable: true,
      autoHighlight: true,
      highlightColor: [255, 255, 255, 60],
      onHover: (info) => {
        if (info.object) {
          const d = info.object as Hotspot;
          setTip({
            x: info.x, y: info.y,
            content: (
              <div style={{ minWidth: 220 }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                  <span style={{ fontWeight: 600 }}>{d.zone_id} · {d.ward_name}</span>
                  <span className={`badge badge-${d.kind}`}>{d.kind}</span>
                </div>
                <div style={{ fontFamily: "var(--font-mono)" }}>PM2.5: {d.pm25_med.toFixed(1)} µg/m³</div>
                <div style={{ color: "var(--text-tertiary)", fontSize: "0.75rem", marginTop: 4 }}>
                  Severity: {(d.severity * 100).toFixed(0)}% ·{" "}
                  {d.attributable ? "Enforceable" : "Diffuse"}
                  {d.fires_6h > 0 && (
                    <span style={{ display: "inline-flex", alignItems: "center", gap: 3, marginLeft: 4, color: "var(--persist-acute)" }}>
                      <Flame {...icon.sm} aria-hidden /> {d.fires_6h}
                    </span>
                  )}
                </div>
                <div style={{ color: "var(--text-tertiary)", fontSize: "0.7rem", marginTop: 4, maxWidth: 240 }}>
                  {d.detection_basis}
                </div>
              </div>
            ),
          });
        } else setTip(null);
      },
      onClick: (info) => info.object && onCellClick((info.object as Hotspot).cell),
      updateTriggers: { getFillColor: [hotspots], getLineColor: [hotspots] },
    });
  }, [layers.hotspots, hotspots, setTip, onCellClick]);

  // ── Secondary layers (one memoized block; rebuilt when visibility or data change)
  const stationLayer  = useMemo(() => layers.stations ? buildStationLayer(stations, setTip, onCellClick) : null,  [layers.stations, stations, setTip, onCellClick]);
  const fireLayer     = useMemo(() => layers.fires     ? buildFireLayer(fires, setTip)               : null,  [layers.fires, fires, setTip]);
  const wardLayer     = useMemo(() => layers.wards     ? buildWardLayer(wardCells, setTip)           : null,  [layers.wards, wardCells, setTip]);
  const blindLayer    = useMemo(() => layers.blindspots? buildBlindSpotLayer(blindSpots, setTip)     : null,  [layers.blindspots, blindSpots, setTip]);
  const satelliteLayer= useMemo(() => layers.satellite ? buildSatelliteLayer(satellite, setTip)      : null,  [layers.satellite, satellite, setTip]);
  const dispatchLayers= useMemo(() => layers.dispatch  ? buildDispatchLayers(dispatchRoutes, setTip) : [],   [layers.dispatch, dispatchRoutes, setTip]);

  // ── Selected cell highlight ──────────────────────────────────────────────────
  const selectedLayer = useMemo(() => {
    if (!selectedCell) return null;
    return new H3HexagonLayer({
      id: "selected-cell",
      parameters: OVERLAY_PARAMETERS,   // the selection ring must never be occluded
      data: [{ cell: selectedCell }],
      getHexagon: (d) => (d as { cell: string }).cell,
      getFillColor: [255, 255, 255, 30],
      getLineColor: [255, 255, 255, 230],
      lineWidthMinPixels: 2.5,
      extruded: false,
      wireframe: true,
      pickable: false,
    });
  }, [selectedCell]);

  // ── Compose all active layers (order matters: bottom → top) ─────────────────
  const deckLayers = useMemo(
    () => [
      fusionLayer,
      satelliteLayer,
      wardLayer,
      hotspotLayer,
      stationLayer,
      fireLayer,
      blindLayer,
      ...dispatchLayers,
      selectedLayer,
    ].filter(Boolean),
    [fusionLayer, satelliteLayer, wardLayer, hotspotLayer, stationLayer, fireLayer, blindLayer, dispatchLayers, selectedLayer]
  );

  return (
    <div className="map-container" style={{ background: "var(--bg-base)" }}>
      <DeckGL
        viewState={viewState}
        onViewStateChange={({ viewState: vs }) => {
          const v = vs as { longitude: number; latitude: number; zoom: number; pitch: number; bearing: number };
          setViewState({ longitude: v.longitude, latitude: v.latitude, zoom: v.zoom, pitch: v.pitch, bearing: v.bearing });
        }}
        controller={true}
        layers={deckLayers}
        getCursor={({ isDragging }) => (isDragging ? "grabbing" : "crosshair")}
        onClick={(info) => {
          if (!info.object) onCellClick(null);
        }}
      >
        <Map
          mapStyle={mapStyle}
          attributionControl={{ compact: true }}
        />
      </DeckGL>

      {/* Recentre — snap back to the city after panning away (Google-Maps style).
          Sits above the legend so the two never collide. */}
      {showOverlays && (
        <button
          onClick={recenter}
          title="Recentre on the city"
          aria-label="Recentre map on the city"
          className="map-btn"
          style={{
            position: "absolute",
            right: 12,
            bottom: isMobile ? 172 : 232,
            zIndex: "var(--z-overlay)",
            width: isMobile ? 40 : 32,
            height: isMobile ? 40 : 32,
          }}
        >
          <Crosshair {...icon.md} aria-hidden />
        </button>
      )}

      {/* 2D / 3D — map TOP-right, matching the citizen explore map so the same
          control lives in the same corner on both surfaces.
          Not bottom-right: that corner is already a stack of the legend (274px
          tall) and the recentre button, and measured, a toggle there overlapped
          the legend by 82x30px. Only rendered while the fusion layer is on — it
          is the only extruded layer, so with it off the control would visibly do
          nothing. */}
      {showOverlays && layers.fusion && (
        <div
          className="glass"
          style={{
            position: "absolute",
            right: 12,
            top: 12,
            zIndex: "var(--z-overlay)",
            display: "flex",
            gap: 3,
            padding: 3,
            borderRadius: "var(--radius-lg)",
            boxShadow: "var(--shadow-md)",
          }}
        >
          <button
            className="chip"
            data-active={!extruded}
            aria-pressed={!extruded}
            onClick={() => setExtruded(false)}
            title="Flat view"
          >
            2D
          </button>
          <button
            className="chip"
            data-active={extruded}
            aria-pressed={extruded}
            onClick={() => setExtruded(true)}
            title="Extrude the PM2.5 field by its value"
          >
            3D
          </button>
        </div>
      )}

      {/* Context-sensitive legend */}
      {showOverlays && (
        <LegendBar
          layers={layers}
          /* Driven by the DATA, not by the slider position: the row appears
             exactly when the map actually contains an unvalued cell, so the key
             can never advertise a swatch that is not on screen. */
          hasNoForecastCells={fusionCells.some((c) => c.pm25 == null)}
        />
      )}

      {/* Hover tooltip */}
      {tooltip && (
        <div
          className="glass"
          style={{
            position: "absolute",
            left: Math.min(tooltip.x + 12, window.innerWidth - 300),
            top: Math.min(tooltip.y + 12, window.innerHeight - 200),
            padding: "9px 12px",
            borderRadius: "var(--radius-lg)",
            pointerEvents: "none",
            zIndex: "var(--z-overlay)",
            maxWidth: 280,
            fontSize: "0.8125rem",
            lineHeight: 1.5,
            boxShadow: "var(--shadow-lg)",
            animation: "fadeIn 0.1s var(--ease) both",
          }}
        >
          {tooltip.content}
        </div>
      )}
    </div>
  );
}
