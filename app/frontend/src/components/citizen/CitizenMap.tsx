"use client";
/**
 * CitizenMap — a compact, tappable map for the citizen view.
 *
 * Two mutually-exclusive layers, picked by which data prop is passed:
 *   - `cells`: H3 choropleth by fusion PM2.5 (what a person there is breathing).
 *     Tapping a cell calls onPickWard.
 *   - `pins`: point markers (e.g. enforcement cases) at explicit lat/lon.
 *     Tapping a pin calls onPickPin.
 * Both read off the same deck.gl/maplibre instance, theme observer, and
 * fit-to-data logic, so CitizenExploreMap can switch what it's showing without
 * remounting a map. Auto-fits to whichever data is present (any city).
 * Deliberately lightweight — not the admin MapContainer, which needs the whole
 * layer/filter machinery.
 */
import { useMemo, useState, useEffect, useRef } from "react";
import Map from "react-map-gl/maplibre";
import DeckGL from "@deck.gl/react";
import { H3HexagonLayer } from "@deck.gl/geo-layers";
import { ScatterplotLayer } from "@deck.gl/layers";
import { cellToLatLng } from "h3-js";
import { initialViewFor } from "@/lib/constants";
import { useCity } from "@/lib/CityContext";
import "maplibre-gl/dist/maplibre-gl.css";

import { pm25ToRgbaArray, hexToRgba } from "@/lib/colors";

interface FusionCell { cell: string; pm25: number; ward_id?: string }
export interface MapPin { id: string; lat: number; lon: number; color: string; selected?: boolean }

const DARK_STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";
const LIGHT_STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json";

export default function CitizenMap({
  cells,
  highlightWard,
  onPickWard,
  pins,
  onPickPin,
  height = 340,
  interactive = true,
  extruded = false,
  bleed = false,
}: {
  cells: FusionCell[];
  highlightWard?: string | null;
  onPickWard?: (wardId: string, cell: string) => void;
  pins?: MapPin[];
  onPickPin?: (id: string) => void;
  height?: number | string;
  interactive?: boolean;
  /** Extrude the hex layer by PM2.5 and open on a tilted camera, so the height
   *  itself reads as "how bad" instead of only colour. Pins are unaffected —
   *  they're always flat markers. Off by default: the small embedded maps
   *  (ward-picker, ward-page preview) stay the flat, lightweight view they
   *  were built as; this is for the full-size explore map only. */
  extruded?: boolean;
  /** Edge-to-edge: no border/corner radius. For a map that fills the page,
   *  matching the admin console's borderless MapContainer, instead of a
   *  card-like embedded preview. */
  bleed?: boolean;
}) {
  // Opens on the ACTIVE city, not a hardcoded Delhi. The fit below still flies to
  // the real cells once they load; this stops the first frame being an empty map
  // 1,700 km from a Chennai or Bengaluru ward.
  const { city } = useCity();
  const [view, setView] = useState(() => ({
    ...initialViewFor(city), zoom: 10.5,
    pitch: extruded ? 45 : 0, bearing: extruded ? -12 : 0,
  }));
  const fittedFor = useRef<string | null>(null);

  // Re-tilt the camera when `extruded` flips after mount (a live 2D/3D toggle,
  // not just an initial setting) — dropping to 2D snaps back to top-down even
  // if the citizen had rotated the 3D view, since "2D" should mean flat.
  const wasExtruded = useRef(extruded);
  useEffect(() => {
    if (wasExtruded.current === extruded) return;
    wasExtruded.current = extruded;
    setView((v) => ({ ...v, pitch: extruded ? 45 : 0, bearing: extruded ? -12 : 0 }));
  }, [extruded]);

  // theme-aware base map — reads the same data-theme the app toggles
  const [mapStyle, setMapStyle] = useState(DARK_STYLE);
  useEffect(() => {
    const apply = () =>
      setMapStyle(document.documentElement.getAttribute("data-theme") === "light" ? LIGHT_STYLE : DARK_STYLE);
    apply();
    const obs = new MutationObserver(apply);
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => obs.disconnect();
  }, []);

  // Fit to the highlighted ward if given, else to all cells. Re-fits when the data
  // moves to a new city or a new ward — keyed on a signature of (first cell + ward),
  // so switching Delhi -> Chennai actually flies the map there.
  useEffect(() => {
    if (!cells.length) {
      // No hex data (pins mode) — fit to the pins instead, so a switch into
      // "cases" mode doesn't strand the view wherever the hex layer last was.
      if (!pins?.length) return;
      const sig = `pins:${pins.length}:${pins[0].id}`;
      if (fittedFor.current === sig) return;
      let sLat = 0, sLon = 0;
      for (const p of pins) { sLat += p.lat; sLon += p.lon; }
      fittedFor.current = sig;
      setView((v) => ({ ...v, latitude: sLat / pins.length, longitude: sLon / pins.length, zoom: 11 }));
      return;
    }
    const sig = `${cells[0].cell}|${highlightWard ?? ""}`;
    if (fittedFor.current === sig) return;
    const target = highlightWard ? cells.filter((c) => c.ward_id === highlightWard) : cells;
    const use = target.length ? target : cells;
    let sLat = 0, sLon = 0, n = 0;
    for (const c of use) {
      try { const [lat, lon] = cellToLatLng(c.cell); sLat += lat; sLon += lon; n++; } catch {}
    }
    if (!n) return;
    fittedFor.current = sig;
    setView((v) => ({ ...v, latitude: sLat / n, longitude: sLon / n, zoom: highlightWard ? 12.5 : 10.5 }));
  }, [cells, highlightWard, pins]);

  const layer = useMemo(() => {
    if (!cells.length) return null;
    return new H3HexagonLayer<FusionCell>({
      id: "citizen-fusion",
      data: cells,
      getHexagon: (d) => d.cell,
      getFillColor: (d) => {
        const base = pm25ToRgbaArray(d.pm25, highlightWard ? 90 : 150);
        if (highlightWard && d.ward_id === highlightWard) return pm25ToRgbaArray(d.pm25, 235);
        return base;
      },
      getLineColor: (d) =>
        highlightWard && d.ward_id === highlightWard ? [255, 255, 255, 200] : [255, 255, 255, 14],
      lineWidthMinPixels: highlightWard ? 1 : 0.5,
      stroked: true,
      // Elevation = PM2.5 itself, so height reads directly as "how much worse
      // than clean air" rather than an arbitrary scale. elevationScale 12 puts
      // a severe cell (~300-400 ug/m3) at ~4-5km — a few hex-widths tall at
      // this zoom, tall enough to read as a skyline without dwarfing the map.
      extruded,
      getElevation: extruded ? (d: FusionCell) => Math.max(d.pm25, 0) : 0,
      elevationScale: 12,
      pickable: interactive,
      autoHighlight: interactive,
      highlightColor: [255, 255, 255, 50],
      updateTriggers: { getFillColor: highlightWard, getLineColor: highlightWard },
      onClick: (info) => {
        if (interactive && info.object && onPickWard) {
          const d = info.object as FusionCell;
          if (d.ward_id) onPickWard(d.ward_id, d.cell);
        }
      },
    });
  }, [cells, highlightWard, onPickWard, interactive, extruded]);

  // Point markers (e.g. enforcement cases). Fixed pixel size — a pin has to stay
  // tappable at every zoom, unlike the hex layer which is meant to fill the view.
  const pinLayer = useMemo(() => {
    if (!pins?.length) return null;
    return new ScatterplotLayer<MapPin>({
      id: "citizen-pins",
      data: pins,
      getPosition: (d) => [d.lon, d.lat],
      getFillColor: (d) => hexToRgba(d.color, d.selected ? 235 : 190),
      getLineColor: (d) => (d.selected ? [255, 255, 255, 230] : [255, 255, 255, 120]),
      getRadius: (d) => (d.selected ? 9 : 7),
      radiusUnits: "pixels",
      radiusMinPixels: 6,
      radiusMaxPixels: 14,
      lineWidthMinPixels: 1.5,
      stroked: true,
      pickable: interactive,
      autoHighlight: interactive,
      highlightColor: [255, 255, 255, 60],
      updateTriggers: { getFillColor: pins, getRadius: pins, getLineColor: pins },
      onClick: (info) => {
        if (interactive && info.object && onPickPin) onPickPin((info.object as MapPin).id);
      },
    });
  }, [pins, onPickPin, interactive]);

  const layers = [layer, pinLayer].filter(Boolean);

  return (
    <div
      style={{
        position: "relative", width: "100%", height, overflow: "hidden",
        borderRadius: bleed ? 0 : "var(--radius-lg)",
        border: bleed ? "none" : "1px solid var(--border-subtle)",
      }}
    >
      <DeckGL
        viewState={view}
        onViewStateChange={interactive ? ({ viewState: vs }) => {
          const v = vs as typeof view;
          setView({ longitude: v.longitude, latitude: v.latitude, zoom: v.zoom, pitch: v.pitch, bearing: v.bearing });
        } : undefined}
        controller={interactive}
        layers={layers}
        getCursor={() => (interactive ? "pointer" : "grab")}
      >
        <Map reuseMaps mapStyle={mapStyle} attributionControl={false} />
      </DeckGL>
    </div>
  );
}
