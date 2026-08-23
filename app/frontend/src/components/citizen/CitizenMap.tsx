"use client";
/**
 * CitizenMap — a compact, tappable ward map for the citizen view.
 *
 * Colours every H3 cell by its fusion PM2.5 (what a person there is breathing),
 * highlights the citizen's ward, and calls onPickWard when they tap a cell. Auto-
 * fits to the data (any city). Deliberately lightweight — not the admin
 * MapContainer, which needs the whole layer/filter machinery.
 */
import { useMemo, useState, useEffect, useRef } from "react";
import Map from "react-map-gl/maplibre";
import DeckGL from "@deck.gl/react";
import { H3HexagonLayer } from "@deck.gl/geo-layers";
import { cellToLatLng } from "h3-js";
import { initialViewFor } from "@/lib/constants";
import { useCity } from "@/lib/CityContext";
import "maplibre-gl/dist/maplibre-gl.css";

import { pm25ToRgbaArray } from "@/lib/colors";

// pm25 is nullable for the same reason it is in lib/types: at a forecast horizon
// some cells have no value, and that is a gap to draw rather than a zero.
interface FusionCell { cell: string; pm25: number | null; ward_id?: string }

const DARK_STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";
const LIGHT_STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json";

export default function CitizenMap({
  cells,
  highlightWard,
  onPickWard,
  height = 340,
  interactive = true,
}: {
  cells: FusionCell[];
  highlightWard?: string | null;
  onPickWard?: (wardId: string, cell: string) => void;
  height?: number | string;
  interactive?: boolean;
}) {
  // Opens on the ACTIVE city, not a hardcoded Delhi. The fit below still flies to
  // the real cells once they load; this stops the first frame being an empty map
  // 1,700 km from a Chennai or Bengaluru ward.
  const { city } = useCity();
  const [view, setView] = useState(() => ({
    ...initialViewFor(city), zoom: 10.5, pitch: 0, bearing: 0,
  }));
  const fittedFor = useRef<string | null>(null);

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
    if (!cells.length) return;
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
  }, [cells, highlightWard]);

  const layer = useMemo(() => {
    if (!cells.length) return null;
    return new H3HexagonLayer<FusionCell>({
      id: "citizen-fusion",
      data: cells,
      getHexagon: (d) => d.cell,
      getFillColor: (d) => {
        // No value is a gap, not clean air — same treatment as the admin map.
        if (d.pm25 == null) return [120, 124, 130, highlightWard ? 18 : 26];
        const base = pm25ToRgbaArray(d.pm25, highlightWard ? 90 : 150);
        if (highlightWard && d.ward_id === highlightWard) return pm25ToRgbaArray(d.pm25, 235);
        return base;
      },
      getLineColor: (d) =>
        highlightWard && d.ward_id === highlightWard ? [255, 255, 255, 200] : [255, 255, 255, 14],
      lineWidthMinPixels: highlightWard ? 1 : 0.5,
      stroked: true,
      extruded: false,
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
  }, [cells, highlightWard, onPickWard, interactive]);

  return (
    <div style={{ position: "relative", width: "100%", height, borderRadius: "var(--radius-lg)", overflow: "hidden", border: "1px solid var(--border-subtle)" }}>
      <DeckGL
        viewState={view}
        onViewStateChange={interactive ? ({ viewState: vs }) => {
          const v = vs as typeof view;
          setView({ longitude: v.longitude, latitude: v.latitude, zoom: v.zoom, pitch: v.pitch, bearing: v.bearing });
        } : undefined}
        controller={interactive}
        layers={layer ? [layer] : []}
        getCursor={() => (interactive ? "pointer" : "grab")}
      >
        <Map reuseMaps mapStyle={mapStyle} attributionControl={false} />
      </DeckGL>
    </div>
  );
}
