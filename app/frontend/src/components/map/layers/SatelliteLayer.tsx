"use client";
/**
 * SatelliteLayer — Sentinel-5P NO2 tropospheric column, per H3 cell.
 *
 * NO2 ONLY. SO2 and the aerosol index are measured noise over a city (see
 * docs/architecture.md) and the backend does not serve them. This is the exact
 * field the detector runs neighbourhood-contrast on, so showing it explains WHY a
 * hotspot fired — the raw signal beneath the attribution.
 *
 * A distinct desaturated violet ramp (not the fusion green→maroon) so the two
 * choropleths stay legible when both are toggled on. Muted on purpose: this is a
 * CONTEXT layer explaining why a hotspot fired, and it must never out-shout the
 * hotspots themselves. It used to run to near-fluorescent magenta.
 */
import { H3HexagonLayer } from "@deck.gl/geo-layers";
import { OVERLAY_PARAMETERS } from "./overlay";

export interface SatelliteCell {
  cell: string;
  no2: number;
}

// Low -> high NO2: deep indigo to a soft heather. Alpha climbs so faint columns
// don't clutter the map. ↔ --sat-no2 family in lib/colors.ts.
function no2ToColor(no2: number, min: number, max: number): [number, number, number, number] {
  const t = max > min ? (no2 - min) / (max - min) : 0.5;
  const r = Math.round(0x4a + t * 0x7f);  // 74 -> 201
  const g = Math.round(0x45 + t * 0x69);  // 69 -> 174
  const b = Math.round(0x78 + t * 0x62);  // 120 -> 218
  const a = Math.round(60 + t * 130);     //  60 -> 190
  return [r, g, b, a];
}

export function buildSatelliteLayer(
  cells: SatelliteCell[],
  onHover: (info: { x: number; y: number; content: React.ReactNode } | null) => void
) {
  if (!cells.length) return null;
  const vals = cells.map((c) => c.no2);
  const min = Math.min(...vals);
  const max = Math.max(...vals);

  return new H3HexagonLayer<SatelliteCell>({
    id: "satellite-no2",
    parameters: OVERLAY_PARAMETERS,
    data: cells,
    getHexagon: (d) => d.cell,
    getFillColor: (d) => no2ToColor(d.no2, min, max),
    extruded: false,
    wireframe: false,
    pickable: true,
    autoHighlight: true,
    highlightColor: [255, 255, 255, 40],
    onHover: (info) => {
      if (info.object) {
        const d = info.object as SatelliteCell;
        onHover({
          x: info.x, y: info.y,
          content: (
            <div>
              <div style={{ fontWeight: 600 }}>NO₂ column</div>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.8rem" }}>
                {d.no2.toFixed(1)} µmol/m²
              </div>
              <div style={{ fontSize: "0.7rem", opacity: 0.7, marginTop: 2 }}>
                Sentinel-5P · the signal detection reads
              </div>
            </div>
          ),
        });
      } else {
        onHover(null);
      }
    },
  });
}
