"use client";
/**
 * DispatchLayer — draws per-team inspection routes on the map.
 *
 * When the backend provides route_geometry (from OSRM), the route follows the
 * actual road path using a PathLayer.  When route_geometry is absent or empty,
 * it falls back to straight LineLayer segments between stop centroids.
 *
 * Stop positions are rendered as pins with sequence labels.
 */
import { PathLayer, LineLayer, ScatterplotLayer, TextLayer } from "@deck.gl/layers";
import { icon, Truck } from "@/components/Icon";
import type { DispatchRoute, DispatchStop } from "@/lib/types";
import { OVERLAY_PARAMETERS } from "./overlay";

// One hue per team, up to 6. Muted and evenly spaced around the wheel: these
// identify a route, they do not rank it, so none of them may read as an alert.
// (The old set was the Tailwind-500 rainbow, and its red/amber members collided
// with the severity colours the hotspot layer draws underneath.)
const TEAM_COLORS: [number, number, number, number][] = [
  [110, 143, 192, 225],  // steel blue  — T1
  [ 94, 168, 138, 225],  // sage        — T2
  [186, 150,  86, 225],  // brass       — T3
  [174, 122, 158, 225],  // mauve       — T4
  [123, 156, 173, 225],  // slate cyan  — T5
  [163, 149, 118, 225],  // khaki       — T6
];

function teamColor(teamId: string): [number, number, number, number] {
  const idx = parseInt(teamId.replace(/\D/g, ""), 10) - 1;
  return TEAM_COLORS[idx % TEAM_COLORS.length];
}

// Build stop-pair lines for each route (fallback when no road geometry)
interface RouteLine { from: [number, number]; to: [number, number]; team_id: string }
interface StopPoint {
  stop: DispatchStop;
  team_id: string;
  route_km: number;
  route_duration_min: number;
  /** This team has exactly one stop, so there is no route to draw — see below. */
  solo: boolean;
}
interface RoadPath  { path: [number, number][]; team_id: string }

export function buildDispatchLayers(
  routes: DispatchRoute[],
  onHover: (info: { x: number; y: number; content: React.ReactNode } | null) => void
) {
  if (!routes.length) return [];

  const roadPaths: RoadPath[] = [];
  const fallbackLines: RouteLine[] = [];
  const points: StopPoint[] = [];

  for (const route of routes) {
    const hasGeometry = route.route_geometry && route.route_geometry.length > 1;

    if (hasGeometry) {
      // Use OSRM road geometry
      roadPaths.push({
        path: route.route_geometry as [number, number][],
        team_id: route.team_id,
      });
    } else {
      // Fallback: straight lines between consecutive stops
      for (let i = 0; i < route.stops.length - 1; i++) {
        fallbackLines.push({
          from: [route.stops[i].lon, route.stops[i].lat],
          to:   [route.stops[i + 1].lon, route.stops[i + 1].lat],
          team_id: route.team_id,
        });
      }
    }

    for (const stop of route.stops) {
      points.push({
        stop,
        team_id: route.team_id,
        route_km: route.route_km,
        route_duration_min: route.route_duration_min ?? 0,
        // A team with ONE stop has no path: nothing to route between, so
        // route_km and route_geometry are legitimately 0/empty. Without a visual
        // cue that reads as a broken route rather than a short assignment —
        // which is exactly how Delhi looked, where 4 actions across 4 teams gave
        // every team a single stop.
        solo: route.stops.length === 1,
      });
    }
  }

  const layers = [];

  // Road-path layer (actual road geometry from OSRM)
  if (roadPaths.length) {
    layers.push(
      new PathLayer<RoadPath>({
        id: "dispatch-road-paths",
        parameters: OVERLAY_PARAMETERS,
        data: roadPaths,
        getPath: (d) => d.path,
        getColor: (d) => teamColor(d.team_id),
        getWidth: 4,
        widthMinPixels: 2,
        widthMaxPixels: 8,
        jointRounded: true,
        capRounded: true,
        pickable: false,
      })
    );
  }

  // Fallback straight-line layer
  if (fallbackLines.length) {
    layers.push(
      new LineLayer<RouteLine>({
        id: "dispatch-routes-fallback",
        parameters: OVERLAY_PARAMETERS,
        data: fallbackLines,
        getSourcePosition: (d) => d.from,
        getTargetPosition: (d) => d.to,
        getColor: (d) => teamColor(d.team_id),
        getWidth: 2,
        widthMinPixels: 2,
        widthMaxPixels: 5,
        pickable: false,
      })
    );
  }

  // Halo behind single-stop assignments.
  //
  // A one-stop team draws no line, so its pin sat on the map looking like a
  // route that had failed to render. The ring says "this IS the whole
  // assignment" — a destination, not a broken path. Drawn first so it sits
  // under the pin.
  const solos = points.filter((p) => p.solo);
  if (solos.length) {
    layers.push(
      new ScatterplotLayer<StopPoint>({
        id: "dispatch-solo-halo",
        parameters: OVERLAY_PARAMETERS,
        data: solos,
        getPosition: (d) => [d.stop.lon, d.stop.lat],
        getRadius: 620,
        radiusMinPixels: 15,
        radiusMaxPixels: 30,
        filled: false,
        stroked: true,
        getLineColor: (d) => {
          const [r, g, b] = teamColor(d.team_id);
          return [r, g, b, 130];
        },
        lineWidthMinPixels: 1.5,
        pickable: false,
      })
    );
  }

  // Stop markers
  const stopLayer = new ScatterplotLayer<StopPoint>({
    id: "dispatch-stops",
    parameters: OVERLAY_PARAMETERS,
    data: points,
    getPosition: (d) => [d.stop.lon, d.stop.lat],
    getRadius: 320,
    radiusMinPixels: 8,
    radiusMaxPixels: 16,
    getFillColor: (d) => teamColor(d.team_id),
    getLineColor: [255, 255, 255, 200],
    lineWidthMinPixels: 2,
    stroked: true,
    pickable: true,
    autoHighlight: true,
    highlightColor: [255, 255, 255, 80],
    onHover: (info) => {
      if (info.object) {
        const { stop, team_id, route_km, route_duration_min, solo } = info.object as StopPoint;
        onHover({
          x: info.x,
          y: info.y,
          content: (
            <div style={{ minWidth: 200 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 6, fontWeight: 600, marginBottom: 4 }}>
                <Truck {...icon.sm} aria-hidden />
                Dispatch stop #{stop.seq}
              </div>
              <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: 2 }}>
                Team {team_id} · Zone {stop.zone_id}
              </div>
              <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
                Ward {stop.ward_id}
              </div>
              <div className="mono" style={{ fontSize: "0.8rem", color: "var(--caution)", marginTop: 4 }}>
                EPS: {stop.eps.toFixed(1)}
              </div>
              <div style={{
                display: "flex", gap: 12, marginTop: 6, paddingTop: 6,
                borderTop: "1px solid var(--border-subtle)",
                fontSize: "0.74rem", color: "var(--text-tertiary)",
              }}>
                {route_km > 0 ? (
                  <>
                    <span>{route_km} km</span>
                    {route_duration_min > 0 && <span>~{Math.round(route_duration_min)} min</span>}
                  </>
                ) : solo ? (
                  // Say WHY there is no route, rather than reporting 0 km as if
                  // a measurement had failed.
                  <span>Only stop for this team — no travel between sites</span>
                ) : (
                  <span>Route distance unavailable</span>
                )}
              </div>
            </div>
          ),
        });
      } else onHover(null);
    },
    updateTriggers: { getFillColor: [routes] },
  });

  // Sequence labels on each stop
  const labelLayer = new TextLayer<StopPoint>({
    id: "dispatch-labels",
    parameters: OVERLAY_PARAMETERS,
    data: points,
    getPosition: (d) => [d.stop.lon, d.stop.lat],
    getText: (d) => `${d.stop.seq}`,
    getSize: 11,
    getColor: [255, 255, 255, 240],
    getTextAnchor: "middle",
    getAlignmentBaseline: "center",
    fontFamily: "\"JetBrains Mono\", monospace",
    fontWeight: "bold",
    pickable: false,
    getPixelOffset: [0, 0],
  });

  layers.push(stopLayer, labelLayer);
  return layers;
}
