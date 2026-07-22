"use client";
/**
 * FireLayer — scatter plot of FIRMS thermal anomaly detections.
 * Sized by FRP (fire radiative power), pulsing red/orange.
 */
import { ScatterplotLayer } from "@deck.gl/layers";
import { FIRE_HEX, hexToRgba } from "@/lib/colors";
import { icon, Flame } from "@/components/Icon";
import { FIRE_CONFIDENCE, type FireDetection } from "@/hooks/useMapData";

/** FIRMS confidence is a letter code, not a number. */
const conf = (d: FireDetection) => FIRE_CONFIDENCE[d.confidence] ?? { label: "Unknown", weight: 0.5 };

/**
 * Absolute timestamp, never "N hours ago".
 *
 * The detections belong to the ANALYSIS WINDOW (Delhi's is November 2025), not
 * to this minute, so an age measured against the wall clock is either NaN — as
 * it was, from a field name that does not exist — or a true-but-useless "6,800
 * hours ago". A date is the honest answer for a historical window and stays
 * correct in a live one.
 */
function overpass(ts: string): string {
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return "time unavailable";
  return d.toLocaleString(undefined, {
    day: "numeric", month: "short", year: "numeric",
    hour: "2-digit", minute: "2-digit", timeZone: "UTC",
  }) + " UTC";
}

export function buildFireLayer(
  fires: FireDetection[],
  onHover: (info: { x: number; y: number; content: React.ReactNode } | null) => void
) {
  if (!fires.length) return null;
  return new ScatterplotLayer<FireDetection>({
    id: "fires",
    data: fires,
    getPosition: (d) => [d.lon, d.lat],
    getRadius: (d) => Math.max(300, Math.min(1200, d.frp * 30)),
    radiusMinPixels: 4,
    radiusMaxPixels: 24,
    getFillColor: (d) => {
      const w = conf(d).weight;
      const [r, g, b] = hexToRgba(w >= 0.7 ? FIRE_HEX.high : FIRE_HEX.low);
      return [r, g, b, Math.round(175 + w * 60)];
    },
    getLineColor: [255, 255, 255, 90],
    lineWidthMinPixels: 1,
    stroked: true,
    pickable: true,
    autoHighlight: true,
    highlightColor: [255, 255, 255, 80],
    onHover: (info) => {
      if (info.object) {
        const d = info.object as FireDetection;
        onHover({
          x: info.x,
          y: info.y,
          content: (
            <div style={{ minWidth: 180 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 6, fontWeight: 600, marginBottom: 4 }}>
                <Flame {...icon.sm} aria-hidden />
                FIRMS detection
              </div>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.85rem", marginBottom: 4 }}>
                FRP: {d.frp.toFixed(1)} MW
              </div>
              <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
                Confidence: {conf(d).label}
              </div>
              <div style={{ fontSize: "0.7rem", color: "var(--text-tertiary)", marginTop: 6 }}>
                {overpass(d.ts)}
              </div>
            </div>
          ),
        });
      } else onHover(null);
    },
    updateTriggers: { getFillColor: [fires], getRadius: [fires] },
  });
}
