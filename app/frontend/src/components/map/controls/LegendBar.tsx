"use client";
/**
 * LegendBar — bottom-right key, showing whichever sections the active layers
 * make relevant.
 *
 * ⚠ It reads AQI_CATEGORIES / PERSISTENCE_HEX / FIRE_HEX from lib/colors.ts —
 * the same module the deck.gl layers colour from. It used to hardcode its OWN
 * AQI swatches (#22c55e, #fde047, #7f1d1d…), so the legend and the choropleth it
 * explained were different colours. A legend that lies is worse than no legend.
 */
import { useState } from "react";
import { AQI_CATEGORIES, PERSISTENCE_HEX, FIRE_HEX, BLINDSPOT_HEX, NO_FORECAST_HEX } from "@/lib/colors";
import { useIsMobile } from "@/hooks/useMediaQuery";
import { icon, X, ListChecks } from "@/components/Icon";
import type { LayerVisibility } from "@/lib/types";

interface Props {
  layers: LayerVisibility;
  /** True when the map currently holds cells with no value — see NO_FORECAST_HEX. */
  hasNoForecastCells?: boolean;
}

function Swatch({ color, round = false }: { color: string; round?: boolean }) {
  return (
    <span
      aria-hidden
      style={{
        width: 10, height: 10, flexShrink: 0,
        borderRadius: round ? "50%" : 2,
        background: color,
        boxShadow: "inset 0 0 0 1px rgba(0,0,0,0.18)",
      }}
    />
  );
}

function Row({ color, label, value, round }: {
  color: string; label: string; value?: string; round?: boolean;
}) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 7, minHeight: 18 }}>
      <Swatch color={color} round={round} />
      <span style={{ fontSize: "0.7rem", color: "var(--text-secondary)", flex: 1 }}>{label}</span>
      {value && (
        <span className="mono" style={{ fontSize: "0.65rem", color: "var(--text-tertiary)" }}>
          {value}
        </span>
      )}
    </div>
  );
}

function Group({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="section-label" style={{ marginBottom: 6 }}>{title}</div>
      <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>{children}</div>
    </div>
  );
}

export default function LegendBar({ layers, hasNoForecastCells = false }: Props) {
  const isMobile = useIsMobile();
  const [open, setOpen] = useState(true);

  const sections: React.ReactNode[] = [];

  if (layers.fusion || layers.hotspots) {
    sections.push(
      <Group key="aqi" title="PM2.5 · India NAQI">
        {AQI_CATEGORIES.map((c) => (
          <Row key={c.label} color={c.color} label={c.label} value={c.range} />
        ))}
      </Group>
    );
  }
  if (hasNoForecastCells) {
    sections.push(
      <Group key="nofc" title="Coverage">
        <Row color={NO_FORECAST_HEX} label="No forecast here" />
        <div style={{ fontSize: "0.65rem", color: "var(--text-tertiary)", marginTop: 2 }}>
          Upwind of every monitor at this hour
        </div>
      </Group>
    );
  }
  if (layers.hotspots) {
    sections.push(
      <Group key="persist" title="Hotspot type">
        <Row color={PERSISTENCE_HEX.chronic}  label="Chronic · 30d" />
        <Row color={PERSISTENCE_HEX.emerging} label="Emerging · 7d" />
        <Row color={PERSISTENCE_HEX.acute}    label="Acute · 24h" />
      </Group>
    );
  }
  if (layers.fires) {
    sections.push(
      <Group key="fire" title="FIRMS detections">
        <Row round color={FIRE_HEX.high} label="High confidence" />
        <Row round color={FIRE_HEX.low}  label="Low confidence" />
      </Group>
    );
  }
  if (layers.blindspots) {
    sections.push(
      <Group key="blind" title="Blind spots">
        <Row color={BLINDSPOT_HEX} label="No monitor, satellite high" />
        <div style={{ fontSize: "0.65rem", color: "var(--text-tertiary)", marginTop: 2 }}>
          Brighter = higher placement rank
        </div>
      </Group>
    );
  }

  if (sections.length === 0) return null;

  if (isMobile && !open) {
    return (
      <button
        className="glass btn btn-sm"
        onClick={() => setOpen(true)}
        style={{ position: "absolute", bottom: 116, right: 12, zIndex: "var(--z-overlay)" }}
      >
        <ListChecks {...icon.sm} aria-hidden />
        Legend
      </button>
    );
  }

  return (
    <div
      className="glass"
      aria-label="Map legend"
      style={{
        position: "absolute",
        bottom: isMobile ? 116 : 80,
        right: 12,
        zIndex: "var(--z-overlay)",
        padding: "10px 12px",
        borderRadius: "var(--radius-lg)",
        boxShadow: "var(--shadow-md)",
        minWidth: 168,
        maxWidth: 208,
        maxHeight: isMobile ? "42vh" : undefined,
        overflowY: isMobile ? "auto" : undefined,
        display: "flex",
        flexDirection: "column",
        gap: 12,
      }}
    >
      {isMobile && (
        <button
          onClick={() => setOpen(false)}
          className="btn btn-quiet btn-icon"
          aria-label="Hide legend"
          style={{ position: "absolute", top: 4, right: 4, minHeight: 0, padding: 3 }}
        >
          <X {...icon.sm} aria-hidden />
        </button>
      )}
      {sections.map((s, i) => (
        <div key={i}>
          {i > 0 && <hr style={{ margin: "0 0 12px" }} />}
          {s}
        </div>
      ))}
    </div>
  );
}
