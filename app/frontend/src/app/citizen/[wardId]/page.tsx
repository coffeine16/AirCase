"use client";
import { use, useMemo } from "react";
import Link from "next/link";
import dynamic from "next/dynamic";
import useSWR from "swr";
import { api } from "@/lib/api";
import { pm25ToAqi, getAqiCategory, AQI_CATEGORIES } from "@/lib/colors";
import { useCity } from "@/lib/CityContext";
import { AQI_ADVICE } from "@/lib/constants";
import type { FusionResponse, ForecastCell } from "@/lib/types";

import VoiceAdvisory from "@/components/citizen/VoiceAdvisory";
import WardAccountability from "@/components/citizen/WardAccountability";
import { icon, ArrowLeft, ArrowRight, Camera, ClipboardList, Megaphone, TrendingUp } from "@/components/Icon";

const CitizenMap = dynamic(() => import("@/components/citizen/CitizenMap"), {
  ssr: false,
  loading: () => <div className="skeleton" style={{ height: 300, borderRadius: "var(--radius-lg)" }} />,
});

interface Params { wardId: string }

export default function WardDashboardPage({ params }: { params: Promise<Params> }) {
  const { wardId } = use(params);

  const { city } = useCity();
  const { data: fusion, isLoading } = useSWR<FusionResponse>([city, "fusion"], () => api.cityFusion(city));
  const { data: wardsResp } = useSWR([city, "wards"], () => api.cityWards(city));
  const cells = useMemo(() => fusion?.cells ?? [], [fusion]);

  // Everything is derived from THIS CITY's fusion field, so the page follows the
  // city switcher and stays consistent (no separate backend summary to drift).
  const wardCells = useMemo(() => new Set(cells.filter((c) => c.ward_id === wardId).map((c) => c.cell)), [cells, wardId]);
  const median = (v: number[]) => (v.length ? [...v].sort((a, b) => a - b)[Math.floor(v.length / 2)] : NaN);
  const wardPm25 = useMemo(() => median(cells.filter((c) => c.ward_id === wardId).map((c) => c.pm25)), [cells, wardId]);
  const wardName = useMemo(
    () => wardsResp?.cells.find((c) => c.ward_id === wardId)?.ward_name ?? wardId,
    [wardsResp, wardId]
  );
  const summary = Number.isFinite(wardPm25) ? { pm25: wardPm25, ward_name: wardName } : null;

  // The ward's real CPCB advisory text (English), from this city's advisories.
  const { data: advisories } = useSWR([city, "advisories"], () => api.cityAdvisories(city));
  const advisoryText = useMemo(
    () => advisories?.find((a) => a.ward_id === wardId)?.texts?.en ?? null,
    [advisories, wardId]
  );

  const { data: fc24 } = useSWR<ForecastCell[]>([city, "forecast", 24], () => api.cityForecast(city, 24));
  const { data: fc48 } = useSWR<ForecastCell[]>([city, "forecast", 48], () => api.cityForecast(city, 48));
  const { data: fc72 } = useSWR<ForecastCell[]>([city, "forecast", 72], () => api.cityForecast(city, 72));
  const horizonAqi = (fc?: ForecastCell[]): number | null => {
    if (!fc) return null;
    const vals = fc.filter((f) => wardCells.has(f.cell)).map((f) => f.pm25_hat);
    if (!vals.length) return null;
    return pm25ToAqi(median(vals));
  };
  const forecast = [
    { label: "Now", aqi: summary ? pm25ToAqi(summary.pm25) : null },
    { label: "+24h", aqi: horizonAqi(fc24) },
    { label: "+48h", aqi: horizonAqi(fc48) },
    { label: "+72h", aqi: horizonAqi(fc72) },
  ];

  const aqi = summary ? pm25ToAqi(summary.pm25) : null;
  const category = aqi != null ? getAqiCategory(aqi) : null;
  // Advice is keyed by CATEGORY, not aqi/100: the NAQI bands aren't every-100
  // (Good is 0–50, Satisfactory 51–100), so floor(aqi/100) mislabelled them.
  const adviceIndex = category ? Math.max(0, AQI_CATEGORIES.indexOf(category)) : 0;

  return (
    <div className="page" style={{ maxWidth: 680 }}>
      <Link href="/citizen" className="nav-link" style={{ marginBottom: "var(--space-md)", marginLeft: -10 }}>
        <ArrowLeft {...icon.sm} aria-hidden />
        Change ward
      </Link>

      {/* Ward name */}
      <div style={{ marginBottom: "var(--space-md)" }}>
        <h1 style={{ marginBottom: 3 }}>
          {isLoading ? (
            <span className="skeleton" style={{ display: "inline-block", width: 200, height: 32, borderRadius: 6 }} />
          ) : (
            summary?.ward_name ?? wardId
          )}
        </h1>
        <span className="mono" style={{ fontSize: "0.78rem", color: "var(--text-tertiary)" }}>
          {wardId}
        </span>
      </div>

      {/* Map of THIS ward — leads the page: you land on your area */}
      <div style={{ marginBottom: "var(--space-lg)" }}>
        <CitizenMap cells={cells} highlightWard={wardId} interactive={false} height={280} />
        <p style={{ fontSize: "0.72rem", color: "var(--text-tertiary)", textAlign: "center", marginTop: 6 }}>
          Live PM2.5 across your ward — brighter cells are dirtier air.
        </p>
      </div>

      {/* AQI Card */}
      <div
        className="card card-rail"
        style={{
          marginBottom: "var(--space-lg)",
          ["--rail" as string]: category?.color ?? "var(--border-strong)",
          padding: "var(--space-lg)",
        }}
      >
        {isLoading ? (
          <div className="skeleton" style={{ height: 120, borderRadius: 8 }} />
        ) : (
          <div style={{ display: "flex", gap: "var(--space-xl)", alignItems: "center" }}>
            {/* AQI reading. A 100px glowing disc read as a game score; a
                square swatch reads as a measurement, and matches the band
                swatches in the map legend. */}
            <div style={{ flexShrink: 0 }}>
              <div
                style={{
                  width: 86, height: 86, borderRadius: "var(--radius-lg)",
                  background: category?.color ?? "var(--bg-tertiary)",
                  display: "flex", flexDirection: "column",
                  alignItems: "center", justifyContent: "center",
                }}
              >
                <div
                  className="mono"
                  style={{ fontSize: "1.7rem", fontWeight: 600, color: category?.textColor ?? "var(--text-secondary)", lineHeight: 1 }}
                >
                  {aqi ?? "—"}
                </div>
                <div style={{ fontSize: "0.58rem", color: category?.textColor ?? "var(--text-tertiary)", opacity: 0.8, marginTop: 3, letterSpacing: "0.1em" }}>
                  AQI
                </div>
              </div>
            </div>
            {/* Details */}
            <div style={{ flex: 1 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                <span style={{ fontSize: "1.1rem", fontWeight: 600 }}>{category?.label ?? "No data"}</span>
                {(() => {
                  const now = forecast[0].aqi, next = forecast[1].aqi;
                  if (now == null || next == null) return null;
                  const d = next - now;
                  if (Math.abs(d) < 3) {
                    return <span className="badge badge-diffuse">steady · 24h</span>;
                  }
                  const worse = d > 0;
                  return (
                    <span className={`badge ${worse ? "badge-critical" : "badge-positive"}`}>
                      <TrendingUp
                        {...icon.sm}
                        aria-hidden
                        style={{ transform: worse ? "none" : "scaleY(-1)" }}
                      />
                      {worse ? "worsening" : "improving"} · 24h
                    </span>
                  );
                })()}
              </div>
              <div style={{ fontSize: "0.875rem", color: "var(--text-secondary)", marginBottom: 8 }}>
                PM2.5 <span className="mono" style={{ color: "var(--text-primary)" }}>
                  {summary?.pm25 != null ? `${summary.pm25.toFixed(1)} µg/m³` : "—"}
                </span>
              </div>
              {/* Only when the ward has NO real advisory. The pipeline's advisory
                  says the same thing better — named ward, actual AQI, and it is
                  the text the voice note reads — so rendering both printed the
                  same sentence twice, once generically and once for real. */}
              {!advisoryText && (
                <div style={{ fontSize: "0.85rem", color: "var(--text-secondary)", lineHeight: 1.5 }}>
                  {AQI_ADVICE[adviceIndex]}
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Why / what is being done / how sure — placed directly under the number,
          because "why" is the question the number provokes and the one every
          other AQI product leaves unanswered. */}
      <WardAccountability city={city} wardId={wardId} cells={cells} />

      {/* Advisory */}
      {advisoryText && (
        <div
          className="card card-rail"
          style={{ marginBottom: "var(--space-lg)", ["--rail" as string]: "var(--caution)" }}
        >
          <h5 style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 8 }}>
            <Megaphone {...icon.sm} aria-hidden />
            Advisory
          </h5>
          <p style={{ fontSize: "0.875rem", color: "var(--text-primary)", lineHeight: 1.6 }}>
            {advisoryText}
          </p>
        </div>
      )}

      {/* Spoken advisory — the IVR/voice-note deliverable, per language */}
      <VoiceAdvisory city={city} wardId={wardId} />

      {/* 72-hour forecast — median predicted AQI across the ward's cells */}
      <div className="card" style={{ marginBottom: "var(--space-lg)" }}>
        <h5 style={{ marginBottom: "var(--space-md)" }}>72-hour forecast</h5>
        <div style={{ display: "flex", gap: "var(--space-sm)" }}>
          {forecast.map((f) => {
            const cat = f.aqi != null ? getAqiCategory(f.aqi) : null;
            return (
              <div
                key={f.label}
                style={{
                  flex: 1, textAlign: "center", padding: "12px 4px",
                  borderRadius: "var(--radius-md)",
                  background: "var(--bg-tertiary)",
                  borderTop: `2px solid ${cat ? cat.color : "var(--border-default)"}`,
                }}
              >
                <div style={{ fontSize: "0.7rem", color: "var(--text-tertiary)", marginBottom: 6 }}>{f.label}</div>
                <div className="mono" style={{ fontSize: "1.35rem", fontWeight: 600, color: "var(--text-primary)", lineHeight: 1 }}>
                  {f.aqi ?? "—"}
                </div>
                <div style={{ fontSize: "0.62rem", color: "var(--text-tertiary)", marginTop: 5 }}>{cat?.label ?? ""}</div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Actions */}
      <div style={{ display: "flex", gap: "var(--space-md)", flexWrap: "wrap" }}>
        <Link
          href={`/citizen/${wardId}/report`}
          className="btn btn-primary"
          style={{ flex: 1, textDecoration: "none" }}
        >
          <Camera {...icon.md} aria-hidden />
          Report pollution
        </Link>
        <Link
          href="/citizen/reports"
          className="btn btn-ghost"
          style={{ flex: 1, textDecoration: "none" }}
        >
          <ClipboardList {...icon.md} aria-hidden />
          My reports
          <ArrowRight {...icon.sm} aria-hidden />
        </Link>
      </div>
    </div>
  );
}
