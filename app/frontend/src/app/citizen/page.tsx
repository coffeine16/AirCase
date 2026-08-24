"use client";
/**
 * Citizen home — find YOUR ward, no scrolling.
 *   1. "Use my location" -> geolocate -> resolve to ward via H3 (client-side).
 *   2. or tap the map on your area.
 *   3. or type to search (kept as a fallback, collapsed by default).
 * The 227-ward scroll grid is gone: a citizen should land on their map, not hunt
 * a list.
 */
import { useState, useMemo, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import dynamic from "next/dynamic";
import useSWR from "swr";
import { api } from "@/lib/api";
import { pm25ToAqi, getAqiCategory } from "@/lib/colors";
import { useCitizenWard } from "@/hooks/useReports";
import { useWardLocator } from "@/hooks/useWardLocator";
import { useCity } from "@/lib/CityContext";
import { nearestCity } from "@/lib/constants";
import { icon, MapPin, Search, ArrowLeft, ArrowRight, Layers, TriangleAlert } from "@/components/Icon";
import type { FusionResponse } from "@/lib/types";

const CitizenMap = dynamic(() => import("@/components/citizen/CitizenMap"), {
  ssr: false,
  loading: () => <div className="skeleton" style={{ height: 340, borderRadius: "var(--radius-lg)" }} />,
});

export default function CitizenHomePage() {
  const router = useRouter();
  const { city, setCity } = useCity();
  const { wardId: savedWard, setWardId } = useCitizenWard();
  const { locate, hasData } = useWardLocator();

  const { data: fusion } = useSWR<FusionResponse>([city, "fusion"], () => api.cityFusion(city));
  const cells = useMemo(() => fusion?.cells ?? [], [fusion]);

  // Ward id -> real name. Fusion cells carry only ward_id; the names live in
  // wards.json, so without this every ward shows as "W256" instead of its name.
  const { data: wardsResp } = useSWR([city, "wards"], () => api.cityWards(city));
  const wardName = useMemo(() => {
    const m = new Map<string, string>();
    for (const c of wardsResp?.cells ?? []) m.set(c.ward_id, c.ward_name);
    return m;
  }, [wardsResp]);

  const [locating, setLocating] = useState(false);
  const [geoError, setGeoError] = useState<string | null>(null);
  const [search, setSearch] = useState("");

  const go = (wardId: string) => {
    setWardId(wardId);
    router.push(`/citizen/${wardId}`);
  };

  // Coordinates waiting for a city's ward data to arrive. Set only when
  // geolocation lands in a DIFFERENT city than the one currently loaded: the
  // ward locator searches one city at a time, so we have to switch first and
  // resolve the ward once that city's cells are in.
  const [pendingFix, setPendingFix] = useState<{ lat: number; lon: number } | null>(null);

  const useMyLocation = () => {
    setGeoError(null);
    if (!("geolocation" in navigator)) {
      setGeoError("Your browser doesn't support location. Tap the map instead.");
      return;
    }
    setLocating(true);
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const { latitude: lat, longitude: lon } = pos.coords;

        // Which of our eight cities is this? Without this step the lookup runs
        // against whatever city happens to be selected, so a citizen in
        // Hyderabad on the Delhi default was told we couldn't find their ward —
        // when we have Hyderabad's full pipeline sitting right there.
        const near = nearestCity(lat, lon);
        if (!near) {
          setLocating(false);
          setGeoError(
            "We don't cover your city yet — we run eight Indian cities so far. " +
            "Tap the map to look at one of them."
          );
          return;
        }
        if (near.city !== city) {
          // Switch, then finish in the effect below once the wards land.
          setCity(near.city as typeof city);
          setPendingFix({ lat, lon });
          return;                      // stays in the locating state deliberately
        }
        setLocating(false);
        const hit = locate(lat, lon);
        if (hit) go(hit.ward_id);
        else setGeoError("Couldn't match your location to a ward. Tap the map instead.");
      },
      () => {
        setLocating(false);
        setGeoError("Location permission denied. Tap the map to pick your area.");
      },
      { enableHighAccuracy: true, timeout: 8000 }
    );
  };

  // Second half of a cross-city locate: the city switched, its wards have now
  // loaded, so the held coordinates can finally be resolved. `hasData` is the
  // gate — running before it is in would miss and show a false "not found".
  useEffect(() => {
    if (!pendingFix || !hasData) return;
    const hit = locate(pendingFix.lat, pendingFix.lon);
    setPendingFix(null);
    setLocating(false);
    if (hit) go(hit.ward_id);
    else setGeoError("Couldn't match your location to a ward. Tap the map instead.");
    // `go` and `locate` are stable enough for this one-shot resolve; re-running
    // on their identity would re-navigate after the push.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingFix, hasData]);

  // Ward list for search — real names from wards.json, matched by name or id.
  const wards = useMemo(() => {
    const seen = new Set<string>();
    const out: { ward_id: string; ward_name: string }[] = [];
    for (const c of cells) {
      if (!c.ward_id || c.ward_id === "unassigned" || seen.has(c.ward_id)) continue;
      seen.add(c.ward_id);
      out.push({ ward_id: c.ward_id, ward_name: wardName.get(c.ward_id) ?? c.ward_id });
    }
    return out.sort((a, b) => a.ward_name.localeCompare(b.ward_name));
  }, [cells, wardName]);
  const searchHits = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return [];
    return wards
      .filter((w) => w.ward_name.toLowerCase().includes(q) || w.ward_id.toLowerCase().includes(q))
      .slice(0, 8);
  }, [wards, search]);

  // Live city snapshot — the citywide median AQI, and its worst wards right now.
  const cityAqi = useMemo(() => {
    if (!cells.length) return null;
    // Cells with no value are excluded, not treated as zero — see FusionCell.pm25.
    const vals = cells.map((c) => c.pm25).filter((v): v is number => v != null)
                      .sort((a, b) => a - b);
    if (!vals.length) return null;
    return pm25ToAqi(vals[Math.floor(vals.length / 2)]);
  }, [cells]);
  const cityCat = cityAqi != null ? getAqiCategory(cityAqi) : null;

  const worstWards = useMemo(() => {
    const byWard = new Map<string, number[]>();
    for (const c of cells) {
      if (!c.ward_id || c.ward_id === "unassigned") continue;
      if (c.pm25 == null) continue;
      (byWard.get(c.ward_id) ?? byWard.set(c.ward_id, []).get(c.ward_id)!).push(c.pm25);
    }
    return [...byWard.entries()]
      .map(([wid, v]) => ({ wid, pm25: v.sort((a, b) => a - b)[Math.floor(v.length / 2)] }))
      .sort((a, b) => b.pm25 - a.pm25)
      .slice(0, 6);
  }, [cells]);

  return (
    <div className="page" style={{ maxWidth: 720 }}>
      <div style={{ textAlign: "center", marginBottom: "var(--space-lg)" }}>
        <h1 style={{ marginBottom: 6 }}>Your air, your ward</h1>
        <p>
          Find your area to see live air quality, a 72-hour forecast, and to report pollution.
        </p>
      </div>

      {/* Live city AQI band — substance before you even pick a ward */}
      {cityCat && (
        <div
          className="card card-rail"
          style={{
            ["--rail" as string]: cityCat.color,
            display: "flex", alignItems: "center", gap: "var(--space-md)",
            marginBottom: "var(--space-md)", padding: "var(--space-md)",
          }}
        >
          {/* The AQI badge was a circle with a 24px coloured glow around it —
              a glow is a light source, and nothing here emits light. A flat
              tile with the band colour reads as a measurement instead. */}
          <div style={{
            width: 50, height: 50, borderRadius: "var(--radius-md)", flexShrink: 0,
            background: cityCat.color, color: cityCat.textColor,
            display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
          }}>
            <div className="mono" style={{ fontSize: "1.1rem", fontWeight: 600, lineHeight: 1 }}>{cityAqi}</div>
            <div style={{ fontSize: "0.5rem", opacity: 0.8, letterSpacing: "0.08em", marginTop: 2 }}>AQI</div>
          </div>
          <div>
            <div style={{ fontSize: "0.975rem", fontWeight: 550, color: "var(--text-primary)" }}>
              Citywide air is {cityCat.label}
            </div>
            <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
              Median across every ward, live now. Find yours below.
            </div>
          </div>
        </div>
      )}

      {savedWard && (
        <button
          className="btn btn-primary"
          onClick={() => router.push(`/citizen/${savedWard}`)}
          style={{ width: "100%", marginBottom: "var(--space-md)" }}
        >
          <ArrowLeft {...icon.md} aria-hidden />
          Back to my ward
        </button>
      )}

      {/* Primary action: geolocate */}
      {/* The page's primary action, and the only filled button on it. */}
      <button
        className="btn btn-primary"
        onClick={useMyLocation}
        disabled={locating || !hasData}
        style={{ width: "100%", marginBottom: "var(--space-sm)", padding: "11px" }}
      >
        <MapPin {...icon.md} aria-hidden />
        {locating ? "Locating…" : "Use my location"}
      </button>
      {geoError && (
        <div role="alert" className="alert alert-caution" style={{ marginBottom: "var(--space-sm)" }}>
          <TriangleAlert {...icon.md} aria-hidden />
          <div className="alert-body">{geoError}</div>
        </div>
      )}

      {/* Search by ward name — always visible, live-matching */}
      <div style={{ position: "relative", marginBottom: "var(--space-sm)" }}>
        <Search
          {...icon.md}
          aria-hidden
          style={{
            position: "absolute", left: 11, top: "50%", transform: "translateY(-50%)",
            color: "var(--text-tertiary)", pointerEvents: "none",
          }}
        />
        <input
          type="search"
          aria-label="Search your ward by name"
          placeholder="Search your ward by name…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          style={{ paddingLeft: 34, width: "100%" }}
        />
        {searchHits.length > 0 && (
          <div
            role="listbox"
            aria-label="Matching wards"
            className="menu"
            style={{ position: "absolute", top: "calc(100% + 4px)", left: 0, right: 0 }}
          >
            {searchHits.map((w) => (
              <button
                key={w.ward_id}
                role="option"
                aria-selected={false}
                className="menu-item"
                onClick={() => go(w.ward_id)}
              >
                <span className="truncate">{w.ward_name}</span>
                <span className="menu-meta mono">{w.ward_id}</span>
              </button>
            ))}
          </div>
        )}
      </div>

      <p style={{ textAlign: "center", fontSize: "0.8rem", color: "var(--text-tertiary)", margin: "var(--space-sm) 0" }}>
        …or tap your area on the map
      </p>

      {/* Tappable map */}
      <CitizenMap cells={cells} onPickWard={(wardId) => go(wardId)} height={360} />

      {/* Entry point to the multi-layer explorer — a different map, not this
          ward-picker. Kept as a quiet link, not a competing primary button:
          finding YOUR ward is still this page's one job. */}
      <Link
        href="/citizen/explore"
        className="nav-link"
        style={{
          display: "flex", alignItems: "center", justifyContent: "center", gap: 6,
          margin: "var(--space-sm) 0 0", padding: "8px", textDecoration: "none",
          fontSize: "0.8rem",
        }}
      >
        <Layers {...icon.sm} aria-hidden />
        Explore air quality, forecast &amp; cases citywide
        <ArrowRight {...icon.sm} aria-hidden />
      </Link>

      {/* Worst wards right now — live, tappable, fills the space with substance */}
      {worstWards.length > 0 && (
        <div style={{ marginTop: "var(--space-lg)" }}>
          <div className="section-label" style={{ marginBottom: "var(--space-sm)" }}>
            Worst air right now
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))", gap: "var(--space-sm)" }}>
            {worstWards.map((w) => {
              const aqi = pm25ToAqi(w.pm25);
              const cat = getAqiCategory(aqi);
              const name = wardName.get(w.wid) ?? w.wid;
              return (
                <button
                  key={w.wid}
                  onClick={() => go(w.wid)}
                  className="card card-hover"
                  style={{
                    display: "flex", alignItems: "center", gap: 10, padding: "10px 12px",
                    cursor: "pointer", textAlign: "left", fontFamily: "inherit",
                  }}
                >
                  <div className="mono" style={{
                    minWidth: 38, height: 38, borderRadius: "var(--radius-sm)", flexShrink: 0,
                    background: cat.color, color: cat.textColor,
                    display: "flex", alignItems: "center", justifyContent: "center",
                    fontWeight: 600, fontSize: "0.9rem",
                  }}>
                    {aqi}
                  </div>
                  <div style={{ minWidth: 0 }}>
                    <div style={{ fontSize: "0.82rem", fontWeight: 600, color: "var(--text-primary)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {name}
                    </div>
                    <div style={{ fontSize: "0.68rem", color: "var(--text-tertiary)" }}>{cat.label}</div>
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      )}

    </div>
  );
}
