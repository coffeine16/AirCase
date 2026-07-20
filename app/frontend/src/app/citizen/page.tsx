"use client";
/**
 * Citizen home — find YOUR ward, no scrolling.
 *   1. "Use my location" -> geolocate -> resolve to ward via H3 (client-side).
 *   2. or tap the map on your area.
 *   3. or type to search (kept as a fallback, collapsed by default).
 * The 227-ward scroll grid is gone: a citizen should land on their map, not hunt
 * a list.
 */
import { useState, useMemo } from "react";
import { useRouter } from "next/navigation";
import dynamic from "next/dynamic";
import useSWR from "swr";
import { api } from "@/lib/api";
import { useCitizenWard } from "@/hooks/useReports";
import { useWardLocator } from "@/hooks/useWardLocator";
import { useCity } from "@/lib/CityContext";
import type { FusionResponse } from "@/lib/types";

const CitizenMap = dynamic(() => import("@/components/citizen/CitizenMap"), {
  ssr: false,
  loading: () => <div className="skeleton" style={{ height: 340, borderRadius: "var(--radius-lg)" }} />,
});

export default function CitizenHomePage() {
  const router = useRouter();
  const { city } = useCity();
  const { wardId: savedWard, setWardId } = useCitizenWard();
  const { locate, hasData } = useWardLocator();

  const { data: fusion } = useSWR<FusionResponse>([city, "fusion"], () => api.cityFusion(city));
  const cells = useMemo(() => fusion?.cells ?? [], [fusion]);

  const [locating, setLocating] = useState(false);
  const [geoError, setGeoError] = useState<string | null>(null);
  const [showSearch, setShowSearch] = useState(false);
  const [search, setSearch] = useState("");

  const go = (wardId: string) => {
    setWardId(wardId);
    router.push(`/citizen/${wardId}`);
  };

  const useMyLocation = () => {
    setGeoError(null);
    if (!("geolocation" in navigator)) {
      setGeoError("Your browser doesn't support location. Tap the map instead.");
      return;
    }
    setLocating(true);
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setLocating(false);
        const hit = locate(pos.coords.latitude, pos.coords.longitude);
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

  // search fallback (collapsed)
  const wards = useMemo(() => {
    const seen = new Set<string>();
    return cells
      .filter((c) => c.ward_id && !seen.has(c.ward_id) && seen.add(c.ward_id))
      .map((c) => ({ ward_id: c.ward_id!, ward_name: (c as { ward_name?: string }).ward_name ?? c.ward_id! }));
  }, [cells]);
  const searchHits = useMemo(
    () => (search.trim() ? wards.filter((w) => w.ward_name.toLowerCase().includes(search.toLowerCase())).slice(0, 8) : []),
    [wards, search]
  );

  return (
    <div style={{ padding: "var(--space-xl)", maxWidth: 720, margin: "0 auto", width: "100%" }}>
      <div style={{ textAlign: "center", marginBottom: "var(--space-lg)" }}>
        <h1 style={{ marginBottom: 8 }}>Your air, your ward</h1>
        <p style={{ color: "var(--text-secondary)" }}>
          Find your area to see live air quality, a 72-hour forecast, and to report pollution.
        </p>
      </div>

      {savedWard && (
        <button
          className="btn btn-primary"
          onClick={() => router.push(`/citizen/${savedWard}`)}
          style={{ width: "100%", justifyContent: "center", marginBottom: "var(--space-md)" }}
        >
          ↩ Back to my ward
        </button>
      )}

      {/* Primary action: geolocate */}
      <button
        className="btn btn-emerald"
        onClick={useMyLocation}
        disabled={locating || !hasData}
        style={{ width: "100%", justifyContent: "center", marginBottom: "var(--space-sm)", padding: "var(--space-md)" }}
      >
        {locating ? "Locating…" : "📍 Use my location"}
      </button>
      {geoError && (
        <p style={{ fontSize: "0.8rem", color: "var(--accent-amber)", textAlign: "center", marginBottom: "var(--space-sm)" }}>
          {geoError}
        </p>
      )}

      <p style={{ textAlign: "center", fontSize: "0.8rem", color: "var(--text-tertiary)", margin: "var(--space-sm) 0" }}>
        …or tap your area on the map
      </p>

      {/* Tappable map */}
      <CitizenMap cells={cells} onPickWard={(wardId) => go(wardId)} height={360} />

      {/* Search fallback (collapsed) */}
      <div style={{ marginTop: "var(--space-md)", textAlign: "center" }}>
        <button className="btn btn-ghost btn-sm" onClick={() => setShowSearch((s) => !s)}>
          {showSearch ? "Hide search" : "Prefer to type? Search by name"}
        </button>
        {showSearch && (
          <div style={{ marginTop: "var(--space-sm)", textAlign: "left" }}>
            <input
              type="search"
              placeholder="Type your ward or locality…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              autoFocus
            />
            {searchHits.length > 0 && (
              <div style={{ marginTop: 6, display: "flex", flexDirection: "column", gap: 4 }}>
                {searchHits.map((w) => (
                  <button
                    key={w.ward_id}
                    className="card card-hover"
                    onClick={() => go(w.ward_id)}
                    style={{ textAlign: "left", padding: "8px 12px", cursor: "pointer", fontSize: "0.88rem" }}
                  >
                    {w.ward_name}
                    <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.72rem", color: "var(--text-tertiary)", marginLeft: 8 }}>
                      {w.ward_id}
                    </span>
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
