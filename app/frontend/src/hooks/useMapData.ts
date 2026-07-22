"use client";
/**
 * useMapData — generic SWR hooks for all secondary map layers.
 * Satellite, wind, and stations are not-yet-built endpoints;
 * they fall back gracefully to empty arrays / stubs.
 *
 * ⚠️ These hooks are CITY-BLIND and currently unmounted — the console reads the
 * per-city bundles via `api.city*(city)` instead. Do not wire one of these into
 * a page as-is: it would render the API's default city's data underneath
 * whatever the city switcher says. Take a `city` argument first, key the SWR
 * cache on it, and call the matching `api.city*` getter.
 * (Kept because api.ts imports the Station / FireDetection contracts below.)
 */
import useSWR from "swr";
import { api, apiFetch } from "@/lib/api";
import type { WardsResponse, AuditResponse, DispatchRoute, Action } from "@/lib/types";

// ── Wards ──────────────────────────────────────────────────────────────────────
export function useWards() {
  const { data, error, isLoading } = useSWR<WardsResponse>(
    "wards",
    () => api.getWards(),
    { revalidateOnFocus: false, dedupingInterval: 600_000 }
  );
  return { wards: data, cells: data?.cells ?? [], error, isLoading };
}

// ── Stations ──────────────────────────────────────────────────────────────────
export interface Station {
  cell: string;
  ward_id: string;
  station_name: string;
  lat: number;
  lon: number;
  pm25: number;
  pm10?: number;
  no2?: number;
  freshness_h: number;   // hours since last reading
}

export function useStations() {
  const { data, error, isLoading } = useSWR<Station[]>(
    "stations",
    () => apiFetch<Station[]>("/stations").catch(() => [] as Station[]),
    { refreshInterval: 300_000, revalidateOnFocus: false }
  );
  return { stations: data ?? [], error, isLoading };
}

// ── Fires ──────────────────────────────────────────────────────────────────────
/**
 * A FIRMS thermal anomaly, as the pipeline actually writes it.
 *
 * This interface used to claim `confidence: number` and `acquired_at`. Neither
 * exists: FIRMS reports confidence as a LETTER CODE and the contract's time
 * field is `ts`. TypeScript could not catch the lie because the rows arrive as
 * runtime JSON, so the popup rendered `"n" * 100` as **NaN%** and
 * `new Date(undefined)` as **NaN h ago**. Keep this shape pinned to the file.
 */
export interface FireDetection {
  cell?: string;
  lat: number;
  lon: number;
  frp: number;                    // fire radiative power (MW)
  confidence: "l" | "n" | "h";    // FIRMS: low / nominal / high
  ts: string;                     // ISO timestamp of the satellite overpass
}

/** FIRMS confidence letter -> label + a weight for styling. */
export const FIRE_CONFIDENCE: Record<string, { label: string; weight: number }> = {
  l: { label: "Low", weight: 0.3 },
  n: { label: "Nominal", weight: 0.6 },
  h: { label: "High", weight: 0.9 },
};

export function useFires() {
  const { data, error, isLoading } = useSWR<FireDetection[]>(
    "fires",
    () => apiFetch<FireDetection[]>("/fires").catch(() => [] as FireDetection[]),
    { refreshInterval: 600_000, revalidateOnFocus: false }
  );
  return { fires: data ?? [], error, isLoading };
}

// ── Satellite (Sentinel-5P NO2 column) ───────────────────────────────────────
export function useSatellite() {
  const { data, error, isLoading } = useSWR<{ cell: string; no2: number }[]>(
    "satellite",
    () => api.getSatellite(),
    { revalidateOnFocus: false }
  );
  return { satellite: data ?? [], error, isLoading };
}

// ── Audit (blind spots + sensor flags) ───────────────────────────────────────
export function useAudit() {
  const { data, error, isLoading } = useSWR<AuditResponse>(
    "audit",
    () => api.getAudit(),
    { revalidateOnFocus: false }
  );
  return {
    blindSpots: data?.blind_spots ?? [],
    sensorFlags: data?.sensor_flags ?? [],
    recommendations: data?.placement_recommendations ?? [],
    error,
    isLoading,
  };
}

// ── Dispatch routes ───────────────────────────────────────────────────────────
export function useDispatch() {
  const { data, error, isLoading } = useSWR<DispatchRoute[]>(
    "dispatch",
    () => api.getDispatch(),
    { revalidateOnFocus: false }
  );
  return { routes: data ?? [], error, isLoading };
}

// ── Actions (zone-level EPS queue) ───────────────────────────────────────────
export function useActions() {
  const { data, error, isLoading, mutate } = useSWR<Action[]>(
    "actions",
    () => api.getActions(),
    { revalidateOnFocus: false }
  );
  return { actions: data ?? [], error, isLoading, refresh: mutate };
}
