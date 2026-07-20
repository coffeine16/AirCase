"use client";
/**
 * Resolve a lat/lon (from geolocation or a map tap) to a ward — entirely
 * client-side via H3, no backend round-trip. The wards payload already maps every
 * H3 cell to a ward; latLngToCell at the same resolution gives the cell, and we
 * look it up. If the exact cell isn't in the city (tapped outside the bbox), we
 * fall back to the nearest ward centroid so a tap always resolves to *something*.
 */
import { useMemo } from "react";
import useSWR from "swr";
import { latLngToCell, cellToLatLng, getResolution } from "h3-js";
import { api } from "@/lib/api";
import { useCity } from "@/lib/CityContext";
import type { WardsResponse } from "@/lib/types";

export interface WardHit {
  ward_id: string;
  ward_name: string;
  cell: string;
}

export function useWardLocator() {
  const { city } = useCity();
  const { data, isLoading } = useSWR<WardsResponse>([city, "wards"], () => api.cityWards(city));

  // cell -> {ward_id, ward_name}, plus a flat list of ward centroids for fallback.
  const { cellMap, res, wardCentroids } = useMemo(() => {
    const cellMap = new Map<string, { ward_id: string; ward_name: string }>();
    const acc = new Map<string, { ward_id: string; ward_name: string; latSum: number; lonSum: number; n: number }>();
    let res = 8;
    for (const c of data?.cells ?? []) {
      cellMap.set(c.cell, { ward_id: c.ward_id, ward_name: c.ward_name });
      const [lat, lon] = cellToLatLng(c.cell);
      const e = acc.get(c.ward_id);
      if (e) { e.latSum += lat; e.lonSum += lon; e.n++; }
      else acc.set(c.ward_id, { ward_id: c.ward_id, ward_name: c.ward_name, latSum: lat, lonSum: lon, n: 1 });
      res = getResolution(c.cell);
    }
    const wardCentroids = [...acc.values()].map((w) => ({
      ward_id: w.ward_id, ward_name: w.ward_name, lat: w.latSum / w.n, lon: w.lonSum / w.n,
    }));
    return { cellMap, res, wardCentroids };
  }, [data]);

  /** lat/lon -> the ward containing it, or the nearest ward if outside the grid. */
  const locate = (lat: number, lon: number): WardHit | null => {
    if (!cellMap.size) return null;
    const cell = latLngToCell(lat, lon, res);
    const exact = cellMap.get(cell);
    if (exact) return { ...exact, cell };
    // nearest ward centroid (simple squared-degree distance is fine at city scale)
    let best: WardHit | null = null;
    let bestD = Infinity;
    for (const w of wardCentroids) {
      const d = (w.lat - lat) ** 2 + (w.lon - lon) ** 2;
      if (d < bestD) { bestD = d; best = { ward_id: w.ward_id, ward_name: w.ward_name, cell }; }
    }
    return best;
  };

  return { locate, isLoading, hasData: cellMap.size > 0, wardCentroids };
}
