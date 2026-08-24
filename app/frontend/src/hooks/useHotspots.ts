"use client";
import useSWR from "swr";
import { api } from "@/lib/api";
import type { Hotspot } from "@/lib/types";
import type { MapFilters } from "@/lib/types";

/**
 * Apply the map filters to the hotspot cells.
 *
 * `sourceByZone` maps zone_id -> attributed source, and is REQUIRED for the
 * Source Type chips to do anything: a Hotspot row carries no `primary_source`.
 * Source is a property of the ZONE (one inspector, one case, one attributed
 * cause), and lives in attributions.json — so the filter is a join, not a field
 * lookup. Without the map passed in, the source chips are silently inert, which
 * is exactly how they shipped: they highlighted, the active-filter count went
 * up, and the map did not change.
 *
 * Cells in a zone with no attribution (the diffuse ones) are excluded whenever
 * a source filter is active. That is the honest reading of "show me waste
 * burning": a zone we could not attribute is not a zone of that source, and
 * showing it would put unexplained cells under a specific accusation.
 */
export function filterHotspots(
  hotspots: Hotspot[],
  filters: Partial<MapFilters>,
  sourceByZone?: Map<string, string>,
): Hotspot[] {
  return hotspots.filter((h) => {
    if (filters.ward_ids?.length && !filters.ward_ids.includes(h.ward_id)) return false;
    if (filters.persistence_types?.length && !filters.persistence_types.includes(h.kind)) return false;
    if (filters.attributable_only && !h.attributable) return false;
    if (filters.source_types?.length) {
      const src = sourceByZone?.get(h.zone_id);
      if (!src || !filters.source_types.includes(src as never)) return false;
    }
    return true;
  });
}

export function useHotspots(filters?: Partial<MapFilters>) {
  const { data, error, isLoading, mutate } = useSWR<Hotspot[]>(
    "hotspots",
    () => api.getHotspots(),
    { refreshInterval: 60_000, revalidateOnFocus: false }
  );

  const filtered = data && filters ? filterHotspots(data, filters) : (data ?? []);
  return { hotspots: filtered, raw: data ?? [], error, isLoading, refresh: mutate };
}
