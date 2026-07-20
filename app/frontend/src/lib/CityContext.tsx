"use client";
/**
 * CityContext — the app's current city, shared across admin + citizen.
 *
 * The whole platform is multi-city (three real live runs). Switching city here
 * moves EVERYTHING: the map re-centres, every contract reloads from that city's
 * static bundle in public/data/<city>/. Persisted so a reload keeps your choice.
 * City data is served static (public/data/<city>/) so switching needs no backend
 * — the same demo-insurance principle the rest of the app follows.
 */
import { createContext, useContext, useEffect, useState, useCallback } from "react";

export const CITIES = [
  { id: "delhi", label: "Delhi" },
  { id: "chennai", label: "Chennai" },
  { id: "bengaluru", label: "Bengaluru" },
] as const;

export type CityId = (typeof CITIES)[number]["id"];
export const DEFAULT_CITY: CityId = "delhi";

interface CityCtx {
  city: CityId;
  cityLabel: string;
  setCity: (c: CityId) => void;
}

const Ctx = createContext<CityCtx>({
  city: DEFAULT_CITY,
  cityLabel: "Delhi",
  setCity: () => {},
});

const KEY = "aq-city";

export function CityProvider({ children }: { children: React.ReactNode }) {
  const [city, setCityState] = useState<CityId>(DEFAULT_CITY);

  useEffect(() => {
    try {
      const saved = localStorage.getItem(KEY) as CityId | null;
      if (saved && CITIES.some((c) => c.id === saved)) setCityState(saved);
    } catch { /* ignore */ }
  }, []);

  const setCity = useCallback((c: CityId) => {
    setCityState(c);
    try { localStorage.setItem(KEY, c); } catch { /* ignore */ }
  }, []);

  const cityLabel = CITIES.find((c) => c.id === city)?.label ?? "Delhi";

  return <Ctx.Provider value={{ city, cityLabel, setCity }}>{children}</Ctx.Provider>;
}

export const useCity = () => useContext(Ctx);
