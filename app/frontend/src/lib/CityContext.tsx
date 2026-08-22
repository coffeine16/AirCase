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
import { createContext, useContext, useEffect, useLayoutEffect, useState, useCallback } from "react";

/**
 * useLayoutEffect on the client, useEffect on the server.
 *
 * A "use client" component is still server-rendered for the initial HTML, and
 * React warns that useLayoutEffect does nothing there. The restore below only
 * needs to beat the first PAINT, which is a client-only concern — so use the
 * layout effect where it exists and fall back where it doesn't.
 */
const useIsomorphicLayoutEffect =
  typeof window !== "undefined" ? useLayoutEffect : useEffect;

// The three the product was built and tuned on come first; the five added with
// the 8-city forecast model follow. Order is the order of the city switcher.
export const CITIES = [
  { id: "delhi", label: "Delhi" },
  { id: "chennai", label: "Chennai" },
  { id: "bengaluru", label: "Bengaluru" },
  { id: "mumbai", label: "Mumbai" },
  { id: "kolkata", label: "Kolkata" },
  { id: "hyderabad", label: "Hyderabad" },
  { id: "pune", label: "Pune" },
  { id: "ahmedabad", label: "Ahmedabad" },
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

/** The saved city, or the default. Safe to call during render on the client. */
function savedCity(): CityId {
  try {
    const saved = localStorage.getItem(KEY) as CityId | null;
    if (saved && CITIES.some((c) => c.id === saved)) return saved;
  } catch { /* localStorage unavailable — fall through */ }
  return DEFAULT_CITY;
}

export function CityProvider({ children }: { children: React.ReactNode }) {
  const [city, setCityState] = useState<CityId>(DEFAULT_CITY);

  /**
   * Restore in a LAYOUT effect, not a passive one.
   *
   * The city has to start as DEFAULT_CITY: this renders on the server too, and
   * reading localStorage during render would make the server and client markup
   * disagree — a hydration error. But with a passive useEffect the restore
   * landed a paint LATE, so the map mounted with Delhi's viewport, then the city
   * flipped to (say) Bengaluru underneath it. The map's recentre effect only
   * fires once data exists, so on a cold load the viewport stayed 1,700 km from
   * the hexagons and the map looked empty until you switched city by hand.
   *
   * useLayoutEffect runs before the browser paints, so the corrected city is in
   * place for the first visible frame and the map never renders over the wrong
   * city. Hydration still matches, because the first render is still the default.
   */
  useIsomorphicLayoutEffect(() => {
    const saved = savedCity();
    if (saved !== city) setCityState(saved);
    // Deliberately once, on mount: this restores a preference, it does not track
    // it. Re-running on `city` would fight the user's own switch.
  }, []);

  const setCity = useCallback((c: CityId) => {
    setCityState(c);
    try { localStorage.setItem(KEY, c); } catch { /* ignore */ }
  }, []);

  const cityLabel = CITIES.find((c) => c.id === city)?.label ?? "Delhi";

  return <Ctx.Provider value={{ city, cityLabel, setCity }}>{children}</Ctx.Provider>;
}

export const useCity = () => useContext(Ctx);
