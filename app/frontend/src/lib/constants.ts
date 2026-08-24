import type { LucideIcon } from "lucide-react";
import {
  Building2, ClipboardList, Construction, Crosshair, Eye, FileText, Flame,
  Gauge, Layers, MapPin, Megaphone, RadioTower, Route, SatelliteDish,
  ScanLine, Target, TrendingUp, TriangleAlert, Truck, Volume2,
} from "./../components/Icon";
import type { AgentName, HotspotKind, SourceCategory, ReportCategory, LayerId, ActionStatus } from "./types";

// ─── Map ──────────────────────────────────────────────────────────────────────

/**
 * Opening viewport per city.
 *
 * This used to be a single hardcoded Delhi position. The map therefore always
 * opened over Delhi, and when the saved city was Chennai or Bengaluru the
 * viewport sat ~1,700 km from the data — hexagons rendered off-screen and the
 * map looked empty until you switched city, which forced a recentre. It only
 * showed up once localStorage held a non-Delhi city, which is why it appeared
 * to be intermittent.
 *
 * The recentre effect in MapContainer still fits to the loaded data; this makes
 * the FIRST frame land in the right place instead of relying on that.
 */
export const CITY_VIEW_STATE: Record<string, { longitude: number; latitude: number }> = {
  delhi:     { longitude: 77.15, latitude: 28.60 },
  chennai:   { longitude: 80.24, latitude: 13.05 },
  bengaluru: { longitude: 77.59, latitude: 12.97 },
  // Each is the centre of that city's shared/config.py BBOX, so the opening
  // frame matches the extent the pipeline actually generated data for.
  mumbai:    { longitude: 72.88, latitude: 19.08 },
  kolkata:   { longitude: 88.37, latitude: 22.55 },
  hyderabad: { longitude: 78.47, latitude: 17.43 },
  pune:      { longitude: 73.85, latitude: 18.55 },
  ahmedabad: { longitude: 72.60, latitude: 23.05 },
};

export const INITIAL_VIEW_STATE = {
  ...CITY_VIEW_STATE.delhi,
  zoom: 11,
  pitch: 30,
  bearing: 0,
} as const;

/** Opening viewport for a given city, falling back to the default. */
export function initialViewFor(city: string) {
  return { ...INITIAL_VIEW_STATE, ...(CITY_VIEW_STATE[city] ?? {}) };
}

/**
 * Which of our cities is this coordinate in — or nearest to?
 *
 * Ward lookup is per-city: it searches the wards of whichever city is currently
 * selected. So a citizen in Hyderabad opening the app on the Delhi default was
 * told "couldn't match your location to a ward" — not because we lack their
 * data, but because we were looking in the wrong city's ward list. That reads
 * as "this app doesn't cover me" when in fact it does.
 *
 * Great-circle distance to each city centre, nearest wins, and the answer is
 * rejected beyond MAX_CITY_DISTANCE_KM — someone in Kochi should be told we do
 * not cover them yet, not silently handed Chennai's air as if it were theirs.
 * A wrong city is worse than no city: every number on the page would be real,
 * and about somewhere else.
 */
export const MAX_CITY_DISTANCE_KM = 120;

export function nearestCity(
  lat: number,
  lon: number,
): { city: string; km: number } | null {
  const R = 6371;
  const rad = Math.PI / 180;
  let best: { city: string; km: number } | null = null;

  for (const [city, c] of Object.entries(CITY_VIEW_STATE)) {
    const dLat = (c.latitude - lat) * rad;
    const dLon = (c.longitude - lon) * rad;
    const a =
      Math.sin(dLat / 2) ** 2 +
      Math.cos(lat * rad) * Math.cos(c.latitude * rad) * Math.sin(dLon / 2) ** 2;
    const km = 2 * R * Math.asin(Math.min(1, Math.sqrt(a)));
    if (!best || km < best.km) best = { city, km };
  }

  return best && best.km <= MAX_CITY_DISTANCE_KM ? best : null;
}

/** Carto dark tiles (no token required) */
export const MAP_STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";

/** Delhi bounding box */
export const CITY_BBOX = {
  lat_min: 28.45, lat_max: 28.75,
  lon_min: 76.95, lon_max: 77.35,
} as const;

// ─── Layers ───────────────────────────────────────────────────────────────────

export const LAYER_LABELS: Record<LayerId, string> = {
  fusion:     "Fusion PM2.5",
  stations:   "Stations",
  satellite:  "Satellite",
  hotspots:   "Hotspots",
  fires:      "Fires",
  wards:      "Ward Boundaries",
  blindspots: "Blind Spots",
  dispatch:   "Dispatch Routes",
};

export const LAYER_ICONS: Record<LayerId, LucideIcon> = {
  fusion:     Gauge,
  stations:   MapPin,
  satellite:  SatelliteDish,
  hotspots:   Target,
  fires:      Flame,
  wards:      Layers,
  blindspots: Eye,
  dispatch:   Route,
};

export const DEFAULT_LAYERS: Record<LayerId, boolean> = {
  fusion:     true,
  stations:   true,
  satellite:  false,
  hotspots:   true,
  fires:      false,
  wards:      true,
  blindspots: false,
  dispatch:   false,
};

// ─── Agents ───────────────────────────────────────────────────────────────────

export const AGENT_LABELS: Record<AgentName, string> = {
  detection:      "Detection",
  attribution:    "Attribution",
  forecast:       "Forecast",
  prioritisation: "Prioritisation",
  memo:           "Enforcement Memo",
  advisory:       "Advisory",
  voice:          "Voice",
  ledger:         "Ledger",
  audit:          "Network Audit",
};

export const AGENT_ICONS: Record<AgentName, LucideIcon> = {
  detection:      ScanLine,
  attribution:    Crosshair,
  forecast:       TrendingUp,
  prioritisation: ClipboardList,
  memo:           FileText,
  advisory:       Megaphone,
  voice:          Volume2,
  ledger:         Layers,
  audit:          RadioTower,
};

export const AGENT_DESCRIPTIONS: Record<AgentName, string> = {
  detection:      "Satellite contrast + fire persistence → chronic / emerging / acute zones",
  attribution:    "Names the responsible source with evidence chain + confidence",
  // NB: describe what is BUILT, not what the architecture doc imagined. The STGCN
  // was cut; the forecast is gradient boosting on lags + met vs a persistence baseline.
  forecast:       "24–72h PM2.5 forecast per cell (gradient boosting vs persistence baseline)",
  prioritisation: "Enforcement Priority Score + inspection route assignment",
  memo:           "Generates dispatch-ready enforcement memo with legal citation",
  advisory:       "Ward-level health advisories in Kannada / Hindi / Tamil / English",
  voice:          "Synthesizes the highest-risk ward advisories to speech (Cloud TTS) for IVR / voice-note delivery",
  ledger:         "Freezes the counterfactual forecast per action; tracks signal→memo→dispatch response chain",
  audit:          "Audits the monitor network: dirty unmonitored cells → next-sensor placement + tamper flags",
};

export const AGENT_ORDER: AgentName[] = [
  "detection",
  "attribution",
  "forecast",
  "prioritisation",
  "memo",
  "advisory",
  "voice",
  "ledger",
  "audit",
];

// ─── Hotspot / Source ─────────────────────────────────────────────────────────

export const PERSISTENCE_LABELS: Record<HotspotKind, string> = {
  chronic:  "Chronic (30d)",
  emerging: "Emerging (7d)",
  acute:    "Acute (24h)",
};

export const PERSISTENCE_DESCRIPTIONS: Record<HotspotKind, string> = {
  chronic:  "Elevated over 30 days — build the case file",
  emerging: "Elevated over 7 days, not 30 — act now",
  acute:    "Elevated in 24h only — send a truck, not a notice",
};

export const SOURCE_LABELS: Record<SourceCategory, string> = {
  industrial:    "Industrial",
  waste_burning: "Waste Burning",
  construction:  "Construction",
  traffic:       "Traffic",
};

// ─── Citizen reports ──────────────────────────────────────────────────────────

export const REPORT_CATEGORY_LABELS: Record<ReportCategory, string> = {
  waste_burning:    "Waste / Garbage Burning",
  construction_dust: "Construction Dust",
  industrial:       "Industrial Smoke",
  traffic:          "Vehicle Exhaust",
  other:            "Other",
};

export const REPORT_CATEGORY_ICONS: Record<ReportCategory, LucideIcon> = {
  waste_burning:     Flame,
  construction_dust: Construction,
  industrial:        Building2,
  traffic:           Truck,
  other:             TriangleAlert,
};

export const REPORT_STATUS_LABELS: Record<string, string> = {
  submitted:    "Submitted",
  under_review: "Under Review",
  corroborated: "Corroborated",
  action_taken: "Action Taken",
  resolved:     "Resolved",
};

/** Badge variant per status — resolves to a design token, never a raw hex, so
 *  the five statuses stay inside the app's three semantic colours instead of
 *  spanning grey→blue→amber→emerald→green as five unrelated hues. */
export const REPORT_STATUS_BADGE: Record<string, string> = {
  submitted:    "badge-diffuse",
  under_review: "badge-accent",
  corroborated: "badge-caution",
  action_taken: "badge-positive",
  resolved:     "badge-positive",
};

// ─── Satellite ────────────────────────────────────────────────────────────────

export const SAT_CHANNEL_LABELS = {
  no2_col: "NO₂ Column",
  so2_col: "SO₂ Column",
  aai:     "Aerosol Index",
} as const;

// ─── AQI Category health advice ───────────────────────────────────────────────

export const AQI_ADVICE = [
  "Air quality is Good. Great day to be outside!",
  "Air quality is Satisfactory. Unusually sensitive people should consider limiting prolonged exertion.",
  "Air quality is Moderate. Reduce prolonged or heavy exertion. Take more breaks during outdoor activities.",
  "Air quality is Poor. People with heart or lung disease, the elderly and children should avoid prolonged exertion.",
  "Air quality is Very Poor. Everyone should avoid prolonged exertion. Sensitive groups: avoid all outdoor activity.",
  "Air quality is Severe. Everyone should avoid any outdoor exertion; sensitive groups should remain indoors.",
];

// ─── Enforcement action status (citizen-facing) ────────────────────────────────

/** Plain-language status, for the citizen surfaces.
 *
 *  Deliberately NOT the admin vocabulary. `pending`/`dispatched`/`actioned` are
 *  internal queue states; a resident reading a map of their own city needs to
 *  know whether anyone is coming, not what our scheduler calls the row. */
export const ACTION_STATUS_LABELS: Record<ActionStatus, string> = {
  pending:    "Awaiting inspection",
  dispatched: "Inspector assigned",
  actioned:   "Action taken",
  resolved:   "Resolved",
};

/** Badge variant per status — a design token, never a raw hex, so the four
 *  statuses stay inside the app's semantic colours instead of becoming four
 *  unrelated hues. Mirrors REPORT_STATUS_BADGE's progression above. */
export const ACTION_STATUS_BADGE: Record<ActionStatus, string> = {
  pending:    "badge-diffuse",
  dispatched: "badge-accent",
  actioned:   "badge-positive",
  resolved:   "badge-positive",
};

/** Attribution confidence as words, not a bare percentage.
 *
 *  The thresholds are the ones `scripts/eval_attribution.py` actually validated
 *  against truth — 100% precision above 0.70, and a median of 0.66 on hits vs
 *  0.42 on misses — so "High confidence" here means something measured, not a
 *  designer's guess at where a number starts sounding good. A citizen reading
 *  "67%" has no way to know whether that is strong or weak; this does. */
export function confidenceLabel(confidence: number): string {
  if (!Number.isFinite(confidence)) return "Not assessed";
  if (confidence >= 0.70) return "High confidence";
  if (confidence >= 0.50) return "Likely";
  return "Under review";
}
