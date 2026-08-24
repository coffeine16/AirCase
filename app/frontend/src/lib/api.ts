/**
 * API client — wraps every fetch with a static JSON fallback.
 * If the FastAPI backend is unreachable, reads from /data/*.json (demo insurance).
 */

// Default to "" (relative paths → fall through to the committed static JSON) so a
// deploy with NO backend is clean: Vercel won't accept an empty env value, so the
// var is simply absent, and absent must mean "no backend", not "localhost". For
// local dev against the backend, set NEXT_PUBLIC_API_URL=http://localhost:8000 in
// .env.local (see .env.example).
const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "";

/**
 * Is a pipeline backend reachable at all?
 *
 * The agent pipeline is a BATCH job (minutes: satellite pulls, model training),
 * not something an HTTP request can run. A static deploy has no backend, so
 * offering a "Run pipeline" button there produces nine red FAILED badges for a
 * request that was never going to work. Read-only surfaces are unaffected — they
 * read the precomputed JSON that the batch run already produced.
 */
export const HAS_BACKEND = Boolean(process.env.NEXT_PUBLIC_API_URL);

/** How long to wait on the live API before serving the static bundle instead.
 *  Read-only contracts are precomputed JSON on both sides, so a slow answer and
 *  no answer are worth the same: whichever arrives first is correct. */
const LIVE_API_TIMEOUT_MS = 4000;

// Citizen reports do NOT go to the read-only FastAPI backend. They go to the n8n
// channel layer (live, HTTPS), which validates, canonicalises the ward against the
// official list, and writes to Supabase — the same table the pipeline syncs down
// for attribution's citizen_corroboration. "Channels are dumb, agents are smart":
// intake is the channel's job, not the serving API's.
const N8N_WEBHOOK_URL =
  process.env.NEXT_PUBLIC_N8N_WEBHOOK_URL ||
  "https://aq-intel.duckdns.org/webhook/citizen-report";

/** Map from API path to the fallback static file in /public/data/ */
const FALLBACK_MAP: Record<string, string> = {
  "/hotspots": "/data/hotspots.json",
  "/attributions": "/data/attributions.json",
  "/fusion": "/data/fusion_field.json",
  "/forecast": "/data/forecast.json",
  "/actions": "/data/actions.json",
  "/dispatch": "/data/dispatch.json",
  "/ledger": "/data/ledger.json",
  "/audit": "/data/audit.json",
  "/wards": "/data/wards.json",
  "/stations": "/data/stations.json",
  "/fires": "/data/fires.json",
  "/loso": "/data/loso.json",
  "/compare": "/data/city_comparison.json",
  "/satellite": "/data/satellite.json",
};

function getFallbackPath(endpoint: string): string | null {
  // Strip query params for matching
  const path = endpoint.split("?")[0];
  return FALLBACK_MAP[path] ?? null;
}

/**
 * City-scoped fetch: reads a contract from that city's static bundle
 * (public/data/<city>/<file>). This is how city-switching works — every map
 * contract lives per-city and needs no backend.
 *
 * A missing file yields the EMPTY fallback, never another city's data. This
 * used to fall through to the flat /data/<file> bundle, which is one city's
 * export: a Chennai console missing one contract would then have rendered
 * Delhi's hotspots under a Chennai label. An empty layer is a visible absence;
 * the wrong city's layer is an invisible lie.
 */
export async function cityFetch<T>(
  city: string,
  file: string,
  fallback: T,
  /** API path serving the SAME contract live, e.g. "/dispatch". Omit for
   *  contracts that only exist as static files. */
  live?: string,
): Promise<T> {
  // Prefer the live API when a backend is configured AND this contract has an
  // endpoint. Without this the console could never show the result of its own
  // work: the agents write to the container's disk, while the map read the
  // static bundle baked into the frontend build at deploy time. A run went
  // green, the data genuinely changed on the server, and the map redrew the
  // identical file — which is exactly how "I changed the team count and nothing
  // happened" looked.
  if (HAS_BACKEND && live) {
    try {
      const res = await fetch(`${API_BASE}${live}?city=${encodeURIComponent(city)}`, {
        next: { revalidate: 0 },
        // FAIL FAST, then fall back. A configured-but-unreachable backend is the
        // dangerous case, not an absent one: an absent backend skips this block
        // entirely, while a dead host that still resolves in DNS can hang the
        // request until the browser's own timeout (tens of seconds) — once per
        // contract, per city switch. The static bundle underneath is complete,
        // so waiting for the corpse buys nothing. Budget is deliberately short:
        // a live backend on the same stage network answers well inside it.
        signal: AbortSignal.timeout(LIVE_API_TIMEOUT_MS),
      });
      if (res.ok) return (await res.json()) as T;
      // A 404 means this city has no pipeline output on the server; the static
      // bundle below may still have it, so fall through rather than fail.
    } catch { /* unreachable OR timed out — the static bundle is the whole point */ }
  }

  try {
    const res = await fetch(`/data/${city}/${file}`, { next: { revalidate: 0 } });
    if (res.ok) return (await res.json()) as T;
  } catch { /* fall through to the empty fallback */ }
  return fallback;
}

export async function apiFetch<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE}${endpoint}`;
  try {
    const res = await fetch(url, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...(options?.headers ?? {}),
      },
      // Don't cache dynamic data
      next: { revalidate: 0 },
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    return res.json() as Promise<T>;
  } catch (err) {
    const fallback = getFallbackPath(endpoint);
    if (fallback) {
      console.warn(`[api] Backend unreachable (${endpoint}), loading fallback: ${fallback}`);
      try {
        const res = await fetch(fallback);
        if (res.ok) return res.json() as Promise<T>;
      } catch {
        // Static fallback also failed
      }
    }
    throw err;
  }
}

// ─── Typed endpoint helpers ────────────────────────────────────────────────────

import type {
  FusionResponse,
  WardsResponse,
  Hotspot,
  Attribution,
  LOSOResponse,
  AuditResponse,
  LedgerEntry,
  Ledger,
  ForecastCell,
  WardForecastPoint,
  Action,
  DispatchRoute,
  DispatchConfig,
  CitizenReport,
  CreateReportPayload,
  PipelineRunResult,
  AgentName,
  CityComparison,
  Memo,
} from "./types";
import type { Station, FireDetection } from "@/hooks/useMapData";

export const api = {
  // ─── City-scoped contracts (drive the multi-city map) ─────────────────────────
  // These read that city's static bundle so switching city needs no backend.
  cityHotspots: (city: string) => cityFetch<Hotspot[]>(city, "hotspots.json", [], "/hotspots"),
  cityFusion: (city: string) => cityFetch<FusionResponse>(city, "fusion_field.json", { ts: "", n_hours: 0, cells: [] }),
  cityWards: (city: string) => cityFetch<WardsResponse>(city, "wards.json", { synthetic: false, n_wards: 0, cells: [] } as unknown as WardsResponse),
  cityStations: (city: string) => cityFetch<Station[]>(city, "stations.json", []),
  cityFires: (city: string) => cityFetch<FireDetection[]>(city, "fires.json", []),
  citySatellite: (city: string) => cityFetch<{ cell: string; no2: number }[]>(city, "satellite.json", []),
  cityAudit: (city: string) => cityFetch<AuditResponse>(city, "audit.json", { blind_spots: [], sensor_flags: [], placement_recommendations: [] }, "/audit"),
  cityDispatch: (city: string) => cityFetch<DispatchRoute[]>(city, "dispatch.json", [], "/dispatch"),
  cityActions: (city: string) => cityFetch<Action[]>(city, "actions.json", [], "/actions"),
  /** Ward-level 3-hourly forecast (24 lead times to +72h), for the citizen
   *  timeline. Ward-scale and medianed, so it is ~1% the size of the cell grid —
   *  the phone downloads kilobytes, not megabytes. */
  cityWardForecast: (city: string) =>
    cityFetch<WardForecastPoint[]>(city, "forecast_ward.json", []),

  /** `h` is a lead time in hours: 3..72 in steps of 3 (was 24|48|72 only). */
  cityForecast: (city: string, h: number) =>
    cityFetch<ForecastCell[]>(city, "forecast.json", []).then((all) => all.filter((f) => f.horizon_h === h)),
  // NO live path on purpose: GET /attributions returns a SUMMARY subset (cell,
  // ward_id, ts, primary_source, confidence, reason) and drops evidence, scores,
  // zone_id and evidence_factors — which the evidence chain and the citizen
  // "why your air is like this" panel both read. The static bundle carries the
  // full record. Wiring this to the API would silently empty those panels.
  cityAttributions: (city: string) => cityFetch<Attribution[]>(city, "attributions.json", []),
  cityMemos: (city: string) => cityFetch<Memo[]>(city, "memos.json", [], "/memos"),
  cityLedger: (city: string) => cityFetch<Ledger | null>(city, "ledger.json", null),
  cityAdvisories: (city: string) => cityFetch<{ ward_id: string; texts?: Record<string, string> }[]>(city, "advisories.json", []),

  // ─── Map data (legacy, single-city via backend) ──────────────────────────────
  getHotspots: () => apiFetch<Hotspot[]>("/hotspots"),
  getAttributions: () => apiFetch<Attribution[]>("/attributions"),
  getAttribution: (cell: string) => apiFetch<Attribution>(`/attribution/${cell}`),
  getFusion: (hourOffset = 0) => apiFetch<FusionResponse>(`/fusion?hour_offset=${hourOffset}`),
  getWards: () => apiFetch<WardsResponse>("/wards"),
  getLoso: () => apiFetch<LOSOResponse>("/loso"),

  // ─── Not-yet-built endpoints (return empty gracefully) ───────────────────────────────────
  // Forecast: FORECAST_WEIGHT = 0.0 until forecast agent is built.
  // severity = clip(base_severity + 0.0 * forecast_delta, 0, 1)
  // A zero we can explain beats a number we invented.
  getForecast: (horizon: 24 | 48 | 72 = 24) =>
    apiFetch<ForecastCell[]>(`/forecast?h=${horizon}`).catch(() => [] as ForecastCell[]),

  // Actions: zone-level (NOT cell-level). ~4–5 zones, not 74 cells.
  // Only attributable:true zones appear here (diffuse excluded from queue).
  getActions: () =>
    apiFetch<Action[]>("/actions").catch(() => [] as Action[]),
  getDispatch: () =>
    apiFetch<DispatchRoute[]>("/dispatch").catch(() => [] as DispatchRoute[]),
  getLedger: () =>
    apiFetch<Ledger>("/ledger").catch(() => null),

  // Multi-city comparison — each row is a full LIVE pipeline run over a real city.
  getComparison: () =>
    apiFetch<CityComparison[]>("/compare").catch(() => [] as CityComparison[]),

  // Which city does the deployed API default to? The container now ships every
  // city's artifacts and `?city=` selects among them, so this is the fallback
  // when no city is named — not a limit. Null when there is no backend.
  getApiHealth: () =>
    apiFetch<{
      ok: boolean;
      city?: string;
      default_city?: string;
      cities_available?: string[];
    }>("/health").catch(() => null),

  // Sentinel-5P NO2 column per cell — the raw signal detection runs on.
  getSatellite: () =>
    apiFetch<{ cell: string; no2: number }[]>("/satellite").catch(() => [] as { cell: string; no2: number }[]),
  getAudit: () =>
    apiFetch<AuditResponse>("/audit").catch(() => ({
      blind_spots: [],
      sensor_flags: [],
      placement_recommendations: [],
    } as AuditResponse)),

  // ─── Agent pipeline ───────────────────────────────────────────────────────────
  runAgent: async (
    agent: AgentName | "all",
    city?: string,
    dispatchConfig?: DispatchConfig,
  ): Promise<PipelineRunResult> => {
    // `city` is not optional in spirit. The API serves whichever city it is asked
    // for and falls back to its own default — so a run without it always targets
    // the DEFAULT city. This parameter was lost once already, and the failure is
    // silent and destructive: running from a Chennai console rewrote Delhi's
    // contracts and looked like it had worked.
    const q = city ? `?city=${encodeURIComponent(city)}` : "";
    const body: Record<string, unknown> = { agent };
    if (dispatchConfig) body.dispatch_config = dispatchConfig;
    return apiFetch<PipelineRunResult>(`/run/agent${q}`, {
      method: "POST",
      body: JSON.stringify(body),
    });
  },

  // ─── Memo ────────────────────────────────────────────────────────────────────
  // GET, not POST: the memo is PRECOMPUTED by the batch pipeline (serving stays
  // read-only). "Generate memo" fetches a document that already exists. Accepts an
  // action_id OR a zone_id — the backend matches either. When the backend is down,
  // fall back to the static memos.json and match client-side (demo insurance).
  // `city` matters on BOTH paths: memos are per-city, so a Chennai zone must not
  // resolve against Delhi's memo file — and the backend defaults to its own city
  // when the query param is absent, which would have returned a Delhi memo for a
  // Chennai zone with no visible error.
  getMemo: async (id: string, city?: string): Promise<Memo> => {
    const local = async () => {
      const memos = city
        ? await cityFetch<Memo[]>(city, "memos.json", [])
        : ((await (await fetch("/data/memos.json")).json()) as Memo[]);
      const memo = memos.find(
        (m) => m.action_id === id || m.zone_id === id || m.memo_id === id
      );
      if (!memo) throw new Error(`no memo for ${id}`);
      return memo;
    };
    const q = city ? `?city=${encodeURIComponent(city)}` : "";
    try {
      return await apiFetch<Memo>(`/memo/${id}${q}`);
    } catch {
      const memo = await local();
      return memo;
    }
  },

  // ─── Citizen ─────────────────────────────────────────────────────────────────
  getWardSummary: (wardId: string) =>
    apiFetch<{ ward_id: string; ward_name: string; aqi: number; pm25: number; advisory?: string }>
      (`/ward/${wardId}/summary`).catch(() => null),

  // Scoped to ONE device. Supabase's anon key can read every row, so the filter
  // has to happen server-side before anything reaches a browser — hence the id
  // is a query param on our own API, not a client-side .filter().
  getReports: (deviceId: string) =>
    apiFetch<CitizenReport[]>(
      `/reports?device_id=${encodeURIComponent(deviceId)}`,
    ).catch(() => [] as CitizenReport[]),

  // Submit a citizen report to the n8n webhook (NOT the FastAPI backend). n8n
  // validates, canonicalises the ward, and writes to Supabase. The webhook returns
  // {ok, status}, so we synthesize a CitizenReport for the optimistic UI update —
  // the authoritative row lives in Supabase and appears after the next sync.
  submitReport: async (payload: CreateReportPayload, deviceId: string): Promise<CitizenReport> => {
    // Frontend uses "construction_dust"; the schema/pipeline category is "construction".
    const category =
      payload.category === "construction_dust" ? "construction" : payload.category;

    // A bare fetch here hangs on the browser's default timeout — tens of seconds
    // of a dead Submit button — whenever the n8n host is unreachable, which is a
    // single self-hosted box and therefore a real possibility. Fail in 8s with a
    // sentence the citizen can act on instead of a spinner that never resolves.
    let res: Response;
    try {
      res = await fetch(N8N_WEBHOOK_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ward_id: payload.ward_id,
          category,
          description: payload.description ?? "",
          lat: payload.lat ?? null,
          lon: payload.lon ?? null,
          device_id: deviceId,
          source: "web",
        }),
        signal: AbortSignal.timeout(8000),
      });
    } catch {
      // Network-level failure (unreachable host, DNS, timeout) — never a status
      // code, so it cannot be distinguished below.
      throw new Error(
        "Could not reach the reporting service. Your report was not sent — " +
        "please try again later."
      );
    }
    if (!res.ok) throw new Error(`Report submission failed (HTTP ${res.status})`);
    const ack = (await res.json().catch(() => ({}))) as { status?: string };

    // Optimistic local record. report_id is provisional until the Supabase sync
    // returns the authoritative row.
    const now = new Date().toISOString();
    return {
      report_id: `local-${Date.now()}`,
      ward_id: payload.ward_id,
      ward_name: payload.ward_id,
      category: payload.category,
      description: payload.description ?? "",
      status: (ack.status as CitizenReport["status"]) ?? "submitted",
      created_at: now,
      updated_at: now,
    };
  },
};
