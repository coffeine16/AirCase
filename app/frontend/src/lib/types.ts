/** TypeScript interfaces matching all JSON contracts from the AirCase backend.
 * Aligned with data contracts defined in docs/architecture.md and agent outputs.
 */

// ─── Spatial ──────────────────────────────────────────────────────────────────

export interface Ward {
  cell: string;
  ward_id: string;
  ward_name: string;
}

export interface WardsResponse {
  synthetic: boolean;
  n_wards: number;
  cells: Ward[];
}

// ─── Fusion Field ─────────────────────────────────────────────────────────────

export interface FusionCell {
  // ⚠ The field is `pm25`, NOT `pm25_hat`. No prediction interval is computed.
  // Source: architecture.md data contracts ("pm25, NOT pm25_hat").
  cell: string;
  ward_id: string;
  pm25: number;
}

export interface FusionResponse {
  ts: string;
  n_hours: number;
  cells: FusionCell[];
}

// ─── Hotspots ─────────────────────────────────────────────────────────────────

export type HotspotKind = "chronic" | "emerging" | "acute";
export type SourceCategory = "industrial" | "construction" | "waste_burning" | "traffic";

export interface Hotspot {
  cell: string;
  ward_id: string;
  ward_name: string;
  zone_id: string;
  kind: HotspotKind;
  cell_kind: string;
  attributable: boolean;
  severity: number;       // [0, 1]
  pm25_med: number;
  nearest_candidate_km: number;
  z_w24h: number;
  z_w7d: number;
  z_w30d: number;
  fires_6h: number;
  detection_basis: string;
  ts: string;
}

// ─── Attribution ──────────────────────────────────────────────────────────────

export interface SourceCandidate {
  name: string;
  type: SourceCategory;
  distance_km: number;
  wind_alignment: number;
}

export interface PollutantReading {
  value: number;
  city_percentile: number;
}

export interface FireActivity {
  fire_hours: number;
  fire_hour_fraction: number;
  frp_p90: number;
}

export interface Meteorology {
  wind_from_deg: number;
  wind_ms: number;
  blh_m: number;
  hour_local: number;
  air_trapped: boolean;
}

export interface EvidenceProfile {
  cell: string;
  ts: string;
  ward_id: string;
  pm25_estimate: number;
  candidates: SourceCandidate[];
  pollutant_signature: Record<string, PollutantReading>;
  fire_activity: FireActivity;
  landuse_context: Record<string, number>;
  meteorology: Meteorology;
  hotspot_kind: HotspotKind;
  evidence_window_hours: number;
}

export interface Attribution {
  cell: string;
  ts: string;
  ward_id: string;
  pm25_estimate: number;
  primary_source: SourceCategory;
  confidence: number;     // [0, 1]
  scores: Record<SourceCategory, number>;
  reason: string;
  evidence_factors: string[];
  evidence: EvidenceProfile;
  explained_by: string;
  zone_id: string;
}

// ─── Action Queue (EPS) ───────────────────────────────────────────────────────
// An ACTION is a ZONE, not a cell. ~5 zones, not 74 cells.
// `legal_basis` is deliberately absent — that belongs to the memo agent.
// Source: eps-spec.md output contract + architecture.md data contracts.

export type ActionStatus = "pending" | "dispatched" | "actioned" | "resolved";

export interface EPSComponents {
  severity: number;           // [0,1] max across zone's cells
  attribution_conf: number;   // [0,1] max over zone's cells
  actionability: number;      // [0,1] base × kind_weight × locatable
  vulnerability: number;      // [0,1] min(school+hospital count / 5, 1.0)
}

export interface ZoneCentroid {
  lat: number;
  lon: number;
}

export interface Action {
  action_id: string;
  zone_id: string;            // e.g. "Z00" — group key from hotspots.json
  ward_id: string;
  ward_name: string;
  cells: string[];            // all H3 cells belonging to this zone
  centroid: ZoneCentroid;     // lat/lon for map pin and dispatch routing
  eps: number;                // 0–100
  components: EPSComponents;
  kind: HotspotKind;          // chronic | emerging | acute
  source: SourceCategory;     // from attribution
  confidence: number;         // attribution confidence [0,1]
  n_cells: number;            // zone size
  pm25_med: number;           // median PM2.5 across zone cells
  status: ActionStatus;
  // NOTE: forecast_delta is 0 until the forecast agent is built.
  // severity = clip(base_severity + FORECAST_WEIGHT * forecast_delta, 0, 1)
  // FORECAST_WEIGHT = 0.0 for now — a zero we can explain beats a number we invented.
}

// ─── Dispatch ─────────────────────────────────────────────────────────────────

export interface DispatchStop {
  seq: number;
  action_id: string;   // links back to actions.json
  zone_id: string;
  ward_id: string;
  eps: number;
  lat: number;         // centroid lat — frontend draws route without resolving cells
  lon: number;         // centroid lon
}

export interface DispatchRoute {
  team_id: string;
  route_km: number;
  route_duration_min: number;        // estimated travel time (from OSRM or fallback)
  coverage_pct: number;
  stops: DispatchStop[];
  route_geometry: [number, number][]; // [lon, lat] pairs — actual road path from OSRM
}

/** Configuration passed to the prioritisation agent for dispatch routing. */
export interface DispatchConfig {
  n_teams: number;       // number of inspection teams (default 4)
  stop_budget: number;   // max total stops across all teams (default 10)
}

// ─── Forecast ─────────────────────────────────────────────────────────────────

/**
 * The forecaster is a QUANTILE model: it predicts a distribution, not a point.
 * `pm25_hat` is the median (p50); p10/p90 are the 80% prediction interval.
 *
 * Optional because a bundle generated before the quantile model was promoted
 * carries only pm25_hat — every consumer must degrade to the point estimate
 * rather than render NaN.
 */
export interface ForecastCell {
  cell: string;
  horizon_h: number;      // 3-hourly: 3, 6, ... 72
  pm25_hat: number;       // p50 — the median prediction
  pm25_p10?: number;      // lower bound of the 80% interval
  pm25_p90?: number;      // upper bound
}

/** One ward, one lead time. The citizen timeline's unit. */
export interface WardForecastPoint {
  ward_id: string;
  horizon_h: number;      // 3-hourly: 3, 6, ... 72
  pm25_hat: number;       // MEDIAN across the ward's cells (p50)
  pm25_p10?: number;      // 80% interval, medianed across the ward's cells
  pm25_p90?: number;      // optional: bundles predating the quantile model omit both
  n_cells: number;
}

// ─── Ledger ───────────────────────────────────────────────────────────────────

export interface LedgerEntry {
  action_id: string;
  zone_id: string;
  ward_id: string;
  ward_name?: string;
  source?: string;
  eps?: number;
  response: {
    signal_at?: string | null;
    memo_drafted_at?: string | null;
    dispatched_at?: string | null;
    actioned_at?: string | null;
    response_hours?: number | null;   // dispatch -> actioned, only once the inspector loop reports
    manual_baseline?: string;
    automated?: string;
  };
  counterfactual: {
    horizon_h: number;
    pm25_counterfactual: number;
    aqi_counterfactual: number;
    band_counterfactual: string;
    frozen_at?: string | null;
  } | null;
  // deliberately null until a real intervention sits between forecast and outcome
  observed_change: number | null;
  our_impact: number | null;
  status: "actioned" | "dispatched" | "awaiting_outcome";
}

// The /ledger endpoint returns this envelope, not a bare array.
export interface Ledger {
  generated_at: string;
  response_time_claim: string;
  effectiveness_claim: string;
  n_actions: number;
  n_actioned: number;
  counterfactual_horizon_h: number;
  entries: LedgerEntry[];
}

// ─── Audit ────────────────────────────────────────────────────────────────────

export interface BlindSpot {
  cell: string;
  ward_id: string;
  satellite_signal: number;
  rank: number;
}

export interface SensorFlag {
  cell: string;
  ward_id: string;
  station_name: string;
  reason: "flat_while_satellite_spikes" | "no_data";
}

export interface AuditResponse {
  blind_spots: BlindSpot[];
  sensor_flags: SensorFlag[];
  placement_recommendations: string[];
}

// ─── LOSO Validation ──────────────────────────────────────────────────────────

export interface LOSOStation {
  rmse: number;
  r2: number;
  n: number;
}

export interface LOSOOverall {
  rmse: number;
  r2: number;
  n_stations: number;
  mean_pm25: number;
  naive_citymean_rmse: number;
}

export interface LOSOResponse {
  overall: LOSOOverall;
  per_station: Record<string, LOSOStation>;
}

// ─── Citizen Reports ──────────────────────────────────────────────────────────

export type ReportCategory = "waste_burning" | "construction_dust" | "industrial" | "traffic" | "other";
export type ReportStatus = "submitted" | "under_review" | "corroborated" | "action_taken" | "resolved";

export interface CitizenReport {
  report_id: string;
  ward_id: string;
  ward_name: string;
  category: ReportCategory;
  description?: string;
  photo_url?: string;
  status: ReportStatus;
  created_at: string;
  updated_at: string;
  status_message?: string;
}

export interface CreateReportPayload {
  ward_id: string;
  category: ReportCategory;
  description?: string;
  photo?: File;
  lat?: number;
  lon?: number;
}

// ─── Enforcement memo ─────────────────────────────────────────────────────────
// The memo agent's output: a dispatch-ready enforcement notice with a
// rule-matched legal basis. GET /memo/{action_id or zone_id} returns one.
export interface LegalBasis {
  id: string;
  statute: string;
  provision: string;
  summary: string;
  authority: string;
  action: string;
  penalty: string;
}

export interface Memo {
  memo_id: string;
  action_id: string;
  zone_id: string;
  ward_id: string;
  ward_name: string;
  issued_at: string;
  status: string;
  subject: string;
  background: string;
  finding: string;
  directive: string;
  legal_basis: LegalBasis[];
  evidence_chain: string[];
  drafted_by: string;
  disclaimer: string;
  /** The deterministic situation the memo agent scored, shipped alongside the
   *  prose. Optional because it is the agent's own summary, not part of the
   *  document — the modal renders it as a scannable header strip so an officer
   *  gets severity/source/confidence before reading a paragraph. */
  situation?: MemoSituation;
}

export interface MemoSituation {
  source: string;
  confidence: number;
  pm25: number;
  aqi: number;
  aqi_category: string;
  kind: HotspotKind;
  n_cells: number;
}

// ─── Multi-city comparison ────────────────────────────────────────────────────
// One row per city, each a full LIVE pipeline run (scripts/city_summary.py).
export interface CityComparison {
  city: string;
  window_end: string;
  mode: string;
  detection: {
    hotspot_cells: number;
    zones: number;
    enforceable_cells: number;
  };
  attribution: {
    named: number;
    by_source: Record<string, number>;
    mean_confidence: number | null;
  };
  forecast_skill_vs_persistence_pct: {
    h24: number | null;
    h48: number | null;
    h72: number | null;
  };
  fusion_loso_r2: number | null;
  actions_queued: number;
  advisory_wards: number;
}

// ─── Agent Pipeline ───────────────────────────────────────────────────────────

export type AgentName =
  | "detection"
  | "attribution"
  | "forecast"
  | "prioritisation"
  | "memo"
  | "advisory"
  | "voice"
  | "ledger"
  | "audit";

export type AgentStatus = "idle" | "running" | "done" | "error";

export interface AgentState {
  name: AgentName;
  status: AgentStatus;
  duration_ms?: number;
  error?: string;
}

export interface PipelineRunResult {
  run_id: string;
  started_at: string;
  completed_at: string;
  agents: AgentState[];
}

// ─── Map Layer State ──────────────────────────────────────────────────────────

export type LayerId =
  | "fusion"
  | "stations"
  | "satellite"
  | "hotspots"
  | "fires"
  | "wards"
  | "blindspots"
  | "dispatch";

export type SatelliteChannel = "no2_col" | "so2_col" | "aai";

export interface LayerVisibility {
  fusion: boolean;
  stations: boolean;
  satellite: boolean;
  hotspots: boolean;
  fires: boolean;
  wards: boolean;
  blindspots: boolean;
  dispatch: boolean;
}

export interface MapFilters {
  ward_ids: string[];
  source_types: SourceCategory[];
  persistence_types: HotspotKind[];
  attributable_only: boolean;
  date?: string;
  /** Hours ahead: 0 (live fusion field), else 3..72 in steps of 3.
   *  Was `0 | 24 | 48 | 72` when the agent produced only three horizons. */
  forecast_horizon: number;
  satellite_channel: SatelliteChannel;
}
