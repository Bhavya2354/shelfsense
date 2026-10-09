// Mirrors backend/app/api/schemas.py.

export interface ReleaseInfo {
  id: string;
  forecast_origin: string;
  created_at: string;
  notes: string | null;
}

export interface Store {
  store_nbr: number;
  city: string;
  state: string;
  store_type: string;
  cluster: number;
  latitude: number | null;
  longitude: number | null;
  forecast_units: number | null;
}

export interface SeriesSummary {
  store_nbr: number;
  item_nbr: number;
  family: string;
  perishable: boolean;
  forecast_units: number;
  recent_units: number;
}

export interface Page<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
}

export interface DailyPoint {
  day: string;
  units: number;
  onpromotion?: boolean | null;
}

export interface ForecastPoint {
  day: string;
  p10: number | null;
  p50: number;
  p90: number | null;
  onpromotion?: boolean | null;
}

export interface OrderLine {
  store_nbr: number;
  item_nbr: number;
  family: string | null;
  cover_days: number;
  service_level: number;
  expected_demand: number;
  order_quantity: number;
  safety_stock: number;
  expected_lost_units: number;
  expected_leftover_units: number;
}

export interface SeriesDetail {
  store_nbr: number;
  item_nbr: number;
  family: string;
  item_class: number;
  perishable: boolean;
  history: DailyPoint[];
  forecast: ForecastPoint[];
  order: OrderLine | null;
}

export interface ModelForecast {
  model_name: string;
  points: ForecastPoint[];
}

export interface HierarchyDetail {
  store_nbr: number;
  family: string;
  history: DailyPoint[];
  models: ModelForecast[];
}

export interface PolicyResult {
  policy: string;
  segment: string;
  demand_units: number;
  lost_units: number;
  leftover_units: number;
  fill_rate: number;
  total_cost: number;
}

export interface ModelRun {
  id: string;
  model_name: string;
  level: "item" | "family";
  metrics: Record<string, number>;
  created_at: string;
}

export interface Score {
  cutoff: string;
  horizon: number;
  metric: string;
  value: number;
}

export interface Finding {
  subject: string;
  payload: Record<string, unknown>;
}

export interface Analysis {
  analysis: string;
  findings: Finding[];
}

export interface QualityCheck {
  check_name: string;
  table_name: string;
  severity: "error" | "warning";
  violations: number;
  checked_at: string;
}

export interface Overview {
  release: ReleaseInfo;
  series: number;
  stores: number;
  forecast_units_next_7d: number;
  actual_units_last_7d: number;
  best_item_model: string | null;
  item_scores: Record<string, Record<string, number>>;
  policy_totals: PolicyResult[];
}
