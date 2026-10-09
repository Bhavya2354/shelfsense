const compact = new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 });
const whole = new Intl.NumberFormat("en", { maximumFractionDigits: 0 });
const oneDecimal = new Intl.NumberFormat("en", { maximumFractionDigits: 1, minimumFractionDigits: 1 });
const dayFormat = new Intl.DateTimeFormat("en", { month: "short", day: "numeric", timeZone: "UTC" });
const longDay = new Intl.DateTimeFormat("en", {
  weekday: "short",
  month: "short",
  day: "numeric",
  year: "numeric",
  timeZone: "UTC",
});

export const fmtCompact = (n: number) => compact.format(n);
export const fmtUnits = (n: number) => whole.format(n);
export const fmtOne = (n: number) => oneDecimal.format(n);
export const fmtPct = (fraction: number, digits = 1) => `${(fraction * 100).toFixed(digits)}%`;
export const fmtSignedPct = (fraction: number, digits = 1) =>
  `${fraction > 0 ? "+" : fraction < 0 ? "−" : ""}${Math.abs(fraction * 100).toFixed(digits)}%`;
export const fmtMetric = (n: number) => n.toFixed(4);
export const fmtDay = (iso: string) => dayFormat.format(new Date(`${iso}T00:00:00Z`));
export const fmtLongDay = (iso: string) => longDay.format(new Date(`${iso}T00:00:00Z`));

const MODEL_LABELS: Record<string, string> = {
  ensemble: "Ensemble",
  lightgbm: "LightGBM",
  embedding_mlp: "Embedding network",
  moving_average: "14-day average",
  weekday_average: "Weekday average",
  SeasonalNaive: "Seasonal naive",
  AutoETS: "ETS",
  AutoARIMA: "ARIMA",
  NBEATS: "N-BEATS",
  TFT: "TFT",
  Ensemble: "Mean of ETS, ARIMA, N-BEATS, TFT",
  ensemble_mint: "Ensemble, MinT reconciled",
  chronos2_zero_shot: "Chronos-2 zero-shot",
  chronos2_finetuned: "Chronos-2 fine-tuned",
  item_ensemble_bottom_up: "Item ensemble, summed up",
};

export const modelLabel = (name: string) => MODEL_LABELS[name] ?? name;

export const titleCase = (text: string) =>
  text
    .toLowerCase()
    .split(/[\s_]+/)
    .map((w) => (w ? w[0]!.toUpperCase() + w.slice(1) : w))
    .join(" ");
