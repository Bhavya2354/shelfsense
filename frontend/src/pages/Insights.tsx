import { useCallback } from "react";

import { useAnalyses, useQuality } from "../api/hooks";
import type { Analysis } from "../api/types";
import { baseOption, Chart, type ChartTokens } from "../components/Chart";
import { ErrorNote, Loading, PageHeader, Panel, Stat, Table, td, tdNum, th, thNum } from "../components/ui";
import { titleCase } from "../lib/format";

interface Effect {
  effect_pct: number;
  ci_low_pct: number;
  ci_high_pct: number;
  p_value: number;
}

const ALPHA = 0.05;

function findings(analyses: Analysis[], name: string) {
  return analyses.find((a) => a.analysis === name)?.findings ?? [];
}

function payload<T>(analyses: Analysis[], name: string, subject: string): T | undefined {
  return findings(analyses, name).find((f) => f.subject === subject)?.payload as T | undefined;
}

const pct = (v: number) => `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(1)}%`;
const interval = (e: Effect) => `${pct(e.ci_low_pct)} to ${pct(e.ci_high_pct)}`;
const pValue = (p: number) => (p < 0.001 ? "< 0.001" : p.toFixed(3));

function barOption(t: ChartTokens, labels: string[], values: number[], horizontal = false) {
  const base = baseOption(t);
  const category = { type: "category", data: labels, axisLabel: { color: t.muted, fontSize: 11 } };
  const value = { ...(base.yAxis as object), type: "value", axisLabel: { color: t.muted, formatter: (v: number) => `${v}%` } };
  return {
    ...base,
    legend: { show: false },
    tooltip: { ...(base.tooltip as object), valueFormatter: (v: unknown) => (typeof v === "number" ? pct(v) : "–") },
    xAxis: horizontal ? value : { ...(base.xAxis as object), ...category },
    yAxis: horizontal ? { ...category, inverse: true, axisLine: { show: false }, axisTick: { show: false } } : value,
    series: [
      {
        type: "bar",
        data: values.map((v) => ({ value: v, itemStyle: { color: v >= 0 ? t.accent : t.amber } })),
        barMaxWidth: 22,
      },
    ],
  };
}

function Seasonality({ analyses }: { analyses: Analysis[] }) {
  const national = payload<{
    weekday_effect_pct: Record<string, number>;
    month_effect_pct: Record<string, number>;
    weekly_strength: number;
    yearly_strength: number;
    trend_growth_pct: number;
  }>(analyses, "seasonality", "national");
  const weekly = useCallback(
    (t: ChartTokens) =>
      barOption(t, Object.keys(national?.weekday_effect_pct ?? {}), Object.values(national?.weekday_effect_pct ?? {})),
    [national],
  );
  const monthly = useCallback(
    (t: ChartTokens) =>
      barOption(t, Object.keys(national?.month_effect_pct ?? {}), Object.values(national?.month_effect_pct ?? {})),
    [national],
  );
  if (!national) return null;
  return (
    <>
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-3">
        <Stat label="Weekly seasonality strength" value={national.weekly_strength.toFixed(2)} detail="0 none, 1 dominant" />
        <Stat label="Yearly seasonality strength" value={national.yearly_strength.toFixed(2)} detail="0 none, 1 dominant" />
        <Stat label="Trend, last year vs first" value={pct(national.trend_growth_pct)} detail="From the MSTL trend" />
      </div>
      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <Panel title="Day-of-week effect" note="Seasonal component of log units, relative to an average day.">
          <Chart build={weekly} height={240} label="Sales effect by day of week" />
        </Panel>
        <Panel title="Month-of-year effect" note="December runs well above an average month.">
          <Chart build={monthly} height={240} label="Sales effect by month" />
        </Panel>
      </div>
    </>
  );
}

function PromotionLift({ analyses }: { analyses: Analysis[] }) {
  const rows = findings(analyses, "promotion_lift").map((f) => ({ family: f.subject, ...(f.payload as unknown as Effect) }));
  const build = useCallback(
    (t: ChartTokens) => barOption(t, rows.map((r) => titleCase(r.family)), rows.map((r) => r.effect_pct), true),
    [rows],
  );
  if (!rows.length) return null;
  return (
    <Panel
      title="Promotion lift by product family"
      note="Within-series estimate: each store-item is compared with itself on promoted vs regular days, net of day effects. 95% intervals use errors clustered by store-item."
    >
      <Chart build={build} height={Math.max(260, rows.length * 22)} label="Promotion lift by family" />
    </Panel>
  );
}

const EFFECT_LABELS: Record<string, string> = {
  payday: "Payday (15th and month end)",
  payday_next: "Day after payday",
  earthquake_relief_period: "April 2016 earthquake relief period",
};

function CalendarEffects({ analyses }: { analyses: Analysis[] }) {
  const rows = findings(analyses, "calendar_effects").filter((f) => f.subject !== "model");
  const rain = payload<Effect & { threshold_mm: number }>(analyses, "rain_effect", "heavy_rain_day");
  const oil = payload<{ min_p_value: number; weeks: number }>(analyses, "oil_granger", "oil_to_sales");
  const fit = payload<{ r_squared: number }>(analyses, "calendar_effects", "model");
  return (
    <div className="grid gap-6 xl:grid-cols-[3fr_2fr]">
      <Panel
        title="Calendar events"
        note={`Regression of national log sales on weekday, month, trend and event days, with Newey-West errors${
          fit ? `; R² ${fit.r_squared.toFixed(2)}` : ""
        }.`}
      >
        <Table caption="Calendar effects on national sales">
          <thead>
            <tr>
              <th className={th}>Event</th>
              <th className={thNum}>Effect</th>
              <th className={thNum}>95% interval</th>
              <th className={thNum}>p-value</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((f) => {
              const e = f.payload as unknown as Effect;
              const significant = e.p_value < ALPHA;
              return (
                <tr key={f.subject}>
                  <td className={td}>
                    {EFFECT_LABELS[f.subject] ?? titleCase(f.subject.replace(/^national_/, "National "))}
                  </td>
                  <td className={`${tdNum} ${significant ? "" : "text-muted"}`}>{pct(e.effect_pct)}</td>
                  <td className={`${tdNum} text-muted`}>{interval(e)}</td>
                  <td className={tdNum}>{pValue(e.p_value)}</td>
                </tr>
              );
            })}
          </tbody>
        </Table>
      </Panel>
      <div className="grid content-start gap-3">
        {rain && (
          <Stat
            label={`Heavy rain day (≥ ${rain.threshold_mm} mm)`}
            value={pct(rain.effect_pct)}
            detail={`Store transactions; 95% interval ${interval(rain)}. Compares cities on the same day.`}
            tone={rain.p_value < ALPHA ? "bad" : "neutral"}
          />
        )}
        {oil && (
          <Stat
            label="Oil price → sales (Granger)"
            value={oil.min_p_value < ALPHA ? "Predictive" : "Not predictive"}
            detail={`Smallest p-value ${pValue(oil.min_p_value)} over 1–4 week lags, ${oil.weeks} weeks.`}
          />
        )}
      </div>
    </div>
  );
}

function Stationarity({ analyses }: { analyses: Analysis[] }) {
  const rows = findings(analyses, "stationarity");
  if (!rows.length) return null;
  return (
    <Table caption="Stationarity tests">
      <thead>
        <tr>
          <th className={th}>Series</th>
          <th className={thNum}>ADF p-value</th>
          <th className={thNum}>KPSS p-value</th>
          <th className={th}>Verdict</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((f) => {
          const p = f.payload as { adf_pvalue: number; kpss_pvalue: number; verdict: string };
          return (
            <tr key={f.subject}>
              <td className={td}>{f.subject === "log_level" ? "Log units" : "Daily change in log units"}</td>
              <td className={tdNum}>{pValue(p.adf_pvalue)}</td>
              <td className={tdNum}>{pValue(p.kpss_pvalue)}</td>
              <td className={td}>{titleCase(p.verdict)}</td>
            </tr>
          );
        })}
      </tbody>
    </Table>
  );
}

function DataQuality() {
  const quality = useQuality();
  if (quality.isPending) return <Loading rows={4} />;
  if (quality.isError) return <ErrorNote error={quality.error} />;
  return (
    <Table caption="Data quality checks from the last ingestion">
      <thead>
        <tr>
          <th className={th}>Check</th>
          <th className={th}>Table</th>
          <th className={th}>Severity</th>
          <th className={thNum}>Rows flagged</th>
        </tr>
      </thead>
      <tbody>
        {quality.data.map((q) => (
          <tr key={q.check_name}>
            <td className={td}>{q.check_name.replace(/_/g, " ")}</td>
            <td className={td}>{q.table_name}</td>
            <td className={`${td} ${q.violations && q.severity === "error" ? "text-red" : ""}`}>{q.severity}</td>
            <td className={tdNum}>{q.violations.toLocaleString()}</td>
          </tr>
        ))}
      </tbody>
    </Table>
  );
}

export function InsightsPage() {
  const analyses = useAnalyses();
  if (analyses.isPending) return <Loading rows={8} />;
  if (analyses.isError) return <ErrorNote error={analyses.error} />;
  const a = analyses.data;
  return (
    <>
      <PageHeader
        title="What drives demand"
        lede="Statistical findings behind the model features: seasonality, calendar events, promotions, weather and oil. Effects are percent changes in units with 95% confidence intervals; grey values are not significant at the 5% level."
      />
      <Seasonality analyses={a} />
      <div className="mt-6">
        <CalendarEffects analyses={a} />
      </div>
      <div className="mt-6">
        <PromotionLift analyses={a} />
      </div>
      <div className="mt-6 grid gap-6 xl:grid-cols-2">
        <Panel title="Stationarity" note="ADF tests for a unit root, KPSS for stationarity; using both separates the cases.">
          <Stationarity analyses={a} />
        </Panel>
        <Panel title="Data quality" note="Errors stop the pipeline; warnings are reported and handled.">
          <DataQuality />
        </Panel>
      </div>
    </>
  );
}
