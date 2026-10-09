import { useCallback } from "react";

import { useHierarchy, useOverview } from "../api/hooks";
import type { PolicyResult } from "../api/types";
import { Chart, type ChartTokens } from "../components/Chart";
import { forecastOption } from "../components/forecastChart";
import { ErrorNote, Loading, PageHeader, Panel, Stat, Table, td, tdNum, th, thNum } from "../components/ui";
import { fmtCompact, fmtLongDay, fmtMetric, fmtPct, fmtSignedPct, fmtUnits, modelLabel, titleCase } from "../lib/format";

const BASELINE = "moving_average";
const POLICY_ORDER = ["naive", "forecast", "newsvendor"];
const POLICY_LABEL: Record<string, string> = {
  naive: "Order the 14-day average",
  forecast: "Order the median forecast",
  newsvendor: "Newsvendor (cost-optimal)",
};

function NationalChart() {
  const national = useHierarchy(0, "ALL");
  const build = useCallback(
    (t: ChartTokens) =>
      forecastOption(
        t,
        national.data?.history ?? [],
        (national.data?.models ?? [])
          .filter((m) => m.model_name === "ensemble_mint" || m.model_name === "chronos2_finetuned")
          .sort((a) => (a.model_name === "ensemble_mint" ? -1 : 1))
          .map((m, i) => ({ name: modelLabel(m.model_name), points: m.points, showBand: i === 1 })),
      ),
    [national.data],
  );
  if (national.isPending) return <Loading rows={6} />;
  if (national.isError) return <ErrorNote error={national.error} />;
  return <Chart build={build} height={320} label="National daily unit sales with 16-day forecast" />;
}

function PolicyTable({ rows }: { rows: PolicyResult[] }) {
  const segments = ["all", "perishable", "shelf_stable"];
  return (
    <Table caption="Replenishment policies replayed against actual demand">
      <thead>
        <tr>
          <th className={th}>Segment</th>
          <th className={th}>Policy</th>
          <th className={thNum}>Fill rate</th>
          <th className={thNum}>Units short</th>
          <th className={thNum}>Units left over</th>
          <th className={thNum}>Mismatch cost</th>
        </tr>
      </thead>
      <tbody>
        {segments.flatMap((segment) =>
          POLICY_ORDER.map((policy) => rows.find((r) => r.segment === segment && r.policy === policy))
            .filter((r): r is PolicyResult => Boolean(r))
            .map((r, i) => (
              <tr key={`${segment}-${r.policy}`} className={r.policy === "newsvendor" ? "bg-accent-soft/40" : ""}>
                <td className={`${td} font-medium text-ink`}>{i === 0 ? titleCase(segment) : ""}</td>
                <td className={td}>{POLICY_LABEL[r.policy] ?? r.policy}</td>
                <td className={tdNum}>{fmtPct(r.fill_rate)}</td>
                <td className={tdNum}>{fmtUnits(r.lost_units)}</td>
                <td className={tdNum}>{fmtUnits(r.leftover_units)}</td>
                <td className={tdNum}>{fmtUnits(r.total_cost)}</td>
              </tr>
            )),
        )}
      </tbody>
    </Table>
  );
}

export function OverviewPage() {
  const overview = useOverview();
  if (overview.isPending) return <Loading rows={8} />;
  if (overview.isError) return <ErrorNote error={overview.error} />;
  const o = overview.data;

  const ensemble = o.item_scores.ensemble?.nwrmsle;
  const baseline = o.item_scores[BASELINE]?.nwrmsle;
  const accuracyGain = ensemble && baseline ? 1 - ensemble / baseline : null;
  const totals = (policy: string) => o.policy_totals.find((p) => p.segment === "all" && p.policy === policy);
  const naive = totals("naive");
  const optimal = totals("newsvendor");
  const costCut = naive && optimal ? 1 - optimal.total_cost / naive.total_cost : null;
  const weekChange = o.actual_units_last_7d ? o.forecast_units_next_7d / o.actual_units_last_7d - 1 : null;

  return (
    <>
      <PageHeader
        title="Next 16 days at a glance"
        lede={
          <>
            Forecasts for <span className="num">{o.series.toLocaleString()}</span> store-items across{" "}
            {o.stores} stores, made with sales up to {fmtLongDay(o.release.forecast_origin)}. Accuracy and
            ordering results below come from backtests on weeks the models never saw.
          </>
        }
      />
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat
          label="Units, next 7 days"
          value={fmtCompact(o.forecast_units_next_7d)}
          detail={weekChange !== null ? `${fmtSignedPct(weekChange)} vs last 7 days` : undefined}
        />
        <Stat
          label="Forecast error (NWRMSLE)"
          value={ensemble ? fmtMetric(ensemble) : "–"}
          detail={accuracyGain !== null ? `${fmtPct(accuracyGain)} lower than a 14-day average` : undefined}
          tone="good"
        />
        <Stat
          label="Fill rate, newsvendor"
          value={optimal ? fmtPct(optimal.fill_rate) : "–"}
          detail={naive ? `vs ${fmtPct(naive.fill_rate)} ordering the average` : undefined}
        />
        <Stat
          label="Mismatch cost"
          value={costCut !== null ? fmtSignedPct(-costCut) : "–"}
          detail="Newsvendor vs ordering the average"
          tone={costCut !== null && costCut > 0 ? "good" : "neutral"}
        />
      </div>

      <div className="mt-6 grid gap-6 xl:grid-cols-[2fr_1fr]">
        <Panel
          title="National unit sales"
          note="Daily units across all stores. The shaded band is the fine-tuned Chronos-2 80% interval."
        >
          <NationalChart />
        </Panel>
        <Panel title="Item-level models" note="Mean NWRMSLE over backtest folds; lower is better.">
          <Table caption="Item model leaderboard">
            <thead>
              <tr>
                <th className={th}>Model</th>
                <th className={thNum}>NWRMSLE</th>
                <th className={thNum}>WAPE</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(o.item_scores)
                .sort(([, a], [, b]) => (a.nwrmsle ?? 9) - (b.nwrmsle ?? 9))
                .map(([name, s]) => (
                  <tr key={name}>
                    <td className={`${td} ${name === o.best_item_model ? "font-medium text-ink" : ""}`}>
                      {modelLabel(name)}
                    </td>
                    <td className={tdNum}>{s.nwrmsle !== undefined ? fmtMetric(s.nwrmsle) : "–"}</td>
                    <td className={tdNum}>{s.wape !== undefined ? fmtPct(s.wape) : "–"}</td>
                  </tr>
                ))}
            </tbody>
          </Table>
        </Panel>
      </div>

      <Panel
        className="mt-6"
        title="What the forecasts are worth"
        note="Each policy replayed on the latest backtest fold. Costs are in shelf-price units: a unit short loses its margin, a unit left over costs write-off (perishable) or holding (shelf-stable)."
      >
        <PolicyTable rows={o.policy_totals} />
      </Panel>
    </>
  );
}
