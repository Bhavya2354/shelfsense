import { useQueries } from "@tanstack/react-query";
import { useCallback } from "react";

import { getJson } from "../api/client";
import { useModels } from "../api/hooks";
import type { ModelRun, Score } from "../api/types";
import { baseOption, Chart, type ChartTokens } from "../components/Chart";
import { ErrorNote, Loading, PageHeader, Panel, Table, td, tdNum, th, thNum } from "../components/ui";
import { fmtMetric, fmtPct, fmtSignedPct, modelLabel } from "../lib/format";

const BASELINE = "moving_average";

function HorizonChart({ runs }: { runs: ModelRun[] }) {
  const scores = useQueries({
    queries: runs.map((r) => ({
      queryKey: ["scores", r.id],
      queryFn: ({ signal }: { signal: AbortSignal }) => getJson<Score[]>(`/models/${r.id}/scores`, {}, signal),
    })),
  });
  const ready = scores.every((s) => s.isSuccess);
  const build = useCallback(
    (t: ChartTokens) => {
      const base = baseOption(t);
      const days = Array.from({ length: 16 }, (_, i) => i + 1);
      return {
        ...base,
        legend: { ...(base.legend as object), type: "scroll" },
        xAxis: { ...(base.xAxis as object), type: "category", data: days.map((d) => `+${d}`), name: "days ahead" },
        yAxis: { ...(base.yAxis as object), type: "value", scale: true, name: "NWRMSLE" },
        tooltip: { ...(base.tooltip as object), valueFormatter: (v: unknown) => (typeof v === "number" ? fmtMetric(v) : "–") },
        series: runs.map((run, i) => {
          const rows = (scores[i]?.data ?? []).filter((s) => s.metric === "nwrmsle" && s.horizon > 0);
          const mean = days.map((d) => {
            const vals = rows.filter((r) => r.horizon === d).map((r) => r.value);
            return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
          });
          const color = t.series[i % t.series.length];
          return {
            name: modelLabel(run.model_name),
            type: "line",
            data: mean,
            showSymbol: false,
            lineStyle: { width: run.model_name === "ensemble" ? 2.5 : 1.5, color },
            itemStyle: { color },
          };
        }),
      };
    },
    [runs, scores],
  );
  if (!ready) return <Loading rows={6} />;
  return <Chart build={build} height={300} label="Forecast error by days ahead for each item-level model" />;
}

export function ModelsPage() {
  const models = useModels("item");
  if (models.isPending) return <Loading rows={8} />;
  if (models.isError) return <ErrorNote error={models.error} />;
  const runs = [...models.data].sort((a, b) => (a.metrics.nwrmsle ?? 9) - (b.metrics.nwrmsle ?? 9));
  const baseline = runs.find((r) => r.model_name === BASELINE)?.metrics.nwrmsle;

  return (
    <>
      <PageHeader
        title="Model evaluation"
        lede="Store-item models scored on rolling-origin backtests: each model trains only on data before a fold's first day, then forecasts the next 16 days. NWRMSLE is the Favorita competition metric (perishables weigh 1.25)."
      />
      <Panel title="Item-level leaderboard" note="Mean over backtest folds.">
        <Table caption="Item-level model scores">
          <thead>
            <tr>
              <th className={th}>Model</th>
              <th className={thNum}>NWRMSLE</th>
              <th className={thNum}>vs 14-day average</th>
              <th className={thNum}>WAPE</th>
              <th className={thNum}>Bias</th>
              <th className={thNum}>MAE (units)</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => {
              const m = r.metrics;
              return (
                <tr key={r.id}>
                  <td className={`${td} ${r.model_name === "ensemble" ? "font-medium text-ink" : ""}`}>
                    {modelLabel(r.model_name)}
                  </td>
                  <td className={tdNum}>{m.nwrmsle !== undefined ? fmtMetric(m.nwrmsle) : "–"}</td>
                  <td className={tdNum}>
                    {m.nwrmsle !== undefined && baseline ? fmtSignedPct(m.nwrmsle / baseline - 1) : "–"}
                  </td>
                  <td className={tdNum}>{m.wape !== undefined ? fmtPct(m.wape) : "–"}</td>
                  <td className={tdNum}>{m.bias !== undefined ? fmtSignedPct(m.bias) : "–"}</td>
                  <td className={tdNum}>{m.mae_units !== undefined ? m.mae_units.toFixed(2) : "–"}</td>
                </tr>
              );
            })}
          </tbody>
        </Table>
      </Panel>
      <Panel
        className="mt-6"
        title="Error by days ahead"
        note="Errors grow with the horizon; weekly bumps come from weekend demand being harder to call."
      >
        <HorizonChart runs={runs} />
      </Panel>
    </>
  );
}
