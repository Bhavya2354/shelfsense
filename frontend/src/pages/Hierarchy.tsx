import { useCallback, useMemo, useState } from "react";
import { useSearchParams } from "react-router";

import { useFamilies, useHierarchy, useModels, useStores } from "../api/hooks";
import { Chart, type ChartTokens } from "../components/Chart";
import { forecastOption } from "../components/forecastChart";
import {
  ErrorNote,
  Field,
  inputClass,
  Loading,
  PageHeader,
  Panel,
  Table,
  td,
  tdNum,
  th,
  thNum,
} from "../components/ui";
import { fmtMetric, fmtPct, modelLabel, titleCase } from "../lib/format";

const DEFAULT_MODELS = ["chronos2_finetuned", "Ensemble", "NBEATS"];

function ModelTable() {
  const models = useModels("family");
  if (models.isPending) return <Loading rows={6} />;
  if (models.isError) return <ErrorNote error={models.error} />;
  const metric = (m: Record<string, number>, k: string) => m[`store_family:${k}`];
  const rows = [...models.data].sort((a, b) => (metric(a.metrics, "rmsle") ?? 9) - (metric(b.metrics, "rmsle") ?? 9));
  return (
    <Table caption="Store-family model comparison">
      <thead>
        <tr>
          <th className={th}>Model</th>
          <th className={thNum}>RMSLE</th>
          <th className={thNum}>WAPE</th>
          <th className={thNum}>MASE</th>
          <th className={thNum}>Bias</th>
          <th className={thNum}>80% coverage</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => {
          const v = (k: string) => metric(r.metrics, k);
          return (
            <tr key={r.id}>
              <td className={td}>{modelLabel(r.model_name)}</td>
              <td className={tdNum}>{v("rmsle") !== undefined ? fmtMetric(v("rmsle")!) : "–"}</td>
              <td className={tdNum}>{v("wape") !== undefined ? fmtPct(v("wape")!) : "–"}</td>
              <td className={tdNum}>{v("mase") !== undefined ? v("mase")!.toFixed(3) : "–"}</td>
              <td className={tdNum}>{v("bias") !== undefined ? fmtPct(v("bias")!) : "–"}</td>
              <td className={tdNum}>{v("coverage_80") !== undefined ? fmtPct(v("coverage_80")!) : "–"}</td>
            </tr>
          );
        })}
      </tbody>
    </Table>
  );
}

export function HierarchyPage() {
  const [params, setParams] = useSearchParams();
  const store = Number(params.get("store") ?? 0);
  const family = params.get("family") ?? "ALL";
  const [selected, setSelected] = useState<string[]>(DEFAULT_MODELS);

  const stores = useStores();
  const families = useFamilies();
  const hierarchy = useHierarchy(store, family);
  const available = useMemo(() => hierarchy.data?.models.map((m) => m.model_name) ?? [], [hierarchy.data]);

  const build = useCallback(
    (t: ChartTokens) =>
      forecastOption(
        t,
        hierarchy.data?.history ?? [],
        selected
          .map((name) => hierarchy.data?.models.find((m) => m.model_name === name))
          .filter((m): m is NonNullable<typeof m> => Boolean(m))
          .map((m, i) => ({ name: modelLabel(m.model_name), points: m.points, showBand: i === 0 })),
      ),
    [hierarchy.data, selected],
  );

  const set = (key: string, value: string) =>
    setParams((prev) => {
      const next = new URLSearchParams(prev);
      next.set(key, value);
      return next;
    });
  const toggle = (name: string) =>
    setSelected((cur) => (cur.includes(name) ? cur.filter((n) => n !== name) : [...cur, name]));

  return (
    <>
      <PageHeader
        title="Store and family forecasts"
        lede="Totals for a whole store or product family, forecast by statistical models, networks trained from scratch and a pretrained foundation model. The reconciled ensemble is adjusted so store-family forecasts add up to store and national totals."
      />
      <div className="mb-4 flex flex-wrap gap-3">
        <Field label="Store">
          <select className={inputClass} value={store} onChange={(e) => set("store", e.target.value)}>
            <option value={0}>All stores (national)</option>
            {stores.data?.map((s) => (
              <option key={s.store_nbr} value={s.store_nbr}>
                {s.store_nbr} · {titleCase(s.city)}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Product family">
          <select
            className={inputClass}
            value={family}
            onChange={(e) => set("family", e.target.value)}
            disabled={store === 0}
          >
            <option value="ALL">All families</option>
            {families.data?.map((f) => (
              <option key={f} value={f}>
                {titleCase(f)}
              </option>
            ))}
          </select>
        </Field>
      </div>

      <Panel
        title={store === 0 ? "National total" : `Store ${store} · ${family === "ALL" ? "all families" : titleCase(family)}`}
        note="The first selected model shows its 80% interval where it produces one."
      >
        <div className="mb-3 flex flex-wrap gap-2" role="group" aria-label="Models shown">
          {available.map((name) => {
            const on = selected.includes(name);
            return (
              <button
                key={name}
                type="button"
                aria-pressed={on}
                onClick={() => toggle(name)}
                className={`rounded-full border px-3 py-1 text-xs ${
                  on ? "border-accent bg-accent-soft text-ink" : "border-line text-muted hover:bg-sunken"
                }`}
              >
                {modelLabel(name)}
              </button>
            );
          })}
        </div>
        {hierarchy.isPending ? (
          <Loading rows={6} />
        ) : hierarchy.isError ? (
          <ErrorNote error={hierarchy.error} />
        ) : (
          <Chart build={build} height={340} label="Store or family unit sales with model forecasts" />
        )}
      </Panel>

      <Panel
        className="mt-6"
        title="How the models compare"
        note="Backtest errors at store-family level, averaged over folds. MASE below 1 beats a same-weekday-last-week forecast."
      >
        <ModelTable />
      </Panel>
    </>
  );
}
