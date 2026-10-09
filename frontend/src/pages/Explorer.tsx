import { useCallback } from "react";
import { useSearchParams } from "react-router";

import { useFamilies, useSeries, useSeriesDetail, useStores } from "../api/hooks";
import type { SeriesDetail } from "../api/types";
import { Chart, type ChartTokens } from "../components/Chart";
import { forecastOption } from "../components/forecastChart";
import {
  ErrorNote,
  Field,
  inputClass,
  Loading,
  PageHeader,
  Pager,
  Panel,
  Table,
  td,
  tdNum,
  th,
  thNum,
} from "../components/ui";
import { fmtOne, fmtPct, fmtUnits, titleCase } from "../lib/format";

const PAGE_SIZE = 25;

function numberParam(params: URLSearchParams, key: string): number | undefined {
  const raw = params.get(key);
  return raw === null || raw === "" ? undefined : Number(raw);
}

function DetailPanel({ detail }: { detail: SeriesDetail }) {
  const build = useCallback(
    (t: ChartTokens) =>
      forecastOption(t, detail.history, [{ name: "Ensemble forecast", points: detail.forecast, showBand: true }]),
    [detail],
  );
  const order = detail.order;
  return (
    <Panel
      title={`Item ${detail.item_nbr} at store ${detail.store_nbr}`}
      note={`${titleCase(detail.family)} · class ${detail.item_class} · ${detail.perishable ? "perishable" : "shelf-stable"}`}
    >
      <Chart build={build} height={300} label="Daily sales history and forecast for the selected store-item" />
      {order ? (
        <dl className="mt-4 grid grid-cols-2 gap-x-6 gap-y-3 border-t border-line pt-4 text-sm sm:grid-cols-4">
          {[
            ["Order now", `${fmtUnits(order.order_quantity)} units`],
            ["Covers", `${order.cover_days} day${order.cover_days > 1 ? "s" : ""}`],
            ["Expected demand", fmtOne(order.expected_demand)],
            ["Target in-stock", fmtPct(order.service_level, 0)],
            ["Safety stock", fmtOne(order.safety_stock)],
            ["Expected short", fmtOne(order.expected_lost_units)],
            ["Expected left over", fmtOne(order.expected_leftover_units)],
          ].map(([k, v]) => (
            <div key={k}>
              <dt className="text-xs text-muted">{k}</dt>
              <dd className="num mt-0.5 text-ink">{v}</dd>
            </div>
          ))}
        </dl>
      ) : (
        <p className="mt-4 text-sm text-muted">No order plan for this series.</p>
      )}
    </Panel>
  );
}

export function ExplorerPage() {
  const [params, setParams] = useSearchParams();
  const store = numberParam(params, "store");
  const family = params.get("family") ?? undefined;
  const page = numberParam(params, "page") ?? 1;
  const selStore = numberParam(params, "s");
  const selItem = numberParam(params, "i");

  const stores = useStores();
  const families = useFamilies();
  const series = useSeries({ store_nbr: store, family, page, page_size: PAGE_SIZE });
  const detail = useSeriesDetail(selStore, selItem);

  const update = (changes: Record<string, string | number | undefined>) =>
    setParams((prev) => {
      const next = new URLSearchParams(prev);
      for (const [k, v] of Object.entries(changes)) {
        if (v === undefined || v === "") next.delete(k);
        else next.set(k, String(v));
      }
      return next;
    });

  return (
    <>
      <PageHeader
        title="Store-item explorer"
        lede="Every forecast store-item, busiest first. Pick one to see its recent sales, the 16-day forecast with its 80% interval, and the order the newsvendor policy recommends."
      />
      <div className="mb-4 flex flex-wrap gap-3">
        <Field label="Store">
          <select
            className={inputClass}
            value={store ?? ""}
            onChange={(e) => update({ store: e.target.value, page: undefined })}
          >
            <option value="">All stores</option>
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
            value={family ?? ""}
            onChange={(e) => update({ family: e.target.value, page: undefined })}
          >
            <option value="">All families</option>
            {families.data?.map((f) => (
              <option key={f} value={f}>
                {titleCase(f)}
              </option>
            ))}
          </select>
        </Field>
      </div>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <Panel title="Store-items">
          {series.isPending ? (
            <Loading rows={8} />
          ) : series.isError ? (
            <ErrorNote error={series.error} />
          ) : (
            <>
              <Table caption="Store-items in the current release">
                <thead>
                  <tr>
                    <th className={th}>Store</th>
                    <th className={th}>Item</th>
                    <th className={th}>Family</th>
                    <th className={thNum}>Last 8 wks</th>
                    <th className={thNum}>Next 16 d</th>
                  </tr>
                </thead>
                <tbody>
                  {series.data.items.map((r) => {
                    const active = r.store_nbr === selStore && r.item_nbr === selItem;
                    return (
                      <tr
                        key={`${r.store_nbr}-${r.item_nbr}`}
                        className={`cursor-pointer ${active ? "bg-accent-soft" : "hover:bg-sunken"}`}
                        onClick={() => update({ s: r.store_nbr, i: r.item_nbr })}
                        aria-selected={active}
                      >
                        <td className={tdNum.replace("text-right", "text-left")}>{r.store_nbr}</td>
                        <td className={tdNum.replace("text-right", "text-left")}>{r.item_nbr}</td>
                        <td className={td}>{titleCase(r.family)}</td>
                        <td className={tdNum}>{fmtUnits(r.recent_units)}</td>
                        <td className={tdNum}>{fmtUnits(r.forecast_units)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </Table>
              <Pager
                page={page}
                pageSize={PAGE_SIZE}
                total={series.data.total}
                onPage={(p) => update({ page: p })}
              />
            </>
          )}
        </Panel>
        <div>
          {selStore === undefined ? (
            <Panel>
              <p className="text-sm text-muted">Select a store-item to see its forecast and order plan.</p>
            </Panel>
          ) : detail.isPending ? (
            <Panel>
              <Loading rows={6} />
            </Panel>
          ) : detail.isError ? (
            <ErrorNote error={detail.error} />
          ) : (
            <DetailPanel detail={detail.data} />
          )}
        </div>
      </div>
    </>
  );
}
