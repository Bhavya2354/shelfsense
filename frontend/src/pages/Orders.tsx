import { useSearchParams } from "react-router";

import { useFamilies, useOrders, usePolicies, useStores } from "../api/hooks";
import type { OrderLine } from "../api/types";
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

const PAGE_SIZE = 50;
const CSV_COLUMNS: (keyof OrderLine)[] = [
  "store_nbr",
  "item_nbr",
  "family",
  "cover_days",
  "service_level",
  "expected_demand",
  "order_quantity",
  "safety_stock",
  "expected_lost_units",
  "expected_leftover_units",
];

function downloadCsv(rows: OrderLine[]) {
  const escape = (v: unknown) => {
    const text = v === null || v === undefined ? "" : String(v);
    return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  };
  const body = [CSV_COLUMNS.join(","), ...rows.map((r) => CSV_COLUMNS.map((c) => escape(r[c])).join(","))];
  const url = URL.createObjectURL(new Blob([body.join("\n")], { type: "text/csv" }));
  const link = Object.assign(document.createElement("a"), { href: url, download: "order-plan.csv" });
  link.click();
  URL.revokeObjectURL(url);
}

function ServiceLevels() {
  const policies = usePolicies("perishable");
  const shelf = usePolicies("shelf_stable");
  if (policies.isPending || shelf.isPending) return null;
  const pick = (rows: typeof policies.data, policy: string) =>
    rows?.find((r) => r.policy === policy && !r.segment.includes(":"));
  const rows = [
    ["Perishable, ordered daily", pick(policies.data, "naive"), pick(policies.data, "newsvendor")],
    ["Shelf-stable, ordered weekly", pick(shelf.data, "naive"), pick(shelf.data, "newsvendor")],
  ] as const;
  return (
    <Table caption="Backtested fill rate by item class">
      <thead>
        <tr>
          <th className={th}>Item class</th>
          <th className={thNum}>Fill rate, 14-day average</th>
          <th className={thNum}>Fill rate, newsvendor</th>
          <th className={thNum}>Cost change</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([label, naive, nv]) => (
          <tr key={label}>
            <td className={td}>{label}</td>
            <td className={tdNum}>{naive ? fmtPct(naive.fill_rate) : "–"}</td>
            <td className={tdNum}>{nv ? fmtPct(nv.fill_rate) : "–"}</td>
            <td className={tdNum}>
              {naive && nv ? fmtPct(nv.total_cost / naive.total_cost - 1) : "–"}
            </td>
          </tr>
        ))}
      </tbody>
    </Table>
  );
}

export function OrdersPage() {
  const [params, setParams] = useSearchParams();
  const store = params.get("store") ? Number(params.get("store")) : undefined;
  const family = params.get("family") ?? undefined;
  const page = Number(params.get("page") ?? 1);
  const stores = useStores();
  const families = useFamilies();
  const orders = useOrders({ store_nbr: store, family, page, page_size: PAGE_SIZE });

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
        title="Order plan"
        lede="Recommended order for each store-item's next cover period: one day for perishables, a week for shelf-stable goods. Each quantity is the demand level worth stocking to, given what a lost sale costs against what a leftover unit costs."
      />
      <Panel className="mb-6" title="Why these quantities" note="Backtested on the latest fold against the simple rule of ordering recent average sales.">
        <ServiceLevels />
      </Panel>
      <div className="mb-4 flex flex-wrap items-end gap-3">
        <Field label="Store">
          <select className={inputClass} value={store ?? ""} onChange={(e) => update({ store: e.target.value, page: undefined })}>
            <option value="">All stores</option>
            {stores.data?.map((s) => (
              <option key={s.store_nbr} value={s.store_nbr}>
                {s.store_nbr} · {titleCase(s.city)}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Product family">
          <select className={inputClass} value={family ?? ""} onChange={(e) => update({ family: e.target.value, page: undefined })}>
            <option value="">All families</option>
            {families.data?.map((f) => (
              <option key={f} value={f}>
                {titleCase(f)}
              </option>
            ))}
          </select>
        </Field>
        <button
          type="button"
          disabled={!orders.data?.items.length}
          onClick={() => orders.data && downloadCsv(orders.data.items)}
          className="h-9 rounded-md border border-line bg-surface px-3 text-sm text-ink-soft enabled:hover:bg-sunken disabled:opacity-40"
        >
          Download this page (CSV)
        </button>
      </div>
      <Panel>
        {orders.isPending ? (
          <Loading rows={10} />
        ) : orders.isError ? (
          <ErrorNote error={orders.error} />
        ) : (
          <>
            <Table caption="Recommended orders, largest first">
              <thead>
                <tr>
                  <th className={th}>Store</th>
                  <th className={th}>Item</th>
                  <th className={th}>Family</th>
                  <th className={thNum}>Cover</th>
                  <th className={thNum}>In-stock target</th>
                  <th className={thNum}>Expected demand</th>
                  <th className={thNum}>Order</th>
                  <th className={thNum}>Safety stock</th>
                  <th className={thNum}>Exp. short</th>
                  <th className={thNum}>Exp. left over</th>
                </tr>
              </thead>
              <tbody>
                {orders.data.items.map((r) => (
                  <tr key={`${r.store_nbr}-${r.item_nbr}`} className="hover:bg-sunken">
                    <td className={tdNum.replace("text-right", "text-left")}>{r.store_nbr}</td>
                    <td className={tdNum.replace("text-right", "text-left")}>{r.item_nbr}</td>
                    <td className={td}>{r.family ? titleCase(r.family) : "–"}</td>
                    <td className={tdNum}>{r.cover_days} d</td>
                    <td className={tdNum}>{fmtPct(r.service_level, 0)}</td>
                    <td className={tdNum}>{fmtOne(r.expected_demand)}</td>
                    <td className={`${tdNum} font-medium`}>{fmtUnits(r.order_quantity)}</td>
                    <td className={tdNum}>{fmtOne(r.safety_stock)}</td>
                    <td className={tdNum}>{fmtOne(r.expected_lost_units)}</td>
                    <td className={tdNum}>{fmtOne(r.expected_leftover_units)}</td>
                  </tr>
                ))}
              </tbody>
            </Table>
            <Pager page={page} pageSize={PAGE_SIZE} total={orders.data.total} onPage={(p) => update({ page: p })} />
          </>
        )}
      </Panel>
    </>
  );
}
