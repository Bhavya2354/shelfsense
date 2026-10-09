import type { ReactNode } from "react";

import { ApiError } from "../api/client";

export function PageHeader({ title, lede, aside }: { title: string; lede?: ReactNode; aside?: ReactNode }) {
  return (
    <header className="mb-6 flex flex-wrap items-end justify-between gap-4 border-b border-line pb-4">
      <div className="max-w-2xl">
        <h1 className="text-2xl font-semibold tracking-tight text-ink">{title}</h1>
        {lede && <p className="mt-1 text-sm leading-relaxed text-muted">{lede}</p>}
      </div>
      {aside}
    </header>
  );
}

export function Panel({
  title,
  note,
  children,
  className = "",
  actions,
}: {
  title?: string;
  note?: ReactNode;
  children: ReactNode;
  className?: string;
  actions?: ReactNode;
}) {
  return (
    <section className={`rounded-lg border border-line bg-surface p-4 sm:p-5 ${className}`}>
      {(title || actions) && (
        <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
          <div>
            {title && <h2 className="text-sm font-semibold text-ink">{title}</h2>}
            {note && <p className="mt-0.5 text-xs text-muted">{note}</p>}
          </div>
          {actions}
        </div>
      )}
      {children}
    </section>
  );
}

export function Stat({
  label,
  value,
  detail,
  tone = "neutral",
}: {
  label: string;
  value: string;
  detail?: ReactNode;
  tone?: "neutral" | "good" | "bad";
}) {
  const toneClass = tone === "good" ? "text-accent" : tone === "bad" ? "text-red" : "text-ink";
  return (
    <div className="rounded-lg border border-line bg-surface px-4 py-3">
      <div className="text-xs font-medium tracking-wide text-muted uppercase">{label}</div>
      <div className={`num mt-1 text-2xl font-medium ${toneClass}`}>{value}</div>
      {detail && <div className="mt-1 text-xs text-muted">{detail}</div>}
    </div>
  );
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="flex flex-col gap-1 text-xs font-medium text-muted">
      {label}
      {children}
    </label>
  );
}

export const inputClass =
  "h-9 min-w-36 rounded-md border border-line bg-surface px-2.5 text-sm text-ink focus:border-accent focus:outline-none";

export function Loading({ rows = 3 }: { rows?: number }) {
  return (
    <div className="space-y-2" aria-busy="true" aria-label="Loading">
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} className="h-4 animate-pulse rounded bg-sunken" style={{ width: `${90 - i * 12}%` }} />
      ))}
    </div>
  );
}

export function ErrorNote({ error }: { error: unknown }) {
  const noRelease = error instanceof ApiError && error.status === 503;
  return (
    <div role="alert" className="rounded-md border border-line bg-sunken px-3 py-2 text-sm text-ink-soft">
      {noRelease
        ? "No forecast release has been published yet. Run the pipeline, then refresh."
        : `Could not load this data${error instanceof Error ? `: ${error.message}` : "."}`}
    </div>
  );
}

export function Pager({
  page,
  pageSize,
  total,
  onPage,
}: {
  page: number;
  pageSize: number;
  total: number;
  onPage: (page: number) => void;
}) {
  const pages = Math.max(1, Math.ceil(total / pageSize));
  const button =
    "h-8 rounded-md border border-line px-3 text-sm text-ink-soft enabled:hover:bg-sunken disabled:opacity-40";
  return (
    <nav className="mt-3 flex items-center justify-between gap-2 text-xs text-muted" aria-label="Pagination">
      <span className="num">
        {total === 0 ? "No rows" : `${(page - 1) * pageSize + 1}–${Math.min(page * pageSize, total)} of ${total.toLocaleString()}`}
      </span>
      <div className="flex gap-2">
        <button type="button" className={button} disabled={page <= 1} onClick={() => onPage(page - 1)}>
          Previous
        </button>
        <button type="button" className={button} disabled={page >= pages} onClick={() => onPage(page + 1)}>
          Next
        </button>
      </div>
    </nav>
  );
}

export function Table({ children, caption }: { children: ReactNode; caption: string }) {
  return (
    <div className="-mx-4 overflow-x-auto px-4 sm:mx-0 sm:px-0">
      <table className="w-full border-collapse text-sm">
        <caption className="sr-only">{caption}</caption>
        {children}
      </table>
    </div>
  );
}

export const th = "border-b border-line px-2 py-2 text-left text-xs font-medium text-muted whitespace-nowrap";
export const thNum = `${th} text-right`;
export const td = "border-b border-line/70 px-2 py-2 text-ink-soft whitespace-nowrap";
export const tdNum = `${td} num text-right text-ink`;
