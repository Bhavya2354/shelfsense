import { useContext } from "react";
import { NavLink, Outlet } from "react-router";

import { useOverview } from "../api/hooks";
import { fmtLongDay } from "../lib/format";
import type { ThemeChoice } from "../lib/theme";
import { ThemeContext } from "./ThemeContext";

const NAV = [
  { to: "/", label: "Overview", end: true },
  { to: "/explorer", label: "Store-item explorer" },
  { to: "/hierarchy", label: "Store & family" },
  { to: "/orders", label: "Order plan" },
  { to: "/models", label: "Model evaluation" },
  { to: "/insights", label: "Demand drivers" },
];

const THEME_LABEL: Record<ThemeChoice, string> = { system: "Auto", light: "Light", dark: "Dark" };

export function Layout() {
  const overview = useOverview();
  const { choice: theme, cycle: onCycleTheme } = useContext(ThemeContext);
  const link = ({ isActive }: { isActive: boolean }) =>
    `block rounded-md px-3 py-1.5 text-sm whitespace-nowrap ${
      isActive ? "bg-accent-soft font-medium text-ink" : "text-ink-soft hover:bg-sunken"
    }`;

  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[15rem_1fr]">
      <aside className="border-b border-line bg-surface lg:sticky lg:top-0 lg:h-screen lg:border-r lg:border-b-0">
        <div className="flex items-center justify-between gap-3 px-4 py-4 lg:flex-col lg:items-start">
          <a href="#/" className="flex items-center gap-2 text-ink">
            <img src={`${import.meta.env.BASE_URL}favicon.svg`} alt="" className="h-6 w-6" />
            <span className="text-base font-semibold tracking-tight">ShelfSense</span>
          </a>
          <button
            type="button"
            onClick={onCycleTheme}
            className="rounded-md border border-line px-2 py-1 text-xs text-muted hover:bg-sunken lg:hidden"
          >
            Theme: {THEME_LABEL[theme]}
          </button>
        </div>
        <nav aria-label="Main" className="flex gap-1 overflow-x-auto px-3 pb-3 lg:flex-col lg:overflow-visible">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.end} className={link}>
              {n.label}
            </NavLink>
          ))}
        </nav>
        <div className="hidden px-4 pt-6 text-xs leading-relaxed text-muted lg:block">
          {overview.data && (
            <p>
              Forecasting from
              <br />
              <span className="num text-ink-soft">{fmtLongDay(overview.data.release.forecast_origin)}</span>
            </p>
          )}
          <button
            type="button"
            onClick={onCycleTheme}
            className="mt-4 rounded-md border border-line px-2 py-1 text-xs hover:bg-sunken"
          >
            Theme: {THEME_LABEL[theme]}
          </button>
        </div>
      </aside>
      <main className="mx-auto w-full max-w-7xl px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
        <Outlet />
      </main>
    </div>
  );
}
