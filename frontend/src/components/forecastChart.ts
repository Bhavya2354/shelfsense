import type { DailyPoint, ForecastPoint } from "../api/types";
import { fmtDay, fmtUnits } from "../lib/format";
import { baseOption, type ChartOption, type ChartTokens } from "./Chart";

export interface ForecastSeries {
  name: string;
  points: ForecastPoint[];
  showBand: boolean;
}

/** History as a quiet line, forecasts as coloured lines, the first one with its P10–P90 band. */
export function forecastOption(t: ChartTokens, history: DailyPoint[], forecasts: ForecastSeries[]): ChartOption {
  const days = [
    ...new Set([...history.map((p) => p.day), ...forecasts.flatMap((f) => f.points.map((p) => p.day))]),
  ].sort();
  const index = new Map(days.map((d, i) => [d, i]));
  const column = <T,>(points: T[], day: (p: T) => string, value: (p: T) => number | null) => {
    const out: (number | null)[] = days.map(() => null);
    for (const p of points) out[index.get(day(p))!] = value(p);
    return out;
  };

  const series: object[] = [
    {
      name: "Actual",
      type: "line",
      data: column(history, (p) => p.day, (p) => p.units),
      showSymbol: false,
      lineStyle: { color: t.inkSoft, width: 1.5 },
      itemStyle: { color: t.inkSoft },
    },
  ];
  forecasts.forEach((f, i) => {
    const color = t.series[i % t.series.length]!;
    if (f.showBand && f.points.some((p) => p.p10 !== null)) {
      series.push(
        {
          name: `${f.name} band floor`,
          type: "line",
          stack: `band-${i}`,
          data: column(f.points, (p) => p.day, (p) => p.p10),
          lineStyle: { opacity: 0 },
          showSymbol: false,
          silent: true,
          tooltip: { show: false },
        },
        {
          name: "P10–P90",
          type: "line",
          stack: `band-${i}`,
          data: column(f.points, (p) => p.day, (p) => (p.p90 ?? p.p50) - (p.p10 ?? p.p50)),
          lineStyle: { opacity: 0 },
          areaStyle: { color: t.band },
          showSymbol: false,
          silent: true,
          tooltip: { show: false },
        },
      );
    }
    series.push({
      name: f.name,
      type: "line",
      data: column(f.points, (p) => p.day, (p) => p.p50),
      showSymbol: false,
      lineStyle: { color, width: 2, type: i === 0 ? "solid" : "dashed" },
      itemStyle: { color },
    });
  });

  const promoDays = history.filter((p) => p.onpromotion).map((p) => p.day);
  forecasts[0]?.points.filter((p) => p.onpromotion).forEach((p) => promoDays.push(p.day));
  if (promoDays.length) {
    series.push({
      name: "On promotion",
      type: "scatter",
      data: promoDays.map((d) => [index.get(d), 0]),
      symbol: "rect",
      symbolSize: [6, 3],
      itemStyle: { color: t.amber },
      tooltip: { show: false },
    });
  }

  const firstForecast = forecasts[0]?.points[0]?.day;
  const base = baseOption(t);
  return {
    ...base,
    legend: {
      ...(base.legend as object),
      data: ["Actual", ...forecasts.map((f) => f.name), ...(promoDays.length ? ["On promotion"] : [])],
    },
    xAxis: { ...(base.xAxis as object), type: "category", data: days, axisLabel: { ...(base.xAxis as { axisLabel: object }).axisLabel, formatter: fmtDay } },
    yAxis: { ...(base.yAxis as object), type: "value", name: "units", nameTextStyle: { color: t.muted, fontSize: 11 } },
    tooltip: {
      ...(base.tooltip as object),
      valueFormatter: (v: unknown) => (typeof v === "number" ? fmtUnits(v) : "–"),
    },
    series: firstForecast
      ? [
          ...series,
          {
            type: "line",
            data: [],
            markLine: {
              silent: true,
              symbol: "none",
              label: { formatter: "forecast", color: t.muted, fontSize: 11, position: "insideEndTop" },
              lineStyle: { color: t.muted, type: "dotted" },
              data: [{ xAxis: firstForecast }],
            },
          },
        ]
      : series,
  };
}
