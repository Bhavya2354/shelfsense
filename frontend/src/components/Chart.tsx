import { BarChart, LineChart, ScatterChart } from "echarts/charts";
import {
  GridComponent,
  LegendComponent,
  MarkAreaComponent,
  MarkLineComponent,
  TooltipComponent,
} from "echarts/components";
import * as echarts from "echarts/core";
import { SVGRenderer } from "echarts/renderers";
import { useContext, useEffect, useRef } from "react";

import { ThemeContext } from "./ThemeContext";

echarts.use([
  LineChart,
  BarChart,
  ScatterChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  MarkLineComponent,
  MarkAreaComponent,
  SVGRenderer,
]);

export type ChartOption = echarts.EChartsCoreOption;

export interface ChartTokens {
  ink: string;
  inkSoft: string;
  muted: string;
  line: string;
  surface: string;
  accent: string;
  band: string;
  amber: string;
  red: string;
  series: string[];
}

function readTokens(): ChartTokens {
  const css = getComputedStyle(document.documentElement);
  const v = (name: string) => css.getPropertyValue(name).trim();
  return {
    ink: v("--ink"),
    inkSoft: v("--ink-soft"),
    muted: v("--muted"),
    line: v("--line"),
    surface: v("--surface"),
    accent: v("--accent"),
    band: v("--band"),
    amber: v("--amber"),
    red: v("--red"),
    series: [1, 2, 3, 4, 5, 6].map((i) => v(`--chart-${i}`)),
  };
}

/** Shared axis, grid and tooltip styling so every chart reads as one system. */
export function baseOption(t: ChartTokens): ChartOption {
  const axis = {
    axisLine: { lineStyle: { color: t.line } },
    axisTick: { show: false },
    axisLabel: { color: t.muted, fontFamily: "IBM Plex Sans", fontSize: 11 },
    splitLine: { lineStyle: { color: t.line, type: "dashed" as const } },
  };
  return {
    animationDuration: 300,
    textStyle: { fontFamily: "IBM Plex Sans", color: t.inkSoft },
    grid: { left: 8, right: 16, top: 28, bottom: 8, containLabel: true },
    tooltip: {
      trigger: "axis",
      backgroundColor: t.surface,
      borderColor: t.line,
      textStyle: { color: t.ink, fontSize: 12 },
      axisPointer: { lineStyle: { color: t.muted } },
    },
    legend: {
      top: 0,
      left: 0,
      icon: "roundRect",
      itemWidth: 12,
      itemHeight: 3,
      textStyle: { color: t.muted, fontSize: 12 },
    },
    xAxis: { ...axis, splitLine: { show: false } },
    yAxis: { ...axis, axisLine: { show: false } },
  };
}

interface ChartProps {
  build: (tokens: ChartTokens) => ChartOption;
  height: number;
  label: string;
}

export function Chart({ build, height, label }: ChartProps) {
  const host = useRef<HTMLDivElement>(null);
  const instance = useRef<echarts.ECharts | null>(null);
  const scheme = useContext(ThemeContext).resolved;

  useEffect(() => {
    if (!host.current) return;
    const chart = echarts.init(host.current, undefined, { renderer: "svg" });
    instance.current = chart;
    const observer = new ResizeObserver(() => chart.resize());
    observer.observe(host.current);
    return () => {
      observer.disconnect();
      chart.dispose();
      instance.current = null;
    };
  }, []);

  useEffect(() => {
    instance.current?.setOption(build(readTokens()), true);
  }, [build, scheme]);

  return <div ref={host} role="img" aria-label={label} style={{ height, width: "100%" }} />;
}
