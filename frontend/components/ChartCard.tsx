"use client";

import type { EChartsCoreOption } from "echarts/core";
import type { ChartPayload } from "@/types";
import { compactNumber, displayValue } from "@/lib/format";
import { EChartCanvas, type ChartColors } from "./EChartCanvas";
import { ResultTable } from "./ResultTable";

export function ChartCard({ chart }: { chart: ChartPayload }) {
  const { spec, source } = chart;
  const data = source.rows.slice(0, 1000).map(row => Object.fromEntries(source.columns.map((column, i) => [column, row[i]])));
  if (spec.type === "table") return <ResultTable result={source} title={spec.title} expanded />;
  if (spec.type === "kpi") return <div className="metric-card"><span>{spec.title}</span><strong>{displayValue(data[0]?.[spec.y ?? spec.x])}</strong>{source.truncated && <small>Limited result</small>}</div>;
  if (!data.length) return <div className="empty-note">{spec.title} · No data to plot</div>;
  const subtitle = `${spec.x.replaceAll("_", " ")}${spec.y ? ` · ${spec.y.replaceAll("_", " ")}` : ""}${source.truncated ? " · limited data" : ""}`;
  return <figure className="chart-card chart-card--premium" aria-label={spec.title}>
    <figcaption className="chart-title"><div><strong>{spec.title}</strong><span>{subtitle}</span></div><span className="chart-engine">Interactive</span></figcaption>
    <div className="chart-area"><EChartCanvas label={spec.title} buildOption={colors => buildOption(spec, data, colors)} /></div>
  </figure>;
}

function base(colors: ChartColors, length: number): EChartsCoreOption {
  return {
    animationDuration: length > 180 ? 0 : 520,
    animationEasing: "cubicOut",
    color: [colors.accent, colors.accent2, "#d49b68", "#a88be8", colors.success, colors.danger],
    aria: { enabled: true, decal: { show: false } },
    grid: { left: 18, right: 22, top: 20, bottom: length > 24 ? 58 : 34, containLabel: true },
    tooltip: { trigger: "axis", confine: true, backgroundColor: colors.surface, borderColor: colors.border, borderWidth: 1, textStyle: { color: colors.text, fontSize: 12 }, axisPointer: { type: "shadow", shadowStyle: { color: `${colors.accent}12` } } },
    toolbox: { right: 8, top: -4, itemSize: 13, iconStyle: { borderColor: colors.muted }, emphasis: { iconStyle: { borderColor: colors.accent } }, feature: { restore: { title: "Reset" }, saveAsImage: { title: "Save image", pixelRatio: 2, backgroundColor: colors.surface } } },
    dataZoom: length > 28 ? [{ type: "inside", filterMode: "none", zoomOnMouseWheel: "shift", moveOnMouseMove: true }, { type: "slider", height: 13, bottom: 4, borderColor: "transparent", backgroundColor: colors.grid, fillerColor: `${colors.accent}45`, handleStyle: { color: colors.accent }, textStyle: { color: colors.muted } }] : [],
  };
}

function categoryAxis(name: string, data: unknown[], colors: ChartColors) {
  return { type: "category" as const, name: name.replaceAll("_", " "), nameLocation: "middle" as const, nameGap: 29, data, axisTick: { show: false }, axisLine: { lineStyle: { color: colors.grid } }, axisLabel: { color: colors.muted, fontSize: 11, hideOverlap: true, formatter: (value: string) => value.length > 18 ? `${value.slice(0, 17)}…` : value } };
}

function valueAxis(name: string, colors: ChartColors) {
  return { type: "value" as const, name: name.replaceAll("_", " "), nameTextStyle: { color: colors.muted, fontSize: 10 }, splitLine: { lineStyle: { color: colors.grid } }, axisLabel: { color: colors.muted, formatter: (value: number) => compactNumber(value) }, axisLine: { show: false }, axisTick: { show: false } };
}

function buildOption(spec: ChartPayload["spec"], data: Record<string, unknown>[], colors: ChartColors): EChartsCoreOption {
  const y = spec.y ?? spec.x;
  const common = base(colors, data.length);
  if (spec.type === "scatter") return { ...common, tooltip: { ...(common.tooltip as object), trigger: "item" }, xAxis: valueAxis(spec.x, colors), yAxis: valueAxis(y, colors), series: [{ type: "scatter", name: y, data: data.filter(row => typeof row[spec.x] === "number" && typeof row[y] === "number").map(row => [row[spec.x], row[y]]), symbolSize: 9, itemStyle: { color: colors.accent, opacity: .72, shadowBlur: 8, shadowColor: `${colors.accent}55` }, emphasis: { scale: 1.6, itemStyle: { opacity: 1 } }, large: data.length > 500, largeThreshold: 500 }] };
  if (spec.type === "pie") return { ...common, grid: undefined, tooltip: { ...(common.tooltip as object), trigger: "item", valueFormatter: (value: unknown) => displayValue(value) }, legend: { bottom: 3, textStyle: { color: colors.muted }, type: "scroll" }, series: [{ type: "pie", radius: ["48%", "76%"], center: ["50%", "46%"], padAngle: 2, itemStyle: { borderRadius: 6, borderColor: colors.surface, borderWidth: 2 }, label: { color: colors.muted, formatter: "{b}\n{d}%" }, emphasis: { scaleSize: 8 }, data: data.map(row => ({ name: String(row[spec.x]), value: row[y] })) }] };

  const categories = [...new Set(data.map(row => row[spec.x]))];
  if (spec.series) {
    const names = [...new Set(data.map(row => String(row[spec.series!])))].slice(0, 8);
    const lookup = new Map<unknown, Map<string, unknown>>();
    for (const row of data) { const values = lookup.get(row[spec.x]) ?? new Map<string, unknown>(); values.set(String(row[spec.series]), row[y]); lookup.set(row[spec.x], values); }
    return { ...common, legend: { top: 0, left: 8, textStyle: { color: colors.muted }, type: "scroll" }, grid: { ...(common.grid as object), top: 45 }, xAxis: categoryAxis(spec.x, categories, colors), yAxis: valueAxis(y, colors), series: names.map(name => ({ type: "bar", name, stack: spec.type === "stacked_bar" ? "total" : undefined, data: categories.map(category => lookup.get(category)?.get(name) ?? null), barMaxWidth: 42, itemStyle: { borderRadius: spec.type === "stacked_bar" ? 0 : [5, 5, 1, 1] }, emphasis: { focus: "series" } })) };
  }
  if (spec.type === "line" || spec.type === "area") return { ...common, xAxis: categoryAxis(spec.x, categories, colors), yAxis: valueAxis(y, colors), series: [{ type: "line", name: y, data: data.map(row => row[y]), smooth: .28, showSymbol: data.length < 18, symbolSize: 6, sampling: "lttb", connectNulls: false, lineStyle: { width: 3, cap: "round", color: colors.accent, shadowBlur: 8, shadowColor: `${colors.accent}35` }, itemStyle: { color: colors.accent }, areaStyle: spec.type === "area" ? { opacity: 1, color: { type: "linear", x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: `${colors.accent}70` }, { offset: 1, color: `${colors.accent}05` }] } } : undefined, emphasis: { focus: "series" } }] };

  const horizontal = categories.length <= 30 && (categories.length > 9 || categories.some(value => String(value).length > 14));
  const series = [{ type: "bar" as const, name: y, data: data.map(row => row[y]), barMaxWidth: 44, realtimeSort: false, itemStyle: { borderRadius: horizontal ? [0, 6, 6, 0] : [6, 6, 1, 1], color: { type: "linear", x: 0, y: horizontal ? 0 : 1, x2: horizontal ? 1 : 0, y2: 0, colorStops: [{ offset: 0, color: colors.accent }, { offset: 1, color: colors.accent2 }] }, shadowBlur: 7, shadowColor: `${colors.accent}28` }, emphasis: { itemStyle: { shadowBlur: 16, shadowColor: `${colors.accent}55` } } }];
  return horizontal
    ? { ...common, grid: { ...(common.grid as object), bottom: 24 }, dataZoom: [], xAxis: valueAxis(y, colors), yAxis: { ...categoryAxis(spec.x, categories, colors), inverse: true }, series }
    : { ...common, xAxis: categoryAxis(spec.x, categories, colors), yAxis: valueAxis(y, colors), series };
}
