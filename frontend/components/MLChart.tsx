"use client";

import type { EChartsCoreOption } from "echarts/core";
import type { MLChart as Chart } from "@/types";
import { compactNumber } from "@/lib/format";
import { EChartCanvas, type ChartColors } from "./EChartCanvas";
import { ResultTable } from "./ResultTable";

const axis = (colors: ChartColors) => ({ axisLine: { lineStyle: { color: colors.grid } }, axisTick: { show: false }, axisLabel: { color: colors.muted, fontSize: 10 }, splitLine: { lineStyle: { color: colors.grid } } });
const valueAxis = (colors: ChartColors) => ({ type: "value" as const, ...axis(colors), axisLabel: { color: colors.muted, formatter: (value: number) => compactNumber(value) } });
const base = (colors: ChartColors): EChartsCoreOption => ({ animationDuration: 480, color: [colors.accent, colors.accent2, colors.danger], aria: { enabled: true }, grid: { left: 18, right: 22, top: 18, bottom: 34, containLabel: true }, tooltip: { trigger: "axis", confine: true, backgroundColor: colors.surface, borderColor: colors.border, textStyle: { color: colors.text, fontSize: 11 } }, toolbox: { right: 8, top: -5, itemSize: 13, iconStyle: { borderColor: colors.muted }, feature: { restore: {}, saveAsImage: { pixelRatio: 2, backgroundColor: colors.surface } } } });

export function MLChart({ chart }: { chart: Chart }) {
  const data = (chart.data ?? []).slice(0, 600);
  if (chart.type === "confusion_matrix") return <figure className="chart-card"><figcaption className="chart-title"><strong>{chart.title}</strong><span>Rows: actual · Columns: predicted</span></figcaption><div className="table-scroll"><table className="confusion-matrix"><thead><tr><th>Actual / predicted</th>{chart.labels?.map(label => <th key={label}>{label}</th>)}</tr></thead><tbody>{chart.matrix?.map((row, i) => <tr key={i}><th>{chart.labels?.[i]}</th>{row.map((value, j) => <td key={j} className={i === j ? "matrix-correct" : "numeric"}>{value.toLocaleString()}</td>)}</tr>)}</tbody></table></div></figure>;
  if (!data.length) return null;
  const columns = [...new Set(data.flatMap(row => Object.keys(row)))];
  const source = { result_id: "", sql: "", duration_ms: 0, returned_row_count: data.length, truncated: (chart.data?.length ?? 0) > data.length, columns, column_types: columns.map(column => data.every(row => row[column] == null || typeof row[column] === "number") ? "DOUBLE" : "VARCHAR"), rows: data.map(row => columns.map(column => row[column] ?? null)) };
  return <><figure className="chart-card chart-card--premium" aria-label={chart.title}><figcaption className="chart-title"><strong>{chart.title}</strong><span className="chart-engine">Interactive</span></figcaption><div className="chart-area"><EChartCanvas label={chart.title} buildOption={colors => option(chart, data, colors)} /></div></figure><ResultTable result={source} title="Chart data" /></>;
}

function option(chart: Chart, data: Record<string, unknown>[], colors: ChartColors): EChartsCoreOption {
  const common = base(colors);
  if (chart.type === "feature_importance") {
    const rows = data.slice(0, 12).reverse();
    return { ...common, grid: { left: 16, right: 24, top: 16, bottom: 24, containLabel: true }, xAxis: valueAxis(colors), yAxis: { type: "category", data: rows.map(row => row.feature), ...axis(colors), inverse: false }, series: [{ type: "bar", data: rows.map(row => row.importance), barMaxWidth: 25, itemStyle: { borderRadius: [0, 6, 6, 0], color: { type: "linear", x: 0, y: 0, x2: 1, y2: 0, colorStops: [{ offset: 0, color: colors.accent }, { offset: 1, color: colors.accent2 }] } } }] };
  }
  if (chart.type === "actual_vs_predicted" || chart.type === "cluster_scatter") {
    const x = chart.type === "actual_vs_predicted" ? "actual" : "x", y = chart.type === "actual_vs_predicted" ? "predicted" : "y";
    return { ...common, tooltip: { ...(common.tooltip as object), trigger: "item" }, xAxis: valueAxis(colors), yAxis: valueAxis(colors), series: [{ type: "scatter", data: data.map(row => [row[x], row[y]]), symbolSize: 8, large: data.length > 400, itemStyle: { color: colors.accent, opacity: .7, shadowBlur: 7, shadowColor: `${colors.accent}55` }, emphasis: { scale: 1.7 } }] };
  }
  if (chart.type === "forecast") {
    const dates = data.map(row => String(row.date ?? ""));
    return { ...common, legend: { top: 0, textStyle: { color: colors.muted } }, grid: { ...(common.grid as object), top: 42 }, xAxis: { type: "category", data: dates, ...axis(colors), axisLabel: { color: colors.muted, hideOverlap: true, formatter: (value: string) => value.slice(0, 10) } }, yAxis: valueAxis(colors), series: [
      { type: "line", name: "Observed", data: data.map(row => row.actual), showSymbol: false, smooth: .2, lineStyle: { color: colors.muted, width: 2 } },
      { type: "line", name: "Forecast", data: data.map(row => row.prediction), showSymbol: true, symbolSize: 5, smooth: .2, lineStyle: { color: colors.accent, width: 3, type: "dashed" }, itemStyle: { color: colors.accent } },
      { type: "line", name: "Upper range", data: data.map(row => row.upper), showSymbol: false, lineStyle: { opacity: 0 }, stack: "range", areaStyle: { color: `${colors.accent}20` } },
    ] };
  }
  if (["roc_curve", "precision_recall_curve", "anomaly_timeline"].includes(chart.type)) {
    const category = chart.type === "anomaly_timeline";
    return { ...common, xAxis: category ? { type: "category", data: data.map(row => row.x), ...axis(colors), axisLabel: { color: colors.muted, hideOverlap: true } } : { ...valueAxis(colors), min: 0, max: 1 }, yAxis: chart.type === "anomaly_timeline" ? valueAxis(colors) : { ...valueAxis(colors), min: 0, max: 1 }, series: [{ type: "line", data: data.map(row => category ? row.y : [row.x, row.y]), smooth: .2, showSymbol: data.length < 30, sampling: "lttb", lineStyle: { color: colors.accent, width: 3 }, areaStyle: chart.type === "anomaly_timeline" ? { color: `${colors.accent}20` } : undefined }] };
  }
  return { ...common, xAxis: { type: "category", data: data.map(row => row.label), ...axis(colors), axisLabel: { color: colors.muted, hideOverlap: true } }, yAxis: valueAxis(colors), series: [{ type: "bar", data: data.map(row => row.value), barMaxWidth: 44, itemStyle: { borderRadius: [6, 6, 1, 1], color: { type: "linear", x: 0, y: 1, x2: 0, y2: 0, colorStops: [{ offset: 0, color: colors.accent }, { offset: 1, color: colors.accent2 }] } } }] };
}
