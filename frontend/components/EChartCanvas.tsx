"use client";

import { useEffect, useRef } from "react";
import * as echarts from "echarts/core";
import { BarChart, LineChart, PieChart, ScatterChart } from "echarts/charts";
import { AriaComponent, DataZoomComponent, GridComponent, LegendComponent, MarkLineComponent, MarkPointComponent, ToolboxComponent, TooltipComponent } from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import type { EChartsCoreOption } from "echarts/core";

echarts.use([BarChart, LineChart, PieChart, ScatterChart, AriaComponent, DataZoomComponent, GridComponent, LegendComponent, MarkLineComponent, MarkPointComponent, ToolboxComponent, TooltipComponent, CanvasRenderer]);

export type ChartColors = { accent: string; accent2: string; text: string; muted: string; grid: string; surface: string; border: string; success: string; danger: string };

function colors(): ChartColors {
  const style = getComputedStyle(document.documentElement);
  const read = (name: string, fallback: string) => style.getPropertyValue(name).trim() || fallback;
  return {
    accent: read("--accent", "#8391ff"), accent2: read("--chart-accent-2", "#58c7b0"),
    text: read("--text", "#eef0f4"), muted: read("--muted", "#838b98"), grid: read("--chart-grid", "#262b33"),
    surface: read("--surface-strong", "#191d23"), border: read("--border", "#292e37"),
    success: read("--success", "#55c2a3"), danger: read("--danger", "#e07b8c"),
  };
}

export function EChartCanvas({ buildOption, label }: { buildOption: (colors: ChartColors) => EChartsCoreOption; label: string }) {
  const element = useRef<HTMLDivElement>(null);
  const builder = useRef(buildOption);
  builder.current = buildOption;
  useEffect(() => {
    if (!element.current) return;
    const chart = echarts.init(element.current, undefined, { renderer: "canvas", useDirtyRect: true });
    const render = () => chart.setOption(builder.current(colors()), { notMerge: true, lazyUpdate: true });
    render();
    const resize = new ResizeObserver(() => chart.resize());
    resize.observe(element.current);
    const theme = new MutationObserver(render);
    theme.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => { resize.disconnect(); theme.disconnect(); chart.dispose(); };
  }, []);
  return <div ref={element} className="echart-canvas" role="img" aria-label={label} />;
}
