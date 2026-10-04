import type { AgentMode, ColumnProfile, Dataset } from "@/types";

const targetWords = /(^|_)(target|label|outcome|status|churn|fraud|default|converted|renewed|approved|cancelled|canceled)($|_)/i;
const businessMeasure = /(revenue|sales|amount|spend|value|profit|cost|price|units|quantity|score|rate)/i;
const pretty = (name: string) => name.replaceAll("_", " ");

function take(columns: ColumnProfile[], kind: ColumnProfile["kind"]) {
  return columns.filter(column => column.kind === kind);
}

export function datasetRecommendations(dataset: Dataset | undefined, mode: AgentMode): string[] {
  if (mode === "chat") return [
    "Research the latest developments in a topic",
    "Compare two products using current sources",
    "Summarize what reliable sources say",
    "Fact-check a claim and show the evidence",
  ];
  const columns = dataset?.profile?.columns ?? [];
  const dates = take(columns, "date");
  const measures = take(columns, "measure").sort((a, b) => Number(businessMeasure.test(b.name)) - Number(businessMeasure.test(a.name)));
  const dimensions = take(columns, "dimension");
  const prompts: string[] = [];
  const add = (text?: string) => { if (text && !prompts.includes(text)) prompts.push(text); };

  if (mode === "ml") {
    const target = [...dimensions, ...measures]
      .filter(column => (column.approximate_distinct ?? Infinity) >= 2 && (column.approximate_distinct ?? Infinity) <= 20)
      .sort((a, b) => Number(targetWords.test(b.name)) - Number(targetWords.test(a.name)))[0];
    if (target) add(`Predict ${pretty(target.name)} using the other available fields`);
    if (dates[0] && measures[0]) add(`Forecast ${pretty(measures[0].name)} over ${pretty(dates[0].name)}`);
    if (measures.length >= 2) add(`Find natural groups using ${measures.slice(0, 3).map(column => pretty(column.name)).join(", ")}`);
    if (measures[0]) add(`Detect unusual records based on ${measures.slice(0, 3).map(column => pretty(column.name)).join(", ")}`);
    if (dimensions[0] && !target) add(`Help me define a prediction target from ${pretty(dimensions[0].name)}`);
  } else {
    if (dates[0] && measures[0]) add(`How has ${pretty(measures[0].name)} changed over ${pretty(dates[0].name)}?`);
    if (dimensions[0] && measures[0]) add(`Compare ${pretty(measures[0].name)} by ${pretty(dimensions[0].name)}`);
    if (dimensions[0]) add(`Which ${pretty(dimensions[0].name)} values are most common?`);
    if (measures[0]) add(`Show the distribution and range of ${pretty(measures[0].name)}`);
    if (measures.length >= 2) add(`Are ${pretty(measures[0].name)} and ${pretty(measures[1].name)} related?`);
    if (columns.some(column => (column.null_percent ?? 0) > 0)) add("Which columns have missing values, and how serious is it?");
  }
  if (!prompts.length && columns[0]) add(`Summarize the patterns in ${pretty(columns[0].name)}`);
  return prompts.slice(0, 4);
}
