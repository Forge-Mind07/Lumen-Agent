"use client";

import { Check, LoaderCircle } from "lucide-react";
import dynamic from "next/dynamic";
import type { MLResult as Result } from "@/types";
import { displayValue } from "@/lib/format";
import { ResultTable } from "./ResultTable";
import { SQLDetails } from "./SQLDetails";
import { SuggestedFollowups, Warning } from "./AnswerSummary";

const MLChart = dynamic(() => import("./MLChart").then(module => module.MLChart), { ssr: false, loading: () => <div className="chart-skeleton" aria-label="Loading chart" /> });

const labels: Record<string, string> = { roc_auc: "ROC AUC", pr_auc: "PR AUC", recall: "Recall", precision: "Precision", f1: "F1", accuracy: "Accuracy", rmse: "RMSE", mae: "MAE", r2: "R²", silhouette: "Silhouette", predicted_rows: "Rows scored", matching_rows: "Matching rows", flagged_rows: "Unusual rows", flagged_share: "Flagged share", segments: "Segments" };
const percentages = new Set(["recall", "precision", "accuracy", "flagged_share"]);

export function ModelMetricCard({ name, value, baseline }: { name: string; value: number | null; baseline?: number | null }) {
  const format = (v: number | null) => v == null ? "Unavailable" : percentages.has(name) ? new Intl.NumberFormat("en", { style: "percent", maximumFractionDigits: 1 }).format(v) : displayValue(Math.abs(v) < 10 ? Number(v.toFixed(3)) : v);
  return <div className="metric-card"><span>{labels[name] ?? name.replaceAll("_", " ")}</span><strong>{format(value)}</strong>{baseline != null && <small>Baseline {format(baseline)}</small>}</div>;
}

export function MLRunProgress({ label, status }: { label: string; status: string }) {
  return <div className={`ml-progress ${status === "failed" ? "ml-progress--failed" : ""}`} role="status">{status === "completed" ? <Check size={14} /> : status === "failed" ? <span>!</span> : <LoaderCircle size={14} className="spin" />}<span>{label}</span></div>;
}

function DetailsList({ values }: { values: Record<string, unknown> }) {
  return <dl className="details-list">{Object.entries(values).map(([key, value]) => <div key={key}><dt>{key.replaceAll("_", " ")}</dt><dd>{Array.isArray(value) ? value.join(", ") : value != null && typeof value === "object" ? Object.entries(value).map(([k, v]) => `${k}: ${displayValue(v)}`).join("; ") : displayValue(value)}</dd></div>)}</dl>;
}

export function MLResultCard({ result, onPrompt, busy = false }: { result: Result; onPrompt: (text: string) => void; busy?: boolean }) {
  const metrics = result.metrics ?? {};
  const preferred = result.task === "classification" ? ["roc_auc", "recall", "precision", "f1"] : result.task === "regression" || result.task === "forecasting" ? ["rmse", "mae", "r2"] : Object.keys(metrics);
  const keys = [...preferred.filter(k => k in metrics), ...Object.keys(metrics).filter(k => !preferred.includes(k))].slice(0, 4);
  const charts = result.charts ?? [];
  const primaryCharts = charts.filter(c => ["feature_importance", "actual_vs_predicted", "forecast", "cluster_size", "prediction_distribution", "anomaly_distribution"].includes(c.type)).slice(0, 2);
  const detailCharts = charts.filter(c => !primaryCharts.includes(c));
  const warnings = result.warnings ?? [];
  return <section className="ml-result">
    <div className="ml-result-heading"><span className="eyebrow">{result.task?.replaceAll("_", " ") ?? (result.models ? "Model comparison" : result.target_distribution ? "Data quality" : "Dataset prepared")}</span><h3>{result.target ? `${result.target.replaceAll("_", " ")}` : result.algorithm?.replaceAll("_", " ") ?? (result.rows != null ? `${result.rows.toLocaleString()} rows ready` : "Analysis result")}</h3><p>{result.summary}</p></div>
    {result.positive_class && <p className="muted-note">Positive class: <strong>{result.positive_class}</strong></p>}
    {keys.length > 0 && <div className="metrics-grid">{keys.map(key => <ModelMetricCard key={key} name={key} value={metrics[key]} baseline={result.baseline_metrics?.[key]} />)}</div>}
    {result.prediction != null && <p>Prediction: <strong>{result.prediction}</strong>{result.score != null ? ` · Score ${displayValue(result.score)}` : ""}</p>}
    {warnings.some(w => /did not outperform|leakage|post-outcome|near-perfect/i.test(w)) && <Warning>{warnings.find(w => /did not outperform|duplicates the target|near-perfect|post-outcome/i.test(w)) ?? "Review feature availability and leakage warnings before trusting performance."}</Warning>}
    {primaryCharts.map((chart, i) => <MLChart key={i} chart={chart} />)}
    {result.target_distribution && <MLChart chart={{ type: "class_distribution", title: "Target distribution", data: Object.entries(result.target_distribution).map(([label, value]) => ({ label, value })) }} />}
    {result.cluster_profiles && <div className="segment-list">{result.cluster_profiles.map(p => <details className="expandable-details" key={p.name}><summary>{p.name}<span>{Math.round(p.share * 100)}% · {p.size.toLocaleString()} rows</span></summary><DetailsList values={p.means} /></details>)}</div>}
    {result.predictions && <ResultTable result={result.predictions} title="Predictions" />}
    {result.models && <div className="table-scroll"><table><thead><tr><th>Model</th><th>Task</th><th>Evaluation</th></tr></thead><tbody>{result.models.map(m => <tr key={m.id}><td>{m.name}</td><td>{m.task}</td><td>{Object.entries(m.metrics).slice(0, 3).map(([k,v]) => `${labels[k] ?? k}: ${displayValue(v)}`).join(" · ")}</td></tr>)}</tbody></table></div>}
    <details className="expandable-details"><summary>Experiment details<span>{result.algorithm?.replaceAll("_", " ") ?? "Dataset, methodology & warnings"}</span></summary>
      <DetailsList values={{ ...(result.run_id ? { run: result.run_id, model: result.model_id } : {}), ...(result.dataset ? { dataset: result.dataset.dataset_id, rows: result.dataset.rows, fingerprint: result.dataset.fingerprint } : {}), ...(result.target ? { target: result.target } : {}), ...(result.features ? { features: result.features } : {}), ...(result.split ? { split: result.split } : {}), ...(result.preprocessing ? { preprocessing: result.preprocessing } : {}), ...(result.importance_method ? { explanation: result.importance_method } : {}), ...(result.prediction_dataset_id ? { predictions: result.prediction_dataset_id } : {}), ...(result.evaluation_dataset_id ? { held_out_rows: result.evaluation_dataset_id } : {}) }} />
      {result.excluded_features && <details className="expandable-details"><summary>Excluded features</summary><DetailsList values={result.excluded_features} /></details>}
      {result.columns && <DetailsList values={Object.fromEntries(result.columns.map(c => [c.name, `${c.type}${c.missing_percent != null ? ` · ${c.missing_percent}% missing` : ""}`]))} />}
      {result.candidates && <><h4>Models tested · validation metrics</h4><div className="table-scroll"><table><thead><tr><th>Candidate</th><th>Result</th></tr></thead><tbody>{result.candidates.map(c => <tr key={c.algorithm}><td>{c.algorithm.replaceAll("_", " ")}</td><td>{c.status === "failed" ? c.error : Object.entries(c.validation_metrics ?? {}).map(([k, v]) => `${labels[k] ?? k}: ${displayValue(v)}`).join(" · ")}</td></tr>)}</tbody></table></div></>}
      {result.model_configuration && <details className="expandable-details"><summary>Model configuration</summary><DetailsList values={result.model_configuration} /></details>}
      {result.runtime_versions && <details className="expandable-details"><summary>Runtime versions</summary><DetailsList values={result.runtime_versions} /></details>}
      {Object.keys(metrics).length > 0 && <DetailsList values={metrics} />}
      {detailCharts.map((chart, i) => <MLChart key={i} chart={chart} />)}
      {(result.dataset?.source_sql || result.source_sql) && <SQLDetails sql={result.dataset?.source_sql ?? result.source_sql ?? ""} />}
      {warnings.length > 0 && <div className="ml-warnings"><h4>Warnings & limitations</h4><ul>{warnings.map((w, i) => <li key={i}>{w}</li>)}</ul></div>}
    </details>
    {!busy && result.model_id && <SuggestedFollowups onSelect={onPrompt} prompts={result.task === "classification" || result.task === "regression" ? ["What are the strongest drivers?", "Compare with the previous model", "Run predictions on this dataset"] : result.task === "clustering" ? ["View customers in each segment", "Explain the segment differences"] : ["Explain the methodology", "Show the predictions"]} />}
  </section>;
}
