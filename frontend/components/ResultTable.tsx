"use client";

import { useMemo, useState } from "react";
import { ArrowDownUp, ChevronLeft, ChevronRight, Copy, Download } from "lucide-react";
import { API_URL } from "@/lib/api";
import { displayValue } from "@/lib/format";
import type { QueryResult } from "@/types";
import { SQLDetails } from "./SQLDetails";

export function ResultTable({ result, title = "Data", expanded = false }: { result: QueryResult; title?: string; expanded?: boolean }) {
  const [sort, setSort] = useState<{ column: number; direction: 1 | -1 } | null>(null);
  const [page, setPage] = useState(0);
  const [copied, setCopied] = useState(false);
  const rows = useMemo(() => !sort ? result.rows : [...result.rows].sort((a, b) => {
    const left = a[sort.column], right = b[sort.column];
    if (left == null) return right == null ? 0 : 1;
    if (right == null) return -1;
    return (typeof left === "number" && typeof right === "number" ? left - right : String(left).localeCompare(String(right), undefined, { numeric: true })) * sort.direction;
  }), [result.rows, sort]);
  const pages = Math.max(1, Math.ceil(rows.length / 12));
  const current = Math.min(page, pages - 1);
  return <div className="data-result" id={result.result_id ? `result-${result.result_id}` : undefined}>
    <details className="expandable-details data-details" open={expanded || undefined}>
      <summary>{title}<span>{result.returned_row_count.toLocaleString()} rows{result.truncated ? " · limited preview" : ""} · View data</span></summary>
      <div className="result-card">
        <div className="result-toolbar"><div><strong>{title}</strong><span>Sort applies to the returned rows</span></div><div className="table-actions">
          <button className="table-action" onClick={async () => { try { await navigator.clipboard.writeText([result.columns, ...rows].map(r => r.map(v => String(v ?? "").replace(/[\t\n]/g, " ")).join("\t")).join("\n")); setCopied(true); } catch { setCopied(false); } }}><Copy size={13} />{copied ? "Copied" : "Copy"}</button>
          {result.result_id && <a href={`${API_URL}/api/artifacts/${result.result_id}/csv`} className="table-action"><Download size={13} />CSV</a>}
        </div></div>
        <div className="table-scroll"><table><thead><tr>{result.columns.map((column, index) => <th key={column} aria-sort={sort?.column === index ? (sort.direction === 1 ? "ascending" : "descending") : "none"} className={/INT|FLOAT|DOUBLE|DECIMAL|REAL/.test(result.column_types[index] ?? "") ? "numeric" : ""}>
          <button onClick={() => { setSort({ column: index, direction: sort?.column === index && sort.direction === 1 ? -1 : 1 }); setPage(0); }}>{column}<ArrowDownUp size={11} /></button>
        </th>)}</tr></thead><tbody>{rows.slice(current * 12, current * 12 + 12).map((row, ri) => <tr key={ri}>{row.map((value, ci) => <td key={ci} className={`${typeof value === "number" ? "numeric" : ""} ${value == null ? "null-cell" : ""}`}><span title={displayValue(value)}>{displayValue(value, result.column_types[ci], result.columns[ci])}</span></td>)}</tr>)}</tbody></table>
        {!rows.length && <p className="empty-note">No rows match this query.</p>}</div>
        <div className="table-foot"><span>{rows.length ? `${current * 12 + 1}–${Math.min(current * 12 + 12, rows.length)} of ${rows.length}` : "0 rows"}{result.truncated ? " · Result capped; refine the query for more detail. Export contains this preview." : ""}</span><div><button aria-label="Previous page" disabled={!current} onClick={() => setPage(current - 1)}><ChevronLeft size={15} /></button><button aria-label="Next page" disabled={current + 1 >= pages} onClick={() => setPage(current + 1)}><ChevronRight size={15} /></button></div></div>
      </div>
    </details>
    {result.sql && <SQLDetails sql={result.sql} duration={result.duration_ms} />}
  </div>;
}
