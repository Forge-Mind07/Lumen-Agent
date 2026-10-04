"use client";

import { X } from "lucide-react";
import { useState } from "react";
import type { Dataset } from "@/types";
import { compactNumber, displayValue, fileSize } from "@/lib/format";

export function DatasetOverview({ dataset, onClose }: { dataset: Dataset; onClose: () => void }) {
  const [search, setSearch] = useState("");
  const groups = ["date", "dimension", "measure", "other"] as const;
  return (
    <aside className="inspector">
      <div className="inspector-head"><div><span>Dataset</span><strong>{dataset.filename}</strong></div><button onClick={onClose}><X size={17} /></button></div>
      <div className="overview-stats">
        <div><span>Rows</span><strong>{compactNumber(dataset.row_count)}</strong></div>
        <div><span>Columns</span><strong>{dataset.column_count}</strong></div>
        <div><span>Size</span><strong>{fileSize(dataset.size_bytes)}</strong></div>
      </div>
      <div className="schema-list">
        <div className="section-label">Table · data</div>
        {(dataset.column_count ?? 0) > 10 && <label className="dataset-search"><input value={search} onChange={e => setSearch(e.target.value)} placeholder="Find a column" aria-label="Find a column" /></label>}
        {dataset.profile?.profile_is_partial && <p className="empty-note">Schema preview is limited. Ask the agent about other columns.</p>}
        {groups.map((group) => {
          const columns = dataset.profile?.columns.filter((column) => column.kind === group && column.name.toLowerCase().includes(search.toLowerCase())) ?? [];
          if (!columns.length) return null;
          return <section key={group}><h3>{group === "date" ? "Dates" : group === "dimension" ? "Dimensions" : group === "measure" ? "Measures" : "Other"}</h3>{columns.map((column) => (
            <details className="schema-column" key={column.name}>
              <summary><strong>{column.name}</strong><code>{column.data_type}</code></summary>
              <p>{compactNumber(column.approximate_distinct)} distinct · {column.null_percent ?? 0}% missing</p>
              {(column.minimum != null || column.maximum != null) && <small>{displayValue(column.minimum)} → {displayValue(column.maximum)}</small>}
            </details>
          ))}</section>;
        })}
      </div>
    </aside>
  );
}
