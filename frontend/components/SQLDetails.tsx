"use client";

import { useState } from "react";
import { Check, Copy } from "lucide-react";

export function SQLDetails({ sql, duration, open = false }: { sql: string; duration?: number; open?: boolean }) {
  const [copied, setCopied] = useState(false);
  const parts = sql.split(/('(?:''|[^'])*'|\b(?:SELECT|FROM|WHERE|GROUP BY|ORDER BY|LIMIT|JOIN|LEFT|RIGHT|INNER|ON|AS|AND|OR|NOT|NULL|IS|WITH|CASE|WHEN|THEN|ELSE|END|SUM|COUNT|AVG|MIN|MAX|DISTINCT|DESC|ASC|HAVING)\b)/gi);
  return <details className="expandable-details" open={open || undefined}>
    <summary>SQL used <span>{duration != null ? `${duration} ms` : "View query"}</span></summary>
    <div className="sql-block"><div className="sql-toolbar"><span>Read-only · DuckDB</span><button onClick={async () => { try { await navigator.clipboard.writeText(sql); setCopied(true); } catch { setCopied(false); } }}>{copied ? <Check size={13} /> : <Copy size={13} />}{copied ? "Copied" : "Copy query"}</button></div>
      <pre><code>{parts.map((part, i) => <span key={i} className={i % 2 ? (part.startsWith("'") ? "sql-string" : "sql-keyword") : undefined}>{part}</span>)}</code></pre>
    </div>
  </details>;
}
