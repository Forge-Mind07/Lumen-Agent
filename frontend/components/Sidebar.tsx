"use client";

import { useState } from "react";
import { Database, FileSpreadsheet, History, Moon, PanelLeftClose, PanelLeftOpen, Plus, Search, Sun, Trash2 } from "lucide-react";
import type { AgentMode, Dataset, Session } from "@/types";
import { compactNumber } from "@/lib/format";

type Props = { mode: AgentMode; collapsed: boolean; datasets: Dataset[]; sessions: Session[]; selectedId?: string; sessionId?: string; theme: "dark" | "light"; loading?: boolean; error?: string; disabled?: boolean; onCollapse: () => void; onSelect: (dataset: Dataset) => void; onDelete: (dataset: Dataset) => void; onSession: (session: Session) => void; onNew: () => void; onUpload: () => void; onTheme: () => void; onRetry: () => void; };
export function Sidebar({ mode, collapsed, datasets, sessions, selectedId, sessionId, theme, loading, error, disabled, onCollapse, onSelect, onDelete, onSession, onNew, onUpload, onTheme, onRetry }: Props) {
  const [search, setSearch] = useState("");
  const filtered = datasets.filter(d => d.filename.toLowerCase().includes(search.toLowerCase()));
  return <aside className={`sidebar ${collapsed ? "sidebar--collapsed" : ""}`}>
    <div className="brand-row"><div className="brand-mark"><span /></div>{!collapsed && <span className="brand-name">Lumen</span>}<button className="icon-button collapse-button" onClick={onCollapse} aria-label="Toggle sidebar">{collapsed ? <PanelLeftOpen size={17} /> : <PanelLeftClose size={17} />}</button></div>
    <button className="new-analysis" disabled={disabled} onClick={onNew} title="New analysis"><Plus size={17} />{!collapsed && <span>New conversation</span>}</button>
    <nav className="sidebar-scroll" aria-label="Datasets and conversations">
      {mode !== "chat" && !collapsed && <div className="section-label">Local workspace</div>}
      {mode === "chat" && !collapsed && <div className="section-label">Research conversations</div>}
      {mode !== "chat" && <>
      {!collapsed && datasets.length > 5 && <label className="dataset-search"><Search size={14} /><input value={search} onChange={e => setSearch(e.target.value)} placeholder="Find a dataset" aria-label="Find a dataset" /></label>}
      {loading && !collapsed && <p className="empty-note">Loading datasets…</p>}
      {error && !collapsed && <div className="connection-error"><p>{error}</p><button onClick={onRetry}>Reconnect</button></div>}
      {!loading && !error && !datasets.length && !collapsed && <p className="empty-note">Add a CSV or Parquet file to get started.</p>}
      <div className="nav-list">{filtered.map(dataset => <div key={dataset.id}>
        <div className="dataset-row"><button disabled={disabled || dataset.status !== "ready"} aria-current={selectedId === dataset.id ? "true" : undefined} className={`nav-item ${selectedId === dataset.id ? "nav-item--active" : ""}`} onClick={() => onSelect(dataset)} title={dataset.error ?? dataset.filename}><FileSpreadsheet size={16} />{!collapsed && <span className="nav-copy"><span>{dataset.filename}</span><small>{dataset.status === "ready" ? `${compactNumber(dataset.row_count)} rows · ${dataset.column_count} columns` : dataset.status === "error" ? "Could not read file" : "Preparing…"}</small></span>}</button>{!collapsed && <button className="dataset-delete" disabled={disabled} onClick={() => onDelete(dataset)} aria-label={`Delete ${dataset.filename}`} title="Delete dataset"><Trash2 size={14} /></button>}</div>
        {!collapsed && selectedId === dataset.id && dataset.profile && <details className="sidebar-schema"><summary>data <span>View schema</span></summary>{dataset.profile.columns.map(c => <div key={c.name} title={`${c.name} · ${c.null_percent ?? "Unknown"}% missing`}><span>{c.name}</span><code>{c.data_type}</code></div>)}{dataset.profile.profile_is_partial && <small>Partial schema preview</small>}</details>}
      </div>)}{!filtered.length && search && <p className="empty-note">No matching datasets</p>}<button className="nav-item nav-item--muted" disabled={disabled} onClick={onUpload} title="Add dataset"><Database size={16} />{!collapsed && <span>Add dataset</span>}</button></div></>}
      {!collapsed && sessions.length > 0 && <><div className="section-label section-label--spaced">Recent conversations</div><div className="nav-list">{sessions.slice(0, 12).map(session => <button disabled={disabled} className={`nav-item recent-item ${session.id === sessionId ? "nav-item--active" : ""}`} key={session.id} onClick={() => onSession(session)} title={session.title}><History size={15} /><span className="nav-copy"><span>{session.title}</span><small>{session.mode === "chat" ? "Research Chat" : session.mode === "ml" ? "ML Agent" : "SQL Agent"}</small></span></button>)}</div></>}
    </nav><div className="sidebar-footer"><button className="nav-item" onClick={onTheme} title="Toggle theme">{theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}{!collapsed && <span>{theme === "dark" ? "Light theme" : "Dark theme"}</span>}</button></div>
  </aside>;
}
