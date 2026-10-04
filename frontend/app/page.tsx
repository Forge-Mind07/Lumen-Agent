"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { AgentEvent, AgentMode, ChartPayload, ConversationItem, Dataset, MLProject, MLResult, QueryResult, Session } from "@/types";
import { deleteDataset, getSession, listDatasets, listSessions, streamAnalysis, uploadDataset } from "@/lib/api";
import { restoreConversation } from "@/lib/conversation";
import { Sidebar } from "@/components/Sidebar";
import { DatasetUpload } from "@/components/DatasetUpload";
import { DatasetHeader } from "@/components/DatasetHeader";
import { Conversation } from "@/components/Conversation";
import { Composer } from "@/components/Composer";
import { DatasetOverview } from "@/components/DatasetOverview";
import { ModeSwitch } from "@/components/ModeSwitch";

export default function Home() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [selected, setSelected] = useState<Dataset>();
  const [items, setItems] = useState<ConversationItem[]>([]);
  const [sessionId, setSessionId] = useState<string>();
  const [mode, setMode] = useState<AgentMode>("sql");
  const [project, setProject] = useState<MLProject | null>(null);
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [connectionError, setConnectionError] = useState<string>();
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string>();
  const [collapsed, setCollapsed] = useState(false);
  const [overview, setOverview] = useState(false);
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  const fileInput = useRef<HTMLInputElement>(null);
  const controller = useRef<AbortController | undefined>(undefined);
  const activeSession = useRef<string | undefined>(undefined);
  const activeRequest = useRef(0);

  const refresh = useCallback(async () => {
    setLoading(true); setConnectionError(undefined);
    try {
      const [ds, ss] = await Promise.all([listDatasets(), listSessions()]);
      setDatasets(ds); setSessions(ss);
      setSelected(current => current ?? ds.find(d => d.status === "ready"));
    } catch { setConnectionError("Cannot reach the data service."); }
    finally { setLoading(false); }
  }, []);
  useEffect(() => {
    const saved = localStorage.getItem("lumen-theme") === "light" ? "light" : "dark";
    setTheme(saved); document.documentElement.dataset.theme = saved;
    const requestedMode = new URLSearchParams(window.location.search).get("mode");
    const savedMode = localStorage.getItem("lumen-mode");
    const initialMode: AgentMode = requestedMode === "ml" || requestedMode === "chat" || requestedMode === "sql" ? requestedMode : savedMode === "ml" || savedMode === "chat" ? savedMode : "sql";
    setMode(initialMode); localStorage.setItem("lumen-mode", initialMode);
    setCollapsed(localStorage.getItem("lumen-sidebar") === "collapsed");
    void refresh();
    return () => { controller.current?.abort(); };
  }, [refresh]);

  const handleEvent = useCallback((event: AgentEvent) => {
    const data = event.data;
    if (event.type === "session_started") { setSessionId(event.session_id); activeSession.current = event.session_id; return; }
    if (event.type === "tool_started") {
      setItems(current => [...current, { id: event.id, type: "activity", toolCallId: String(data.tool_call_id), label: String(data.label), tool: String(data.tool), state: "running" }]); return;
    }
    if (event.type === "tool_finished") {
      setItems(current => current.map(item => item.type === "activity" && item.toolCallId === data.tool_call_id ? { ...item, state: data.ok ? "done" : "error", sql: data.sql ? String(data.sql) : item.sql, durationMs: typeof data.duration_ms === "number" ? data.duration_ms : undefined, rowsReturned: typeof data.rows_returned === "number" ? data.rows_returned : undefined, error: data.error ? String(data.error) : undefined } : item.type === "ml_progress" && item.toolCallId === data.tool_call_id ? { ...item, status: data.ok ? "completed" : "failed", label: data.ok ? item.label : String(data.error ?? "Operation failed") } : item)); return;
    }
    if (event.type === "ml_progress") {
      const toolCallId = String(data.tool_call_id), stage = String(data.stage);
      setItems(current => {
        const without = current.map(item => item.type === "ml_progress" && item.toolCallId === toolCallId && item.status === "running" ? { ...item, status: "completed" } : item);
        const id = `progress-${toolCallId}-${stage}`;
        const progress: ConversationItem = { id, type: "ml_progress", toolCallId, stage, label: String(data.label), status: String(data.status) };
        return without.some(i => i.id === id) ? without.map(i => i.id === id ? progress : i) : [...without, progress];
      }); return;
    }
    if (event.type === "query_result") { setItems(current => [...current, { id: event.id, type: "result", result: data as QueryResult }]); return; }
    if (event.type === "chart_created") { setItems(current => [...current, { id: event.id, type: "chart", chart: data as ChartPayload }]); return; }
    if (event.type === "ml_result") { setItems(current => {
      const result = data as MLResult;
      const lastUser = current.map(item => item.type).lastIndexOf("user");
      const prior = result.run_id ? current.findIndex((item, index) => index > lastUser && item.type === "ml" && item.result.run_id === result.run_id) : -1;
      const next: ConversationItem = { id: event.id, type: "ml", result };
      return prior < 0 ? [...current, next] : current.map((item, index) => index === prior ? next : item);
    }); return; }
    if (event.type === "message_delta") {
      setItems(current => {
        const existing = current.findIndex(i => i.id === "streaming-answer");
        if (existing < 0) return [...current, { id: "streaming-answer", type: "assistant", text: String(data.delta), streaming: true }];
        return current.map((item, i) => i === existing && item.type === "assistant" ? { ...item, text: item.text + String(data.delta) } : item);
      }); return;
    }
    if (event.type === "analysis_complete") {
      setItems(current => {
        const found = current.some(i => i.id === "streaming-answer");
        return found ? current.map(item => item.id === "streaming-answer" && item.type === "assistant" ? { ...item, id: event.id, text: String(data.message ?? item.text), streaming: false } : item) : [...current, { id: event.id, type: "assistant", text: String(data.message ?? "") }];
      }); return;
    }
    if (event.type === "error") setItems(current => [...current, { id: event.id, type: "error", text: String(data.message || "Analysis failed") }]);
  }, []);

  async function send(message: string) {
    if ((mode !== "chat" && !selected) || busy) return;
    const token = ++activeRequest.current;
    setItems(current => [...current.map(item => item.id === "streaming-answer" && item.type === "assistant" ? { ...item, id: `answer-${Date.now()}`, streaming: false } : item), { id: `user-${Date.now()}`, type: "user", text: message }]);
    setBusy(true);
    const abort = new AbortController(); controller.current = abort;
    try {
      await streamAnalysis({ dataset_id: mode === "chat" ? undefined : selected?.id, message, session_id: sessionId, mode }, event => { if (activeRequest.current === token) handleEvent(event); }, abort.signal);
      if (activeRequest.current === token) {
        setSessions(await listSessions());
        if (activeSession.current) setProject((await getSession(activeSession.current)).project);
      }
    } catch (error) {
      if (!abort.signal.aborted && activeRequest.current === token) setItems(current => [...current, { id: `error-${Date.now()}`, type: "error", text: error instanceof Error ? error.message : "Analysis failed" }]);
    } finally {
      if (activeRequest.current === token) {
        setBusy(false); controller.current = undefined;
        setItems(current => current.map(item => item.type === "activity" && item.state === "running" ? { ...item, state: "error", error: abort.signal.aborted ? "Stopped" : "Request ended before completion" } : item.type === "ml_progress" && item.status === "running" ? { ...item, status: "failed", label: abort.signal.aborted ? "Stopped" : "Request ended" } : item.type === "assistant" && item.streaming ? { ...item, streaming: false } : item));
      }
    }
  }
  async function removeDataset(dataset: Dataset) {
    if (busy || !window.confirm(`Delete ${dataset.filename}? This also removes its saved conversations, ML runs, and local artifacts.`)) return;
    setUploadError(undefined);
    try {
      await deleteDataset(dataset.id);
      const remaining = datasets.filter(item => item.id !== dataset.id);
      setDatasets(remaining);
      setSessions(current => current.filter(session => session.dataset_id !== dataset.id));
      if (selected?.id === dataset.id) {
        newAnalysis();
        setSelected(remaining.find(item => item.status === "ready"));
        setOverview(false);
      }
    } catch (error) { setUploadError(error instanceof Error ? error.message : "Could not delete dataset"); }
  }
  async function acceptFile(file: File) {
    if (busy) return;
    setUploading(true); setUploadError(undefined);
    try { const dataset = await uploadDataset(file); setDatasets(current => [dataset, ...current]); selectDataset(dataset); }
    catch (error) { setUploadError(error instanceof Error ? error.message : "Upload failed"); }
    finally { setUploading(false); if (fileInput.current) fileInput.current.value = ""; }
  }
  function newAnalysis() {
    ++activeRequest.current; controller.current?.abort(); setItems([]); setSessionId(undefined); activeSession.current = undefined; setProject(null); setBusy(false);
  }
  function selectDataset(dataset: Dataset) { newAnalysis(); setSelected(dataset); setOverview(false); }
  function switchMode(next: AgentMode) { if (next === mode || busy) return; newAnalysis(); setOverview(false); setMode(next); localStorage.setItem("lumen-mode", next); }
  async function openSession(session: Session) {
    if (busy) return;
    const dataset = datasets.find(d => d.id === session.dataset_id);
    if (!dataset && session.mode !== "chat") return;
    const token = ++activeRequest.current;
    setBusy(true);
    try {
      const detail = await getSession(session.id);
      if (activeRequest.current !== token) return;
      if (dataset) setSelected(dataset); setMode(detail.mode ?? "sql"); localStorage.setItem("lumen-mode", detail.mode ?? "sql");
      setSessionId(session.id); activeSession.current = session.id; setProject(detail.project); setItems(restoreConversation(detail));
    } catch { setUploadError("Could not restore this conversation. Try again."); }
    finally { if (activeRequest.current === token) setBusy(false); }
  }
  function toggleTheme() { const next = theme === "dark" ? "light" : "dark"; setTheme(next); localStorage.setItem("lumen-theme", next); document.documentElement.dataset.theme = next; }
  return <main className="app-shell">
    <Sidebar mode={mode} collapsed={collapsed} datasets={datasets} sessions={sessions.filter(s => (s.mode ?? "sql") === mode)} selectedId={selected?.id} sessionId={sessionId} theme={theme} loading={loading} error={connectionError} disabled={busy || uploading} onCollapse={() => { setCollapsed(!collapsed); localStorage.setItem("lumen-sidebar", collapsed ? "expanded" : "collapsed"); }} onSelect={selectDataset} onDelete={removeDataset} onSession={openSession} onNew={newAnalysis} onUpload={() => fileInput.current?.click()} onTheme={toggleTheme} onRetry={refresh} />
    <input ref={fileInput} type="file" accept=".csv,.parquet" hidden onChange={event => event.target.files?.[0] && acceptFile(event.target.files[0])} />
    <section className="workspace">
      <div className="workspace-mode-bar"><ModeSwitch mode={mode} onChange={switchMode} disabled={busy} /><span>{mode === "chat" ? "Chat and research the web" : mode === "ml" ? "Learn from your data" : "Understand your data"}</span>{mode !== "chat" && <><button className="mobile-dataset-action" disabled={busy || uploading} onClick={() => fileInput.current?.click()}>Add data</button><select className="mobile-dataset-select" aria-label="Selected dataset" disabled={busy} value={selected?.id ?? ""} onChange={e => { const ds = datasets.find(d => d.id === e.target.value); if (ds) selectDataset(ds); }}><option value="" disabled>Select a dataset</option>{datasets.filter(d => d.status === "ready").map(d => <option key={d.id} value={d.id}>{d.filename}</option>)}</select></> }</div>
      {uploadError && <div className="message-error" role="alert">{uploadError}<button onClick={() => setUploadError(undefined)} aria-label="Dismiss">×</button></div>}
      {connectionError && <div className="connection-banner" role="alert">Data service disconnected. <button onClick={refresh}>Reconnect</button></div>}
      {uploading && selected && <div className="connection-banner" role="status">Preparing your dataset…</div>}
      {mode === "chat" ? <>
        <header className="research-header"><div><span>Qwen</span><strong>Research Chat</strong></div><small>Plain conversation · web sources when needed</small></header>
        <Conversation mode={mode} items={items} busy={busy} onPrompt={send} />
        <Composer key={mode} mode={mode} busy={busy} onSend={send} onStop={() => controller.current?.abort()} onUpload={() => fileInput.current?.click()} />
      </> : selected ? <>
        <DatasetHeader dataset={selected} onOverview={() => setOverview(!overview)} />
        {mode === "ml" && project && <div className="project-strip"><span className="project-dot" /><strong>{project.target?.replaceAll("_", " ") ?? project.name}</strong><span>{project.task?.replaceAll("_", " ") ?? "New project"} · {project.status.replaceAll("_", " ")}</span></div>}
        <Conversation dataset={selected} mode={mode} items={items} busy={busy} onPrompt={send} />
        <Composer key={`${mode}-${selected.id}`} datasetName={selected.filename} mode={mode} busy={busy} onSend={send} onStop={() => controller.current?.abort()} onUpload={() => fileInput.current?.click()} />
      </> : loading ? <div className="empty-note">Loading your workspace…</div> : <DatasetUpload uploading={uploading} error={uploadError} onFile={acceptFile} />}
    </section>{mode !== "chat" && selected && overview && <DatasetOverview dataset={selected} onClose={() => setOverview(false)} />}
  </main>;
}
