"use client";
import { BrainCircuit, Database, Search } from "lucide-react";
import type { AgentMode } from "@/types";

export function ModeSwitch({ mode, onChange, disabled = false }: { mode: AgentMode; onChange: (mode: AgentMode) => void; disabled?: boolean }) {
  return <div className="mode-switch" role="group" aria-label="Agent mode">
    <button disabled={disabled} aria-pressed={mode === "chat"} onClick={() => onChange("chat")}><Search size={14} /><span>Research</span></button>
    <button disabled={disabled} aria-pressed={mode === "sql"} onClick={() => onChange("sql")}><Database size={14} /><span>SQL Agent</span></button>
    <button disabled={disabled} aria-pressed={mode === "ml"} onClick={() => onChange("ml")}><BrainCircuit size={14} /><span>ML Agent</span></button>
  </div>;
}
