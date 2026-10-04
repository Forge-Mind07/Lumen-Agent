"use client";

import { useState } from "react";
import { Check, ChevronDown, CircleAlert, Clock3, Copy, LoaderCircle, TerminalSquare } from "lucide-react";
import type { ActivityItem } from "@/types";
import { SQLDetails } from "./SQLDetails";

export function ActivityStep({ item }: { item: ActivityItem }) {
  const [open, setOpen] = useState(false);
  const expandable = Boolean(item.sql || item.error);
  return (
    <div className={`activity ${item.state === "error" ? "activity--error" : ""}`}>
      <button className="activity-summary" onClick={() => expandable && setOpen(!open)}>
        <span className={`activity-state activity-state--${item.state}`}>
          {item.state === "running" ? <LoaderCircle className="spin" size={13} /> : item.state === "done" ? <Check size={13} /> : <CircleAlert size={13} />}
        </span>
        <span>{item.label}</span>
        {item.durationMs != null && <small><Clock3 size={12} /> {item.durationMs} ms</small>}
        {item.rowsReturned != null && <small>{item.rowsReturned} rows</small>}
        {expandable && <ChevronDown className={open ? "rotate" : ""} size={14} />}
      </button>
      {open && (
        <div className="activity-detail">
          {item.sql && (
            <div className="sql-block">
              <div className="sql-toolbar">
                <span><TerminalSquare size={13} /> SQL</span>
                <button onClick={() => navigator.clipboard.writeText(item.sql ?? "")}><Copy size={13} /> Copy</button>
              </div>
              <SQLDetails sql={item.sql} open />
            </div>
          )}
          {item.error && <div className="tool-error">{item.error}</div>}
        </div>
      )}
    </div>
  );
}
