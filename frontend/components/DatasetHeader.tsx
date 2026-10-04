"use client";

import { ChevronDown, Columns3, Database, TableProperties } from "lucide-react";
import type { Dataset } from "@/types";
import { compactNumber } from "@/lib/format";

export function DatasetHeader({ dataset, onOverview }: { dataset: Dataset; onOverview: () => void }) {
  return (
    <header className="dataset-header">
      <div className="dataset-heading">
        <span className="dataset-icon"><Database size={16} /></span>
        <div>
          <button className="dataset-title dataset-title-button" onClick={onOverview}>{dataset.filename}<ChevronDown size={14} /></button>
          <div className="dataset-meta">
            {compactNumber(dataset.row_count)} rows <span>·</span> {dataset.column_count} columns
            <span className="ready-status"><i /> Ready</span>
          </div>
        </div>
      </div>
      <button className="quiet-button" onClick={onOverview}>
        <TableProperties size={15} /> Data overview
      </button>
    </header>
  );
}
