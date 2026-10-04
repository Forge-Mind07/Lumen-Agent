"use client";

import { useRef, useState } from "react";
import { FileUp, LoaderCircle, ShieldCheck } from "lucide-react";

type Props = {
  uploading: boolean;
  error?: string;
  onFile: (file: File) => void;
};

export function DatasetUpload({ uploading, error, onFile }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  function choose(files: FileList | null) {
    const file = files?.[0];
    if (file) onFile(file);
  }

  return (
    <div className="upload-page">
      <div className="upload-intro">
        <span className="eyebrow">Local data workspace</span>
        <h1>Start with a dataset.</h1>
        <p>Upload a CSV or Parquet file. Your data stays on this machine and is queried directly with DuckDB.</p>
      </div>
      <button
        className={`dropzone ${dragging ? "dropzone--active" : ""}`}
        onClick={() => !uploading && input.current?.click()}
        onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          if (!uploading) choose(event.dataTransfer.files);
        }}
      >
        <input
          ref={input}
          type="file"
          accept=".csv,.parquet,text/csv,application/vnd.apache.parquet"
          hidden
          onChange={(event) => choose(event.target.files)}
        />
        <span className="upload-icon">
          {uploading ? <LoaderCircle className="spin" size={23} /> : <FileUp size={23} />}
        </span>
        <strong>{uploading ? "Profiling dataset…" : "Drop a dataset here"}</strong>
        <span>{uploading ? "Detecting types, ranges, and missing values" : "or click to choose a file · CSV or Parquet"}</span>
      </button>
      {error && <div className="upload-error">{error}</div>}
      <div className="privacy-note"><ShieldCheck size={14} /> Stored and analyzed locally</div>
    </div>
  );
}
