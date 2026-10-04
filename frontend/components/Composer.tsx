"use client";

import { FormEvent, KeyboardEvent, useRef, useState } from "react";
import { ArrowUp, Paperclip, Square } from "lucide-react";

type Props = {
  datasetName?: string;
  mode?: "sql" | "ml" | "chat";
  busy: boolean;
  onSend: (message: string) => void;
  onStop: () => void;
  onUpload: () => void;
};

export function Composer({ datasetName, mode = "sql", busy, onSend, onStop, onUpload }: Props) {
  const [value, setValue] = useState("");
  const textarea = useRef<HTMLTextAreaElement>(null);

  function submit(event?: FormEvent) {
    event?.preventDefault();
    const message = value.trim();
    if (!message || busy) return;
    setValue("");
    onSend(message);
    if (textarea.current) textarea.current.style.height = "auto";
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); submit(); }
  }

  return (
    <div className="composer-wrap">
      <form className="composer" onSubmit={submit}>
        <textarea
          ref={textarea}
          value={value}
          rows={1}
          aria-label={mode === "chat" ? "Message Research Chat" : mode === "ml" ? "Message ML Agent" : "Message SQL Agent"}
          placeholder={mode === "chat" ? "Ask anything, or request web research…" : mode === "ml" ? "What would you like to predict or discover?" : "Ask anything about this dataset…"}
          onKeyDown={onKeyDown}
          onChange={(event) => {
            setValue(event.target.value);
            event.target.style.height = "auto";
            event.target.style.height = `${Math.min(event.target.scrollHeight, 160)}px`;
          }}
        />
        <div className="composer-bar">
          {mode !== "chat" && <button type="button" className="attach-button" onClick={onUpload} aria-label="Upload dataset"><Paperclip size={17} /></button>}
          <span className="composer-dataset"><i /> {mode === "chat" ? "Qwen · web research" : datasetName}</span>
          <button type={busy ? "button" : "submit"} onClick={busy ? onStop : undefined} className={`send-button ${value.trim() || busy ? "send-button--ready" : ""}`} aria-label={busy ? "Stop" : "Send"}>
            {busy ? <Square size={13} fill="currentColor" /> : <ArrowUp size={17} />}
          </button>
        </div>
      </form>
      <p>{mode === "chat" ? "Qwen answers directly and searches the web when current evidence is needed." : mode === "ml" ? "Models run locally. Review evaluation and data quality before using predictions." : "Answers grounded in your data. Queries and results are always inspectable."}</p>
    </div>
  );
}
