"use client";

import { useEffect, useRef } from "react";
import dynamic from "next/dynamic";
import { LoaderCircle, Sparkles } from "lucide-react";
import type { AgentMode, ConversationItem, Dataset } from "@/types";
import { ActivityStep } from "./ActivityStep";
import { ResultTable } from "./ResultTable";
import { AnswerSummary } from "./AnswerSummary";
import { MLResultCard, MLRunProgress } from "./MLResult";
import { conversationTurns } from "@/lib/conversation";
import { datasetRecommendations } from "@/lib/recommendations";

const ChartCard = dynamic(() => import("./ChartCard").then(module => module.ChartCard), { ssr: false, loading: () => <div className="chart-skeleton" aria-label="Loading chart" /> });

type Props = { dataset?: Dataset; mode: AgentMode; items: ConversationItem[]; busy: boolean; onPrompt: (prompt: string) => void };

export function Conversation({ dataset, mode, items, busy, onPrompt }: Props) {
  const viewport = useRef<HTMLDivElement>(null);
  const follow = useRef(true);
  useEffect(() => { if (follow.current && viewport.current) viewport.current.scrollTop = viewport.current.scrollHeight; }, [items, busy]);
  if (!items.length) {
    const prompts = datasetRecommendations(dataset, mode);
    const label = mode === "chat" ? "Research Chat" : mode === "ml" ? "ML Agent" : "SQL Agent";
    const heading = mode === "chat" ? "Ask Qwen anything." : mode === "ml" ? "Build predictive models from your data." : "Ask questions about your data.";
    const description = mode === "chat" ? "Talk naturally, or ask for current research with linked sources." : `${dataset?.row_count?.toLocaleString()} rows · ${dataset?.column_count} columns\n${mode === "ml" ? "Predict outcomes, explore segments, and understand what drives them." : "Explore patterns, compare results, and find useful answers."}`;
    return <div className="conversation conversation--empty"><div className="ready-copy"><span><Sparkles size={19} /></span><div className="eyebrow">{label}</div><h2>{heading}</h2><p>{description.split("\n").map((line, i) => <span key={line}>{i > 0 && <br />}{line}</span>)}</p></div><div className="prompt-grid">{prompts.map(prompt => <button key={prompt} onClick={() => onPrompt(prompt)}>{prompt}<span>↗</span></button>)}</div></div>;
  }
  function render(item: ConversationItem) {
    switch (item.type) {
      case "user": return <div className="user-message" key={item.id}>{item.text}</div>;
      case "assistant": return <AnswerSummary key={item.id} text={item.text} streaming={item.streaming} />;
      case "activity": return <ActivityStep key={item.id} item={item} />;
      case "result": return <ResultTable key={item.id} result={item.result} />;
      case "chart": return <ChartCard key={item.id} chart={item.chart} />;
      case "ml": return <MLResultCard key={item.id} result={item.result} onPrompt={onPrompt} busy={busy} />;
      case "ml_progress": return <MLRunProgress key={item.id} label={item.label} status={item.status} />;
      case "error": return <div className="message-error" role="alert" key={item.id}>{item.text}</div>;
    }
  }
  return <div ref={viewport} className="conversation" onScroll={() => { const el = viewport.current; if (el) follow.current = el.scrollHeight-el.scrollTop-el.clientHeight < 120; }}><div className="conversation-inner">
    {conversationTurns(items).map((turn, index) => {
      const complete = turn.some(i => i.type === "assistant" && !i.streaming);
      const technical = turn.filter(i => i.type === "activity" || i.type === "ml_progress");
      return <section className="conversation-turn" key={turn[0]?.id ?? index}>
        {turn.filter(i => i.type === "user").map(render)}
        {turn.filter(i => i.type === "assistant").map(render)}
        {!complete && technical.map(render)}
        {turn.filter(i => ["chart", "ml", "error"].includes(i.type)).map(item => item.type === "ml" && !item.result.metrics && !item.result.models && !item.result.prediction ? <details className="expandable-details" key={item.id}><summary>{item.result.target_distribution ? "Data quality checked" : "Dataset prepared"}<span>{item.result.rows?.toLocaleString()} rows · View details</span></summary>{render(item)}</details> : render(item))}
        {turn.filter(i => i.type === "result").map(render)}
        {complete && technical.length > 0 && <details className="expandable-details execution-details"><summary>Activity<span>{technical.filter(i => i.type === "activity").length} steps</span></summary>{technical.map(render)}</details>}
      </section>;
    })}
    {busy && <div className="thinking"><LoaderCircle className="spin" size={15} /><span>{mode === "chat" ? "Thinking and researching…" : mode === "ml" ? "Working with your data…" : "Looking into it…"}</span></div>}
  </div></div>;
}
