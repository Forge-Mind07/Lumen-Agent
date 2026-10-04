"use client";

import { Bot } from "lucide-react";
import { Children } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { shapeAnswer } from "@/lib/conversation";

export function AnswerSummary({ text, streaming }: { text: string; streaming?: boolean }) {
  const answer = shapeAnswer(text);
  return <div className="assistant-row"><div className="assistant-avatar"><Bot size={15} /></div><div className="assistant-message">
    <AnswerMarkdown text={answer.summary} />
    {answer.findings.length > 0 && <KeyFindings findings={answer.findings} />}
    {answer.hasDetails && <details className="expandable-details"><summary>Full explanation</summary><AnswerMarkdown text={text} /></details>}
    {streaming && <span className="cursor" />}
  </div></div>;
}

function AnswerMarkdown({ text }: { text: string }) {
  return <ReactMarkdown remarkPlugins={[remarkGfm]} components={{ table: ({ children }) => <div className="table-scroll markdown-table"><table>{children}</table><div className="table-foot">Answer table · up to 20 rows. Use View data for query results.</div></div>, tbody: ({ children }) => <tbody>{Children.toArray(children).slice(0, 20)}</tbody>, a: ({ href, children }) => <a href={href} onClick={event => { if (href?.startsWith("#result-")) { event.preventDefault(); const target = document.getElementById(href.slice(1)); const details = target?.querySelector("details"); if (details) details.open = true; target?.scrollIntoView({ block: "center" }); } }}>{children}</a> }}>{text.replace(/\[source:\s*([A-Za-z0-9_-]+)\]/g, "[View data](#result-$1)")}</ReactMarkdown>;
}

export function KeyFindings({ findings }: { findings: string[] }) {
  return <ul className="key-findings">{findings.map((finding, index) => <li key={index}><AnswerMarkdown text={finding} /></li>)}</ul>;
}

export function SuggestedFollowups({ prompts, onSelect }: { prompts: string[]; onSelect: (text: string) => void }) {
  return <div className="followups">{prompts.slice(0, 3).map(prompt => <button key={prompt} onClick={() => onSelect(prompt)}>{prompt} ↗</button>)}</div>;
}

export function Warning({ children }: { children: React.ReactNode }) {
  return <div className="warning" role="status">{children}</div>;
}
