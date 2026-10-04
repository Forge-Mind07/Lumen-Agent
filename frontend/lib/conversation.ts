import type { ChartPayload, ConversationItem, MLResult, QueryResult, SessionDetail } from "@/types";

export function restoreConversation(session: SessionDetail): ConversationItem[] {
  const embedded = new Set(session.artifacts.filter(a => a.kind === "ml_result").map(a => (a.payload.predictions as QueryResult | undefined)?.result_id));
  const artifacts = session.artifacts.filter(a => !embedded.has(a.id));
  const entries: { time: string; item: ConversationItem }[] = session.messages.map(m => ({ time: m.created_at, item: { id: m.id, type: m.role, text: m.content } }));
  for (const artifact of artifacts) {
    if (artifact.kind === "query_result") entries.push({ time: artifact.created_at, item: { id: artifact.id, type: "result", result: { ...artifact.payload, result_id: artifact.id } as QueryResult } });
    if (artifact.kind === "ml_result") {
      const result = artifact.payload as MLResult;
      const prior = result.run_id ? entries.findIndex(e => e.item.type === "ml" && e.item.result.run_id === result.run_id) : -1;
      const sameTurn = prior >= 0 && !session.messages.some(m => m.role === "user" && m.created_at > entries[prior].time && m.created_at < artifact.created_at);
      if (sameTurn) entries[prior].item = { id: artifact.id, type: "ml", result };
      else entries.push({ time: artifact.created_at, item: { id: artifact.id, type: "ml", result } });
    }
    if (artifact.kind === "chart") {
      const source = session.artifacts.find(a => a.id === artifact.payload.result_id);
      if (source) entries.push({ time: artifact.created_at, item: { id: artifact.id, type: "chart", chart: { chart_id: artifact.id, spec: artifact.payload, source: { ...source.payload, result_id: source.id } } as ChartPayload } });
    }
  }
  return entries.sort((a, b) => a.time.localeCompare(b.time)).map(e => e.item);
}

export function conversationTurns(items: ConversationItem[]): ConversationItem[][] {
  const turns: ConversationItem[][] = [];
  for (const item of items) {
    if (item.type === "user" || !turns.length) turns.push([]);
    turns[turns.length - 1].push(item);
  }
  return turns;
}

export function shapeAnswer(text: string) {
  if (text.length <= 1400) return { summary: text, findings: [] as string[], hasDetails: false };
  const paragraphs = text.split(/\n\s*\n/).map(p => p.trim()).filter(Boolean);
  const prose = paragraphs.find(p => !/^(#{1,6}\s|\||[-*]\s|\d+[.)]\s|(?:\*\*)?[\w ]{1,25}:(?:\*\*)?)/.test(p));
  const bullets = paragraphs.flatMap(p => /^\d+[.)]\s|^[-*]\s/.test(p) ? p.split(/\n(?=\d+[.)]\s|[-*]\s)/) : []).map(p => p.replace(/^(?:\d+[.)]|[-*])\s*/, ""));
  const summary = prose ?? bullets.shift() ?? paragraphs[0];
  return { summary, findings: bullets.filter(p => p !== summary).slice(0, 3), hasDetails: true };
}
