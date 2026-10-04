import type { AgentEvent, AgentMode, Dataset, Session, SessionDetail } from "@/types";

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

async function checked<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(body.detail || "Request failed");
  }
  return response.json() as Promise<T>;
}

export async function listDatasets(): Promise<Dataset[]> {
  return checked(await fetch(`${API_URL}/api/datasets`, { cache: "no-store" }));
}

export async function listSessions(): Promise<Session[]> {
  return checked(await fetch(`${API_URL}/api/sessions`, { cache: "no-store" }));
}

export async function getSession(id: string): Promise<SessionDetail> {
  return checked(await fetch(`${API_URL}/api/sessions/${encodeURIComponent(id)}`, { cache: "no-store" }));
}

export async function uploadDataset(file: File): Promise<Dataset> {
  const form = new FormData();
  form.append("file", file);
  return checked(await fetch(`${API_URL}/api/datasets`, { method: "POST", body: form }));
}

export async function deleteDataset(id: string): Promise<void> {
  const response = await fetch(`${API_URL}/api/datasets/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(body.detail || "Could not delete dataset");
  }
}

export async function streamAnalysis(
  input: { dataset_id?: string; message: string; session_id?: string; mode: AgentMode },
  onEvent: (event: AgentEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API_URL}/api/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
    signal,
  });
  if (!response.ok || !response.body) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(body.detail || "Analysis request failed");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() ?? "";
    for (const block of blocks) {
      const data = block
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trim())
        .join("\n");
      if (data) onEvent(JSON.parse(data) as AgentEvent);
    }
    if (done) break;
  }
}
