export type ColumnProfile = {
  name: string;
  data_type: string;
  kind: "dimension" | "measure" | "date" | "other";
  null_percent: number | null;
  approximate_distinct: number | null;
  minimum: unknown;
  maximum: unknown;
};

export type Dataset = {
  id: string;
  filename: string;
  file_type: string;
  size_bytes: number;
  status: "processing" | "ready" | "error";
  row_count: number | null;
  column_count: number | null;
  error: string | null;
  created_at: string;
  profile: {
    table_name: string;
    row_count: number;
    columns: ColumnProfile[];
    sample_rows: Record<string, unknown>[];
    profile_is_partial: boolean;
  } | null;
};

export type Session = {
  id: string;
  dataset_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  mode: AgentMode;
};

export type QueryResult = {
  result_id: string;
  sql: string;
  columns: string[];
  column_types: string[];
  rows: unknown[][];
  returned_row_count: number;
  truncated: boolean;
  duration_ms: number;
};

export type ChartPayload = {
  chart_id: string;
  spec: {
    result_id: string;
    type: "kpi" | "table" | "line" | "bar" | "area" | "scatter" | "histogram" | "stacked_bar" | "heatmap" | "pie";
    x: string;
    y?: string;
    series?: string;
    title: string;
  };
  source: QueryResult;
};

export type AgentEvent = {
  version: number;
  id: string;
  type: string;
  timestamp?: string;
  session_id: string;
  data: Record<string, unknown>;
};

export type ActivityItem = {
  id: string;
  type: "activity";
  toolCallId: string;
  label: string;
  tool: string;
  state: "running" | "done" | "error";
  sql?: string;
  durationMs?: number;
  rowsReturned?: number;
  error?: string;
};

export type ConversationItem =
  | { id: string; type: "user"; text: string }
  | { id: string; type: "assistant"; text: string; streaming?: boolean }
  | ActivityItem
  | { id: string; type: "result"; result: QueryResult }
  | { id: string; type: "chart"; chart: ChartPayload }
  | { id: string; type: "ml"; result: MLResult }
  | { id: string; type: "ml_progress"; toolCallId: string; label: string; stage: string; status: string }
  | { id: string; type: "error"; text: string };

export type AgentMode = "sql" | "ml" | "chat";
export type MLChart = { type: string; title: string; data?: Record<string, unknown>[]; labels?: string[]; matrix?: number[][] };
export type MLProject = { id: string; name: string; status: string; target?: string; task?: string; active_model_id?: string; dataset_id?: string };
export type MLResult = {
  run_id?: string; project_id?: string; model_id?: string; dataset_id?: string;
  task?: string; target?: string; algorithm?: string; summary?: string; status?: string;
  metrics?: Record<string, number | null>; baseline_metrics?: Record<string, number | null>;
  charts?: MLChart[]; warnings?: string[]; predictions?: QueryResult;
  dataset?: { dataset_id: string; rows: number; source_sql?: string; fingerprint?: string; columns?: { name: string; type: string }[] };
  rows?: number; columns?: { name: string; type: string; missing_percent?: number; distinct?: number }[];
  source_sql?: string; fingerprint?: string; positive_class?: string; importance_method?: string;
  features?: string[]; excluded_features?: Record<string, string>; preprocessing?: string;
  split?: Record<string, unknown>; model_configuration?: Record<string, unknown>;
  candidates?: { algorithm: string; status: string; validation_metrics?: Record<string, number>; error?: string }[];
  models?: { id: string; name: string; task: string; metrics: Record<string, number> }[];
  cluster_profiles?: { name: string; size: number; share: number; means: Record<string, number> }[];
  target_distribution?: Record<string, number>; prediction_dataset_id?: string; evaluation_dataset_id?: string;
  prediction?: string; score?: number;
  runtime_versions?: Record<string, string>;
};

export type SessionDetail = Session & {
  messages: { id: string; role: "user" | "assistant"; content: string; created_at: string }[];
  artifacts: { id: string; kind: "query_result" | "chart" | "ml_result"; payload: Record<string, unknown>; created_at: string }[];
  project: MLProject | null;
};
