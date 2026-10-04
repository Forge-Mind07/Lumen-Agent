# Research Chat, SQL Agent and ML Agent

Lumen remains a Next.js / FastAPI application, with DuckDB for data access,
SQLite for metadata, Apache ECharts for controlled visualizations, and the existing
OpenAI-compatible provider for tool orchestration. No additional agent framework
or deployment platform is required.

## Architecture

The selected mode is saved in browser storage and on each conversation. Research
Chat uses only general conversation plus controlled `web_search` and
`read_web_page` tools; it has no SQL or ML tools and requires no uploaded dataset. The
server rejects requests that try to change an existing conversation's mode.
SQL mode receives SQL/chart tools only; ML mode receives those same tools plus
typed ML tools. Both use `agent.py`, `provider.py`, `context.py`, `sql_tool.py`
and the existing SSE transport. The LLM receives metadata and bounded previews,
not the training matrix or model artifact.

ML jobs run in a separate Python process. The process runs only predefined
operations. There is no Python/code execution tool. A semaphore permits one job
at a time per API process; cancellation/timeouts kill the worker. Progress comes
from the actual runtime phases, including individual candidate fits. SQL queries
and materialization use the existing validated, timed, read-only query executor.

`create_dataset` produces an immutable JSON snapshot, a physical DuckDB `data`
table for later queries, and metadata including source SQL/query ID, source
dataset, columns, row count, SHA-256 fingerprint, creation time and session.
Oversized queries are rejected with a request to aggregate or sample explicitly;
the runtime does not silently train on the first N rows. Prediction and evaluation
outputs are also queryable snapshots, scoped to their conversation.

## Persistence and migration

`Storage.initialize()` adds `sessions.mode` with default `sql` for existing rows.
It creates four indexed metadata tables without deleting or rewriting prior data:

| Table | Contents |
| --- | --- |
| `ml_projects` | Active dataset, task, target, features/exclusions, model/run IDs, status, prior training configuration |
| `ml_datasets` | Immutable snapshot metadata and provenance |
| `ml_runs` | Request, operation, configuration, dataset fingerprint, status, actual results, errors and timestamps |
| `ml_models` | Algorithm, task, feature schema, preprocessing description, artifact location, measured metrics and dataset version |

These tables contain `id`, `session_id` (foreign key), `payload_json` and
`updated_at`. Artifacts gain the `ml_result` kind. Complete preprocessing and
estimators are saved together as trusted local joblib files under
`data/ml/runs/<run_id>/pipeline.joblib`. Snapshots live under `data/ml/datasets`.
Run records include Python, sklearn, pandas and NumPy versions. Do not load
untrusted joblib files or move artifacts without their registry records.

## Tool and HTTP contracts

Existing HTTP routes remain compatible; `POST /api/chat/stream` adds
`mode: "chat" | "sql" | "ml"`, defaulting to SQL. Chat mode may omit `dataset_id`.
Session detail responses include mode
and active project. New read endpoints:

- `GET /api/ml/projects/{session_id}`: project, dataset references, models and runs.
- `GET /api/ml/runs/{run_id}?session_id=...`: a scoped experiment manifest.

The same chat endpoint drives all mutations. ML tools are defined with Pydantic
schemas in `app/ml/contracts.py`; unknown fields and invalid values are rejected.

| Tool | Purpose |
| --- | --- |
| `create_dataset` | Materialize a validated SQL query into a dataset reference |
| `inspect_dataset` | Profile, target distribution, suggested task, quality/leakage checks |
| `train_model` | Classification/regression candidate comparison, holdout evaluation, explanations, persistence |
| `compare_models`, `list_project_models` | Inspect saved candidates and previous runs |
| `evaluate_model` | Saved holdout results, or score an explicit labeled dataset |
| `explain_model` | Held-out permutation importance and evaluation limitations |
| `explain_prediction` | Individual feature replacement sensitivity |
| `predict` | Individual or batch inference, class probabilities, optional probability filtering |
| `cluster_dataset` | Segment data with bounded KMeans comparison |
| `forecast_series` | Aggregate, backtest, forecast and estimate approximate ranges |
| `detect_anomalies` | Isolation Forest scores and flagged rows |

In ML mode, `run_sql` accepts an optional `dataset_id`. This lets the agent query
prediction outputs using the same SQL engine, e.g. filter positive-class
`prediction_probability > 0.8`, inspect `false_negative` in the saved holdout
dataset, or filter retained business columns. Outputs retain input columns.
CSV export contains the clearly labeled bounded preview. Full bounded results
can be queried through their reference. No tool writes predictions into a user
database.

SSE adds `ml_progress` and `ml_result`, using the same versioned event envelope.
Progress is operational state; model reasoning is not exposed. Persisted ML
artifacts allow result cards to be restored with the conversation.

## Algorithms and evaluation

| Task | Algorithms | Main evaluation |
| --- | --- | --- |
| Classification | Logistic regression, random forest, prior-frequency dummy baseline | ROC AUC/PR AUC for binary targets; accuracy, precision, recall, F1, confusion matrix; macro metrics for multiclass |
| Regression | Ridge, random forest, mean dummy baseline | MAE, RMSE, R²; residuals and actual/predicted plot |
| Clustering | KMeans, compare 2–6 clusters automatically or choose 2–8 | Bounded silhouette estimate, segment sizes and numeric profiles |
| Forecasting | Last value, linear trend; seasonal naive with adequate history | Rolling validation origins and separate final-horizon test; MAE/RMSE |
| Anomaly detection | Isolation Forest | Anomaly scores and flagged share; no unsupported accuracy claim |

Supervised runs use distinct training, validation and test partitions. Candidate
selection uses validation metrics, including the baseline; the selected pipeline
is refit on training+validation and evaluated on the held-out test. A chronological
split is available through `time_column` and keeps equal timestamps together.
All stochastic estimators/splits use a recorded seed (42 by default).

Numerical features use median imputation and scaling. Categorical features use a
missing marker and bounded one-hot encoding with unknown-category handling.
Transformations are fitted on training rows only and saved with the estimator.
Inference validates required columns and numeric types before applying the same
pipeline. Feature importance permutes original columns on at most 500 held-out
rows, with 3 repeats. Individual explanations replace features with training
reference values; these are sensitivity checks, not SHAP or causal explanations.

Exact duplicate rows and missing-target rows are removed before splitting.
Constant, ≥95%-missing, obvious identifier, high-cardinality text, raw timestamp,
target duplicate/encoding and suspiciously correlated features are excluded and
reported. Name-based post-outcome checks are heuristic. Users still need to
confirm prediction-time availability and entity/time split suitability.

Forecasts aggregate daily, weekly or monthly with sum/mean. Missing periods are
rejected until the user explicitly defines their meaning. Approximate ranges
use a validation-error quantile; they are not guaranteed coverage. Cluster names
describe measured high/low feature values rather than invented business labels.

## UI changes and files

- `page.tsx`, `Sidebar`, `DatasetHeader`, `DatasetOverview`, `Composer`: persistent
  mode choice, active project strip, searchable datasets/schema, connection states,
  mobile dataset selector, mode-specific prompts and restorable conversations.
- `Conversation`, `AnswerSummary`: answer-first turn grouping, GFM tables, preserved
  follow-ups, bounded scrolling, secondary execution details.
- `ResultTable`, `SQLDetails`: collapsed data, 12-row pages, type-aware formatting,
  sticky headers, null/long-value handling, sorting, clipboard/CSV and highlighted SQL.
- `ChartCard`: controlled KPI/table/bar/line/area/scatter/histogram rendering;
  retains legacy pie/stacked chart rendering for saved conversations.
- `ModeSwitch`, `MLResult`, `MLChart`: live progress, measured metrics and baselines,
  explainability, confusion matrix, curves, residuals, forecast intervals, segment
  profiles, prediction tables and expandable experiment details.
- `lib/conversation.ts`, `lib/api.ts`, `types/index.ts`, `globals.css`: artifact
  restoration, shared contracts and consistent dark/light/responsive styles.
- Backend additions: `ml/contracts.py`, `datasets.py`, `preprocessing.py`,
  `supervised.py`, `patterns.py`, `inference.py`, `worker.py`, `service.py`.
  Integration updates: `agent.py`, `context.py`, `models.py`, `storage.py`,
  `main.py`, `config.py`, dependencies and example configuration.
- `research.py`: bounded Tavily search/extract tools with compact structured source
  results, URL validation, timeouts and citation-oriented prompting.
- `EChartCanvas`, `ChartCard`, `MLChart`: lazy-loaded, tree-shaken Apache ECharts
  Canvas renderer with responsive resizing, data zoom, accessible descriptions,
  image export, premium styling and controlled chart specifications.

## Run locally

Use one API worker for the current local runtime. Keep the existing local LLM
endpoint configured in `backend/.env`; it must support function/tool calls.

```bash
cd backend
python -m venv .venv                  # only if absent
.venv/bin/pip install -e '.[test]'
cp .env.example .env                  # only if absent; preserve existing settings
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```

```bash
cd frontend
npm ci
npm run dev -- --hostname 127.0.0.1 --port 3000
```

Open `http://127.0.0.1:3000`. The mode switch is above the dataset header.
Development and production builds use separate `.next` and `.next-build` folders.
Start an OpenAI-compatible tool-calling model on the URL configured in
`backend/.env` before launching Lumen.

For web research, set `TAVILY_API_KEY` in `backend/.env` or the launching
environment. The local desktop launcher reuses the machine's existing Tavily
credential without copying it into the repository.

Try these **synthetic** example files:

- SQL: `examples/sales.csv`; ask “What are our top products?”
- ML: `examples/ml_customers.csv`; ask “Predict churned. Exclude order_value.”
  Then “Remove age and retrain”, “Optimize recall”, “Compare with the previous
  model”, or “Show customers above 80% churn probability”.
- Regression: with the same file, start a new ML conversation and ask “Predict
  order_value; exclude churned”.
- Clustering: “Segment customers using monthly_spend, days_since_login and plan”.
- Forecasting: `examples/monthly_sales.csv`; “Forecast monthly revenue for the
  next 3 months”.
- Anomalies: “Find unusual customers using monthly_spend and support_tickets”.

The six-row SQL sample is deliberately too small for reliable ML training.

## Tests

```bash
cd backend
.venv/bin/python -m pytest -q
# Optional: real local LLM; uses a temporary database, not your saved chats
.venv/bin/python scripts/smoke_local_model.py --mode sql
.venv/bin/python scripts/smoke_local_model.py --mode ml
```

```bash
cd frontend
npm run lint
npm run build
npx playwright install chromium       # once, if no compatible browser is installed
npm test
# Alternatively set PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH to an existing Chromium.
```

`tests/test_ml.py` covers scoped references, migration, project persistence,
mode isolation, real worker training/inference, categorical and missing data,
leakage, invalid targets/configuration, candidate failure, row/time limits,
determinism, time splits, model comparison and retraining, predictions, explanations,
clustering, forecasting and anomalies. Existing SQL/provider/API/context tests
remain. Playwright uses deterministic mocked API responses to test UI behavior,
including mode persistence, chart rendering, 500-row pagination, restored markdown,
SQL display, follow-up preservation, light/mobile layouts and disconnected state.
API and research tests also cover dataset-free chat, mode-specific tool isolation,
structured search results, missing search configuration, and physical dataset deletion.

## Limits and next improvements

- Local, single-user file workspace. Authentication, remote SQL connectors,
  multitenancy, cross-source joins and external database write-back are not added.
- Defaults: 20,000 rows, 80 source columns, 50 MB serialized snapshots, 180-second
  runtime deadline, 3 candidates including the baseline. Up to 24 extra derived
  columns are allowed for prediction/evaluation artifacts. Numerical training
  stays in a bounded subprocess, but no distributed queue or OS memory quota
  is installed. Use representative SQL materialization for larger inputs.
- Models are scoped to their conversation; refresh/restoration retains project
  state. To score a subset of the connected source, materialize its rows and
  call predict with the prior model. Cross-conversation model sharing is absent.
- No XGBoost/LightGBM, hyperparameter search, probability calibration, SHAP,
  grouped cross-validation or automatic feature synthesis. The tools reject
  unsupported algorithms rather than pretending to run them.
- Forecasting is intentionally simple and univariate; richer seasonal/exogenous
  models and stronger interval calibration are useful next additions.
- An API restart marks in-flight jobs as interrupted; it does not resume them.
  On Linux, workers terminate when their parent API process dies. Durable
  scheduling/resumption and retention/cleanup controls are future work.
- Per-session concurrency prevention and one-job scheduling apply to a single
  API process. Add a shared queue/locks before deploying multiple API workers.
- Next priorities: authenticated database connectors, grouped/time-series
  cross-validation, calibrated classification thresholds, artifact retention,
  accessible chart data alternatives, and model drift monitoring.
