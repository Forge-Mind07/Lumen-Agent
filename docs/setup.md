# Setup and configuration

## Prerequisites

- Python 3.11 or newer
- Node.js 20.9 or newer
- an OpenAI-compatible `/v1/chat/completions` endpoint with tool calling
- enough local storage for uploaded data and generated model artifacts

## Backend

```bash
cd backend
python -m venv .venv
.venv/bin/pip install -e '.[test]'
cp ../.env.example .env
.venv/bin/uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

`PRIMARY_MODEL_BASE_URL`, `PRIMARY_MODEL_NAME`, and
`PRIMARY_MODEL_API_KEY` define the analytical model. Local servers that ignore
authorization still require a non-empty placeholder value. `TAVILY_API_KEY` is
optional and enables the Research Chat search tools.

Runtime state is written beneath `DATA_DIR`. Do not commit that directory: it
contains uploaded datasets, conversation history, query artifacts, snapshots,
and serialized model pipelines.

## Frontend

```bash
cd frontend
npm ci
cp .env.local.example .env.local
npm run dev
```

The default frontend expects the API at `http://127.0.0.1:8000`.

## Optional launcher

After completing both installations, `scripts/launch-lumen.sh sql` can start
the backend and frontend. It expects the model endpoint to be running unless
`LUMEN_MODEL_START_COMMAND` contains a local command that starts it.

## Verification

```bash
cd backend
.venv/bin/python -m pytest -q
```

```bash
cd frontend
npm run lint
npm run build
npm test
```

The pytest suite uses scripted providers and does not require a live model.
Playwright uses mocked API responses but does require a compatible browser.
