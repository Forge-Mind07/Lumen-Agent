# Design decisions

## Keep model orchestration separate from numerical execution

The model chooses among typed tools, but DuckDB and scikit-learn perform the
actual work. This keeps numerical behavior testable and prevents generated code
from becoming an execution path.

## Use DuckDB for uploaded analytical data

CSV and Parquet are the primary inputs. DuckDB can query both efficiently while
supporting a persistent `data` relation and read-only execution. `sqlglot`
validation, relation allowlisting, timeouts, and result caps provide separate
layers of control.

## Use SQLite for application state

Sessions, messages, result artifacts, and ML registries are relational but
local and single-user. SQLite provides durable state without adding a server;
larger dataset payloads remain in DuckDB or filesystem artifacts.

## Stream progress with server-sent events

The interaction is server-to-browser progress over a single request. SSE fits
that flow and keeps the transport simpler than a bidirectional socket while
still exposing tool activity, query results, charts, ML progress, and answer
deltas.

## Bound prompt context

The model sees compact profiles, summaries, recent messages, and capped tool
results. Raw dataframes remain outside the prompt. This reduces context cost and
makes behavior predictable for local models with finite context windows.

## Isolate ML work in a child process

Training is CPU/memory intensive and may stall. A separate worker lets the API
enforce one job at a time per process, report concrete phases, and terminate on
timeout or cancellation. It is isolation for lifecycle control, not a complete
OS sandbox.

## Persist reproducible experiment inputs

ML snapshots record source provenance and fingerprints; run records include the
request, seed, environment versions, results, and errors. Preprocessing is fitted
only on training data and saved with the estimator so inference reuses the same
transformations.
