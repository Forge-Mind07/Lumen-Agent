import { expect, test, type Page } from "@playwright/test";

const dataset = { id: "ds-test", filename: "customers.csv", status: "ready", file_type: "csv", size_bytes: 20000, row_count: 500, column_count: 4, created_at: "2026-01-01", error: null,
  profile: { table_name: "data", row_count: 500, profile_is_partial: false, sample_rows: [], columns: ["age", "monthly_spend", "plan", "churned"].map(name => ({ name, data_type: name === "plan" ? "VARCHAR" : "DOUBLE", kind: name === "plan" ? "dimension" : "measure", null_percent: 0, approximate_distinct: 20 })) } };
const query = { sql: "SELECT age, monthly_spend FROM data ORDER BY age", columns: ["age", "monthly_spend"], column_types: ["BIGINT", "DOUBLE"], rows: Array.from({ length: 500 }, (_, i) => [i + 1, 1000 + i]), returned_row_count: 500, truncated: true, duration_ms: 12 };
const saved = { id: "ses-sql", title: "Explore customers", mode: "sql", dataset_id: dataset.id, created_at: "2026-01-01", updated_at: "2026-01-01" };
const mlResult = { run_id: "run-test", model_id: "model-test", task: "classification", target: "churned", algorithm: "logistic_regression", positive_class: "1",
  summary: "Compared candidates on validation data and measured performance on a separate test split.", metrics: { roc_auc: .84, recall: .7, precision: .76, f1: .729 }, baseline_metrics: { roc_auc: .5, recall: 0, precision: 0, f1: 0 },
  features: ["age", "monthly_spend", "plan"], excluded_features: { customer_id: "Identifier" }, warnings: ["Confirm features are available at prediction time."],
  split: { strategy: "stratified_random", train_rows: 300, validation_rows: 100, test_rows: 100 },
  candidates: [{ algorithm: "baseline", status: "completed", validation_metrics: { roc_auc: .5 } }, { algorithm: "logistic_regression", status: "completed", validation_metrics: { roc_auc: .85 } }],
  charts: [{ type: "feature_importance", title: "Most influential signals", data: [{ feature: "monthly_spend", importance: .23 }, { feature: "age", importance: .08 }] }, { type: "confusion_matrix", title: "Held-out predictions", labels: ["0", "1"], matrix: [[65, 5], [9, 21]] }],
  dataset: { dataset_id: "mlds-test", rows: 500, source_sql: "SELECT * FROM data", fingerprint: "fixture" },
};

async function mockWorkspace(page: Page) {
  let datasetAvailable = true;
  await page.route("**/api/**", async route => {
    const url = new URL(route.request().url());
    if (url.pathname === "/api/datasets" && route.request().method() === "DELETE") { datasetAvailable = false; return route.fulfill({ status: 204, body: "" }); }
    if (url.pathname === `/api/datasets/${dataset.id}` && route.request().method() === "DELETE") { datasetAvailable = false; return route.fulfill({ status: 204, body: "" }); }
    if (url.pathname === "/api/datasets") return route.fulfill({ json: datasetAvailable ? [dataset] : [] });
    if (url.pathname === "/api/sessions") return route.fulfill({ json: datasetAvailable ? [saved] : [] });
    if (url.pathname === "/api/sessions/ses-sql") return route.fulfill({ json: { ...saved, project: null, messages: [{ id: "u", role: "user", content: "Explore customers", created_at: "2026-01-01T01:00:00" }, { id: "a", role: "assistant", content: "Here is the summary.\n\n| Plan | Rows |\n| --- | --- |\n| Basic | 200 |", created_at: "2026-01-01T01:00:02" }], artifacts: [{ id: "result-1", kind: "query_result", payload: query, created_at: "2026-01-01T01:00:01" }, { id: "chart-1", kind: "chart", payload: { result_id: "result-1", type: "bar", x: "age", y: "monthly_spend", title: "Spend by age" }, created_at: "2026-01-01T01:00:01.500" }] } });
    if (url.pathname === "/api/sessions/ses-ml") return route.fulfill({ json: { ...saved, id: "ses-ml", mode: "ml", project: { id: "proj-test", name: "Customer churn", status: "trained", target: "churned", task: "classification", active_model_id: "model-test" }, messages: [], artifacts: [] } });
    if (url.pathname === "/api/chat/stream") {
      const body = route.request().postDataJSON();
      const ml = body.mode === "ml", chat = body.mode === "chat";
      const answer = chat ? "Here is a researched answer with an [official source](https://example.com)." : ml ? "The model performs better than the baseline on the held-out sample." : "There are 500 rows.";
      const events = [{ type: "session_started", data: {} }, ...(chat ? [{ type: "tool_started", data: { tool_call_id: "web", tool: "web_search", label: "Searching current sources" } }, { type: "tool_finished", data: { tool_call_id: "web", ok: true } }] : ml ? [{ type: "tool_started", data: { tool_call_id: "train", tool: "train_model", label: "Building models" } }, { type: "ml_progress", data: { tool_call_id: "train", stage: "training", label: "Training models", status: "running" } }, { type: "ml_result", data: mlResult }, { type: "tool_finished", data: { tool_call_id: "train", ok: true } }] : [{ type: "query_result", data: { ...query, result_id: "result-1" } }]), { type: "message_delta", data: { delta: answer } }, { type: "analysis_complete", data: { message: answer } }];
      return route.fulfill({ contentType: "text/event-stream", body: events.map((e, i) => `data: ${JSON.stringify({ ...e, id: `evt-${Date.now()}-${i}`, version: 1, session_id: ml ? "ses-ml" : "ses-sql" })}\n\n`).join("") });
    }
    return route.fulfill({ status: 404, json: { detail: "Not found" } });
  });
}

test.beforeEach(async ({ page }) => { await mockWorkspace(page); await page.goto("/"); });

test("mode switching persists and keeps SQL and ML conversations separate", async ({ page }) => {
  await expect(page.getByRole("heading", { name: "Ask questions about your data." })).toBeVisible();
  await page.getByRole("button", { name: "ML Agent", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Build predictive models from your data." })).toBeVisible();
  await page.reload();
  await expect(page.getByRole("button", { name: "ML Agent", exact: true })).toHaveAttribute("aria-pressed", "true");
  await page.getByRole("textbox", { name: "Message ML Agent" }).fill("Predict churn using churned as the target");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect(page.locator(".metric-card").filter({ hasText: "ROC AUC" })).toContainText("0.84");
  await expect(page.locator(".ml-result .echart-canvas canvas").first()).toBeVisible();
  await page.locator(".conversation").evaluate(el => { el.scrollTop = 0; });
  await page.screenshot({ path: "test-results/ml-summary.png", fullPage: true });
  await page.getByText("Experiment details", { exact: false }).first().click();
  await expect(page.getByText("Models tested · validation metrics")).toBeVisible();
  await expect(page.getByText("Held-out predictions", { exact: true })).toBeVisible();
  await page.screenshot({ path: "test-results/ml-workspace.png", fullPage: true });
  await page.getByRole("button", { name: "SQL Agent", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Ask questions about your data." })).toBeVisible();
  await expect(page.locator(".ml-result")).toHaveCount(0);
});

test("research chat is dataset-independent and renders cited answers", async ({ page }) => {
  await page.getByRole("button", { name: "Research", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Ask Qwen anything." })).toBeVisible();
  await expect(page.getByText("Research the latest developments in a topic")).toBeVisible();
  await page.getByRole("textbox", { name: "Message Research Chat" }).fill("Research current battery technology");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect(page.getByRole("link", { name: "official source" })).toHaveAttribute("href", "https://example.com");
  await page.getByText("Activity", { exact: false }).click();
  await expect(page.getByText("Searching current sources")).toBeVisible();
});

test("suggestions use actual schema and uploaded datasets can be deleted", async ({ page }) => {
  await expect(page.getByText("Compare monthly spend by plan")).toBeVisible();
  page.once("dialog", dialog => dialog.accept());
  await page.getByRole("button", { name: "Delete customers.csv" }).click();
  await expect(page.getByRole("heading", { name: "Start with a dataset." })).toBeVisible();
});

test("saved SQL chat restores GFM tables, bounded results, pagination and SQL details", async ({ page }) => {
  await page.getByRole("button", { name: /Explore customers/ }).click();
  await expect(page.locator(".markdown-table tbody tr")).toHaveCount(1);
  await expect(page.locator(".chart-card .echart-canvas canvas")).toBeVisible();
  await expect(page.locator(".result-card tbody tr").first()).not.toBeVisible();
  await page.locator(".data-details > summary").click();
  await expect(page.locator(".result-card tbody tr")).toHaveCount(12);
  await expect(page.locator(".result-card tbody tr").first()).toContainText("1,000");
  await page.getByRole("button", { name: "Next page" }).click();
  await expect(page.locator(".result-card .table-foot")).toContainText("13–24 of 500");
  await page.getByRole("button", { name: "age", exact: true }).click();
  await page.getByRole("button", { name: "age", exact: true }).click();
  await expect(page.locator(".result-card tbody tr").first()).toContainText("500");
  await page.locator(".data-result > details").last().locator("summary").click();
  await expect(page.locator(".sql-keyword").first()).toHaveText("SELECT");
  await expect(page.getByRole("link", { name: "CSV" })).toHaveAttribute("href", /result-1\/csv$/);
  await page.screenshot({ path: "test-results/sql-workspace.png", fullPage: true });
});

test("follow-up preserves the previous answer", async ({ page }) => {
  for (const prompt of ["How many rows?", "Show me the data"]) {
    await page.getByRole("textbox", { name: "Message SQL Agent" }).fill(prompt);
    await page.getByRole("button", { name: "Send", exact: true }).click();
    await expect(page.getByRole("button", { name: "Send", exact: true })).toBeVisible();
  }
  await expect(page.locator(".assistant-message")).toHaveCount(2);
});

test("mobile and light theme keep the chat usable", async ({ page }) => {
  await page.getByRole("button", { name: "Light theme" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByRole("combobox", { name: "Selected dataset" })).toBeVisible();
  await expect(page.getByRole("button", { name: "ML Agent", exact: true })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
  await page.screenshot({ path: "test-results/mobile-workspace.png", fullPage: true });
});

test("disconnected state is actionable", async ({ page }) => {
  await page.route("**/api/datasets", route => route.abort());
  await page.reload();
  await expect(page.locator(".connection-banner")).toContainText("Data service disconnected");
  await expect(page.locator(".connection-banner").getByRole("button", { name: "Reconnect" })).toBeVisible();
});
