import { defineConfig } from "@playwright/test";

const port = process.env.PLAYWRIGHT_PORT ?? "3101";

export default defineConfig({
  testDir: "./tests",
  workers: 1,
  timeout: 30_000,
  use: { baseURL: `http://127.0.0.1:${port}`, headless: true,
    launchOptions: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH } : {},
  },
  webServer: { command: `NEXT_DIST_DIR=.next-playwright npm run dev -- --hostname 127.0.0.1 --port ${port}`, url: `http://127.0.0.1:${port}`, reuseExistingServer: !process.env.CI, timeout: 90_000 },
});
