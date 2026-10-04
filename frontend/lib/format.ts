export function compactNumber(value: number | null | undefined): string {
  if (value == null) return "—";
  return new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(value);
}

export function fileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
}

export function displayValue(value: unknown, type?: string, column?: string): string {
  if (value == null) return "—";
  if (typeof value === "number") {
    const currency = column?.match(/(?:^|_)(usd|eur|gbp|inr)(?:_|$)/i)?.[1]?.toUpperCase();
    if (currency) return new Intl.NumberFormat("en", { style: "currency", currency }).format(value);
    return new Intl.NumberFormat("en", { maximumFractionDigits: 4 }).format(value);
  }
  if (type?.startsWith("DATE") || type?.startsWith("TIMESTAMP")) {
    const date = new Date(String(value));
    if (!Number.isNaN(date.getTime())) return new Intl.DateTimeFormat("en", { dateStyle: "medium", ...(type.startsWith("TIMESTAMP") ? { timeStyle: "short" as const } : {}), timeZone: "UTC" }).format(date);
  }
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}
