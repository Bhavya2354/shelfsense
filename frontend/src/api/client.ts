const BASE_URL = import.meta.env.VITE_API_BASE_URL.replace(/\/+$/, "");

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

type Params = Record<string, string | number | boolean | string[] | null | undefined>;

function toQuery(params: Params): string {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === null || value === undefined || value === "") continue;
    if (Array.isArray(value)) value.forEach((v) => query.append(key, v));
    else query.set(key, String(value));
  }
  const text = query.toString();
  return text ? `?${text}` : "";
}

export async function getJson<T>(path: string, params: Params = {}, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}${toQuery(params)}`, {
    headers: { Accept: "application/json" },
    signal,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // Non-JSON error bodies keep the status text.
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}
