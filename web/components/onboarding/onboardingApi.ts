export async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/analysis${path}`, options);
  const text = response.status === 204 ? "" : await response.text();
  let body: Record<string, unknown> = {};
  try {
    body = text ? JSON.parse(text) : {};
  } catch {
    throw new Error(
      response.ok
        ? "The server returned an invalid response"
        : `The analysis service failed (${response.status})`,
    );
  }
  if (!response.ok) throw new Error(String(body.detail || body.error || "The request failed"));
  return body as T;
}

export const json = (method: string, value: unknown): RequestInit => ({
  method,
  headers: { "content-type": "application/json" },
  body: JSON.stringify(value),
});

export type Connection = { id: string; url: string; username: string };

export type SetupSettings = {
  profile: { username: string; email: string; display_name: string; is_admin: boolean };
  timezone: string;
  navidrome: Connection | null;
};

export type Track = {
  id: string;
  title: string;
  artist?: string;
  album?: string;
  duration: number;
  cover_art?: string;
};

/** Makes the user's current server connection the one their library uses. */
export function selectConnection(connectionId: string, complete = false) {
  return api(
    "/users/me/preferences/onboarding",
    json("PUT", { complete, connection_id: connectionId }),
  );
}
