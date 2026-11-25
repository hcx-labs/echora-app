import defaults from "./runtimeConfig.defaults.json";

export type RuntimeConfig = typeof defaults;
let current: RuntimeConfig = { ...defaults };

export function runtimeConfig(): Readonly<RuntimeConfig> { return current; }

export function applyRuntimeConfig(value: unknown): boolean {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  const next = { ...defaults };
  for (const key of Object.keys(defaults) as (keyof RuntimeConfig)[]) {
    const setting = candidate[key];
    if (typeof setting !== "number" || !Number.isSafeInteger(setting) || setting <= 0 || setting > 2_147_483_647) return false;
    next[key] = setting;
  }
  current = next;
  return true;
}
