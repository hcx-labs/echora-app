"use client";

import { toast } from "sonner";
import { Notice } from "../ui/notice";

import { SettingsNotice } from "./SettingsNotice";
import { FormEvent, useEffect, useState } from "react";
import { SectionHeading, Toggle } from "./Presentation";
import { Button } from "../ui/button";
import { LoadingState } from "../ui/spinner";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { Textarea } from "../ui/textarea";
import { SettingRow } from "./SettingRow";
import styles from "./ExternalAISettings.module.css";
import layout from "./SettingsView.module.css";
import ClearTranslationsWarning from "./ClearTranslationsWarning";

type LanguagePair = { source: string; target: string };
type Configuration = {
  enabled: boolean;
  url: string;
  model: string;
  prompt: string;
  language_pairs: LanguagePair[];
  has_key: boolean;
  transcription_refinement: boolean;
  transcription_prompt: string;
  default_transcription_prompt?: string;
};
type Section = "endpoint" | "translation" | "transcription";
const endpoint = "/analysis/settings/external-ai";
const normalizeTag = (value: string) => value.trim().replaceAll("_", "-").toLowerCase();

function validate(value: Configuration, section: Section): string {
  if (section === "transcription")
    return value.transcription_prompt.trim() ? "" : "Enter a transcription refinement prompt.";
  if (section === "endpoint" && value.url) {
    try {
      const url = new URL(value.url);
      if (
        !/^https?:$/.test(url.protocol) ||
        !url.hostname ||
        url.username ||
        url.password ||
        /[\s\x00-\x1f\\?#]/.test(value.url) ||
        /https?:\/\/[^/]*@/i.test(value.url)
      )
        throw new Error();
    } catch {
      return "Enter an HTTP(S) API base URL without credentials, spaces, query parameters or a fragment.";
    }
  }
  if (section === "endpoint")
    return value.enabled && !value.url ? "Enter an API URL before enabling External AI." : "";
  if (!value.prompt.trim()) return "Enter a translation prompt.";
  const seen = new Set<string>();
  for (const pair of value.language_pairs) {
    const source = normalizeTag(pair.source),
      target = normalizeTag(pair.target);
    if (![source, target].every((tag) => /^[a-z]{2,3}(?:-[a-z0-9]{2,8})*$/.test(tag)))
      return "Use language tags such as ja, en or pt-BR for every source and target.";
    if (source === target) return "Source and target languages must differ.";
    const key = `${source}:${target}`;
    if (seen.has(key)) return "Remove duplicate language pairs.";
    seen.add(key);
  }
  return "";
}

async function request(
  options: RequestInit | undefined,
  section: `/${Section}`,
): Promise<Partial<Configuration>> {
  const response = await fetch(endpoint + section, {
    ...options,
    cache: "no-store",
    credentials: "same-origin",
  });
  // Never display response bodies: validation payloads must not expose credentials.
  if (!response.ok) {
    if (response.status === 401) throw new Error("Your session expired. Sign in again.");
    if (response.status === 403) throw new Error("Administrator access is required.");
    if (response.status === 422)
      throw new Error("Invalid External AI settings. Check the URL, prompt and language pairs.");
    throw new Error("Could not access External AI settings. Try again.");
  }
  const body = await response.json();
  if (section === "/endpoint")
    return { enabled: body.enabled, url: body.url, has_key: body.has_key };
  if (section === "/transcription")
    return {
      enabled: body.enabled,
      transcription_refinement: body.transcription_refinement,
      transcription_prompt: body.transcription_prompt,
      default_transcription_prompt: body.default_transcription_prompt,
    };
  return { model: body.model, prompt: body.prompt, language_pairs: body.language_pairs };
}

export default function ExternalAISettings() {
  return (
    <div className={styles.sections}>
      <SettingsSection section="endpoint" />
      <SettingsSection section="translation" />
      <SettingsSection section="transcription" />
    </div>
  );
}

function SettingsSection({ section }: { section: Section }) {
  const [value, setValue] = useState<Configuration | null>(null);
  const [confirmClear, setConfirmClear] = useState(false);
  const [apiKey, setApiKey] = useState("");
  const [clearKey, setClearKey] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [attempt, setAttempt] = useState(0);
  // Only a failed load stays on the page; action results are toasts.
  const [loadError, setLoadError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    request({ signal: controller.signal }, `/${section}`)
      .then((result) => {
        if (!controller.signal.aborted)
          setValue({
            enabled: false,
            url: "",
            has_key: false,
            model: "",
            prompt: "",
            language_pairs: [],
            transcription_refinement: true,
            transcription_prompt: "",
            ...result,
          });
      })
      .catch((reason) => {
        if (!controller.signal.aborted)
          setLoadError(reason instanceof Error ? reason.message : "Could not load settings.");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [attempt, section]);

  useEffect(() => {
    if (section !== "transcription") return;
    const follow = (event: Event) => update({ enabled: (event as CustomEvent<boolean>).detail });
    window.addEventListener("echora:external-ai-enabled", follow);
    return () => window.removeEventListener("echora:external-ai-enabled", follow);
  }, [section]);

  function update(patch: Partial<Configuration>) {
    setValue((current) => (current ? { ...current, ...patch } : current));
  }
  async function save(event: FormEvent) {
    event.preventDefault();
    if (!value || saving) return;
    const invalid = validate(value, section);
    if (invalid) {
      toast.error("Check these settings", { description: invalid });
      return;
    }
    setSaving(true);
    const payload =
      section === "endpoint"
        ? {
            enabled: value.enabled,
            url: value.url,
            ...(clearKey ? { api_key: "" } : apiKey ? { api_key: apiKey } : {}),
          }
        : section === "transcription"
          ? {
              transcription_refinement: value.transcription_refinement,
              transcription_prompt: value.transcription_prompt.trim(),
            }
          : {
              model: value.model.trim(),
              prompt: value.prompt.trim(),
              language_pairs: value.language_pairs.map((pair) => ({
                source: normalizeTag(pair.source),
                target: normalizeTag(pair.target),
              })),
            };
    try {
      const result = await request(
        {
          method: "PUT",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(payload),
        },
        `/${section}`,
      );
      setValue((current) => (current ? { ...current, ...result } : current));
      setApiKey("");
      setClearKey(false);
      if (section === "endpoint")
        window.dispatchEvent(
          new CustomEvent("echora:external-ai-enabled", { detail: Boolean(result.enabled) }),
        );
      toast.success("External AI settings saved");
    } catch (reason) {
      toast.error("Could not save External AI settings", {
        description: reason instanceof Error ? reason.message : undefined,
      });
    } finally {
      setSaving(false);
    }
  }

  async function clearTranslations() {
    if (saving) return;
    setSaving(true);
    try {
      const response = await fetch(`${endpoint}/translations`, {
        method: "DELETE",
        credentials: "same-origin",
      });
      if (response.status === 409)
        throw new Error("Wait for active sync and lyrics jobs to stop, then try again.");
      if (!response.ok) throw new Error("Could not clear translations.");
      const result = await response.json();
      window.dispatchEvent(new Event("echora:translations-cleared"));
      setConfirmClear(false);
      toast.success(`Cleared ${result.deleted} translations`, {
        description: "Save any prompt changes before running sync.",
      });
    } catch (reason) {
      toast.error("Could not clear translations", {
        description: reason instanceof Error ? reason.message : undefined,
      });
    } finally {
      setSaving(false);
    }
  }

  return (
    <form className={styles.form} onSubmit={save} aria-busy={loading || saving}>
      {loading ? (
        <LoadingState
          label={`Loading ${section === "endpoint" ? "API endpoint" : section === "translation" ? "lyrics translation" : "transcription refinement"} settings…`}
        />
      ) : (
        value && (
          <>
            <fieldset disabled={saving}>
              {section === "endpoint" && (
                <section aria-labelledby="external-ai-endpoint-title">
                  <SectionHeading
                    id="external-ai-endpoint-title"
                    title="API endpoint"
                    description="Configure the OpenAI-compatible endpoint shared by External AI features."
                  />
                  <Toggle
                    label="Enable External AI"
                    checked={value.enabled}
                    onChange={(enabled) => update({ enabled })}
                    disabled={saving}
                  />
                  <SettingRow
                    label="API base URL"
                    htmlFor="external-ai-url"
                    description={
                      <span id="external-ai-url-help">
                        /chat/completions is appended. Local HTTP is allowed; prefer HTTPS off-host.
                      </span>
                    }
                  >
                    <Input
                      id="external-ai-url"
                      type="url"
                      maxLength={2048}
                      required={value.enabled}
                      value={value.url}
                      placeholder="http://localhost:8000/v1"
                      onChange={(event) => update({ url: event.target.value })}
                      aria-describedby="external-ai-url-help"
                    />
                  </SettingRow>
                  <SettingRow
                    label="API key"
                    htmlFor="external-ai-key"
                    description="Leave blank to keep the stored key, or enter a replacement."
                  >
                    <div className={styles.keyControls}>
                      <Input
                        id="external-ai-key"
                        type="password"
                        autoComplete="new-password"
                        maxLength={8192}
                        disabled={clearKey}
                        value={apiKey}
                        placeholder={value.has_key && !clearKey ? "••••••••••••••••" : ""}
                        aria-label={
                          value.has_key
                            ? "API key, stored key unchanged unless replaced"
                            : "API key"
                        }
                        onChange={(event) => {
                          setApiKey(event.target.value);
                        }}
                      />
                      <Button
                        type="button"
                        variant="outline"
                        className="shrink-0"
                        aria-pressed={clearKey}
                        onClick={() => {
                          setClearKey((current) => !current);
                          setApiKey("");
                        }}
                      >
                        {clearKey ? "Cancel removal" : "Clear on save"}
                      </Button>
                    </div>
                  </SettingRow>
                </section>
              )}
              {section === "translation" && (
                <section aria-labelledby="external-ai-translation-title">
                  <SectionHeading
                    id="external-ai-translation-title"
                    title="Lyrics translation"
                    description="Missing translations are generated during sync when External AI is enabled."
                  />
                  <SettingRow
                    label="Model"
                    htmlFor="external-ai-model"
                    description={
                      <span id="external-ai-model-help">
                        Use the model identifier provided by your endpoint.
                      </span>
                    }
                  >
                    <Input
                      id="external-ai-model"
                      maxLength={200}
                      value={value.model}
                      onChange={(event) => update({ model: event.target.value })}
                      aria-describedby="external-ai-model-help"
                    />
                  </SettingRow>
                  <SettingRow
                    label="Language pairs"
                    description="Choose source and target language tags, such as ja, en or pt-BR."
                  >
                    <div className={styles.pairs}>
                      {value.language_pairs.map((pair, index) => (
                        <div className={styles.pair} key={index}>
                          <Label>
                            <span>
                              Source <span className="sr-only">{index + 1}</span>
                            </span>
                            <Input
                              required
                              value={pair.source}
                              placeholder="ja"
                              onChange={(event) =>
                                update({
                                  language_pairs: value.language_pairs.map((item, i) =>
                                    i === index ? { ...item, source: event.target.value } : item,
                                  ),
                                })
                              }
                            />
                          </Label>
                          <Label>
                            <span>
                              Target <span className="sr-only">{index + 1}</span>
                            </span>
                            <Input
                              required
                              value={pair.target}
                              placeholder="en"
                              onChange={(event) =>
                                update({
                                  language_pairs: value.language_pairs.map((item, i) =>
                                    i === index ? { ...item, target: event.target.value } : item,
                                  ),
                                })
                              }
                            />
                          </Label>
                          <Button
                            type="button"
                            variant="outline"
                            aria-label={`Remove language pair ${index + 1}`}
                            onClick={() =>
                              update({
                                language_pairs: value.language_pairs.filter((_, i) => i !== index),
                              })
                            }
                          >
                            Remove
                          </Button>
                        </div>
                      ))}
                      <Button
                        type="button"
                        variant="outline"
                        disabled={value.language_pairs.length >= 64}
                        onClick={() =>
                          update({
                            language_pairs: [...value.language_pairs, { source: "", target: "" }],
                          })
                        }
                      >
                        Add language pair
                      </Button>
                    </div>
                  </SettingRow>
                  <SettingRow
                    label="Instruction prompt"
                    htmlFor="external-ai-prompt"
                    description="Instructions sent to the translation model."
                  >
                    <Textarea
                      className={styles.prompt}
                      id="external-ai-prompt"
                      required
                      maxLength={8000}
                      rows={5}
                      value={value.prompt}
                      onChange={(event) => update({ prompt: event.target.value })}
                    />
                  </SettingRow>
                </section>
              )}
              {section === "transcription" && (
                <section aria-labelledby="external-ai-transcription-title">
                  <SectionHeading
                    id="external-ai-transcription-title"
                    title="Transcription refinement"
                    description="During sync, AI-transcribed lyrics get a second pass that fixes mis-heard words using the whole song as context. Line timing is kept. Uses the model set under Lyrics translation."
                  />
                  {!value.enabled && (
                    <SettingsNotice tone="info" title="External AI is off">
                      Refinement runs only while External AI is enabled.
                    </SettingsNotice>
                  )}
                  <Toggle
                    label="Refine AI transcriptions"
                    checked={value.enabled && value.transcription_refinement}
                    onChange={(transcription_refinement) => update({ transcription_refinement })}
                    disabled={saving || !value.enabled}
                  />
                  <SettingRow
                    label="Refinement prompt"
                    htmlFor="external-ai-transcription-prompt"
                    description={
                      <span id="external-ai-transcription-prompt-help">
                        Instructions for the second pass. The response format is added
                        automatically.{" "}
                        {value.default_transcription_prompt &&
                          value.transcription_prompt.trim() !==
                            value.default_transcription_prompt && (
                            <button
                              type="button"
                              className="underline"
                              onClick={() =>
                                update({ transcription_prompt: value.default_transcription_prompt })
                              }
                            >
                              Restore default
                            </button>
                          )}
                      </span>
                    }
                  >
                    <Textarea
                      className={styles.prompt}
                      id="external-ai-transcription-prompt"
                      required
                      maxLength={8000}
                      rows={8}
                      value={value.transcription_prompt}
                      onChange={(event) => update({ transcription_prompt: event.target.value })}
                      aria-describedby="external-ai-transcription-prompt-help"
                    />
                  </SettingRow>
                </section>
              )}
              <div className={layout.actions}>
                <Button type="submit" loading={saving}>
                  {saving
                    ? "Saving…"
                    : section === "endpoint"
                      ? "Save API settings"
                      : section === "translation"
                        ? "Save translation settings"
                        : "Save refinement settings"}
                </Button>
              </div>
              {section === "translation" && (
                <section className={layout.group}>
                  <SectionHeading title="Stored translations" />
                  <SettingsNotice tone="warning" title="Instance-wide deletion">
                    Clearing translations affects every user and cannot be undone.
                  </SettingsNotice>
                  <SettingRow
                    label="Clear all translations"
                    description="Instance-wide. Original lyrics and karaoke timing are kept."
                  >
                    <Button
                      variant="destructive"
                      type="button"
                      disabled={saving}
                      onClick={() => setConfirmClear(true)}
                    >
                      Clear all translations
                    </Button>
                  </SettingRow>
                </section>
              )}
            </fieldset>
          </>
        )
      )}
      {confirmClear && (
        <ClearTranslationsWarning
          busy={saving}
          onClose={() => setConfirmClear(false)}
          onConfirm={clearTranslations}
        />
      )}
      {!loading && !value && (
        <Notice
          tone="error"
          title="Could not load External AI settings"
          action={
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => {
                setLoading(true);
                setLoadError("");
                setAttempt((current) => current + 1);
              }}
            >
              Try again
            </Button>
          }
        >
          {loadError}
        </Notice>
      )}
    </form>
  );
}
