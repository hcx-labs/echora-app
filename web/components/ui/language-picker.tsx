"use client";

import * as React from "react";
import { Check, ChevronsUpDown, Search, X } from "lucide-react";

import { cn } from "@/lib/utils";
import { Popover, PopoverContent, PopoverTrigger } from "./popover";

// ISO 639-1, plus the regional variants people pick for lyrics (Brazilian Portuguese,
// Traditional Chinese...). Names come from the browser (Intl.DisplayNames), so they need no list.
const CODES = (
  "aa ab af ak am an ar as av ay az ba be bg bi bm bn bo br bs ca ce ch co cr cs cu cv cy da de " +
  "dv dz ee el en eo es et eu fa ff fi fj fo fr fy ga gd gl gn gu gv ha he hi ho hr ht hu hy hz " +
  "ia id ie ig ii ik io is it iu ja jv ka kg ki kj kk kl km kn ko kr ks ku kv kw ky la lb lg li " +
  "ln lo lt lu lv mg mh mi mk ml mn mr ms mt my na nb nd ne ng nl nn no nr nv ny oc oj om or os " +
  "pa pi pl ps pt qu rm rn ro ru rw sa sc sd se sg si sk sl sm sn so sq sr ss st su sv sw ta te " +
  "tg th ti tk tl tn to tr ts tt tw ty ug uk ur uz ve vi vo wa wo xh yi yo za zh zu " +
  "en-US en-GB es-419 es-ES fr-CA pt-BR pt-PT zh-CN zh-TW zh-HK yue fil haw"
).split(" ");

let displayNames: Intl.DisplayNames | null | undefined;

/** The language's English name ("Brazilian Portuguese" for pt-BR), or the code itself. */
export function languageName(code: string): string {
  if (displayNames === undefined) {
    try {
      displayNames = new Intl.DisplayNames(["en"], { type: "language" });
    } catch {
      displayNames = null;
    }
  }
  try {
    const name = displayNames?.of(code);
    return name && name.toLowerCase() !== code.toLowerCase() ? name : code;
  } catch {
    return code;
  }
}

const CODE_PATTERN = /^[a-z]{2,3}(?:-[a-z0-9]{2,8})*$/i;

/** ``pt-br`` → ``pt-BR``: the language lower case, a region upper case, a script title case. */
export function canonicalLanguage(code: string): string {
  const [language, ...rest] = code.trim().replace(/_/g, "-").split("-");
  return [
    language.toLowerCase(),
    ...rest.map((part) =>
      /^[a-z]{2}$|^\d{3}$/i.test(part)
        ? part.toUpperCase()
        : part.length === 4
          ? part[0].toUpperCase() + part.slice(1).toLowerCase()
          : part.toLowerCase(),
    ),
  ].join("-");
}

type Option = { code: string; name: string };

let options: Option[] | null = null;
function allOptions(): Option[] {
  options ??= CODES.map((code) => ({ code, name: languageName(code) })).sort((a, b) =>
    a.name.localeCompare(b.name),
  );
  return options;
}

type Common = {
  /** Codes not offered (the target among its own sources). */
  exclude?: string[];
  placeholder?: string;
  disabled?: boolean;
  "aria-label"?: string;
  className?: string;
  /** How codes are stored: ``pt-BR`` (default) or lower case ``pt-br``. */
  lowercase?: boolean;
};

type Single = Common & { multiple?: false; value: string; onChange: (code: string) => void };
type Multiple = Common & { multiple: true; value: string[]; onChange: (codes: string[]) => void };

/**
 * A searchable language dropdown: people pick names ("Japanese"), the value is the code ("ja").
 * Typing a code that is not listed ("mni") offers it as is.
 */
export function LanguagePicker(props: Single | Multiple) {
  const { exclude = [], placeholder = "Choose a language", disabled, className, lowercase } = props;
  const [open, setOpen] = React.useState(false);
  const [query, setQuery] = React.useState("");
  const [active, setActive] = React.useState(0);
  const listRef = React.useRef<HTMLUListElement>(null);
  const listId = React.useId();
  const store = (code: string) => (lowercase ? code.toLowerCase() : canonicalLanguage(code));
  const selected = props.multiple ? props.value : props.value ? [props.value] : [];
  const isSelected = (code: string) =>
    selected.some((value) => value.toLowerCase() === code.toLowerCase());
  const excludeKey = exclude.map((code) => code.toLowerCase()).join(",");
  const excluded = new Set(excludeKey ? excludeKey.split(",") : []);

  const matches = React.useMemo(() => {
    const needle = query.trim().toLowerCase();
    const list = allOptions().filter((option) => !excluded.has(option.code.toLowerCase()));
    if (!needle) return list;
    return list
      .filter(
        (option) =>
          option.name.toLowerCase().includes(needle) ||
          option.code.toLowerCase().startsWith(needle),
      )
      .sort((a, b) => {
        const rank = (option: Option) =>
          option.code.toLowerCase() === needle
            ? 0
            : option.name.toLowerCase().startsWith(needle)
              ? 1
              : 2;
        return rank(a) - rank(b);
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps -- ``excluded`` is ``excludeKey``
  }, [query, excludeKey]);
  const typed = query.trim();
  const custom =
    CODE_PATTERN.test(typed) &&
    !excluded.has(typed.toLowerCase()) &&
    !matches.some((option) => option.code.toLowerCase() === typed.toLowerCase())
      ? { code: canonicalLanguage(typed), name: languageName(canonicalLanguage(typed)) }
      : null;
  const shown = custom ? [custom, ...matches] : matches;

  React.useEffect(() => {
    listRef.current?.querySelector<HTMLElement>(`[data-index="${active}"]`)?.scrollIntoView({
      block: "nearest",
    });
  }, [active]);

  function choose(code: string) {
    if (props.multiple) {
      props.onChange(
        isSelected(code)
          ? props.value.filter((value) => value.toLowerCase() !== code.toLowerCase())
          : [...props.value, store(code)],
      );
      setQuery("");
      setActive(0);
    } else {
      props.onChange(store(code));
      setOpen(false);
    }
  }

  function keyDown(event: React.KeyboardEvent) {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActive((index) => Math.min(index + 1, shown.length - 1));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setActive((index) => Math.max(index - 1, 0));
    } else if (event.key === "Enter") {
      event.preventDefault();
      if (shown[active]) choose(shown[active].code);
    } else if (event.key === "Backspace" && !query && props.multiple && props.value.length) {
      props.onChange(props.value.slice(0, -1));
    }
  }

  const label = props["aria-label"] ?? placeholder;
  return (
    <Popover
      open={open}
      onOpenChange={(next) => {
        if (disabled) return;
        setOpen(next);
        setActive(0);
      }}
    >
      <PopoverTrigger asChild>
        <button
          type="button"
          role="combobox"
          aria-expanded={open}
          aria-controls={listId}
          aria-label={label}
          disabled={disabled}
          className={cn(
            "flex min-h-9 w-full min-w-0 items-center gap-2 border border-input bg-transparent px-3 py-1.5 text-left text-[13px] outline-none transition-colors hover:border-border-strong focus-visible:border-ring disabled:cursor-not-allowed disabled:opacity-50",
            className,
          )}
        >
          <span className="flex min-w-0 flex-1 flex-wrap gap-1">
            {selected.length === 0 && (
              <span className="truncate text-subtle-foreground">{placeholder}</span>
            )}
            {!props.multiple && selected[0] && (
              <span className="flex min-w-0 flex-1 items-center justify-between gap-2">
                <span className="truncate">{languageName(selected[0])}</span>
                <span className="shrink-0 text-xs text-subtle-foreground">{selected[0]}</span>
              </span>
            )}
            {props.multiple &&
              props.value.map((code) => (
                <span
                  key={code}
                  className="inline-flex max-w-full items-center gap-1 border border-border bg-surface px-1.5 py-0.5 text-xs"
                >
                  <span className="truncate">{languageName(code)}</span>
                  <span
                    role="button"
                    tabIndex={-1}
                    aria-label={`Remove ${languageName(code)}`}
                    onPointerDown={(event) => {
                      event.preventDefault();
                      event.stopPropagation();
                      props.onChange(props.value.filter((value) => value !== code));
                    }}
                    className="text-subtle-foreground hover:text-foreground"
                  >
                    <X className="size-3" />
                  </span>
                </span>
              ))}
          </span>
          <ChevronsUpDown className="size-3.5 shrink-0 text-subtle-foreground" />
        </button>
      </PopoverTrigger>
      <PopoverContent
        align="start"
        className="w-[var(--radix-popover-trigger-width)] min-w-64 p-0"
        onOpenAutoFocus={(event) => {
          event.preventDefault();
          (event.currentTarget as HTMLElement).querySelector("input")?.focus();
        }}
      >
        <label className="flex h-9 items-center gap-2 border-b border-border px-3">
          <Search className="size-3.5 shrink-0 text-subtle-foreground" />
          <input
            aria-label={`Search ${label.toLowerCase()}`}
            aria-controls={listId}
            value={query}
            onChange={(event) => {
              setQuery(event.target.value);
              setActive(0);
            }}
            onKeyDown={keyDown}
            placeholder="Search languages or type a code"
            className="h-full min-w-0 flex-1 bg-transparent text-[13px] outline-none placeholder:text-subtle-foreground"
          />
        </label>
        <ul
          id={listId}
          ref={listRef}
          role="listbox"
          aria-multiselectable={props.multiple || undefined}
          className="max-h-64 overflow-y-auto py-1"
        >
          {shown.map((option, index) => {
            const on = isSelected(option.code);
            return (
              <li key={option.code} role="option" aria-selected={on} data-index={index}>
                <button
                  type="button"
                  onMouseEnter={() => setActive(index)}
                  onClick={() => choose(option.code)}
                  className={cn(
                    "grid h-9 w-full grid-cols-[1rem_minmax(0,1fr)_auto] items-center gap-2 px-3 text-left text-[13px]",
                    index === active ? "bg-hover text-foreground" : "text-foreground/85",
                    on && "text-primary",
                  )}
                >
                  {on ? <Check className="size-3.5" /> : <span />}
                  <span className="truncate">{option.name}</span>
                  <span className="text-xs text-subtle-foreground">{option.code}</span>
                </button>
              </li>
            );
          })}
          {!shown.length && (
            <li className="px-3 py-6 text-center text-xs text-subtle-foreground">
              No language matches “{query}”
            </li>
          )}
        </ul>
      </PopoverContent>
    </Popover>
  );
}
