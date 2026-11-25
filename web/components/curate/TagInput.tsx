"use client";

import { useRef, useState, type KeyboardEvent } from "react";
import { X } from "lucide-react";
import { useId } from "react";
import { Label } from "../ui/label";

export type Tag = { label: string; negative: boolean };

export default function TagInput({ label, hint, placeholder, tags, onChange }: {
  label: string;
  hint?: string;
  placeholder: string;
  tags: Tag[];
  onChange: (tags: Tag[]) => void;
}) {
  const id = useId();
  const [draft, setDraft] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  function commit() {
    const value = draft.trim().replace(/^[-+]+/, "").trim();
    if (!value) { setDraft(""); return; }
    const negative = draft.trim().startsWith("-");
    if (tags.some(tag => tag.label.toLowerCase() === value.toLowerCase() && tag.negative === negative)) { setDraft(""); return; }
    onChange([...tags, { label: value, negative }]);
    setDraft("");
  }

  function onKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === "Enter" || event.key === ",") { event.preventDefault(); commit(); }
    else if (event.key === "Backspace" && !draft && tags.length) { onChange(tags.slice(0, -1)); }
  }

  function remove(index: number) { onChange(tags.filter((_, position) => position !== index)); }
  const excluding = draft.trim().startsWith("-");

  return <div className="grid content-start gap-2">
    <Label htmlFor={id}>{label}{hint && <span className="font-normal text-muted-foreground">· {hint}</span>}</Label>
    {/* The whole field focuses the text input; tags live inline before it. */}
    <div onClick={() => inputRef.current?.focus()} className="flex min-h-9 cursor-text flex-wrap items-center gap-1.5 border border-border-strong bg-surface px-1.5 py-1 transition-colors hover:border-[#444] focus-within:border-ring focus-within:ring-2 focus-within:ring-ring/30">
      {tags.map((tag, index) => <span key={`${tag.label}-${index}`} className={`inline-flex h-7 max-w-full items-center gap-1 border pr-0.5 pl-2 text-[13px] ${tag.negative ? "border-destructive/40 bg-destructive/10 text-destructive" : "border-success/40 bg-success/10 text-success"}`}>
        <span className="truncate">{tag.negative ? "−" : "+"} {tag.label}</span>
        <button type="button" className="grid size-6 place-items-center opacity-70 hover:opacity-100" onClick={event => { event.stopPropagation(); remove(index); }} aria-label={`Remove ${tag.label}`}><X className="size-3.5" /></button>
      </span>)}
      <input ref={inputRef} id={id} value={draft} onChange={event => setDraft(event.target.value)} onKeyDown={onKeyDown} onBlur={commit} placeholder={tags.length ? "" : placeholder} className={`h-7 min-w-28 flex-1 bg-transparent px-1.5 text-base outline-none placeholder:text-subtle-foreground md:text-[13px] ${excluding ? "text-destructive" : ""}`} />
    </div>
  </div>;
}
