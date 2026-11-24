import type { ReactNode } from "react";
import { Notice } from "../ui/notice";

type Tone = "info" | "admin" | "warning" | "error" | "success";

export function SettingsNotice({ title, children, tone = "info" }: { title: string; children?: ReactNode; tone?: Tone }) {
  return <Notice tone={tone === "admin" ? "info" : tone} title={title} className="my-5">{children}</Notice>;
}
