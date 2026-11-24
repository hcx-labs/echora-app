import type { ReactNode } from "react";
import { Pane } from "./pane";

export function SidePanel({ title, subtitle, actions, children }: { title: string; subtitle?: ReactNode; actions?: ReactNode; children: ReactNode }) {
  return <Pane title={title} subtitle={subtitle} actions={actions}>{children}</Pane>;
}
