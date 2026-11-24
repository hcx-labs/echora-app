import type { ReactNode } from "react";
import { Label } from "../ui/label";
import styles from "./SettingRow.module.css";

export function SettingRow({ label, description, htmlFor, children }: { label: ReactNode; description?: ReactNode; htmlFor?: string; children: ReactNode }) {
  return <div className={styles.row}><div><div className={styles.label}>{htmlFor ? <Label htmlFor={htmlFor}>{label}</Label> : label}</div>{description && <div className={styles.description}>{description}</div>}</div><div className={styles.control}>{children}</div></div>;
}
