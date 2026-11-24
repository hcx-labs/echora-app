import type { ReactNode } from "react";
import { PanelHeader, PanelBody, PanelFooter } from "../layout/panel";
import styles from "./GalaxyPanel.module.css";

export default function GalaxyPanel({ title, header, description, actions, children, footer, className = "", active }: {
  title: string;
  /** Replaces the title text with richer content, such as artwork and metadata. */
  header?: ReactNode;
  description?: string;
  actions?: ReactNode;
  children: ReactNode;
  footer?: ReactNode;
  className?: string;
  active?: boolean;
}) {
  return <section aria-label={title} data-active={active} className={`motion-enter ${styles.panel} ${className}`}>
    {header ? <header className={styles.richHeader}><div className={styles.richHeaderContent}>{header}</div>{actions && <div className={styles.richHeaderActions}>{actions}</div>}</header> : <PanelHeader title={title} actions={actions} compact />}
    <PanelBody className="min-h-0 flex-1 overflow-y-auto p-4">{description && <p className={styles.description}>{description}</p>}{children}</PanelBody>
    {footer && <PanelFooter>{footer}</PanelFooter>}
  </section>;
}
