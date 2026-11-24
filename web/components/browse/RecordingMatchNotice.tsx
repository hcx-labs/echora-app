import { TriangleAlert } from "lucide-react";
import { Badge } from "../ui/badge";
import styles from "./RecordingMatchNotice.module.css";

export default function RecordingMatchNotice({ message }: { message: string }) {
  if (!message) return null;
  return <div className={styles.notice} role="status"><Badge variant="outline"><TriangleAlert aria-hidden="true" />Recording match</Badge><p>{message}</p></div>;
}
