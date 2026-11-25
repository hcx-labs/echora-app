"use client";

import { runtimeConfig } from "../runtime/runtimeConfig";

import { useEffect, useState } from "react";
import { jobRequest } from "./durableJobs";
import { batchStatus, type Batch } from "./batchStatus";
import styles from "./BatchStatusList.module.css";

type Response = { batches: Batch[]; total: number };
const PAGE_SIZE = 10;

export default function BatchStatusList({ jobId, active }: { jobId: string; active: boolean }) {
  const [page, setPage] = useState(0);
  const [data, setData] = useState<Response | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      let retry = active;
      try {
        const value = await jobRequest<Response>(`/jobs/${encodeURIComponent(jobId)}/batches?limit=${PAGE_SIZE}&offset=${page * PAGE_SIZE}`, controller.signal);
        if (controller.signal.aborted) return;
        setData(value); setError("");
        if (page > 0 && page * PAGE_SIZE >= value.total) setPage(0);
      } catch {
        if (controller.signal.aborted) return;
        setError("Could not update batch status. Retrying."); retry = true;
      }
      if (retry && !controller.signal.aborted) timer = setTimeout(poll, runtimeConfig().batch_poll_ms);
    }
    void poll();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [jobId, active, page]);
  if (data?.total === 0 && !error) return <p role="status">{active ? "Preparing sync batches." : "This job has no track batches."}</p>;
  return <section className={styles.panel} aria-label="Batch status">
    <header><strong>Batch status</strong></header>
    {error && <p role="status">{error}</p>}
    {!data && !error && <p>Reading batch status</p>}
    <div className={styles.list} tabIndex={0} aria-label="Batches">
      {data?.batches.map(batch => {
        const detail = batchStatus(batch);
        return <article key={batch.job_id} data-status={batch.status}>
          <div><strong>Batch {batch.batch_number}</strong><span>{batch.status}</span></div>
          <div><strong>{detail.stage}</strong><small>{detail.progress}</small>
            {detail.message && <p>{detail.message}</p>}
          </div>
        </article>;
      })}
    </div>
    {data && data.total > PAGE_SIZE && <footer>
      <button type="button" disabled={page === 0} onClick={() => setPage(value => value - 1)}>Previous</button>
      <span>Page {page + 1} of {Math.ceil(data.total / PAGE_SIZE)}</span>
      <button type="button" disabled={(page + 1) * PAGE_SIZE >= data.total} onClick={() => setPage(value => value + 1)}>Next</button>
    </footer>}
  </section>;
}
