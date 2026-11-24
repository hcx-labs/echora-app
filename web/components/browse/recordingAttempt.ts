// Owns server admission independently of React renders. Uploads may resolve after
// cancellation: retain their response long enough to cancel the admitted job.
export class RecordingAttempt {
  generation = 0;
  jobId = "";
  private cancelJob: (id: string) => void;
  constructor(cancelJob: (id: string) => void) { this.cancelJob = cancelJob; }
  invalidate() {
    this.generation++;
    if (this.jobId) this.cancelJob(this.jobId);
    this.jobId = "";
  }
  admit(generation: number, id: string) {
    if (generation !== this.generation) { this.cancelJob(id); return false; }
    this.jobId = id;
    return true;
  }
}
