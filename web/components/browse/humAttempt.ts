// One capture/search lifetime. Aborting invalidates callbacks as well as fetches.
export class HumAttempt {
  readonly controller = new AbortController();
  private stream: MediaStream | null = null;
  recorder: MediaRecorder | null = null;

  get current() { return !this.controller.signal.aborted; }
  get signal() { return this.controller.signal; }

  attachStream(stream: MediaStream) {
    if (!this.current) {
      stream.getTracks().forEach(track => track.stop());
      return false;
    }
    this.stream = stream;
    return true;
  }

  release() {
    if (this.recorder) {
      this.recorder.onstop = null;
      this.recorder.ondataavailable = null;
      this.recorder.onerror = null;
      if (this.recorder.state !== "inactive") this.recorder.stop();
      this.recorder = null;
    }
    this.stream?.getTracks().forEach(track => track.stop());
    this.stream = null;
  }

  cancel() {
    this.controller.abort();
    this.release();
  }
}
