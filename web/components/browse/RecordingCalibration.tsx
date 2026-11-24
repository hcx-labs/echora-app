"use client";

import { toast } from "sonner";
import { useEffect, useRef, useState } from "react";
import TrackReferencePicker, { type ReferenceTrack } from "../curate/TrackReferencePicker";
import { CALIBRATION_SAMPLES, calibrationRequest } from "./recordingCalibration";
import { Checkbox } from "../ui/checkbox";
import { Textarea } from "../ui/textarea";
import { Badge } from "../ui/badge";
import { Button } from "../ui/button";
import styles from "./RecordingCalibration.module.css";

type Sample = { id: string; created_at: string; expires_at: string; expected_track_id?: string; expected_title?: string; expected_artist?: string; not_in_library: boolean; notes: string; duration_seconds: number };
export default function RecordingCalibration({ audio, clearAudio, captureBusy, onSaving }: { audio: Blob | null; clearAudio: () => void; captureBusy: boolean; onSaving: (saving: boolean) => void }) {
  const preview = useRef<HTMLAudioElement>(null);
  const [tracks, setTracks] = useState<ReferenceTrack[]>([]);
  const [negative, setNegative] = useState(false);
  const [notes, setNotes] = useState("");
  const [samples, setSamples] = useState<Sample[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [working, setWorking] = useState("");
  const [reload, setReload] = useState(0);
  const lock = useRef(false);
  useEffect(() => {
    if (!audio) return;
    const url = URL.createObjectURL(audio);
    const player = preview.current;
    if (player) player.src = url;
    return () => { if (player) { player.pause(); player.removeAttribute("src"); player.load(); } URL.revokeObjectURL(url); };
  }, [audio]);
  useEffect(() => {
    const controller = new AbortController();
    fetch(CALIBRATION_SAMPLES, { signal: controller.signal, cache: "no-store" }).then(async response => {
      if (!response.ok) throw new Error("Could not load saved clips. Retry to check your saved samples.");
      const body = await response.json();
      if (!controller.signal.aborted) setSamples(body.samples);
    }).catch(reason => { if (!controller.signal.aborted) setError(reason.message); }).finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [reload]);
  async function save() {
    if (!audio || lock.current || captureBusy) return;
    lock.current = true; setWorking("save"); onSaving(true);
    try {
      const { url, init } = calibrationRequest(audio, tracks[0]?.id, negative, notes);
      const response = await fetch(url, init);
      const body = await response.json().catch(() => null);
      if (response.status !== 201) throw new Error(typeof body?.detail === "string" ? body.detail : "Could not save clip. Check saved samples before retrying.");
      setSamples(current => [{ ...body, expected_title: tracks[0]?.title, expected_artist: tracks[0]?.artist }, ...current]); clearAudio(); setTracks([]); setNegative(false); setNotes("");
      toast.success("Calibration clip saved", { description: "No identification was performed." });
    } catch (reason) { toast.error("Could not save clip", { description: reason instanceof Error ? reason.message : "Check saved samples before retrying." }); }
    finally { lock.current = false; setWorking(""); onSaving(false); }
  }
  async function remove(id: string) {
    if (lock.current) return;
    lock.current = true; setWorking(id);
    try {
      const response = await fetch(`${CALIBRATION_SAMPLES}/${encodeURIComponent(id)}`, { method: "DELETE" });
      if (response.status !== 204) throw new Error("Could not delete clip. Retry deletion.");
      setSamples(current => current.filter(sample => sample.id !== id)); toast.success("Calibration clip deleted");
    } catch (reason) { toast.error("Could not delete clip", { description: reason instanceof Error ? reason.message : undefined }); }
    finally { lock.current = false; setWorking(""); }
  }
  return <div className={styles.panel}>
    {audio && <div className={styles.preview}><Badge variant="secondary">Local preview · not uploaded</Badge><audio ref={preview} controls /><Button type="button" disabled={!!working} onClick={clearAudio}>Discard local clip</Button></div>}
    <fieldset disabled={!!working || captureBusy}>
      <legend>Label the recording yourself</legend>
      <label><Checkbox disabled={!!working || captureBusy} checked={negative} onCheckedChange={checked => { setNegative(checked === true); setTracks([]); }} /> Not in my library</label>
      {!negative && <TrackReferencePicker label="Actual track in your library" value={tracks} onChange={setTracks} single />}
      <label className={styles.notes}>Speaker or background noise (optional)<Textarea maxLength={300} rows={2} value={notes} onChange={event => setNotes(event.target.value)} /></label>
    </fieldset>
    <p id="calibration-consent">Saving uploads the raw clip privately to server disk for 7 days, including any captured speech. Only save audio you are authorized to share. You can delete it below.</p>
    <Button type="button" aria-describedby="calibration-consent" disabled={loading || !audio || captureBusy || !!working || (!negative && !tracks.length)} onClick={() => void save()}>{working === "save" ? "Saving calibration clip…" : "Save calibration clip"}</Button>
    {error && <p role="alert">{error}</p>}
    <section className={styles.saved} aria-label="Saved calibration samples"><h3>Your saved calibration clips</h3><Button type="button" disabled={!!working || loading} onClick={() => { setError(""); setLoading(true); setReload(value => value + 1); }}>Refresh saved clips</Button>
      {loading ? <p role="status">Loading saved clips…</p> : !error && !samples.length && <p>No saved calibration clips.</p>}
      <div className={styles.list}>{samples.map(sample => <article key={sample.id}><strong>{sample.not_in_library ? "Not in my library" : sample.expected_title || tracks.find(track => track.id === sample.expected_track_id)?.title || "Library track"}</strong>{sample.expected_artist && <span>{sample.expected_artist}</span>}<small>{sample.duration_seconds.toFixed(1)} seconds · Saved {new Date(sample.created_at).toLocaleString()} · Expires {new Date(sample.expires_at).toLocaleString()}</small>{sample.notes && <p>{sample.notes}</p>}<Button type="button" disabled={loading || !!working} aria-label={`Delete calibration clip ${sample.expected_title || sample.id}`} onClick={() => void remove(sample.id)}>{working === sample.id ? "Deleting…" : "Delete clip"}</Button></article>)}</div>
    </section>
  </div>;
}
