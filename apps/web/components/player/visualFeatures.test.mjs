import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';
const source = readFileSync(new URL('./visualFeatures.ts', import.meta.url), 'utf8');
const { validateVisualFeatures: validate, visualFrameAt, beatClockAt, neutralVisualFrame, publishVisualFrame, descriptorRhythm, validateStructure, validateVisualEnrichment } = await import(`data:text/javascript;base64,${Buffer.from(stripTypeScriptTypes(source)).toString('base64')}`);
function fixture() {
  const n = 5, hop = 512 / 22050;
  const vector = () => Array(n).fill(.2);
  const matrix = size => Array.from({ length: n }, () => Array(size).fill(.2));
  return { revision: '2', sample_rate: 22050, hop_length: 512, hop_seconds: hop, duration_seconds: n * hop, source_offset_seconds: 0, frame_count: n,
    bands: matrix(24), band_centers_hz: Array.from({ length: 24 }, (_, i) => 30 + i * 100), level: vector(), centroid: vector(), flux: vector(), onset: vector(), chroma: matrix(12), pitch: matrix(84), pitch_midi_start: 24,
    waveform: matrix(64).map(row => row.map((_, i) => i % 2 ? -1 : 1)), waveform_offsets_samples: Array.from({ length: 64 }, (_, i) => Math.round(i * 2047 / 63)), waveform_window_samples: 2048,
    reactivity: matrix(3), attacks: matrix(3), reactivity_edges_hz: [20, 250, 2000, 10000], timbre: Object.fromEntries(['centroid_hz', 'bandwidth_hz', 'rolloff_hz', 'flatness', 'zcr'].map(key => [key, vector()])), onset_times_seconds: [hop], beat_times_seconds: [hop], tempo: { bpm: 120, candidates_bpm: [60, 120, 240] }, kernels: { kind: 'fixed_nonlearned', names: ['temporal_rise', 'temporal_fall', 'spectral_slope', 'spectral_curvature'], values: matrix(4).map(row => { row[2] = -.5; return row; }) } };
}
test('accepts revision 2 including signed spectral slope and real PCM', () => {
  const data = validate(fixture()); assert.ok(data);
  const frame = visualFrameAt(data, data.hop_seconds * 1.1, 0);
  assert.equal(frame.bpm, 120); assert.equal(frame.beat, true); assert.equal(frame.onset, true);
  assert.deepEqual(frame.waveform, data.waveform[1]); assert.equal(frame.features.kernels.values[2], -.5);
});
test('rejects invalid dimensions, bounds, metadata, nonfinite and legacy payloads', () => {
  const changes = [v => v.revision = '1', v => v.duration_seconds = 601, v => v.frame_count = 25841,
    v => v.bands[0].pop(), v => v.level.pop(), v => v.pitch[0][0] = NaN, v => v.waveform[0][0] = -1.01,
    v => v.reactivity_edges_hz[1] = 180, v => v.waveform_offsets_samples[63] = 2048, v => v.waveform_window_samples = 1024,
    v => v.timbre.centroid_hz[0] = 11026, v => v.kernels.values[0][0] = -.1, v => v.kernels.values[0][2] = -1.1,
    v => v.beat_times_seconds = [.1, .05], v => v.beat_times_seconds = [.01, .01], v => v.onset_times_seconds = [1], v => v.tempo.bpm = Infinity,
    v => v.hop_seconds = 0, v => v.source_offset_seconds = 1, v => delete v.chroma[0]];
  for (const change of changes) { const value = fixture(); change(value); assert.equal(validate(value), null, String(change)); }
});
test('no repeated transients; seek/reset cannot generate a false beat', () => {
  const data = validate(fixture()), at = data.hop_seconds * 1.1;
  for (const previous of [null, at, at + 1]) { const frame = visualFrameAt(data, at, previous); assert.equal(frame.beat, false); assert.equal(frame.onset, false); assert.equal(frame.bassAttack, 0); }
});
test('missing/end/out-of-range frames are immediately neutral', () => {
  const data = validate(fixture());
  for (const [timeline, at] of [[null, 0], [data, -1], [data, data.duration_seconds], [data, 100]]) {
    const frame = visualFrameAt(timeline, at, 0); assert.equal(frame.active, false); assert.equal(frame.bpm, null); assert.ok(frame.waveform.every(n => n === 0)); assert.equal(frame.features, null);
  }
});
test('one unified frame plus shell-only compatibility bridge, including resets', () => {
  const events = []; globalThis.window = { dispatchEvent: event => events.push(event) };
  const frame = neutralVisualFrame(); publishVisualFrame(frame);
  assert.deepEqual(events.map(event => event.type), ['echora:visual-frame', 'echora:audio-reactivity']);
  assert.ok(events.every(event => event.detail === frame)); delete globalThis.window;
});

test('descriptor rhythm requires recognized complete matching-source data and otherwise keeps core tempo', () => {
  const data = validate(fixture());
  const enrichment = { descriptors: { revision: '1', status: 'complete', descriptors: { revision: '1', duration_seconds: data.duration_seconds, rhythm: { bpm: 90, beat_times_seconds: [.01, .03] } } } };
  data.rhythm = descriptorRhythm(enrichment, data.duration_seconds);
  assert.equal(visualFrameAt(data, .035, .025).bpm, 90);
  assert.equal(visualFrameAt(data, .035, .025).beat, true);
  assert.equal(visualFrameAt(data, .035, .025).rhythmSource, 'descriptors');
  for (const change of [v => v.descriptors.status = 'partial', v => v.descriptors.revision = '2', v => v.descriptors.descriptors.duration_seconds += 1, v => v.descriptors.descriptors.rhythm.beat_times_seconds = [.03, .01]]) {
    const value = structuredClone(enrichment); change(value); data.rhythm = descriptorRhythm(value, data.duration_seconds);
    assert.equal(data.rhythm, null); assert.equal(visualFrameAt(data, .035, .025).bpm, 120);
  }
});
test('structure validates separately with square dimensions and a hard 192-bin bound', () => {
  const value = { method: 'pooled_mfcc_rbf_chroma_cosine_nonlearned', max_bins: 192, edges_seconds: [0, 1], novelty: [0], chroma_cosine: [[1]], mfcc_rbf: [[1]] };
  assert.equal(validateStructure(value, 1), true);
  assert.equal(validateStructure({ ...value, novelty: Array(193).fill(0) }, 1), false);
  assert.equal(validateStructure({ ...value, mfcc_rbf: [[1, 1]] }, 1), false);
});

test('cached vocal activity preserves gaps/tail and melody preview preserves unvoiced points', () => {
  const data = validate(fixture());
  const enrichment = { vocal_activity: { activity: { duration_seconds: data.duration_seconds, windows: [{ start_seconds: .01, end_seconds: .04, vocal_activation: .8 }] } }, melody: { source: 'vocals', points: [{ time_seconds: 0, pitch: 69 }, { time_seconds: .04, pitch: null }] } };
  data.enrichment = validateVisualEnrichment(enrichment, data.duration_seconds);
  assert.equal(visualFrameAt(data, .02, null).enrichment.vocalActivation, .8);
  assert.equal(visualFrameAt(data, .001, null).enrichment.vocalActivation, null);
  assert.equal(visualFrameAt(data, .05, null).enrichment.vocalActivation, null);
  assert.equal(visualFrameAt(data, .02, null).enrichment.melodyPreviewPitchMidi, 69);
  assert.equal(visualFrameAt(data, .05, null).enrichment.melodyPreviewPitchMidi, null);
  const invalid = structuredClone(enrichment); invalid.vocal_activity.activity.duration_seconds += 1; invalid.melody.points[0].pitch = Infinity;
  assert.deepEqual(validateVisualEnrichment(invalid, data.duration_seconds), { vocal: null, melody: null });
});
test('unified frame includes source units, ambiguity and bounded structural novelty', () => {
  const value = fixture();
  value.structure = { method: 'pooled_mfcc_rbf_chroma_cosine_nonlearned', max_bins: 192, edges_seconds: [0, value.duration_seconds], novelty: [.25], chroma_cosine: [[1]], mfcc_rbf: [[1]] };
  const data = validate(value), frame = visualFrameAt(data, .02, null);
  assert.equal(frame.source.sampleRate, 22050); assert.equal(frame.source.pitchMidiStart, 24);
  assert.deepEqual(frame.tempoCandidatesBpm, [60, 120, 240]); assert.equal(frame.structuralNovelty, .25);
  assert.equal(data.structureSummary.chroma_cosine, undefined);
});

test('beat clock follows a beat grid whose tempo changes', () => {
  // Four beats at 0.5 s, then the song slows to 0.6 s per beat.
  const beats = [1, 1.5, 2, 2.5, 3, 3.6, 4.2, 4.8, 5.4, 6, 6.6];
  const at = time => beatClockAt(beats, time);
  assert.equal(at(1.25).position, 0.5);
  assert.equal(at(3.3).position, 4.5);
  assert.ok(Math.abs(at(1.25).period - 0.5) < 1e-9);
  assert.ok(Math.abs(at(6.3).period - 0.6) < 1e-9);
  // One late beat in a steady grid does not move the local tempo.
  const steady = [0, 0.5, 1, 1.62, 2, 2.5, 3, 3.5, 4];
  assert.ok(Math.abs(beatClockAt(steady, 1.8).period - 0.5) < 1e-9);
  // Before the first beat and after the last, the edge interval continues the grid.
  assert.ok(Math.abs(at(0.5).position + 1) < 1e-9);
  assert.ok(Math.abs(at(7.2).position - 11) < 1e-9);
  assert.equal(beatClockAt([1], 1), null);
});
