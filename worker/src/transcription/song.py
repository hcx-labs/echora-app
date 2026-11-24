"""Shared Roformer vocals + windowed MOSS fallback, with sequential GPU inference."""

import re
from contextlib import contextmanager, nullcontext
from worker.transcription.recovery import WindowDecodeError, stalled_segments
from worker.media.roformer import SEPARATION_REVISION
from worker.media.preprocessing import vocal_waveform

PROMPT = "Transcribe the audio. For each segment, start with the timestamp and speaker ID ([S01], [S02], [S03], ...), then the spoken text, and end with the segment timestamp."


def transcription_prompt(language: str | None) -> str:
    if not language:
        return PROMPT
    return f"Transcribe the audio in {language}. Do not translate it. " + PROMPT[20:]


PATTERN = re.compile(r"\[(\d+(?:\.\d+)?)\]\[(S\d+|MULTI)\]([^\[\]]*)\[(\d+(?:\.\d+)?)\]")
PIPELINE_REVISION = f"roformer-moss-w60-o12-timing-v7:{SEPARATION_REVISION}"


def windows(duration):
    if duration <= 0:
        raise ValueError("Empty audio")
    start = 0.0
    while True:
        end = min(duration, start + 60)
        yield start, end
        if end >= duration:
            return
        start += 48


def parse_window(text, offset, duration):
    matches = list(PATTERN.finditer(text.strip()))
    if not matches or "".join(m[0] for m in matches) != text.strip():
        raise ValueError("Incomplete MOSS transcript")
    segments = []
    for m in matches:
        start, end = float(m[1]), float(m[4])
        if not 0 <= start < end <= duration + 0.01 or not m[3].strip():
            raise ValueError("Invalid MOSS interval")
        segments.append(
            {
                "start_ms": round((start + offset) * 1000),
                "end_ms": round((min(end, duration) + offset) * 1000),
                "speaker": m[2],
                "text": m[3].strip(),
            }
        )
    return segments


class SongTranscriber:
    """MOSS transcription in batch phases: load once, generate many songs, release.

    Phases (see docs/batch-phases.md): vocals are separated first (a separation phase);
    generation runs with MOSS loaded once for every song; timing repair, which needs the
    karaoke aligner instead, runs afterwards for the songs that need it. `transcribe`
    does all three for one song, as before.
    """

    def __init__(self, model_id, revision):
        self.model_id, self.revision = model_id, revision
        self._processor = None
        self._model = None

    def _load_processor(self):
        if self._processor is None:
            from huggingface_hub import snapshot_download
            from .vendor.moss.processor import MossTranscribeDiarizeProcessor

            self._snapshot = snapshot_download(
                self.model_id, revision=self.revision, local_files_only=True
            )
            self._processor = MossTranscribeDiarizeProcessor.from_pretrained(
                self._snapshot, local_files_only=True, trust_remote_code=True
            )
        return self._processor

    @property
    def sample_rate(self) -> int:
        return int(self._load_processor().feature_extractor.sampling_rate)

    def open(self) -> None:
        """Load MOSS onto the device; stays loaded until close()."""
        import torch
        from transformers import AutoModelForCausalLM

        if self._model is not None:
            return
        self._load_processor()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
        self._model = (
            AutoModelForCausalLM.from_pretrained(
                self._snapshot,
                trust_remote_code=True,
                local_files_only=True,
                dtype=self._dtype,
                attn_implementation="sdpa",
            )
            .to(self._device)
            .eval()
        )

    def close(self) -> None:
        import torch

        if self._model is None:
            return
        self._model.to("cpu")
        self._model = None
        if self._device == "cuda":
            torch.cuda.empty_cache()

    @contextmanager
    def loaded(self):
        """MOSS loaded for one generation phase, released at its end."""
        self.open()
        try:
            yield self
        finally:
            self.close()

    def transcribe(
        self,
        audio_bytes,
        language: str | None = None,
        check=lambda: None,
        diagnostic_sink=lambda _: None,
        vocal_activity=None,
        progress=lambda _: None,
    ):
        """One song, all phases: separate, generate with MOSS, release MOSS, repair timing."""
        # Reads the prepared stem, or completes separation and releases its model
        # before ASR loads. Karaoke reuses the same source/configuration artifact.
        vocal_waveform(audio_bytes, sample_rate=self.sample_rate, check=check)
        with self.loaded():
            draft = self.generate(
                audio_bytes, language, check, diagnostic_sink, vocal_activity, progress
            )
        try:
            return self.finish(draft, check, progress)
        finally:
            if draft["repairs"]:
                from worker.kara.pipeline import _stop_fa_kara_worker

                _stop_fa_kara_worker()

    def generate(
        self,
        audio_bytes,
        language: str | None = None,
        check=lambda: None,
        diagnostic_sink=lambda _: None,
        vocal_activity=None,
        progress=lambda _: None,
    ):
        """Generation for one song with the loaded model. Returns a draft for finish().

        The draft keeps the vocals of at most two windows whose timestamps came out
        compressed: timing repair in finish() aligns those, and only those.
        """
        import numpy as np
        import torch
        from transformers import LogitsProcessorList, StoppingCriteriaList
        from worker.transcription.guard import SegmentGuard
        from worker.transcription.budget import generate_song
        from worker.transcription.timing import compressed_timing

        if self._model is None:
            raise RuntimeError("Open the transcriber before generating")
        processor, model, device, dtype = self._processor, self._model, self._device, self._dtype
        check()
        sr = int(processor.feature_extractor.sampling_rate)
        # The separation phase normally cached these vocals. On a miss (a pruned or
        # uncacheable stem), MOSS leaves the GPU while Roformer separates, so two
        # large models never share it.
        offloaded = []

        def make_room():
            if device == "cuda" and not offloaded:
                model.to("cpu")
                torch.cuda.empty_cache()
                offloaded.append(True)

        vocals = vocal_waveform(audio_bytes, sample_rate=sr, check=check, before_separate=make_room)
        if offloaded:
            model.to(device)
        duration = len(vocals) / sr
        ranges = list(windows(duration))
        diagnostics = []
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": "in-memory"},
                    {"type": "text", "text": transcription_prompt(language)},
                ],
            }
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        def decode(start, end, token_limit):
            check()
            with (
                torch.inference_mode(),
                torch.autocast("cuda", dtype=dtype) if device == "cuda" else nullcontext(),
            ):
                inputs = processor(
                    text=prompt,
                    audio=[vocals[round(start * sr) : round(end * sr)]],
                    audio_kwargs={"device": device},
                    return_tensors="pt",
                ).to(device)
                plen = inputs["input_ids"].shape[1]
                guard = SegmentGuard(processor.tokenizer, plen)

                def stop_stall(input_ids, scores, **kwargs):
                    generated = input_ids.shape[1] - plen
                    if generated % 64 == 0:
                        check()
                    if generated % 16:
                        return False
                    return stalled_segments(
                        processor.tokenizer.decode(
                            input_ids[0, plen:].tolist(), skip_special_tokens=True
                        )
                    )

                output = model.generate(
                    **inputs,
                    max_new_tokens=token_limit,
                    do_sample=False,
                    stopping_criteria=StoppingCriteriaList([stop_stall]),
                    repetition_penalty=1.0,
                    no_repeat_ngram_size=0,
                    logits_processor=LogitsProcessorList([guard]),
                )
            ids = output[0, plen:]
            text = processor.tokenizer.decode(ids, skip_special_tokens=True).strip()
            attempt = {"start": start, "end": end, "tokens": len(ids), "guard_events": guard.events}
            # Release decode tensors before recursive retries.
            del inputs, output, ids
            try:
                if stalled_segments(text):
                    raise WindowDecodeError("MOSS repeated a segment without advancing timestamps")
                if attempt["tokens"] >= token_limit:
                    raise WindowDecodeError("MOSS hit token limit")
                segments = parse_window(text, start, end - start)
            except ValueError as error:
                attempt["error"] = str(error)
                diagnostics.append(attempt)
                diagnostic_sink({**attempt, "raw_output": text})
                raise WindowDecodeError(
                    f"{start:.2f}-{end:.2f}s: {error}", tokens=attempt["tokens"]
                ) from error
            diagnostics.append(attempt)
            diagnostic_sink({**attempt, "raw_output": text})
            return segments, attempt["tokens"]

        def retry_policy(start, end):
            from worker.transcription.evidence import retry_evidence

            samples = vocals[round(start * sr) : round(end * sr)]
            rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) if samples.size else 0
            evidence = retry_evidence(
                vocal_activity, start, end, duration, float(20 * np.log10(max(rms, 1e-12)))
            )
            if evidence:
                diagnostic_sink({"start": start, "end": end, "retry_evidence": evidence})
            return evidence

        candidates, unresolved, budget = generate_song(
            ranges, decode, retry_policy, check, progress
        )
        check()
        # Timing repair is budgeted to the first two compressed candidates, in order.
        repairs = {}
        for index, (_, start, end, segments, _) in enumerate(candidates):
            if compressed_timing(segments, start, end) is not None and len(repairs) < 2:
                repairs[index] = np.ascontiguousarray(vocals[round(start * sr) : round(end * sr)])
        return {
            "ranges": ranges,
            "duration": duration,
            "sr": sr,
            "candidates": candidates,
            "unresolved": unresolved,
            "budget": budget,
            "diagnostics": diagnostics,
            "repairs": repairs,
        }

    def finish(self, draft, check=lambda: None, progress=lambda _: None):
        """Timing repair (with the karaoke aligner, only if the draft needs it) and the result.

        Needs no MOSS. The aligner process is left running for the next song; the
        phase that repairs stops it.
        """
        from worker.transcription.budget import reconcile_song
        from worker.transcription.timing import compressed_timing, apply_aligned_times

        ranges, duration, sr = draft["ranges"], draft["duration"], draft["sr"]
        diagnostics = draft["diagnostics"]
        timing_repairs = []
        result = []
        # Candidates are matched to their kept audio by position; reconcile_song passes them in order.
        by_window = {(c[1], c[2], id(c[3])): index for index, c in enumerate(draft["candidates"])}

        def reconcile(start, end, segments):
            evidence = compressed_timing(segments, start, end)
            if evidence is None:
                return segments
            record = {"start": start, "end": end, **evidence}
            if len(timing_repairs) >= 2:
                diagnostics.append({**record, "timing_status": "repair_budget_exhausted"})
                return segments
            timing_repairs.append(record)
            progress(f"Reconciling compressed timestamps at {start:.0f}–{end:.0f}s")
            check()
            import io
            import soundfile as sf
            from worker.kara.pipeline import _run_fa_kara

            buffer = io.BytesIO()
            sf.write(
                buffer,
                draft["repairs"][by_window[(start, end, id(segments))]],
                sr,
                format="WAV",
                subtype="FLOAT",
            )
            try:
                aligned = _run_fa_kara(
                    buffer.getvalue(),
                    "\n".join(s["text"] for s in segments),
                    None,
                    separate_vocals=False,
                )
                corrected = apply_aligned_times(segments, aligned["lines"], start, end)
                if compressed_timing(corrected, start, end) is not None:
                    raise ValueError("Alignment did not resolve timestamp compression")
            except (ValueError, RuntimeError) as error:
                if "out of memory" in str(error).lower():
                    raise
                check()
                record.update(status="unresolved", error=str(error)[-1000:])
                return segments
            check()
            record.update(
                status="repaired",
                aligner_model=aligned["model"],
                aligner_revision=aligned["model_revision"],
            )
            return corrected

        decoded = reconcile_song(ranges, draft["candidates"], reconcile, check)
        for i, (start, end) in enumerate(ranges):
            segments = decoded[i]
            left = 0 if i == 0 else (ranges[i - 1][1] + start) / 2
            right = duration if i == len(ranges) - 1 else (end + ranges[i + 1][0]) / 2
            result.extend(
                s for s in segments if left <= (s["start_ms"] + s["end_ms"]) / 2000 < right
            )
        check()
        if not result:
            raise ValueError("No usable lyrics generated")
        result.sort(key=lambda s: s["start_ms"])
        unresolved, budget = draft["unresolved"], draft["budget"]
        return {
            "text": "\n".join(s["text"] for s in result),
            "status": "available",
            "synced": True,
            "lines": result,
            "ai_generated": True,
            "transcription": {
                "model": self.model_id,
                "revision": self.revision,
                "pipeline_revision": PIPELINE_REVISION,
                "separator": "mel-band-roformer",
                "separator_revision": SEPARATION_REVISION,
                "windows": diagnostics,
                "partial": bool(unresolved),
                "unresolved_windows": unresolved,
                "retry_budget": budget,
                "timing_repairs": timing_repairs,
                "recovery_policy": "Omit unresolved retry windows, never globally deduplicate repeated lyrics",
                "speaker_policy": "Sxx/MULTI labels are unverified",
            },
        }


def needs_repair(draft) -> bool:
    """Whether finishing this draft runs the karaoke aligner (a separate phase)."""
    return bool(draft["repairs"])
