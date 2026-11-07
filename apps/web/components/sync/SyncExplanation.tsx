import { PaneSection, PaneSections } from "../layout/pane";

const steps = [
  ["Read the library", "Fetch track IDs, metadata, artwork references and counts from Navidrome."],
  [
    "Verify track identity",
    "Source-audio SHA-256 hashes resolve duplicates without merging recordings. Entire-library sync can re-verify known audio.",
  ],
  [
    "Fill missing analysis",
    "Existing analysis is kept. New-tracks sync only processes unindexed tracks; Entire library also refreshes metadata and lyrics.",
  ],
] as const;

// Mirrors the analysis service: representations.py, download_models.py, roformer.py,
// hum_search.py, language_detection.py, voice_pipeline.py, the optional MOSS / recording encoders
// and motion artwork. `phases` are the batch progress phases in which each model runs.
const models: {
  name: string;
  role: string;
  detail: string;
  phases: string[];
  optional?: boolean;
}[] = [
  {
    name: "MuQ-MuLan large",
    phases: ["muq"],
    role: "Musical meaning",
    detail:
      "Audio–language embedding for style, mood and instrumentation. Powers semantic Galaxy, concepts, curations and sound shape. Uses MuQ-large-msd-iter and XLM-RoBERTa encoders.",
  },
  {
    name: "MERT v1 95M",
    phases: ["mert"],
    role: "Acoustic structure",
    detail:
      "Self-supervised audio embedding for timbre, rhythm, pitch and production. Powers acoustic Galaxy, journeys and sound shape.",
  },
  {
    name: "BGE-M3",
    phases: ["lyrics"],
    role: "Lyrics meaning",
    detail: "Multilingual text embedding of lyrics for themes and the lyrics Galaxy.",
  },
  {
    name: "Semantic fusion",
    phases: ["building", "writing"],
    role: "Combined space",
    detail:
      "Whitened, weighted blend of MuQ-MuLan and BGE-M3 vectors. Rebuilt after sync; no separate model.",
  },
  {
    name: "Essentia Discogs-EffNet",
    phases: ["voice"],
    role: "Vocals",
    detail: "Voice/instrumental and vocal-gender classification heads.",
  },
  {
    name: "Mel-Band-Roformer",
    phases: ["preprocess"],
    role: "Vocal separation",
    detail: "Isolates vocals for hum search and transcription.",
  },
  {
    name: "Essentia MELODIA",
    phases: ["melody", "melody-index"],
    role: "Melody",
    detail: "Predominant-pitch contours for query by humming.",
  },
  {
    name: "fastText LID-176 · IndicLID",
    phases: ["lyrics"],
    role: "Language",
    detail: "Detects lyrics language, including romanized Indic text.",
  },
  {
    name: "Echora Kara 300M",
    phases: ["karaoke"],
    role: "Karaoke timing",
    detail: "Aligns lyrics to the song for syllable-timed karaoke.",
  },
  {
    name: "Echora MOSS Transcribe + Diarize",
    phases: ["transcription"],
    role: "Lyrics transcription",
    detail: "Transcribes missing lyrics when AI lyrics are enabled in Settings.",
    optional: true,
  },
  {
    name: "Recording encoder",
    phases: ["recording_fingerprint"],
    role: "Identify song",
    detail: "Fingerprints for matching recorded audio. Recognition is separately gated.",
    optional: true,
  },
  {
    name: "LTX-2.5 22B distilled",
    phases: ["motion-artwork"],
    role: "Motion artwork",
    detail:
      "Animates each cover into a seamless loop that starts and ends on the cover, held steady by Lightricks' static-camera LoRA and, when upscaling is on, sharpened by the 2× spatial upscaler. Enable it in Settings → Motion artwork.",
    optional: true,
  },
];

/** `activePhases`: progress phases of the sync's running batches; their models are highlighted. */
export default function SyncExplanation({ activePhases = [] }: { activePhases?: string[] }) {
  return (
    <PaneSections>
      <PaneSection
        title="How sync works"
        hint="Sync reads your Navidrome library without changing tracks or files."
      >
        <ol className="grid gap-4">
          {steps.map(([title, detail], index) => (
            <li key={title} className="grid grid-cols-[20px_minmax(0,1fr)] gap-x-2.5">
              <span className="grid size-5 place-items-center border border-border-strong text-xs tabular-nums text-muted-foreground">
                {index + 1}
              </span>
              <div>
                <p className="text-[13px] font-medium">{title}</p>
                <p className="mt-0.5 text-xs leading-relaxed text-muted-foreground">{detail}</p>
              </div>
            </li>
          ))}
        </ol>
      </PaneSection>
      <PaneSection
        title="Analysis models"
        hint={`${models.length} models and analysis steps power Echora.`}
      >
        <ul className="grid grid-cols-[minmax(0,1fr)] gap-px border border-border bg-border">
          {models.map((model) => {
            const running = model.phases.some((phase) => activePhases.includes(phase));
            return (
              <li
                key={model.name}
                aria-current={running ? "step" : undefined}
                className={`min-w-0 px-3 py-2.5 transition-colors ${running ? "bg-primary/10 shadow-[inset_2px_0_0_var(--accent)]" : "bg-rail"}`}
              >
                <div className="flex items-baseline justify-between gap-2">
                  <p
                    className="flex min-w-0 items-center gap-1.5 text-[13px] font-medium"
                    title={model.name}
                  >
                    {running && (
                      <span
                        aria-hidden
                        className="size-1.5 shrink-0 animate-pulse rounded-full bg-primary"
                      />
                    )}
                    <span className="truncate">{model.name}</span>
                    {running && <span className="sr-only">(running now)</span>}
                  </p>
                  <span className="shrink-0 text-xs text-primary">{model.role}</span>
                </div>
                <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                  {model.detail}
                  {model.optional && <span className="text-subtle-foreground"> Optional.</span>}
                </p>
              </li>
            );
          })}
        </ul>
      </PaneSection>
    </PaneSections>
  );
}
