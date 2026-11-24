"use client";

import { AudioLines, Clapperboard, Languages, MicVocal, Sparkles } from "lucide-react";
import type { ReactNode } from "react";
import { Button } from "../ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "../ui/dialog";

type Props = {
  transcribed: boolean;
  translatedLanguages: string[];
  motionArtwork?: boolean;
  /** Word timing made by Echora's aligner (not supplied with the lyrics). */
  alignedKaraoke?: boolean;
};

/** One kind of AI processing: its icon beside a heading and an explanation. */
function Item({ icon, title, children }: { icon: ReactNode; title: string; children: ReactNode }) {
  return (
    <section className="flex items-start gap-3">
      <span aria-hidden="true" className="mt-0.5 shrink-0 text-muted-foreground [&_svg]:size-4">
        {icon}
      </span>
      <div>
        <h3>{title}</h3>
        <p>{children}</p>
      </div>
    </section>
  );
}

export default function AiLyricsIndicator({
  transcribed,
  translatedLanguages,
  motionArtwork = false,
  alignedKaraoke = false,
}: Props) {
  const languages = [...new Set(translatedLanguages)];
  if (!transcribed && !languages.length && !motionArtwork && !alignedKaraoke) return null;
  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          aria-label="AI processing applied to this song"
          title="AI processing details"
        >
          <Sparkles />
        </Button>
      </DialogTrigger>
      <DialogContent className="z-[1200]">
        <DialogHeader>
          <DialogTitle>AI processing</DialogTitle>
          <DialogDescription>AI-generated results can contain errors.</DialogDescription>
        </DialogHeader>
        {transcribed && (
          <Item icon={<AudioLines />} title="Transcription">
            AI transcribed the original lyrics from audio.
          </Item>
        )}
        {alignedKaraoke && (
          <Item icon={<MicVocal />} title="Karaoke">
            AI timed each word of the lyrics to the singing. Words can light up early or late.
          </Item>
        )}
        {languages.length > 0 && (
          <Item icon={<Languages />} title="Translation">
            <>
              AI translated the lyrics into{" "}
              {languages
                .map((language) => {
                  try {
                    return (
                      new Intl.DisplayNames(["en"], { type: "language" }).of(language) || language
                    );
                  } catch {
                    return language;
                  }
                })
                .join(", ")}
              .
            </>
          </Item>
        )}
        {motionArtwork && (
          <Item icon={<Clapperboard />} title="Motion artwork">
            AI generated this video from the album cover. It may contain visual mistakes or differ
            from the original artwork.
          </Item>
        )}
      </DialogContent>
    </Dialog>
  );
}
