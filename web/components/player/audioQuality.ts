import type { AudioQuality } from "./PlayerProvider";

const codecName = (codec?: string) => {
  const value = codec?.trim().toUpperCase();
  return value === "M4A" ? "AAC" : value || "AUDIO";
};

export function audioQualityLabel(quality: AudioQuality | null): string {
  if (!quality) return "Reading audio quality";
  if (quality.streamQuality !== "original") return `MP3 · ${quality.streamQuality} kbps`;
  const parts = [quality.lossless ? "LOSSLESS" : codecName(quality.codec)];
  if (quality.lossless) parts.push(codecName(quality.codec));
  if (quality.bit_depth) parts.push(`${quality.bit_depth}-bit`);
  if (quality.sample_rate_hz) {
    const khz = quality.sample_rate_hz / 1000;
    parts.push(`${Number.isInteger(khz) ? khz : khz.toFixed(1)} kHz`);
  }
  if (quality.bit_rate_kbps) parts.push(`${quality.bit_rate_kbps} kbps`);
  return parts.join(" · ");
}

/** Compact form for tight spaces, e.g. "FLAC 16/44.1" or "MP3 320". The full label belongs in a tooltip. */
export function audioQualityShortLabel(quality: AudioQuality | null): string {
  if (!quality) return "";
  if (quality.streamQuality !== "original") return `MP3 ${quality.streamQuality}`;
  const codec = codecName(quality.codec);
  if (quality.lossless && quality.bit_depth && quality.sample_rate_hz) {
    const khz = quality.sample_rate_hz / 1000;
    return `${codec} ${quality.bit_depth}/${Number.isInteger(khz) ? khz : khz.toFixed(1)}`;
  }
  return quality.bit_rate_kbps ? `${codec} ${quality.bit_rate_kbps}` : codec;
}
