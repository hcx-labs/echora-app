"use client";

import { useState } from "react";
import { toast } from "sonner";
import {
  defaultPlaybackPreferences,
  PlaybackPreferences,
  readPlaybackPreferences,
  writePlaybackPreferences,
} from "../player/playbackPreferences";
import { SectionHeading, Choice, ChoiceItem, Toggle } from "./Presentation";
import { RangeControl } from "./RangeControl";
import { SettingRow } from "./SettingRow";
import { SettingsNotice } from "./SettingsNotice";

/** Motion, karaoke highlights and the audio-reactive backdrop, saved in this browser.
 * `essentials` leaves out the fine controls (frame rate, opacity, band sensitivity);
 * `quiet` skips the confirmation toast on each change. */
export default function AppearanceSettings({
  essentials = false,
  quiet = false,
}: {
  essentials?: boolean;
  quiet?: boolean;
}) {
  // Rendered only on the client, after the page has loaded its user.
  const [playback, setPlayback] = useState<PlaybackPreferences>(() =>
    typeof window === "undefined" ? defaultPlaybackPreferences : readPlaybackPreferences(),
  );
  function savePlayback(next: PlaybackPreferences) {
    setPlayback(next);
    writePlaybackPreferences(next);
    if (!quiet) toast.success("Playback preferences saved", { id: "playback-preferences" });
  }
  function saveAnimationSpeed(next: PlaybackPreferences["animationSpeed"]) {
    savePlayback({ ...playback, animationSpeed: next });
  }
  return (
    <section>
      <SectionHeading title="Motion and interface" />
      <SettingRow
        label={<>Animation speed</>}
        htmlFor="settings-field-7"
        description={<>This preference applies across the entire application.</>}
      >
        <Choice
          id="settings-field-7"
          value={playback.animationSpeed}
          onChange={(value) => saveAnimationSpeed(value as PlaybackPreferences["animationSpeed"])}
        >
          <ChoiceItem value="slow">Slow</ChoiceItem>
          <ChoiceItem value="normal">Normal</ChoiceItem>
          <ChoiceItem value="fast">Fast</ChoiceItem>
        </Choice>
      </SettingRow>
      <SettingRow
        label={<>Karaoke highlight style</>}
        htmlFor="settings-field-8"
        description={
          <>
            Highlight each syllable at once, fill it with a moving liquid edge, bring words out of a
            blur (Bloom), or punch each word in like a combo hit (Strike). Timing stays the same.
            Saved in this browser.
          </>
        }
      >
        <Choice
          id="settings-field-8"
          value={playback.karaokeHighlightStyle}
          onChange={(value) =>
            savePlayback({
              ...playback,
              karaokeHighlightStyle: value as PlaybackPreferences["karaokeHighlightStyle"],
            })
          }
        >
          <ChoiceItem value="syllable">Syllable highlight</ChoiceItem>
          <ChoiceItem value="lava">Lava fill</ChoiceItem>
        </Choice>
      </SettingRow>
      <section className="pt-6">
        <SectionHeading
          title="Audio-reactive backdrop"
          description="Choose how the desktop background responds to the playing track and its frequency bands."
        />
        <SettingRow
          label={<>Backdrop style</>}
          htmlFor="settings-field-9"
          description={
            <>
              All styles use the playing track and its album-derived color palette. Backdrops are
              hidden in the compact mobile layout.
            </>
          }
        >
          <Choice
            id="settings-field-9"
            value={playback.backdropPreset}
            onChange={(value) =>
              savePlayback({
                ...playback,
                backdropPreset: value as PlaybackPreferences["backdropPreset"],
              })
            }
          >
            <ChoiceItem value="waves">PS3 waves</ChoiceItem>
            <ChoiceItem value="oscilloscope">Oscilloscope</ChoiceItem>
            <ChoiceItem value="void">Void tunnel</ChoiceItem>
            <ChoiceItem value="curtain">Digital curtain</ChoiceItem>
            <ChoiceItem value="ascii">ASCII dance</ChoiceItem>
            <ChoiceItem value="roots">Living roots</ChoiceItem>
            <ChoiceItem value="lightningfall">Lightning Fall</ChoiceItem>
            <ChoiceItem value="meshgrid">Mesh Grid</ChoiceItem>
            <ChoiceItem value="clouds">Storm clouds</ChoiceItem>
            <ChoiceItem value="waterdrops">Water drops</ChoiceItem>
          </Choice>
          {playback.backdropPreset === "waterdrops" && (
            <SettingsNotice tone="info" title="Backdrop behavior">
              Music drives invisible drop impacts on an album-colored water surface. Bass sets
              impact size, vocals change drop velocity and ripple spread, and treble changes surface
              tension and damping. Ripples overlap and interfere, then settle when playback pauses.
            </SettingsNotice>
          )}
          {playback.backdropPreset === "meshgrid" && (
            <SettingsNotice tone="info" title="Backdrop behavior">
              A rippling triangular landscape in album-art colors. Bass raises the waves, vocals
              roughen the mesh, and treble lights the edges.
            </SettingsNotice>
          )}
          {playback.backdropPreset === "lightningfall" && (
            <SettingsNotice tone="info" title="Backdrop behavior">
              Dense falling trails in album-art colors. Bass and tempo drive the flow; individual
              frequency bands subtly vary trail length, thickness, and glow. Pausing dims the
              trails.
            </SettingsNotice>
          )}
          {playback.backdropPreset === "clouds" && (
            <SettingsNotice tone="warning" title="Flashing effects">
              Full-background smoke. Bass drives billowing, vocals drive cloud color, and treble
              lights cloud edges and soft internal lightning. Attacks in any band can trigger
              branching strikes. Drift follows estimated tempo. Strikes are at least 0.8 seconds
              apart. Contains flashes. Set all three sensitivity sliders to zero for no lightning,
              or turn off Animated backdrop.
            </SettingsNotice>
          )}
        </SettingRow>
        {!essentials && (
          <>
            <SettingRow
              label={<>Wave frame rate</>}
              htmlFor="settings-field-10"
              description={
                <>Stored in this browser so each device can use an appropriate rendering load.</>
              }
            >
              <Choice
                id="settings-field-10"
                value={playback.waveFrameRate}
                onChange={(value) =>
                  savePlayback({
                    ...playback,
                    waveFrameRate: value as PlaybackPreferences["waveFrameRate"],
                  })
                }
              >
                <ChoiceItem value="30">30 FPS</ChoiceItem>
                <ChoiceItem value="60">60 FPS</ChoiceItem>
                <ChoiceItem value="uncapped">Uncapped</ChoiceItem>
              </Choice>
            </SettingRow>
            <Toggle
              label="Animated backdrop"
              checked={playback.wavesEnabled}

              onChange={() => savePlayback({ ...playback, wavesEnabled: !playback.wavesEnabled })}
            />
            <div>
              <SettingRow
                label={
                  <>
                    <b>Backdrop opacity</b>
                    <output className="font-normal tabular-nums">
                      {Math.round(playback.backdropOpacity * 100)}%
                    </output>
                  </>
                }
                htmlFor="settings-field-11"
              >
                <RangeControl
                  id="settings-field-11"
                  aria-label="Backdrop opacity"
                  min="0"
                  max="1"
                  step="0.05"
                  value={playback.backdropOpacity}
                  onChange={(event) =>
                    savePlayback({
                      ...playback,
                      backdropOpacity: Number(event.target.value),
                    })
                  }
                />
              </SettingRow>
            </div>
            <div>
              {(
                [
                  ["bassReactivity", "Bass"],
                  ["vocalReactivity", "Vocals"],
                  ["trebleReactivity", "Treble"],
                ] as const
              ).map(([key, label]) => (
                <SettingRow
                  key={key}
                  label={
                    <>
                      <b>{label}</b>
                      <output className="font-normal tabular-nums">
                        {Math.round(playback[key] * 100)}%
                      </output>
                    </>
                  }
                  htmlFor={`settings-${key}`}
                >
                  <RangeControl
                    id={`settings-${key}`}
                    disabled={!playback.wavesEnabled}
                    min="0"
                    max="2"
                    step="0.05"
                    value={playback[key]}
                    onChange={(event) =>
                      savePlayback({ ...playback, [key]: Number(event.target.value) })
                    }
                  />
                </SettingRow>
              ))}
            </div>
          </>
        )}
      </section>
    </section>
  );
}
