# Backdrop journey

Every backdrop moves through the song as a journey instead of only following the current loudness. The song's sections are known before playback starts, so a backdrop can build up before a chorus arrives, land with it, and return to the same place when the chorus comes back.

## Sections

`planSongSections` in `web/components/player/visualFeatures.ts` plans the sections once per track from the cached visual features (`structure`, revision 2). It needs no new analysis.

1. **Boundaries.** Checkerboard novelty on the average of the chroma and MFCC self-similarity matrices (2 s bins, 4-bin kernel). Local peaks above a quarter of the strongest, at least 12 s apart, at most 20.
2. **Scenes.** Each section is compared with every other. When its best match among earlier sections is at least one standard deviation above the song's own between-section similarity, it takes that section's scene. A returning chorus therefore returns to the chorus scene.
3. **Energy.** The mean level of each section, half normalised to the song's range and half by rank, so mastered pop with a narrow loudness range still separates verse, pre-chorus and chorus.

Only the bounded plan is kept on the timeline; the matrices never enter the animation loop. Every visual frame carries its `section` (index, scene, energy, next section's energy, start and end).

## Live state

`SongJourney` in `web/components/player/songJourney.ts` turns frames into a state each backdrop reads once per drawn frame:

| Field                                  | Meaning                                                                                           |
| -------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `travel`                               | Distance travelled. Tempo sets the pace, `drive` the push.                                        |
| `drive`                                | How hard the music pushes: section energy, loudness relative to the song, build-up and impact.    |
| `anticipation`                         | Rises over the last bars (16 beats, 3–9 s) before a section with clearly more energy.             |
| `impact`                               | Lands when that section starts, then decays over about a second. Seeks never trigger it.          |
| `scene`, `previousScene`, `sceneBlend` | The current scene and a 2.5 s blend from the previous one.                                        |
| `beatPhase`, `beatPulse`               | Position between detected beats and a short pulse on each beat.                                   |
| `harmonyShift`                         | Hue offset (±0.5 rad) as the harmony moves away from the song's home key on the circle of fifths. |

`color(palette, slot)` picks a colour for the current scene from the artwork palette (scenes rotate through it), turns it by the harmony shift in OKLab, and lifts chroma and lightness with the drive and impact.

## Presets

| Preset                              | Journey                                                                                                                                                                                                                                                                   |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Void tunnel                         | Flight speed follows the drive. Each scene has its own cross-section (circle, hexagon, petals, octagon…) and twist. Build-ups stretch the rings and draw warp streaks; impacts send a shock ring outward; beats flash the nearest rings. A far light grows over the song. |
| Oscilloscope                        | The trace takes the scene colours and leaves echo traces when a section lands.                                                                                                                                                                                            |
| Mesh grid                           | An endless flight over terrain toward a horizon glow. Each scene has its own landscape. The camera climbs through build-ups and a shockwave rolls across the land on impact.                                                                                              |
| Clouds                              | Forward flight through the banks. The sky behind them takes the scene colour; build-ups charge light inside the clouds and impacts strike lightning.                                                                                                                      |
| Lightning fall                      | Each scene uses three of the artwork's colours. Build-ups pinch the streams together; impacts throw them wide with bright heads.                                                                                                                                          |
| Roots                               | Each new scene grows a new network in its colours while the previous one fades; impacts sprout fresh branches everywhere.                                                                                                                                                 |
| Water drops                         | The drive thickens the rain and a build-up quickens it; an impact drops a stone in the middle of the pool.                                                                                                                                                                |
| Waves, digital curtain, ASCII dance | Scene colours, wave height and flow speed, particle count and brightness follow the journey; impacts flash the curtain and the ASCII grid.                                                                                                                                |

Reactivity preferences, animation speed and frame-rate caps still apply. Songs without structure data play as a single section.
