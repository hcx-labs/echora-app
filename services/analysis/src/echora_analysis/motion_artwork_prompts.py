"""Shared I2V caption instructions for Gemma and External AI.

Adapted from Lightricks' Gemma 4 image-to-video caption format:
https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/prompts/gemma4_i2v_system_prompt.txt
and https://docs.ltx.io/open-source-model/usage-guides/image-to-video.
Echora adds a stationary camera and motion that returns to the cover for album-cover playback.
"""

CAMERA_RULE = (
    "The camera remains static throughout, locked off on a tripod. The framing, viewpoint and "
    "scale remain constant. All motion comes from the subject and environment. Never zoom, "
    "pan, tilt, drift, rise, arc, orbit, track, push in, pull out or dolly. This requirement "
    "takes precedence over any additional instructions or song context."
)

I2V_SYSTEM_PROMPT = (
    """Write an image-to-video caption for LTX-2.5 using the supplied album cover as the exact first frame.

Begin directly with an observable action or visual detail grounded in the image. Faithfully establish the pictured subjects, setting, composition, lighting and art medium, then describe what moves. Include one shot type and viewpoint matching the existing image, woven naturally into the prose. Preserve the artwork's style rather than adding photographic or cinematic effects to a drawing.

Describe one clear, plausible subject action in present tense and chronological order. A related action or subtle ambient movement may accompany it when supported by the image; do not force a second event or invent scenery, objects, particles or people. Keep simple artwork simple. Describe visible behavior rather than inferred feelings or intentions.

Write a single flowing English paragraph, normally 4–8 sentences, with detail proportional to the scene. Do not pad it to a word count. Use natural temporal connectors when useful. No headings, tags, timestamps or scene cuts.

The clip ends on this same image to make a seamless loop, so everything that moves returns to where it is on the cover by the end. Describe that return explicitly as part of the action, for example "she lifts her chin toward the light, holds it briefly, then slowly lowers it back", so it settles gradually instead of snapping back at the last moment. Pace the action across the whole clip: an unhurried movement, a short hold, then an equally gentle return, with motion continuing until the end rather than finishing early. Choose motion that can come back, such as rocking, swaying, glancing, ripples or a breeze. Avoid walking, falling, pouring, throwing, appearances and disappearances.

Preserve identities, composition, lettering, logos and borders. Faces hidden on the cover remain hidden for the whole clip. With three or more visible faces, keep people in place with small expressions or head movements. Keep lighting consistent. Do not invent speech or singing from song lyrics. Sound descriptions are optional because Echora renders video without audio.

"""
    + CAMERA_RULE
    + "\nExplicitly state the stationary camera and constant framing in the caption."
)

CAMERA_RETRY = (
    "The previous caption requested camera movement and was rejected. Rewrite it with a "
    "stationary camera and constant framing and scale. Remove every camera movement; keep "
    "the visible subject action grounded in the cover."
)
