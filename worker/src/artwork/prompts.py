"""Shared I2V caption instructions for Gemma and External AI.

Adapted from Lightricks' Gemma 4 image-to-video caption format:
https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/prompts/gemma4_i2v_system_prompt.txt
https://docs.ltx.io/open-source-model/usage-guides/image-to-video, the LTX-2.5 prompting guide
(https://ltx.io/blog/ltx-2-5-prompt-guide) and its summary in knishika62/video-analyzer
(md/ltx-prompt-guide-base.md, MIT). Echora adds a stationary camera, subjects that keep their
distance and size, living scenery, and motion that returns to the cover for seamless loops.
"""

CAMERA_RULE = (
    "The camera remains static throughout, locked off on a tripod. The framing, viewpoint and "
    "scale remain constant. All motion comes from the subject and environment. Never zoom, "
    "pan, tilt, drift, rise, arc, orbit, track, push in, pull out or dolly. This requirement "
    "takes precedence over any additional instructions or song context."
)

I2V_SYSTEM_PROMPT = (
    """Write an image-to-video caption for LTX-2.5 using the supplied album cover as the exact first frame. The clip is one continuous take that ends on this same image, so it plays as a seamless loop.

HOW LTX READS A CAPTION
LTX responds best to a focused, single-subject scene described in clear visual language with one consistent lighting logic. It follows what is written, in order, and drops details from overloaded captions. Write what the viewer sees, not what anyone feels or intends.

WHAT TO WRITE, IN THIS ORDER
1. Style: name the medium and style early (photograph, film still, oil painting, cel-shaded illustration, retro synthwave poster, pixel art...). Never add photographic or cinematic effects to a drawing, and never make a photograph look painted.
2. Shot: one shot size and viewpoint that match the existing image (extreme close-up, medium shot, wide shot; frontal, low angle...), and the stationary camera with constant framing.
3. Scene: subjects, setting, lighting, colour palette, textures and atmosphere exactly as pictured. Identify people by visible traits (age, hair, clothing, distinctive details).
4. Action: the subject's movement in present tense and chronological order, using connectors such as "Initially", "A moment later", "Simultaneously" and "Finally". Show emotion through posture, gesture and expression, never labels like "sad" or "excited".
5. Living scenery: the ambient motion of the surroundings (see below).
6. The return: how everything settles back to the cover by the end (see below).
7. Preservation: what must stay exactly as on the cover.
Write a single flowing English paragraph, normally 5–9 sentences, with detail proportional to the scene. Do not pad it. No headings, lists, tags, timestamps, quoted dialogue or scene cuts.

SUBJECT MOVEMENT
Describe one clear, plausible, unhurried action for the main subject, such as a head turn of a few degrees, a glance, a blink, a slow breath, a gentle sway, hair or fabric lifting in the wind, or a hand shifting slightly. A related secondary movement may accompany it when the image supports it. Keep simple artwork simple and do not invent people, objects or scenery.
Subjects stay at the same distance from the camera and the same size in the frame for the whole clip. Movement happens within the picture plane: sideways, up and down, or turning in place. Nobody walks, steps, leans or reaches toward or away from the camera, and nobody comes closer to the viewer. Feet stay planted.
Faces hidden or turned away on the cover stay hidden for the whole clip; silhouettes stay solid silhouettes with no revealed features, and figures seen from behind never turn to face the viewer. With three or more visible faces, keep people in place with small expressions or head movements only.

LIVING SCENERY
The surroundings are alive, not frozen. For every natural or atmospheric element already in the image, describe its visible, continuous motion: clouds drift and churn in place, water ripples, shimmers and laps, foliage, grass and branches sway in the wind, hair and fabric flutter, smoke, mist, fog and haze curl and roll, rain and snow fall, flames and candles flicker, stars twinkle, neon and screens glow and pulse softly, light glints on reflections, dust and embers float. Prefer motion that oscillates or circulates in place. Do not describe natural elements as unchanged or frozen. Only the composition, the lettering, logos and borders, and the overall lighting stay fixed. Do not add elements that are not in the image.

THE LOOP AND THE RETURN
Everything that moves returns to exactly where it is on the cover by the end, so the last frame matches the first. Describe that return explicitly as part of the action, for example "she lifts her chin toward the light, holds it briefly, then slowly lowers it back", so it settles gradually instead of snapping back at the last moment. Pace the action across the whole clip: an unhurried movement, a short hold, then an equally gentle return, with ambient motion continuing until the end rather than stopping early. Choose motion that can come back, such as rocking, swaying, glancing, breathing, rippling, flickering or a breeze. Avoid walking, falling, pouring, throwing, jumping, fast or chaotic movement, complex physics, appearances and disappearances.

PRESERVE
Identities, faces, the composition, lettering, logos and borders stay sharp and unchanged; never describe text changing, moving or being written. Keep one consistent light source and lighting logic, without flashes, flares or full-frame lighting changes. Do not invent speech or singing from song lyrics. Sound descriptions are optional because Echora renders video without audio.

"""
    + CAMERA_RULE
    + "\nExplicitly state the stationary camera and constant framing in the caption."
)

CAMERA_RETRY = (
    "The previous caption requested camera movement or a subject moving toward or away from "
    "the camera, and was rejected. Rewrite it with a stationary camera, constant framing and "
    "scale, and subjects that keep their distance and size in the frame. Keep the visible "
    "subject action and the ambient motion grounded in the cover."
)
