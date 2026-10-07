"""LTX-2.5 motion artwork graph and video post-processing.

The cover is pinned as both the first frame (LTXVImgToVideoInplace) and the last (LTXVAddGuide), so
the clip starts and ends on the cover and loops seamlessly. Lightricks' static-camera LoRA holds the
framing. Sampling follows Lightricks' distilled workflow: LTXVPreprocess on the cover, euler_ancestral
and the distilled sigma list. With upscaling, that render is stage one of their two-stage pipeline: the
latent is doubled with the LTX-2.5 spatial upscaler and refined at the larger size in three steps.
In "auto" mode Gemma-4 E2B writes the prompt from the cover in a separate request, using the LTX
caption-style system prompt plus Echora's instructions; Echora tidies it before rendering. Prompts
can be encoded ahead of time (build_encode_graph) so rendering never loads the text encoder.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .motion_artwork_prompts import CAMERA_RULE, I2V_SYSTEM_PROMPT

LTX_REPOSITORY = "Lightricks/LTX-2.5"
LTX_REVISION = "5e6e71018ee1756ed329b697a7b4aedc934dfce9"
GEMMA_REPOSITORY = "Comfy-Org/gemma-4"
GEMMA_REVISION = "63d0f7c476756b88910170c1df75e2384ea1af31"
TRANSFORMER = "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
TEXT_ENCODER = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
VAE = "ltx-2.5-video-vae-conv-bf16.safetensors"
UPSCALER = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
# Trained on LTX-2 19B; its layers match LTX-2.5 22B and it loads without missing keys.
CAMERA_LORA_REPOSITORY = "Lightricks/LTX-2-19b-LoRA-Camera-Control-Static"
CAMERA_LORA_REVISION = "ba623881aa5d59559a8b26c41beb4b28253b8cae"
CAMERA_LORA = "ltx-2-19b-lora-camera-control-static.safetensors"
# The default "camera lock". Full strength holds the camera but damps the subject's motion too.
CAMERA_LORA_STRENGTH = 0.5
PROMPT_MODEL = "gemma4_e2b_it_int8_convrot.safetensors"
# (repository, revision, path in repository); ComfyUI resolves them by file name.
MODEL_FILES = (
    (LTX_REPOSITORY, LTX_REVISION, f"diffusion_models/{TRANSFORMER}"),
    (LTX_REPOSITORY, LTX_REVISION, f"text_encoders/{TEXT_ENCODER}"),
    (LTX_REPOSITORY, LTX_REVISION, f"vae/{VAE}"),
    (LTX_REPOSITORY, LTX_REVISION, f"latent_upscale_models/{UPSCALER}"),
    (CAMERA_LORA_REPOSITORY, CAMERA_LORA_REVISION, CAMERA_LORA),
    (GEMMA_REPOSITORY, GEMMA_REVISION, f"text_encoders/{PROMPT_MODEL}"),
)
# (ComfyUI loader node, input name, file) for availability checks.
REQUIRED_LOADERS = (
    ("UNETLoader", "unet_name", TRANSFORMER),
    ("CLIPLoader", "clip_name", TEXT_ENCODER),
    ("VAELoader", "vae_name", VAE),
    ("LatentUpscaleModelLoader", "model_name", UPSCALER),
    ("LoraLoaderModelOnly", "lora_name", CAMERA_LORA),
    ("CLIPLoader", "clip_name", PROMPT_MODEL),
)

FPS = 24
# The clip starts and ends on the cover; the saved loop drops the closing copy of it.
LOOP = "pinned"
# Bump when either generated-prompt strategy changes, so covers get new loops.
WRITER_REVISION = "ltx-i2v-7-return"
DISTILLED_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
# Lightricks' distilled second stage: the upscaled latent is refined from partway through.
REFINE_SIGMAS = "0.909375, 0.725, 0.421875, 0.0"
# cfg=1 ignores the negative branch, but the conditioning nodes still require one.
NEGATIVE = "static, blurry, low quality, jpeg artifacts, deformed, distorted face, warped text"
DEFAULT_INSTRUCTIONS = (
    "Animate the pictured subject with one clear, natural movement paced across the whole clip, "
    "then a gentle, described return to the pose on the cover by the end. "
    "Include subtle ambient motion in the surroundings only where the artwork supports it. "
    "Preserve the original art style, composition, identities and lettering. "
) + CAMERA_RULE
# Zooms crop the cover and warp it as it enlarges, so prompts that ask for one are written again.
ZOOM_WORDS = re.compile(
    r"\b(zoom\w*|push(?:es|ed|ing)?[- ]?in|pull(?:s|ed|ing)?[- ]?(?:out|back)|dolly\w*|dollies|"
    r"moves? (?:closer|toward|towards|in on))\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Recipe:
    # The size the model renders at; upscaling doubles it.
    resolution: int = 768
    frames: int = 241
    upscale: bool = True
    camera_lock: float = CAMERA_LORA_STRENGTH
    prompt_mode: str = "auto"
    instructions: str = DEFAULT_INSTRUCTIONS
    fixed_prompt: str = ""
    seed: int | None = 42

    @property
    def output_resolution(self) -> int:
        return self.resolution * 2 if self.upscale else self.resolution

    def as_dict(self) -> dict:
        prompt_model = {
            "auto": f"{GEMMA_REPOSITORY}@{GEMMA_REVISION}:{WRITER_REVISION}",
            "external": f"external:{WRITER_REVISION}",
        }
        return {
            "model": f"{LTX_REPOSITORY}@{LTX_REVISION}",
            "prompt_model": prompt_model.get(self.prompt_mode),
            "resolution": self.resolution,
            "output_resolution": self.output_resolution,
            "frames": self.frames,
            "upscale": f"{LTX_REPOSITORY}@{LTX_REVISION}:{UPSCALER}" if self.upscale else None,
            "camera_lora": f"{CAMERA_LORA_REPOSITORY}@{CAMERA_LORA_REVISION}:{self.camera_lock}",
            "fps": FPS,
            "prompt_mode": self.prompt_mode,
            "instructions": self.instructions if self.prompt_mode != "fixed" else None,
            "fixed_prompt": self.fixed_prompt if self.prompt_mode == "fixed" else None,
            "seed": self.seed,
            "loop": LOOP,
        }

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True).encode()).hexdigest()[:32]


def build_prompt_graph(recipe: Recipe, image_name: str, seed: int) -> dict:
    """Use the pinned node's Gemma 4 chat format with our stationary-camera I2V template.

    TextGenerateLTX2Prompt in the pinned ComfyUI does not support a system override.
    TextGenerate with skip-template accepts the same formatted chat and image tokens.
    """
    caption_request = (
        f"<|turn>system\n{I2V_SYSTEM_PROMPT}\nOutput only the caption text.<turn|>\n"
        f"<|turn>user\n<|image><|image|><image|>\n\n{recipe.instructions}\n\n{CAMERA_RULE}<turn|>\n"
        "<|turn>model\n<|channel>final\n"
    )
    return {
        "1_cover": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "30_prompt_model": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": PROMPT_MODEL, "type": "ltxv", "device": "default"},
        },
        "31_write_prompt": {
            "class_type": "TextGenerate",
            "inputs": {
                "clip": ["30_prompt_model", 0],
                "image": ["1_cover", 0],
                "prompt": caption_request,
                "max_length": 600,
                "sampling_mode": "on",
                "sampling_mode.temperature": 0.3,
                "sampling_mode.top_k": 64,
                "sampling_mode.top_p": 0.95,
                "sampling_mode.min_p": 0.05,
                "sampling_mode.repetition_penalty": 1.15,
                "sampling_mode.seed": seed,
                "thinking": False,
                "use_default_template": False,
            },
        },
        "32_show_prompt": {
            "class_type": "PreviewAny",
            "inputs": {"source": ["31_write_prompt", 0]},
        },
    }


# Any described camera movement: LTX tends to turn even a sideways drift into a push-in.
CAMERA_MOVES = re.compile(
    r"\bcamera\b[^.;:]{0,40}?\b(pans?|panning|drifts?|drifting|rises?|rising|arcs?|arcing|tilts?|tilting|"
    r"tracks?|tracking|glides?|gliding|sweeps?|orbits?|orbiting|slides?|sliding|cranes?|moves? (?:slowly|gently|"
    r"sideways|left|right|up|down|forward|back)|moves?|moving|advances?|approaches?|recedes?)\b",
    re.IGNORECASE,
)


NEGATION = re.compile(r"\b(no|not|never|without|nor|neither)\b|n't\b", re.IGNORECASE)
# A negative in one clause must not excuse a later, affirmative camera move.
CLAUSE_BREAK = re.compile(
    r"[.;:]|\b(?:but|then|however|while|whereas)\b|"
    r"\band\s+(?=(?:it|(?:the\s+)?camera)\b)|,\s*(?=(?:it|(?:the\s+)?camera)\b)",
    re.IGNORECASE,
)
# What may sit between a clause break and a camera move whose subject is left out
# ("the camera holds, then gently pans"). Any other word starts a new subject
# ("the camera is locked off while leaves move"), whose movement is not the camera's.
SAME_SUBJECT = re.compile(
    r"^(?:[\s,]|\b(?:it|the|camera|then|and|also|now|slowly|gently|softly|subtly|smoothly|"
    r"gradually|briefly|begins?|starts?|to)\b)*$",
    re.IGNORECASE,
)
FRAME_CHANGES = re.compile(
    r"\b(?:framing|frame|view)\b[^.;:]{0,30}?\b(?:tightens?|widens?|zooms?|enlarges?)\b",
    re.IGNORECASE,
)


def zooms(prompt: str) -> bool:
    """Whether a prompt asks for a zoom-like or any other camera move ("no pan", "never zooms" do not count)."""
    for pattern in (ZOOM_WORDS, CAMERA_MOVES, FRAME_CHANGES):
        for match in pattern.finditer(prompt):
            # Look at the clause before the match too, for "the camera does not zoom" or "no push-in".
            # Camera patterns may start before a clause boundary ("camera does not pan,
            # then zooms"). Evaluate the actual movement word, not the whole sentence.
            movement = match.start(1) if pattern is CAMERA_MOVES else match.start()
            preceding = prompt[:movement]
            breaks = list(CLAUSE_BREAK.finditer(preceding))
            start = breaks[-1].end() if breaks else 0
            if (
                pattern is CAMERA_MOVES
                and start > match.start()
                and not SAME_SUBJECT.match(prompt[start:movement])
            ):
                continue
            if not NEGATION.search(prompt[start : match.end()]):
                return True
    return False


def clean_prompt(text: str, limit: int = 2400) -> str:
    """Tidy a generated caption: collapse whitespace and cut it where the model starts over.

    The small prompt model sometimes repeats its whole caption; LTX should receive it once.
    """
    text = " ".join(str(text).split())
    head = text[:60]
    if len(head) == 60:
        again = text.find(head, 60)
        if again > 0:
            text = text[:again].rstrip()
    if len(text) > limit:
        cut = text.rfind(". ", 0, limit)
        text = text[: cut + 1] if cut > 0 else text[:limit]
    return text


def build_encode_graph(text: str, prefix: str) -> dict:
    """Encode one prompt with the LTX text encoder and save it, so rendering can skip the encoder.

    SaveConditioning writes output/conditioning/<prefix>_00001_.safetensors (about 0.5 MB).
    """
    return {
        "3_text_encoder": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": TEXT_ENCODER, "type": "ltxv", "device": "default"},
        },
        "5_encode": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["3_text_encoder", 0], "text": text},
        },
        "6_save": {
            "class_type": "SaveConditioning",
            "inputs": {
                "conditioning": ["5_encode", 0],
                "filename_prefix": f"conditioning/{prefix}",
            },
        },
    }


def encoding_file(prefix: str) -> str:
    """The file SaveConditioning writes for a prefix in a fresh output directory."""
    return f"{prefix}_00001_.safetensors"


def build_graph(
    recipe: Recipe,
    image_name: str,
    seed: int,
    prefix: str,
    prompt: str | None = None,
    encodings: tuple[str, str] | None = None,
) -> dict:
    """The video graph for one cover, from prompt text or from saved (positive, negative) encodings."""
    if (prompt is None) == (encodings is None):
        raise ValueError("Give either a prompt or saved encodings")
    graph: dict[str, dict] = {
        "1_cover": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2_model": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": TRANSFORMER, "weight_dtype": "default"},
        },
        "2b_camera_lora": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["2_model", 0],
                "lora_name": CAMERA_LORA,
                "strength_model": recipe.camera_lock,
            },
        },
        "4_vae": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "7_conditioning": {
            "class_type": "LTXVConditioning",
            "inputs": {
                "positive": ["5_positive", 0],
                "negative": ["6_negative", 0],
                "frame_rate": float(FPS),
            },
        },
        "8_latent": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {
                "width": recipe.resolution,
                "height": recipe.resolution,
                "length": recipe.frames,
                "batch_size": 1,
            },
        },
        "9_preprocess": {
            "class_type": "LTXVPreprocess",
            "inputs": {"image": ["1_cover", 0], "img_compression": 18},
        },
        "10_first_frame": {
            "class_type": "LTXVImgToVideoInplace",
            "inputs": {
                "vae": ["4_vae", 0],
                "image": ["9_preprocess", 0],
                "latent": ["8_latent", 0],
                "strength": 1.0,
                "bypass": False,
            },
        },
    }
    if encodings:
        graph["5_positive"] = {
            "class_type": "ConditioningLoader",
            "inputs": {"conditioning_name": encodings[0]},
        }
        graph["6_negative"] = {
            "class_type": "ConditioningLoader",
            "inputs": {"conditioning_name": encodings[1]},
        }
    else:
        graph["3_text_encoder"] = {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": TEXT_ENCODER, "type": "ltxv", "device": "default"},
        }
        graph["5_positive"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["3_text_encoder", 0], "text": prompt},
        }
        graph["6_negative"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["3_text_encoder", 0], "text": NEGATIVE},
        }
    # The cover is also the last frame, so the clip returns to it and loops seamlessly.
    graph["12_end_frame"] = {
        "class_type": "LTXVAddGuide",
        "inputs": {
            "positive": ["7_conditioning", 0],
            "negative": ["7_conditioning", 1],
            "vae": ["4_vae", 0],
            "latent": ["10_first_frame", 0],
            "image": ["1_cover", 0],
            "frame_idx": -1,
            "strength": 1.0,
        },
    }
    positive, negative, latent = ["12_end_frame", 0], ["12_end_frame", 1], ["12_end_frame", 2]
    graph.update(
        {
            "13_noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
            "14_guider": {
                "class_type": "CFGGuider",
                "inputs": {
                    "model": ["2b_camera_lora", 0],
                    "positive": positive,
                    "negative": negative,
                    "cfg": 1.0,
                },
            },
            "15_sampler": {
                "class_type": "KSamplerSelect",
                "inputs": {"sampler_name": "euler_ancestral"},
            },
            "16_sigmas": {"class_type": "ManualSigmas", "inputs": {"sigmas": DISTILLED_SIGMAS}},
            "17_sample": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "noise": ["13_noise", 0],
                    "guider": ["14_guider", 0],
                    "sampler": ["15_sampler", 0],
                    "sigmas": ["16_sigmas", 0],
                    "latent_image": latent,
                },
            },
            "18_crop_guides": {
                "class_type": "LTXVCropGuides",
                "inputs": {"positive": positive, "negative": negative, "latent": ["17_sample", 0]},
            },
        }
    )
    final = "18_crop_guides"
    if recipe.upscale:
        # Double the latent, pin the cover to both ends again at the new size, and refine.
        graph.update(
            {
                "22_upscale_model": {
                    "class_type": "LatentUpscaleModelLoader",
                    "inputs": {"model_name": UPSCALER},
                },
                "23_upscale": {
                    "class_type": "LTXVLatentUpsampler",
                    "inputs": {
                        "samples": ["18_crop_guides", 2],
                        "upscale_model": ["22_upscale_model", 0],
                        "vae": ["4_vae", 0],
                    },
                },
                "24_first_frame": {
                    "class_type": "LTXVImgToVideoInplace",
                    "inputs": {
                        "vae": ["4_vae", 0],
                        "image": ["9_preprocess", 0],
                        "latent": ["23_upscale", 0],
                        "strength": 1.0,
                        "bypass": False,
                    },
                },
                "25_end_frame": {
                    "class_type": "LTXVAddGuide",
                    "inputs": {
                        "positive": ["18_crop_guides", 0],
                        "negative": ["18_crop_guides", 1],
                        "vae": ["4_vae", 0],
                        "latent": ["24_first_frame", 0],
                        "image": ["1_cover", 0],
                        "frame_idx": -1,
                        "strength": 1.0,
                    },
                },
                "26_noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed + 1}},
                "27_guider": {
                    "class_type": "CFGGuider",
                    "inputs": {
                        "model": ["2b_camera_lora", 0],
                        "positive": ["25_end_frame", 0],
                        "negative": ["25_end_frame", 1],
                        "cfg": 1.0,
                    },
                },
                "28_sigmas": {"class_type": "ManualSigmas", "inputs": {"sigmas": REFINE_SIGMAS}},
                "29_refine": {
                    "class_type": "SamplerCustomAdvanced",
                    "inputs": {
                        "noise": ["26_noise", 0],
                        "guider": ["27_guider", 0],
                        "sampler": ["15_sampler", 0],
                        "sigmas": ["28_sigmas", 0],
                        "latent_image": ["25_end_frame", 2],
                    },
                },
                "30_crop_guides": {
                    "class_type": "LTXVCropGuides",
                    "inputs": {
                        "positive": ["25_end_frame", 0],
                        "negative": ["25_end_frame", 1],
                        "latent": ["29_refine", 0],
                    },
                },
            }
        )
        final = "30_crop_guides"
    graph.update(
        {
            "19_decode": {
                "class_type": "VAEDecode",
                "inputs": {"samples": [final, 2], "vae": ["4_vae", 0]},
            },
            "20_video": {
                "class_type": "CreateVideo",
                "inputs": {"images": ["19_decode", 0], "fps": FPS},
            },
            "21_save": {
                "class_type": "SaveVideo",
                "inputs": {
                    "video": ["20_video", 0],
                    "filename_prefix": f"echora-motion-artwork/{prefix}",
                    "format": "mp4",
                },
            },
        }
    )
    return graph


def _probe(path: Path) -> tuple[int, int, int]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    return int(stream["width"]), int(stream["height"]), int(stream["nb_read_frames"])


def finish(source: Path, target: Path) -> tuple[int, int, int]:
    """Write the final H.264 loop.

    The last frame is the cover again (pinned), so it is dropped and the loop wraps from the
    second-to-last frame straight back to the first without showing the cover twice.
    """
    width, height, count = _probe(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part.mp4")
    frames = count - 1 if count > 1 else count
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(source),
            "-frames:v",
            str(frames),
            "-r",
            str(FPS),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(partial),
        ]
    )
    if result.returncode:
        partial.unlink(missing_ok=True)
        raise RuntimeError("Could not encode the motion artwork video")
    partial.replace(target)
    return width, height, frames
