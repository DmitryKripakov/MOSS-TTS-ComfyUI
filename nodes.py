"""ComfyUI node definitions for MOSS-TTS v1.5 (Local-Transformer 4B + Delay 8B).

V1 node protocol (NODE_CLASS_MAPPINGS / INPUT_TYPES / ...): the stable,
AST-parseable surface — exactly what the ComfyUI registry crawler expects.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from typing import Any

import torch

from . import compat, director, runtime, selftest
from . import lora as adapters
from .compat import ProgressReporter
from .loader import (
    ATTENTION_OPTIONS,
    DTYPE_OPTIONS,
    bundle_generation,
    list_custom_models,
    load_mosstts_bundle,
)
from .native import DEFAULT_VARIANT, VARIANTS

logger = logging.getLogger("ComfyUI-MOSS-TTS-v15")

MODEL_TYPE = "MOSSTTS_V15_MODEL"

_VARIANT_LABELS = [VARIANTS[k].label for k in VARIANTS]
_LABEL_TO_KEY = {VARIANTS[k].label: k for k in VARIANTS}

_FRAMES_TOOLTIP = (
    "Duration hint in audio frames (12.5 frames/s): 125 ≈ 10 s, 375 ≈ 30 s. "
    "0 = model decides via EOS. Wire the Estimate Tokens node to compute it."
)

_AUDIO_KNOBS = {
    "audio_temperature": ("FLOAT", {"default": 1.7, "min": 0.0, "max": 3.0, "step": 0.05,
                                    "tooltip": "Acoustic sampling temperature (MOSS default 1.7)."}),
    "audio_top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01,
                              "tooltip": "Acoustic nucleus sampling."}),
    "audio_top_k": ("INT", {"default": 25, "min": 0, "max": 1024, "step": 1,
                            "tooltip": "Acoustic top-k."}),
    "audio_repetition_penalty": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.01,
                                           "tooltip": "1.0 = off. Mild values (1.05-1.15) suppress droning / tempo freeze."}),
}

_TEXT_KNOBS = {
    "text_temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                   "tooltip": "Text-stream (alignment/pacing) temperature."}),
    "text_top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                             "tooltip": "Text-stream nucleus sampling."}),
    "text_top_k": ("INT", {"default": 50, "min": 0, "max": 500, "step": 1,
                           "tooltip": "Text-stream top-k."}),
}

_BUDGET_KNOBS = {
    "target_tokens": ("INT", {"default": 0, "min": 0, "max": 45000, "step": 1, "tooltip": _FRAMES_TOOLTIP}),
    "max_new_tokens": ("INT", {"default": 4096, "min": 16, "max": 45000, "step": 1,
                               "tooltip": "Hard generation budget in frames (12.5 fps): 4096 ≈ 5.5 min cap."}),
    "do_sample": ("BOOLEAN", {"default": True,
                              "tooltip": "Stochastic sampling; off = greedy decode (delay variant maps this to temperature=0)."}),
    "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1,
                     "tooltip": "Same seed + same inputs → identical output."}),
}

_CONDITIONING = {
    "language": (runtime.LANGUAGES, {"default": "auto",
                                     "tooltip": "Language hint. v1.5 performs best when it is set explicitly."}),
    "instruction": ("STRING", {"multiline": True, "default": "",
                               "tooltip": "Free-form style instruction, e.g. 'male, warm, elderly narrator'."}),
}


def _all_knobs() -> dict:
    merged: dict = {}
    merged.update(_CONDITIONING)
    merged.update(_AUDIO_KNOBS)
    merged.update(_TEXT_KNOBS)
    merged.update(_BUDGET_KNOBS)
    return merged


_PRESET_WIDGETS = "widgets"
_PRESET_MODEL = "model recommended"
_OPTIONAL_PRESET = {
    "sampling_preset": ([_PRESET_WIDGETS, _PRESET_MODEL], {
        "default": _PRESET_WIDGETS,
        "tooltip": "widgets = the audio_* values above. model recommended = the audio "
                   "temperature / top_p / top_k / repetition penalty from the loaded model's "
                   "card (e.g. LAION sft3: 1.0 / 0.95 / 50 / 1.0); the audio_* widgets are "
                   "then ignored. Custom folders have no card values — widgets are used.",
    }),
}


def _audio_sampling(mosstts_model, preset: str, with_reference: bool, temperature: float,
                    top_p: float, top_k: int, repetition_penalty: float) -> dict:
    """The audio_* knobs to use: the widgets, or the model card's recommendation."""
    values = (temperature, top_p, top_k, repetition_penalty)
    if preset == _PRESET_MODEL:
        spec = mosstts_model.spec
        card = (getattr(spec, "sampling_with_reference", None) if with_reference else None) \
            or getattr(spec, "sampling", None)
        if card:
            values = card
            logger.info("[MOSS-TTS] model recommended audio sampling for %s: "
                        "temperature=%s top_p=%s top_k=%s repetition_penalty=%s",
                        spec.key, *values)
        else:
            logger.info("[MOSS-TTS] %s has no recommended sampling — using the widgets.", spec.key)
    temperature, top_p, top_k, repetition_penalty = values
    return {"audio_temperature": float(temperature), "audio_top_p": float(top_p),
            "audio_top_k": int(top_k), "audio_repetition_penalty": float(repetition_penalty)}


_CUSTOM_NONE = "none"


class MossTTSV15LoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (_VARIANT_LABELS, {"default": VARIANTS[DEFAULT_VARIANT].label,
                                            "tooltip": "Local-Transformer (4B, 48kHz stereo, Qwen3-4B backbone) or Delay (8B, 24kHz) or VoiceGenerator (1.7B, 24kHz, voice design)."}),
                "dtype": (DTYPE_OPTIONS, {"default": "auto", "tooltip": "auto = bf16 on CUDA, fp32 on CPU."}),
                "attention": (ATTENTION_OPTIONS, {"default": "auto",
                                                  "tooltip": "auto = flash_attention_2 when the flash_attn package exists, else sdpa."}),
                "download_if_missing": ("BOOLEAN", {"default": True,
                                                    "tooltip": "Resolve models from $MOSS_TTS_MODELS_DIR / HF cache / models/mosstts first; download only when missing."}),
            },
            "optional": {
                "custom_model": ([_CUSTOM_NONE] + sorted(list_custom_models()), {
                    "default": _CUSTOM_NONE,
                    "tooltip": "A fine-tune in its own folder under models/mosstts (or "
                               "$MOSS_TTS_MODELS_DIR), up to two levels deep — e.g. your own "
                               "model or a training checkpoint. Overrides `model`; the family "
                               "(Local / Delay) is read from its config.json. Refresh the page "
                               "after adding a folder.",
                }),
            },
        }

    RETURN_TYPES = (MODEL_TYPE,)
    RETURN_NAMES = ("mosstts_model",)
    FUNCTION = "load_model"
    CATEGORY = "MOSS-TTS v1.5"
    DESCRIPTION = "Load a MOSS-TTS v1.5 variant (weights + codec), integrated with ComfyUI memory management."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return bundle_generation()

    def load_model(self, model: str, dtype: str, attention: str, download_if_missing: bool,
                   custom_model: str = _CUSTOM_NONE):
        custom_dir = None
        if custom_model and custom_model != _CUSTOM_NONE:
            folders = list_custom_models()
            if custom_model not in folders:
                raise FileNotFoundError(
                    f"custom_model {custom_model!r} is no longer in models/mosstts "
                    f"(found: {sorted(folders) or 'nothing'})."
                )
            custom_dir = str(folders[custom_model])
        bundle = load_mosstts_bundle(
            variant=_LABEL_TO_KEY[model],
            dtype_name=dtype,
            attention=attention,
            download_if_missing=bool(download_if_missing),
            custom_dir=custom_dir,
        )
        return (bundle,)


class MossTTSV15GenerateSpeech:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "mosstts_model": (MODEL_TYPE, {}),
            "text": ("STRING", {"multiline": True, "default": "Hello! This is MOSS-TTS v1.5 running inside ComfyUI.",
                                "tooltip": "Text to synthesize. Supports [pause 3.2s] markers, Pinyin and IPA (delay variant)."}),
        }
        required.update(_all_knobs())
        return {"required": required, "optional": dict(_OPTIONAL_PRESET)}

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("audio", "tokens_generated")
    FUNCTION = "generate"
    CATEGORY = "MOSS-TTS v1.5"
    DESCRIPTION = "Reference-free text-to-speech; the voice is steered by language + instruction."

    def generate(self, mosstts_model, text, language, instruction, audio_temperature,
                 audio_top_p, audio_top_k, audio_repetition_penalty, text_temperature,
                 text_top_p, text_top_k, target_tokens, max_new_tokens, do_sample, seed,
                 sampling_preset=_PRESET_WIDGETS):
        audio_knobs = _audio_sampling(mosstts_model, sampling_preset, False, audio_temperature,
                                      audio_top_p, audio_top_k, audio_repetition_penalty)
        wav, frames = runtime.generate_speech(
            mosstts_model, text=text, language=language, instruction=instruction,
            target_tokens=target_tokens, max_new_tokens=max_new_tokens, seed=seed,
            do_sample=do_sample, text_temperature=text_temperature, text_top_p=text_top_p,
            text_top_k=text_top_k, **audio_knobs,
            progress_callback=ProgressReporter(max_new_tokens),
        )
        return (runtime.tensor_to_comfy_audio(wav, mosstts_model.spec.sample_rate), frames)


class MossTTSV15VoiceClone:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "mosstts_model": (MODEL_TYPE, {}),
            "reference_audio": ("AUDIO", {"tooltip": "Voice reference (5-15 s recommended)."}),
            "text": ("STRING", {"multiline": True, "default": "This line will be spoken in the reference voice."}),
        }
        required.update(_all_knobs())
        return {"required": required, "optional": dict(_OPTIONAL_PRESET)}

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("audio", "tokens_generated")
    FUNCTION = "clone"
    CATEGORY = "MOSS-TTS v1.5"
    DESCRIPTION = "Zero-shot voice cloning from a reference AUDIO input."

    def clone(self, mosstts_model, reference_audio, text, language, instruction,
              audio_temperature, audio_top_p, audio_top_k, audio_repetition_penalty,
              text_temperature, text_top_p, text_top_k, target_tokens, max_new_tokens,
              do_sample, seed, sampling_preset=_PRESET_WIDGETS):
        audio_knobs = _audio_sampling(mosstts_model, sampling_preset, True, audio_temperature,
                                      audio_top_p, audio_top_k, audio_repetition_penalty)
        wav, frames = runtime.voice_clone(
            mosstts_model, reference_audio=reference_audio, text=text, language=language,
            instruction=instruction, target_tokens=target_tokens,
            max_new_tokens=max_new_tokens, do_sample=do_sample,
            text_temperature=text_temperature, text_top_p=text_top_p, text_top_k=text_top_k,
            **audio_knobs,
            seed=seed, progress_callback=ProgressReporter(max_new_tokens),
        )
        return (runtime.tensor_to_comfy_audio(wav, mosstts_model.spec.sample_rate), frames)


class MossTTSV15VoiceDesign:
    """MOSS-VoiceGenerator（MossTTSDelay 家族，1.7B）：用文字描述音色并直接
    发声，不需要参考音频。官方对该模型推荐的采样默认值与 delay 不同
    （1.5 / 0.6 / 50 / 1.1），故本节点单独给出默认值。"""

    @classmethod
    def INPUT_TYPES(cls):
        knobs = _all_knobs()
        instruction = knobs.pop("instruction")
        instruction = ("STRING", {**instruction[1], "default": "A warm, friendly, young female voice.",
                                  "tooltip": "音色描述（必填）：性别/年龄/情绪/语速/口音等，中英文均可。产出音频可直接当 Voice Clone 的参考。"})
        # MOSS-VoiceGenerator recommended decoding defaults.
        for name, default in (("audio_temperature", 1.5), ("audio_top_p", 0.6),
                              ("audio_top_k", 50), ("audio_repetition_penalty", 1.1)):
            typ, kw = knobs[name]
            knobs[name] = (typ, {**kw, "default": default})
        return {"required": {
            "mosstts_model": (MODEL_TYPE, {}),
            "instruction": instruction,
            "text": ("STRING", {"multiline": True, "default": "Hello! This voice was designed from a text description."}),
            **knobs,
        }, "optional": dict(_OPTIONAL_PRESET)}

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("audio", "tokens_generated")
    FUNCTION = "run"
    CATEGORY = "MOSS-TTS v1.5"
    DESCRIPTION = "声音设计：用文本描述音色并直接发声（MOSS-VoiceGenerator），无需参考音频。"

    def run(self, mosstts_model, text, language, instruction, audio_temperature,
            audio_top_p, audio_top_k, audio_repetition_penalty, text_temperature,
            text_top_p, text_top_k, target_tokens, max_new_tokens, do_sample, seed,
            sampling_preset=_PRESET_WIDGETS):
        if mosstts_model.spec.key != "voicegen":
            raise ValueError("Voice Design 需要在 Load Model 里选择 MOSS-VoiceGenerator。")
        if not (instruction or "").strip():
            raise ValueError("Voice Design 必须提供音色描述（instruction）。")
        audio_knobs = _audio_sampling(mosstts_model, sampling_preset, False, audio_temperature,
                                      audio_top_p, audio_top_k, audio_repetition_penalty)
        wav, frames = runtime.generate_speech(
            mosstts_model, text=text, language=language, instruction=instruction,
            target_tokens=target_tokens, max_new_tokens=max_new_tokens, seed=seed,
            do_sample=do_sample, text_temperature=text_temperature, text_top_p=text_top_p,
            text_top_k=text_top_k, **audio_knobs,
            progress_callback=ProgressReporter(max_new_tokens),
        )
        return (runtime.tensor_to_comfy_audio(wav, mosstts_model.spec.sample_rate), frames)


class MossTTSV15ContinueSpeech:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "mosstts_model": (MODEL_TYPE, {}),
            "previous_audio": ("AUDIO", {"tooltip": "Prior clip to continue from."}),
            "previous_text": ("STRING", {"multiline": True, "default": "",
                                         "tooltip": "Exact text that produced previous_audio (word-for-word matters)."}),
            "text": ("STRING", {"multiline": True, "default": "",
                                "tooltip": "Follow-up text to speak next."}),
            "previous_tokens": ("INT", {"default": 0, "min": 0, "max": 45000, "step": 1,
                                        "tooltip": "Frame count of previous_audio; wire tokens_generated from the upstream node. 0 = measure from audio duration."}),
            "head_trim_frames": ("INT", {"default": 1, "min": 0, "max": 10, "step": 1,
                                         "tooltip": "Frames trimmed from the start of the NEW segment (codec receptive-field bleed ≈ 80 ms/frame)."}),
        }
        required.update(_all_knobs())
        return {"required": required, "optional": dict(_OPTIONAL_PRESET)}

    RETURN_TYPES = ("AUDIO", "INT", "AUDIO", "INT")
    RETURN_NAMES = ("audio", "tokens_generated", "full_audio", "full_tokens")
    FUNCTION = "continue_speech"
    CATEGORY = "MOSS-TTS v1.5"
    DESCRIPTION = "Prefix continuation: extend a generated clip in the same voice."

    def continue_speech(self, mosstts_model, previous_audio, previous_text, text,
                        previous_tokens, head_trim_frames, language, instruction,
                        audio_temperature, audio_top_p, audio_top_k,
                        audio_repetition_penalty, text_temperature, text_top_p, text_top_k,
                        target_tokens, max_new_tokens, do_sample, seed,
                        sampling_preset=_PRESET_WIDGETS):
        audio_knobs = _audio_sampling(mosstts_model, sampling_preset, True, audio_temperature,
                                      audio_top_p, audio_top_k, audio_repetition_penalty)
        wav, frames, full_wav, full_frames = runtime.continue_speech(
            mosstts_model, previous_audio=previous_audio, previous_text=previous_text,
            text=text, language=language, instruction=instruction,
            target_tokens=target_tokens, previous_tokens=previous_tokens,
            head_trim_frames=head_trim_frames, max_new_tokens=max_new_tokens,
            do_sample=do_sample, text_temperature=text_temperature, text_top_p=text_top_p,
            text_top_k=text_top_k, **audio_knobs, seed=seed,
            progress_callback=ProgressReporter(max_new_tokens),
        )
        sr = mosstts_model.spec.sample_rate
        return (
            runtime.tensor_to_comfy_audio(wav, sr), frames,
            runtime.tensor_to_comfy_audio(full_wav, sr), full_frames,
        )


class MossTTSV15EstimateTokens:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "words_per_minute": ("FLOAT", {"default": 150.0, "min": 60.0, "max": 400.0, "step": 5.0,
                                               "tooltip": "Speaking-rate assumption. For CJK this is characters-per-minute."}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("target_tokens",)
    FUNCTION = "estimate"
    CATEGORY = "MOSS-TTS v1.5"
    DESCRIPTION = "Heuristic text → target_tokens estimate for duration control."

    def estimate(self, text: str, words_per_minute: float):
        text = (text or "").strip()
        if not text:
            return (0,)
        if _is_cjk(text):
            units = sum(1 for ch in text if not ch.isspace())
        else:
            units = len(text.split())
        seconds = units / max(1e-3, float(words_per_minute) / 60.0)
        return (int(math.ceil(seconds * 12.5)),)


_LORA_NONE = "none"


def _lora_choices() -> tuple[dict, dict]:
    """(your adapters, LAION's). LAION adapters already downloaded into
    mosstts_loras are listed once, under their LAION name."""
    hf = adapters.hf_choices()
    local = adapters.list_local(compat.comfy_models_dir(),
                                tuple(compat.comfy_folder_paths(adapters.LORA_FOLDER_NAME)))
    laion = {adapters.catalog_label(e["repo"], e["subfolder"]) for e in hf.values()}
    return {k: v for k, v in local.items() if k not in laion}, hf


class MossTTSV15LoRA:
    """Adds one LoRA adapter (with its strength) to the model's adapter stack.
    Chain several: voice + emotion + DPO quality."""

    @classmethod
    def INPUT_TYPES(cls):
        local, hf = _lora_choices()
        return {
            "required": {
                "mosstts_model": (MODEL_TYPE, {}),
                "lora": ([_LORA_NONE] + sorted(local) + list(hf), {
                    "default": _LORA_NONE,
                    "tooltip": "Your adapters: folders with adapter_model.safetensors (+ "
                               "adapter_config.json) under models/mosstts_loras. LAION's published "
                               "adapters (voices, emotions, DPO quality, sounds) download on first use. "
                               "LAION adapters are trained on LAION voice-acting sft3 — load that model.",
                }),
                "strength": ("FLOAT", {"default": 1.0, "min": -3.0, "max": 3.0, "step": 0.05,
                                       "tooltip": "Scale of this adapter. LAION: voice 1.0 (trained value, "
                                                  "not swept), emotion 1.0-1.5, DPO quality 1.0."}),
            },
            "optional": {
                "speaker_name": ("STRING", {
                    "default": "",
                    "tooltip": "Puts 'Speaker: <name>' into the prompt when there is no reference "
                               "audio — the name mode LAION's voices were trained with. Empty = the "
                               "LAION voice's own name (shown in the list), nothing for other adapters. "
                               "'none' = no name.",
                }),
            },
        }

    RETURN_TYPES = (MODEL_TYPE,)
    RETURN_NAMES = ("mosstts_model",)
    FUNCTION = "apply"
    CATEGORY = "MOSS-TTS v1.5"
    DESCRIPTION = ("Adds a LoRA adapter to the MOSS-TTS model, unmerged (the output heads share weights "
                   "with the embeddings, so merging would break the model). Chain several, each with its "
                   "own strength. A LAION voice also sets 'Speaker: <name>' for a consistent voice.")

    def apply(self, mosstts_model, lora, strength, speaker_name=""):
        name = (speaker_name or "").strip()
        if lora == _LORA_NONE:
            path, auto_speaker = None, ""
        else:
            local, hf = _lora_choices()
            if lora in local:
                path, auto_speaker = local[lora], adapters.known_speaker(local[lora].parent.name)
            elif lora in hf:
                entry = hf[lora]
                path = adapters.fetch_hf(entry["repo"], entry["subfolder"], entry.get("revision"),
                                         adapters.download_root(compat.comfy_models_dir()))
                auto_speaker = entry["speaker"]
                if not str(getattr(mosstts_model.spec, "key", "")).startswith(("laion_sft3", "custom:")):
                    logger.warning("[MOSS-TTS LoRA] %s is trained on LAION voice-acting sft3; the loaded "
                                   "model is %s.", lora, mosstts_model.spec.key)
            else:
                raise FileNotFoundError(f"LoRA {lora!r} not found (models/mosstts_loras or LAION list).")
            weights = adapters.load(path)  # parse + validate now, not mid-generation
            if isinstance(getattr(mosstts_model, "model", None), torch.nn.Module):
                adapters.check_compatible(mosstts_model.model, weights, lora)
        speaker = mosstts_model.speaker
        if name.lower() == "none":
            speaker = ""
        elif name:
            speaker = name
        elif auto_speaker:
            speaker = auto_speaker
        loras = tuple(mosstts_model.loras)
        if path is not None and float(strength) != 0.0:
            loras += ((lora, str(path), float(strength)),)
        return (dataclasses.replace(mosstts_model, loras=loras, speaker=speaker),)


class MossTTSV15SelfTest:
    """Checks the loaded model where it runs: environment, every weight against
    the checkpoint on disk, and a greedy generation against CPU reference frames."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mosstts_model": (MODEL_TYPE, {})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "run"
    CATEGORY = "MOSS-TTS v1.5"
    OUTPUT_NODE = True
    DESCRIPTION = ("Diagnostics for broken audio: torch / GPU / attention, every model weight vs the "
                   "checkpoint on disk, and a greedy generation of LAION's example vs reference frames "
                   "(LAION sft3). Connect the report to Preview as Text and send it along with the bug.")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, mosstts_model):
        bundle = runtime.resume_model(mosstts_model)
        report = selftest.run(bundle)
        logger.info("[MOSS-TTS] self test:\n%s", report)
        return {"ui": {"text": [report]}, "result": (report,)}


class MossTTSV15DirectorPrompt:
    """System prompt for an LLM that turns plain text into a performance for
    the voice-acting fine-tunes. Use it as the system prompt of any text
    generation node; feed its reply to Script Builder."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target": (director.TARGETS, {
                    "default": director.TARGET_SFT3,
                    "tooltip": "sft3: a script with a delivery cue per sentence, vocal bursts and beats "
                               "(Script Builder adds the timing). 8B / MOSS: one instruction + the text.",
                }),
                "voice": (director.VOICE_MODES, {
                    "default": director.VOICE_DESCRIBE,
                    "tooltip": "describe the voice: the LLM picks age / gender / timbre (plain Generate "
                               "Speech). fixed: Voice Clone or a voice LoRA — the LLM must not describe "
                               "the voice (it would fight the reference / adapter, and a description that "
                               "changes per run changes the voice).",
                }),
                "wording": (director.WORDING_MODES, {
                    "default": director.WORDING_EXACT,
                    "tooltip": "Keep your words exactly (only cues / sounds / beats are added) or let the "
                               "LLM smooth them for speech.",
                }),
                "bursts": (director.BURST_MODES, {
                    "default": director.BURSTS_OFTEN,
                    "tooltip": "Vocal bursts (laughs, sighs, breaths) — sft3 only. LAION: real speech is "
                               "full of them; a line without any sounds read rather than spoken.",
                }),
            },
            "optional": {
                "extra": ("STRING", {"multiline": True, "default": "",
                                     "tooltip": "Your own notes appended to the prompt (character, "
                                                "scene, style)."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("system_prompt",)
    FUNCTION = "build"
    CATEGORY = "MOSS-TTS v1.5"
    DESCRIPTION = ("System prompt for an LLM (e.g. ComfyUI's Generate Text) that turns plain text into a "
                   "voice-acting performance for the LAION fine-tunes. Feed the LLM's reply to Script Builder.")

    def build(self, target, voice, wording, bursts, extra=""):
        return (director.system_prompt(target, voice, wording, bursts, extra),)


class MossTTSV15ScriptBuilder:
    """Director reply -> instruction / text / target_tokens for the MOSS nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "director_reply": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "The LLM's reply (GENERAL: / SCRIPT: or INSTRUCTION: / TEXT:), or a script "
                               "you wrote yourself. <think> blocks and code fences are ignored.",
                }),
                "format": (["auto", "sft3 timed script", "instruction + text"], {
                    "default": "auto",
                    "tooltip": "auto: by the reply's fields (SCRIPT: -> sft3, INSTRUCTION: / TEXT: -> "
                               "instruction + text; unlabelled text -> sft3 script).",
                }),
                "language": (["English", "German"] + [l for l in runtime.LANGUAGES
                                                      if l not in ("auto", "English", "German")], {
                    "default": "English",
                    "tooltip": "Language code at the end of GENERAL (sft3 knows English and German). "
                               "Set the same language on the generate node.",
                }),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05,
                                    "tooltip": "Speech rate for the computed durations: 4 frames "
                                               "(0.32 s) per word at 1.0."}),
                "target_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 600.0, "step": 0.1,
                                             "tooltip": "0 = natural length. Otherwise the speech is "
                                                        "stretched to fit (pauses and sounds kept), within "
                                                        "x0.6-x1.5 — beyond that the model invents words."}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("instruction", "text", "target_tokens", "seconds", "report")
    FUNCTION = "build"
    CATEGORY = "MOSS-TTS v1.5"
    DESCRIPTION = ("Turns a director reply into the generate node's instruction / text / target_tokens. "
                   "sft3: adds every duration, pause and sound length so they add up to target_tokens, "
                   "and puts the timed script into both SCRIPT: and text, byte-identical.")

    def build(self, director_reply, format, language, speed, target_seconds):
        b = director.build(director_reply, fmt=format, language=language, speed=speed,
                           target_seconds=target_seconds)
        logger.info("[MOSS-TTS] Script Builder: %s", b.report.replace("\n", " | "))
        return (b.instruction, b.text, b.target_tokens, b.seconds, b.report)


def _is_cjk(text: str) -> bool:
    for ch in text[:200]:
        code = ord(ch)
        if (0x4E00 <= code <= 0x9FFF or 0x3040 <= code <= 0x309F
                or 0x30A0 <= code <= 0x30FF or 0xAC00 <= code <= 0xD7AF):
            return True
    return False


NODE_CLASS_MAPPINGS = {
    "MossTTSV15_LoadModel": MossTTSV15LoadModel,
    "MossTTSV15_GenerateSpeech": MossTTSV15GenerateSpeech,
    "MossTTSV15_VoiceClone": MossTTSV15VoiceClone,
    "MossTTSV15_VoiceDesign": MossTTSV15VoiceDesign,
    "MossTTSV15_ContinueSpeech": MossTTSV15ContinueSpeech,
    "MossTTSV15_EstimateTokens": MossTTSV15EstimateTokens,
    "MossTTSV15_DirectorPrompt": MossTTSV15DirectorPrompt,
    "MossTTSV15_ScriptBuilder": MossTTSV15ScriptBuilder,
    "MossTTSV15_LoRA": MossTTSV15LoRA,
    "MossTTSV15_SelfTest": MossTTSV15SelfTest,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MossTTSV15_LoadModel": "MOSS-TTS v1.5 Load Model",
    "MossTTSV15_GenerateSpeech": "MOSS-TTS v1.5 Generate Speech",
    "MossTTSV15_VoiceClone": "MOSS-TTS v1.5 Voice Clone",
    "MossTTSV15_VoiceDesign": "MOSS-TTS v1.5 Voice Design",
    "MossTTSV15_ContinueSpeech": "MOSS-TTS v1.5 Continue Speech",
    "MossTTSV15_EstimateTokens": "MOSS-TTS v1.5 Estimate Tokens",
    "MossTTSV15_DirectorPrompt": "MOSS-TTS v1.5 Director System Prompt",
    "MossTTSV15_ScriptBuilder": "MOSS-TTS v1.5 Script Builder",
    "MossTTSV15_LoRA": "MOSS-TTS v1.5 LoRA",
    "MossTTSV15_SelfTest": "MOSS-TTS v1.5 Self Test",
}
