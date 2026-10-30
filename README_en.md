# MOSS-TTS v1.5 for ComfyUI

[中文](README.md)

ComfyUI custom nodes for [OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5)
[OpenMOSS-Team/MOSS-TTS-v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5), and
[OpenMOSS-Team/MOSS-VoiceGenerator](https://huggingface.co/OpenMOSS-Team/MOSS-VoiceGenerator):

- **Local-Transformer** — Qwen3-**4B**-class backbone + nano-GPT2 local transformer, MOSS-Audio-Tokenizer-v2, **48 kHz stereo**, n_vq=12. (Note: the "1.7B" floating around in other packs' READMEs is wrong; the backbone config is Qwen3-4B-shaped, checkpoint ≈ 9.1 GB in bf16.)
- **Delay** — 8B delay-pattern model, MOSS-Audio-Tokenizer (v1), **24 kHz**, n_vq=32.
- **VoiceGenerator** — 1.7B voice-design model (MossTTSDelay family, shares the delay code path): speak a voice described in text, no reference audio needed; **24 kHz**, n_vq=16. Its output can feed Voice Clone as reference — design -> clone -> produce.

All three are loaded through the same nodes; switch variants in Load Model.

## Highlights

- Reference-free TTS, zero-shot voice cloning, continuation, hard duration control (`target_tokens`, 12.5 frames/s), 31 languages with explicit language tags, `[pause 3.2s]` markers.
- No `trust_remote_code`: the model code is vendored under `assets/` (with minimal, clearly marked compatibility patches), so the pack does not depend on whatever happens to be in your HF module cache.
- Works on both transformers 4.x and 5.x for both variants (patched vendored code + feature detection); E2E-validated on both 5.16 and 4.57 across two machines.
- Deep ComfyUI memory-management integration: weights register through `ModelPatcher` / `ModelPatcherDynamic` (AIMDO DynamicVRAM aware) and participate in normal unloads; every integration point is feature-detected and degrades gracefully on older ComfyUI.
- Weights are resolved, in order: `$MOSS_TTS_MODELS_DIR/<Repo-Name>` → `ComfyUI/models/mosstts/<Repo-Name>` → the HF hub cache (respecting `HF_HOME`) → auto-download (when `download_if_missing`).

## Install

**ComfyUI-Manager**: Manager → Custom Nodes Manager → search `moss-tts` → Install. Manager fetches the package and installs dependencies (`install.py`) automatically; restart ComfyUI afterwards.

**Git clone**:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/CloudRipple/MOSS-TTS-ComfyUI.git
# deps (only missing ones get installed):
python install.py   # or: pip install -r requirements.txt
```

Restart ComfyUI. First use resolves weights from your local caches; if absent
and `download_if_missing` is on, they download from Hugging Face
(~9.1 GB + ~8 GB codec for Local; ~17 GB + ~6.7 GB codec for Delay).

## Nodes (category `MOSS-TTS v1.5`)

| Node | Purpose |
|---|---|
| MOSS-TTS v1.5 Load Model | Load variant (`local` / `delay` / `voicegen` / LAION fine-tunes) or a `custom_model` folder, dtype (auto/bf16/fp16/fp32), attention (auto/sdpa/flash_attention_2/eager), download toggle. |
| MOSS-TTS v1.5 Generate Speech | Reference-free TTS. |
| MOSS-TTS v1.5 Voice Clone | Clone from a reference `AUDIO` input. |
| MOSS-TTS v1.5 Voice Design | `instruction` holds the voice description (required); speaks it directly (use with VoiceGenerator). |
| MOSS-TTS v1.5 Continue Speech | Prefix continuation: extend a clip in the same voice. Outputs both the new segment and the stitched full audio (+ exact frame counts for chaining). |
| MOSS-TTS v1.5 Estimate Tokens | Text → `target_tokens` estimate for duration control. |
| MOSS-TTS v1.5 Director System Prompt | System prompt for an LLM (e.g. ComfyUI's Generate Text) that turns plain text into a voice-acting performance for the LAION fine-tunes. |
| MOSS-TTS v1.5 Script Builder | The LLM's reply → `instruction`, `text`, `target_tokens` for the generate nodes; for sft3 it computes every duration / pause / sound length. |

All generator nodes output `tokens_generated` (audio frames; seconds = frames / 12.5) so continuation chains can hand exact prefix lengths forward.

## Fine-tunes (this fork)

**LAION voice-acting** models are in the Load Model list; they run on the same
vendored code as the OpenMOSS originals (checked: identical model code, every
checkpoint tensor name and shape matches):

| Variant | Repo | Notes |
|---|---|---|
| LAION voice-acting sft3 | [laion/moss-tts-local-transformer-4.55b-voice-acting-v2-sft3](https://huggingface.co/laion/moss-tts-local-transformer-4.55b-voice-acting-v2-sft3) | Local family, 48 kHz, English / German, CC-BY-4.0. Timed scripts: `[2.4 seconds duration]`, `[0.3 seconds pause]`, `(breathy giggle, 0.4 seconds)`, `(clearly amused)`. |
| LAION voice-acting 8B | [laion/moss-tts-v1.5-8b-voice-acting](https://huggingface.co/laion/moss-tts-v1.5-8b-voice-acting) | Delay family, 24 kHz, multilingual, Apache-2.0. `instruction` works together with a voice reference. |

**Your own fine-tunes**: put the model folder (config.json + safetensors +
tokenizer files) under `ComfyUI/models/mosstts/` (or `$MOSS_TTS_MODELS_DIR`),
up to two levels deep — e.g. `mosstts/my_voice/` or `mosstts/runs/ckpt-500/` —
and pick it in Load Model's `custom_model` (refresh the page after adding).
The family (Local / Delay), n_vq and sample rate come from its config.json; the
codec is the family's own. `custom_model` overrides `model`.

**Text → performance with an LLM** (the two-layer setup LAION uses):

```
Director System Prompt ──system_prompt──► Generate Text (core) ──text──► Script Builder
                                              ▲ prompt = your plain text      │ instruction, text, target_tokens
                                                                              ▼
                                                     Generate Speech / Voice Clone (LAION sft3 or 8B)
```

- The LLM writes the performance and **never a number**: a delivery cue before every
  sentence `(clearly amused, letting it show)`, vocal bursts in their own brackets
  between sentences `(chuckle)`, beats `[pause]` / `[long pause]`.
- Script Builder adds the arithmetic sft3 was trained on: `[N.N seconds duration]`
  per segment (4 frames per word), `[N.N seconds pause]`, `(label, N.N seconds)`,
  splits segments over 12 s, and makes `target_tokens` the exact sum at 12.5 fps.
  The timed script goes into both `SCRIPT:` (inside `instruction`) and `text`,
  byte-identical, as the format requires. It reproduces LAION's published worked
  examples exactly.
- Director `target`: *sft3 — timed script* or *8B / MOSS — instruction + text*
  (one instruction for the whole text, no tags). `voice`: describe the voice
  (Generate Speech) or leave it to the reference clip (Voice Clone).
- sft3 speaks English and German; cues stay English either way. Set the same
  `language` on the generate node, and `sampling_preset` = model recommended.

**`sampling_preset`** (Generate / Clone / Design / Continue): `widgets` (default,
old behaviour) or `model recommended` — the audio temperature / top_p / top_k /
repetition penalty from the model card, the audio_* widgets are then ignored:

| Model | Without reference | With reference (Clone, Continue) |
|---|---|---|
| OpenMOSS Local / Delay | 1.7 / 0.8 / 25 / 1.0 | same |
| MOSS-VoiceGenerator | 1.5 / 0.6 / 50 / 1.1 | same |
| LAION sft3 | 1.0 / 0.95 / 50 / 1.0 | same |
| LAION 8B | 0.8 / 0.95 / 25 / 1.1 | 1.0 / 0.95 / 25 / 1.1 |
| custom folder | widgets | widgets |

## VRAM

- Local (4B): ~12 GB bf16 active.
- Delay (8B): ~22 GB bf16 active.
- VoiceGenerator (1.7B): ~4 GB bf16 model + ~7 GB fp32 codec.

Keep voice-clone references short (5–15 s); prefix/PKV memory grows linearly with prefix duration.

## Troubleshooting

- **Nothing happens / model can't be found**: the loader prints the exact search paths. Set `$MOSS_TTS_MODELS_DIR` to a directory containing `MOSS-TTS-Local-Transformer-v1.5/` etc., or rely on `HF_HOME` pointing at your HF cache.
- **flash-attn errors**: set attention to `auto` or `sdpa`. flash-attn is optional; sdpa quality is identical.
- **VRAM pressure alongside big ComfyUI workflows**: the pack registers with ComfyUI's memory management, so `Free memory` / unloading works as usual.

## Updating the vendored model code

`assets/` mirrors the remote code of the four HF repos. Patches are marked
`# MOSS-TTS-V15-ComfyUI patch:` and limited to: transformers 4.x/5.x import
compat, an optional progress callback for the delay variant, and Path-accepting
`encode_audios_from_path`. To sync with upstream, recopy the four repos' code
files and re-apply the grep-able patch set.

## License

MIT (this pack). Model weights and upstream model code are Apache-2.0 by OpenMOSS-Team.
