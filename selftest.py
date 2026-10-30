"""Self test for a loaded MOSS-TTS bundle, run where the problem is (the GPU).

1. Environment: torch / CUDA / GPU, attention backends.
2. Weights: every tensor of the loaded model against its safetensors file on
   disk (the tied heads included) — catches weights damaged or re-cast on the
   way into VRAM.
3. Greedy generation of LAION's model-card example, no adapters, compared
   frame by frame with reference frames computed on CPU (selftest_reference.json).
   A healthy GPU matches the first frames; bf16 rounding may make a late frame
   differ, a broken forward pass differs from the first frame on.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import torch

from . import lora, native

# The sft3 model card's own example (Text = SCRIPT, numbers add up to Tokens).
_SCRIPT = ("[0.4 seconds pause] (intensely amused: letting it out / not hiding it, warm and open, unguarded; "
           "bright, relaxed) [2.4 seconds duration] You are not going to believe this. [0.3 seconds pause] "
           "(breathy giggle, 0.4 seconds) [0.2 seconds pause] (still intensely amused) [2.3 seconds duration] "
           "He actually wore it to the wedding.")
PROMPT = {
    "text": _SCRIPT,
    "instruction": "GENERAL: A young adult feminine voice, warm and conversational; reads as amusement; 6.0s, EN.\n"
                   "SCRIPT:\n" + _SCRIPT,
    "tokens": 75,
    "language": "English",
}
GREEDY = {"max_new_tokens": 8, "do_sample": False}
REFERENCE_FILE = Path(__file__).resolve().parent / "selftest_reference.json"


def _env() -> list[str]:
    lines = [f"python {platform.python_version()}, torch {torch.__version__}, "
             f"CUDA {torch.version.cuda}, {platform.machine()}"]
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(torch.cuda.current_device())
        lines.append(f"GPU {p.name}, sm_{p.major}{p.minor}, {p.total_memory / 2**30:.0f} GB")
        b = torch.backends.cuda
        lines.append("SDPA enabled: flash=%s mem_efficient=%s math=%s cudnn=%s" % (
            b.flash_sdp_enabled(), b.mem_efficient_sdp_enabled(), b.math_sdp_enabled(),
            b.cudnn_sdp_enabled() if hasattr(b, "cudnn_sdp_enabled") else "?"))
    return lines


def check_weights(bundle) -> list[str]:
    """Every model tensor vs the checkpoint on disk."""
    from safetensors import safe_open

    model = bundle.model
    live = model.state_dict(keep_vars=True)
    files = native._iter_safetensors_entries(Path(bundle.model_dir))
    checked, bad, missing_on_disk = 0, [], set(live)
    worst = 0.0
    for path, keys in files:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for key in keys or list(f.keys()):      # a single model.safetensors lists no keys
                if key not in live:
                    continue
                missing_on_disk.discard(key)
                disk = f.get_tensor(key).float()
                have = live[key].detach().float().cpu()
                if have.shape != disk.shape:
                    bad.append(f"{key}: shape {tuple(have.shape)} vs disk {tuple(disk.shape)}")
                    continue
                diff = (have - disk).abs().max().item() / max(disk.abs().max().item(), 1e-6)
                worst = max(worst, diff)
                if diff > 1e-2 or not torch.isfinite(have).all():
                    bad.append(f"{key}: relative max diff {diff:.3g}")
                checked += 1
    tied = getattr(model, "audio_lm_heads", None)
    tie = "n/a"
    if tied is not None and hasattr(model, "audio_embeddings"):
        same = all(torch.equal(h.weight.detach().cpu(), e.weight.detach().cpu())
                   for h, e in zip(model.audio_lm_heads, model.audio_embeddings))
        tie = "equal" if same else "DIFFERENT (heads no longer match the embeddings)"
    devices = sorted({str(t.device) for t in live.values()})
    dtypes = sorted({str(t.dtype).replace("torch.", "") for t in live.values()})
    lines = [f"weights: {checked} tensors compared with disk, worst relative diff {worst:.2e}; "
             f"tied heads {tie}; devices {devices}; dtypes {dtypes}"]
    extra = sorted(k for k in missing_on_disk if not k.startswith(("text_lm_head", "audio_lm_heads")))
    if extra:
        lines.append(f"  not in the checkpoint: {extra[:5]}{' …' if len(extra) > 5 else ''}")
    lines += [f"  BAD {b}" for b in bad[:10]]
    return lines


def greedy_check(bundle) -> list[str]:
    """Greedy frames of the card example vs the CPU reference."""
    if not REFERENCE_FILE.is_file():
        return ["greedy: no reference file"]
    ref = json.loads(REFERENCE_FILE.read_text(encoding="utf-8"))
    if getattr(bundle.spec, "key", "") != ref["model"]:
        return [f"greedy: reference is for {ref['model']}, the loaded model is {bundle.spec.key} — skipped"]
    stack = getattr(bundle.model, lora._STACK_ATTR, ())
    lora.clear(bundle.model)                         # the reference has no adapters
    try:
        user = bundle.processor.build_user_message(**PROMPT)
        batch = bundle.processor([[user]], mode="generation")
        with torch.inference_mode():
            out = bundle.model.generate(input_ids=batch["input_ids"].to(bundle.device),
                                        attention_mask=batch["attention_mask"].to(bundle.device), **GREEDY)
    finally:
        if stack:
            lora.apply(bundle.model, stack)
    got = out[0][1].tolist()
    want = ref["frames"]
    n = min(len(got), len(want))
    first_diff = next((i for i in range(n) if got[i] != want[i]), None)
    lines = [f"greedy ({len(got)} frames vs reference {len(want)}, {ref['device']} {ref['dtype']}): "
             + ("all frames match" if first_diff is None else f"first difference at frame {first_diff}")]
    if first_diff is not None:
        lines.append(f"  frame {first_diff}: got {got[first_diff]}")
        lines.append(f"  frame {first_diff}: ref {want[first_diff]}")
        if first_diff <= 1:
            lines.append("  -> the forward pass on this device is wrong from the start "
                         "(weights above are fine? then it is the device's kernels)")
    return lines


def run(bundle) -> str:
    lines = [f"MOSS-TTS self test — {bundle.spec.key} on {bundle.device} ({bundle.torch_dtype}), "
             f"attention {bundle.attn_implementation}"]
    for part in (_env, check_weights, greedy_check):
        try:
            lines += part(bundle) if part is not _env else part()
        except Exception as e:  # report, never crash the test
            lines.append(f"{part.__name__}: failed — {type(e).__name__}: {e}")
    return "\n".join(lines)
