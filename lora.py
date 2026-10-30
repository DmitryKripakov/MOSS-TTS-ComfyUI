"""PEFT LoRA adapters for MOSS-TTS, applied WITHOUT merging.

The local family ties its output heads to the input embeddings
(`audio_lm_heads[i].weight IS audio_embeddings[i].weight`). LAION's adapters
target those heads, so merging `B @ A * scale` into a head weight would write the
same delta into the embedding table and silently ruin the model (the adapter
cards warn about exactly this). Here every adapted Linear gets a forward hook
that adds `scale * B(A(x))` to its output: the weights are never touched, any
number of adapters stack with their own strength, and switching or removing them
is instant.

A bundle carries its adapter stack (`bundle.loras`, set by the LoRA node on a
copy of the bundle); `apply(model, stack)` runs before every generation and
installs exactly that stack (or none), so two branches of a workflow sharing one
loaded model never leak adapters into each other.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger("ComfyUI-MOSS-TTS-v15")

ADAPTER_FILE = "adapter_model.safetensors"
CONFIG_FILE = "adapter_config.json"
_PREFIX = "base_model.model."
_KEY = re.compile(r"^(?:base_model\.model\.)?(.+)\.lora_([AB])(?:\.[^.]+)?\.weight$")

_HOOKS_ATTR = "_mosstts_lora_hooks"
_STACK_ATTR = "_mosstts_lora_stack"
_CACHE_SIZE = 6


class LoraWeights:
    """One adapter file: {module name: (A [r,in], B [out,r], scale)} on CPU."""

    def __init__(self, path: Path):
        from safetensors.torch import load_file

        self.path = Path(path)
        folder = self.path.parent
        cfg = {}
        if (folder / CONFIG_FILE).is_file():
            cfg = json.loads((folder / CONFIG_FILE).read_text(encoding="utf-8"))
        else:
            logger.warning("[MOSS-TTS LoRA] %s has no %s — assuming lora_alpha = r (scale 1).",
                           folder, CONFIG_FILE)
        if cfg.get("use_dora"):
            raise ValueError(f"{folder.name}: DoRA adapters are not supported.")
        peft_type = cfg.get("peft_type", "LORA")
        if peft_type != "LORA":
            raise ValueError(f"{folder.name}: peft_type {peft_type!r} is not supported (LORA only).")
        tensors = load_file(str(self.path), device="cpu")
        pairs: dict[str, dict[str, torch.Tensor]] = {}
        other = []
        for key, value in tensors.items():
            m = _KEY.match(key)
            if m is None:
                other.append(key)
                continue
            pairs.setdefault(m.group(1), {})[m.group(2)] = value
        if other:
            raise ValueError(
                f"{folder.name}: unsupported tensors in the adapter (embedding LoRA / modules_to_save?): "
                f"{other[:4]}{' …' if len(other) > 4 else ''}"
            )
        alpha = float(cfg.get("lora_alpha", 0) or 0)
        rslora = bool(cfg.get("use_rslora"))
        alpha_pattern = cfg.get("alpha_pattern") or {}
        self.modules: dict[str, tuple[torch.Tensor, torch.Tensor, float]] = {}
        for name, ab in pairs.items():
            if set(ab) != {"A", "B"}:
                raise ValueError(f"{folder.name}: {name} has lora_{'A' if 'A' not in ab else 'B'} missing.")
            a, b = ab["A"], ab["B"]
            r = a.shape[0]
            a_ = _pattern_value(alpha_pattern, name, alpha or r)
            scale = a_ / (math.sqrt(r) if rslora else r)
            self.modules[name] = (a.float(), b.float(), float(scale))
        if not self.modules:
            raise ValueError(f"{folder.name}: no LoRA weights in {self.path.name}.")
        self.base = cfg.get("base_model_name_or_path", "")
        self.rank = max(a.shape[0] for a, _, _ in self.modules.values())


def _pattern_value(pattern: dict, name: str, default):
    for key, value in pattern.items():
        if re.match(rf"(.*\.)?{key}$", name):
            return value
    return default


_CACHE: "OrderedDict[tuple, LoraWeights]" = OrderedDict()


def load(path) -> LoraWeights:
    """LoraWeights for an adapter file (cached by path + mtime)."""
    path = Path(path)
    if path.is_dir():
        path = path / ADAPTER_FILE
    key = (str(path), path.stat().st_mtime_ns)
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    weights = LoraWeights(path)
    _CACHE[key] = weights
    while len(_CACHE) > _CACHE_SIZE:
        _CACHE.popitem(last=False)
    return weights


def check_compatible(model: nn.Module, weights: LoraWeights, label: str) -> None:
    """Every adapted module must exist in the model as a Linear of matching shape."""
    modules = dict(model.named_modules())
    missing, bad = [], []
    for name, (a, b, _) in weights.modules.items():
        mod = modules.get(name)
        if not isinstance(mod, nn.Linear):
            missing.append(name)
        elif a.shape[1] != mod.in_features or b.shape[0] != mod.out_features or a.shape[0] != b.shape[1]:
            bad.append(f"{name} {tuple(b.shape[:1]) + tuple(a.shape[1:])} vs "
                       f"{(mod.out_features, mod.in_features)}")
    if missing or bad:
        raise ValueError(
            f"LoRA {label!r} does not fit the loaded model ({len(missing)} missing, {len(bad)} wrong "
            f"shape: {(missing + bad)[:3]}). It was trained for {weights.base or 'another model'} — "
            "load that model (LAION's adapters need LAION voice-acting sft3)."
        )


class _Delta:
    """Forward hook adding the stack's LoRA terms to one Linear's output."""

    def __init__(self, terms: list):
        self.terms = terms          # [(A, B, scale)] on CPU, float32
        self.key = None
        self.cached: list = []

    def _on(self, device, dtype) -> list:
        if self.key != (device, dtype):    # keep one copy: the device the model now runs on
            self.cached = [(a.to(device, dtype), b.to(device, dtype), s) for a, b, s in self.terms]
            self.key = (device, dtype)
        return self.cached

    def __call__(self, module, args, output):
        x = args[0]
        dtype = output.dtype if output.dtype.is_floating_point else x.dtype
        delta = None
        for a, b, s in self._on(x.device, dtype):
            term = (x.to(dtype) @ a.t()) @ b.t()
            delta = term * s if delta is None else delta + term * s
        return output + delta.to(output.dtype)


def clear(model: Optional[nn.Module]) -> None:
    """Remove every adapter hook from the model."""
    if model is None:
        return
    for handle in getattr(model, _HOOKS_ATTR, ()):
        handle.remove()
    setattr(model, _HOOKS_ATTR, [])
    setattr(model, _STACK_ATTR, ())


def apply(model: nn.Module, stack) -> None:
    """Install exactly `stack` = ((label, path, strength), …) on the model —
    nothing else stays. A no-op when it is already installed."""
    stack = tuple(stack or ())
    if getattr(model, _STACK_ATTR, ()) == stack and getattr(model, _HOOKS_ATTR, None) is not None:
        return
    clear(model)
    active = [(label, path, float(strength)) for label, path, strength in stack if float(strength) != 0.0]
    if not active:
        setattr(model, _STACK_ATTR, stack)
        return
    per_module: dict[str, list] = {}
    for label, path, strength in active:
        weights = load(path)
        check_compatible(model, weights, label)
        for name, (a, b, scale) in weights.modules.items():
            per_module.setdefault(name, []).append((a, b, scale * strength))
    modules = dict(model.named_modules())
    handles = [modules[name].register_forward_hook(_Delta(terms)) for name, terms in per_module.items()]
    setattr(model, _HOOKS_ATTR, handles)
    setattr(model, _STACK_ATTR, stack)
    logger.info("[MOSS-TTS LoRA] active: %s (%d modules)",
                ", ".join(f"{label} x{strength:g}" for label, _, strength in active), len(handles))


# ---------------------------------------------------------------------------
# finding adapters
# ---------------------------------------------------------------------------

LORA_FOLDER_NAME = "mosstts_loras"
_SCAN_DEPTH = 4


def local_roots(models_dir: Optional[str]) -> list[Path]:
    roots = []
    env = os.getenv("MOSS_TTS_LORAS_DIR")
    if env:
        roots.append(Path(env))
    if models_dir:
        roots.append(Path(models_dir) / LORA_FOLDER_NAME)
    return roots


def list_local(models_dir: Optional[str]) -> dict[str, Path]:
    """{label (relative folder): adapter file} for every folder holding
    adapter_model.safetensors under models/mosstts_loras (or
    $MOSS_TTS_LORAS_DIR), a few levels deep — a whole downloaded HF repo of
    adapters works as is."""
    found: dict[str, Path] = {}
    for root in local_roots(models_dir):
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            rel = Path(dirpath).relative_to(root)
            dirnames[:] = sorted(d for d in dirnames if not d.startswith(".")) if len(rel.parts) < _SCAN_DEPTH else []
            if ADAPTER_FILE in filenames:
                label = rel.as_posix() if rel.parts else root.name
                found.setdefault(label, Path(dirpath) / ADAPTER_FILE)
    return found


# LAION's published adapters, downloaded on first use (catalog: lora_catalog.json)
_CATALOG_FILE = Path(__file__).resolve().parent / "lora_catalog.json"
_CATALOG: Optional[dict] = None


def catalog() -> dict:
    global _CATALOG
    if _CATALOG is None:
        try:
            _CATALOG = json.loads(_CATALOG_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _CATALOG = {}
    return _CATALOG


def hf_choices() -> dict[str, dict]:
    """{combo label: {repo, subfolder, revision, speaker}} for LAION's adapters."""
    out = {}
    for group in catalog().get("groups", []):
        for item in group["adapters"]:
            out[f"{group['prefix']}{item['label']}"] = {
                "repo": group["repo"], "revision": group.get("revision"),
                "subfolder": item["subfolder"], "speaker": item.get("speaker", ""),
            }
    return out


def known_speaker(folder_name: str) -> str:
    """Speaker name of a LAION voice adapter by its folder name (anime_000 ->
    Emma), for adapters downloaded by hand into models/mosstts_loras."""
    for group in catalog().get("groups", []):
        for item in group["adapters"]:
            if item.get("speaker") and item["subfolder"] == folder_name:
                return item["speaker"]
    return ""


def fetch_hf(repo: str, subfolder: str, revision: Optional[str] = None) -> Path:
    """Download (or find in the HF cache) one adapter -> its safetensors path."""
    from huggingface_hub import hf_hub_download

    kw = {"subfolder": subfolder or None, "revision": revision}
    hf_hub_download(repo, CONFIG_FILE, **kw)
    return Path(hf_hub_download(repo, ADAPTER_FILE, **kw))
