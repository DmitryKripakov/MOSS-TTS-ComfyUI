"""LoRA adapters applied unmerged (lora.py), the LoRA node and name-mode
speakers. A tiny model with the same tie as MOSS-TTS Local (head weight IS
the embedding table) and a synthetic PEFT adapter file."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from moss_tts import lora


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_embeddings = nn.ModuleList([nn.Embedding(10, 8)])
        self.transformer = nn.Module()
        self.transformer.layers = nn.ModuleList([nn.Module()])
        self.transformer.layers[0].q_proj = nn.Linear(8, 8)
        self.audio_lm_heads = nn.ModuleList([nn.Linear(8, 10, bias=False)])
        self.audio_lm_heads[0].weight = self.audio_embeddings[0].weight      # tied, as in MOSS

    def forward(self, ids):
        h = self.transformer.layers[0].q_proj(self.audio_embeddings[0](ids))
        return self.audio_lm_heads[0](h)


def _adapter(folder, r=2, alpha=4, seed=0, modules=("transformer.layers.0.q_proj", "audio_lm_heads.0"),
             shapes={"transformer.layers.0.q_proj": (8, 8), "audio_lm_heads.0": (10, 8)}, **cfg):
    g = torch.Generator().manual_seed(seed)
    folder.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for m in modules:
        out_f, in_f = shapes[m]
        tensors[f"base_model.model.{m}.lora_A.weight"] = torch.randn(r, in_f, generator=g)
        tensors[f"base_model.model.{m}.lora_B.weight"] = torch.randn(out_f, r, generator=g)
    save_file(tensors, str(folder / lora.ADAPTER_FILE))
    (folder / lora.CONFIG_FILE).write_text(json.dumps({"peft_type": "LORA", "r": r, "lora_alpha": alpha,
                                                       "base_model_name_or_path": "tiny", **cfg}))
    return folder / lora.ADAPTER_FILE, tensors


def _merged_reference(model, adapters, ids):
    """What a merge would compute — on a deep copy with the tie broken."""
    ref = _hook_free_copy(model)
    ref.audio_lm_heads[0].weight = nn.Parameter(ref.audio_lm_heads[0].weight.detach().clone())
    mods = dict(ref.named_modules())
    for tensors, scale in adapters:
        for key, a in tensors.items():
            if key.endswith("lora_A.weight"):
                name = key[len("base_model.model."):-len(".lora_A.weight")]
                b = tensors[key.replace("lora_A", "lora_B")]
                mods[name].weight.data += scale * (b @ a)
    return ref(ids)


def test_unmerged_lora_matches_merge_and_never_touches_weights(tmp_path):
    torch.manual_seed(0)
    model, ids = _Tiny(), torch.tensor([[1, 2, 3]])
    emb_before = model.audio_embeddings[0].weight.detach().clone()
    base = model(ids)
    p1, t1 = _adapter(tmp_path / "voice", r=2, alpha=4, seed=1)
    p2, t2 = _adapter(tmp_path / "emo", r=4, alpha=4, seed=2)         # different ranks stack fine
    lora.apply(model, (("voice", str(p1), 1.0), ("emo", str(p2), 1.5)))
    got = model(ids)
    want = _merged_reference(model, [(t1, 4 / 2 * 1.0), (t2, 4 / 4 * 1.5)], ids)
    assert torch.allclose(got, want, atol=1e-4)
    assert torch.equal(model.audio_embeddings[0].weight, emb_before)          # embeddings untouched
    assert model.audio_lm_heads[0].weight is model.audio_embeddings[0].weight  # tie kept
    lora.apply(model, ())                                                     # removed: exact base output
    assert torch.equal(model(ids), base)


def test_apply_installs_exactly_the_stack(tmp_path):
    model, ids = _Tiny(), torch.tensor([[4, 5]])
    p1, t1 = _adapter(tmp_path / "a", seed=3)
    lora.apply(model, (("a", str(p1), 1.0),))
    handles = list(model._mosstts_lora_hooks)
    lora.apply(model, (("a", str(p1), 1.0),))                 # same stack: nothing re-installed
    assert model._mosstts_lora_hooks == handles
    lora.apply(model, (("a", str(p1), 0.5),))                 # new strength: replaced, not added
    assert len(model._mosstts_lora_hooks) == 2
    assert torch.allclose(model(ids), _merged_reference(model, [(t1, 2 * 0.5)], ids), atol=1e-4)
    lora.apply(model, (("a", str(p1), 0.0),))                 # zero strength = off
    assert model._mosstts_lora_hooks == []
    lora.clear(model)


def _hook_free_copy(model):
    import copy

    ref = copy.deepcopy(model)
    for m in ref.modules():
        m._forward_hooks.clear()
    return ref


def test_scaling_variants_and_bad_adapters(tmp_path):
    p, _ = _adapter(tmp_path / "rs", r=4, alpha=8, use_rslora=True, alpha_pattern={"q_proj": 2})
    w = lora.load(p)
    assert w.modules["audio_lm_heads.0"][2] == pytest.approx(8 / 2)          # alpha / sqrt(r)
    assert w.modules["transformer.layers.0.q_proj"][2] == pytest.approx(2 / 2)
    p, _ = _adapter(tmp_path / "dora", use_dora=True)
    with pytest.raises(ValueError, match="DoRA"):
        lora.load(p)
    p, _ = _adapter(tmp_path / "wrong", modules=("audio_lm_heads.0",), shapes={"audio_lm_heads.0": (12, 8)})
    with pytest.raises(ValueError, match="does not fit"):
        lora.apply(_Tiny(), (("wrong", str(p), 1.0),))
    p, _ = _adapter(tmp_path / "missing", modules=("nope.proj",), shapes={"nope.proj": (8, 8)})
    with pytest.raises(ValueError, match="1 missing"):
        lora.apply(_Tiny(), (("missing", str(p), 1.0),))


def test_list_local_and_catalog(tmp_path, monkeypatch):
    root = tmp_path / "models"
    _adapter(root / "mosstts_loras" / "my_voice")
    _adapter(root / "mosstts_loras" / "moss-va-sft3-emotion-loras" / "Anger")
    (root / "mosstts_loras" / "empty").mkdir()
    monkeypatch.delenv("MOSS_TTS_LORAS_DIR", raising=False)
    found = lora.list_local(str(root))
    assert set(found) == {"my_voice", "moss-va-sft3-emotion-loras/Anger"}
    hf = lora.hf_choices()
    voices = [k for k in hf if k.startswith("LAION voice / ")]
    assert len(voices) == 500 and len([k for k in hf if k.startswith("LAION emotion / ")]) == 40
    emma = hf["LAION voice / Emma (female, EN) · anime_000"]
    assert emma["speaker"] == "Emma" and emma["subfolder"] == "anime_000" and emma["revision"]
    assert hf["LAION DPO quality"]["subfolder"] == ""


def test_speaker_name_mode_prompt():
    """`Speaker: <name>` replaces `None` under Reference(s) of the vendored
    template — and only there, only without reference audio."""
    from moss_tts import native, runtime

    native._register_asset_package("moss_tts_local")
    from _mossttsv15_moss_tts_local.processing_moss_tts import UserMessage

    msg = UserMessage(text="hello there", instruction="GENERAL: calm", tokens=20, language="English").to_dict()
    out = runtime._with_speaker(SimpleNamespace(speaker="Emma"), msg)
    assert "<user_inst>\n- Reference(s):\nSpeaker: Emma\n- Instruction:\nGENERAL: calm" in out["content"]
    assert msg["content"].count("Speaker:") == 0                              # original untouched
    assert runtime._with_speaker(SimpleNamespace(speaker=""), msg) is msg
    clone = UserMessage(text="hi", reference=[torch.zeros(5, 12, dtype=torch.long)]).to_dict()
    assert runtime._with_speaker(SimpleNamespace(speaker="Emma"), clone) is clone   # audio reference wins


def test_lora_node_stacks_on_a_copy(tmp_path, monkeypatch):
    import moss_tts.nodes as nodes
    from moss_tts.loader import MossTTSBundle

    root = tmp_path / "models"
    _adapter(root / "mosstts_loras" / "mine")
    monkeypatch.delenv("MOSS_TTS_LORAS_DIR", raising=False)
    monkeypatch.setattr(nodes.compat, "comfy_models_dir", lambda: str(root))
    fetched = []
    real_fetch = nodes.adapters.fetch_hf
    monkeypatch.setattr(nodes.adapters, "fetch_hf",
                        lambda repo, sub, rev=None, local_root=None: fetched.append((repo, sub, local_root))
                        or root / "mosstts_loras" / "mine" / lora.ADAPTER_FILE)
    base = MossTTSBundle(spec=SimpleNamespace(key="laion_sft3"), model=nn.Linear(1, 1), processor=None,
                         codec=nn.Linear(1, 1), model_dir=tmp_path, codec_dir=tmp_path,
                         device=torch.device("cpu"), torch_dtype=torch.float32, dtype_name="fp32",
                         attn_implementation="sdpa")
    node = nodes.MossTTSV15LoRA()
    assert "LAION voice / Emma (female, EN) · anime_000" in node.INPUT_TYPES()["required"]["lora"][0]
    (a,) = node.apply(base, "mine", 0.8)
    (b,) = node.apply(a, "LAION voice / Emma (female, EN) · anime_000", 1.0)
    assert base.loras == () and base.speaker == ""                            # the loaded bundle untouched
    assert [x[0] for x in b.loras] == ["mine", "LAION voice / Emma (female, EN) · anime_000"]
    assert b.loras[0][2] == 0.8 and b.speaker == "Emma" and b.model is base.model
    assert fetched == [("laion/moss-va-sft3-voice-loras", "anime_000", root / "mosstts_loras")]
    (c,) = node.apply(b, "none", 1.0, speaker_name="none")
    assert c.speaker == "" and c.loras == b.loras
    (d,) = node.apply(base, "mine", 1.0, speaker_name="Marlene")
    assert d.speaker == "Marlene"
    # a LAION adapter already in mosstts_loras (downloaded earlier, or by hand) is listed once,
    # under its LAION name, and used from disk
    _adapter(root / "mosstts_loras" / "moss-va-sft3-voice-loras" / "anime_016")
    monkeypatch.setattr(nodes.adapters, "fetch_hf", real_fetch)
    choices = node.INPUT_TYPES()["required"]["lora"][0]
    assert "moss-va-sft3-voice-loras/anime_016" not in choices and "mine" in choices
    (e,) = node.apply(base, "LAION voice / James (male, EN) · anime_016", 1.0)
    assert e.speaker == "James"
    assert e.loras[-1][1] == str(root / "mosstts_loras" / "moss-va-sft3-voice-loras" / "anime_016" / lora.ADAPTER_FILE)
    _adapter(tmp_path / "loose" / "anime_031")                   # outside the LAION repo layout
    assert lora.known_speaker("anime_031") == "Olivia"
    with pytest.raises(FileNotFoundError):
        node.apply(base, "gone", 1.0)


def test_ensure_keeps_overlays(monkeypatch):
    import dataclasses

    from moss_tts import loader

    model, codec = nn.Linear(1, 1), nn.Linear(1, 1)
    active = loader.MossTTSBundle(spec=SimpleNamespace(key="laion_sft3"), model=model, processor=None,
                                  codec=codec, model_dir=None, codec_dir=None, device=torch.device("cpu"),
                                  torch_dtype=torch.float32, dtype_name="fp32", attn_implementation="sdpa")
    copy = dataclasses.replace(active, loras=(("x", "/p", 1.0),), speaker="Emma")
    monkeypatch.setattr(loader, "_ACTIVE_BUNDLE", active)
    assert loader.ensure_mosstts_bundle(copy) is copy                         # a LoRA copy is live, no reload
    monkeypatch.setattr(loader, "_ACTIVE_BUNDLE", None)                       # unloaded meanwhile
    fresh = dataclasses.replace(active)
    monkeypatch.setattr(loader, "load_mosstts_bundle", lambda *a, **k: fresh)
    back = loader.ensure_mosstts_bundle(copy)
    assert back.loras == copy.loras and back.speaker == "Emma" and back.model is fresh.model


def test_fetch_hf_downloads_into_the_comfy_folder(tmp_path, monkeypatch):
    import huggingface_hub

    calls = []

    def fake_download(repo, filename, subfolder=None, revision=None, local_dir=None):
        dest = Path(local_dir) / (subfolder or "") / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("{}")
        calls.append((repo, filename, subfolder, local_dir))
        return str(dest)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    root = tmp_path / "mosstts_loras"
    got = lora.fetch_hf("laion/moss-va-sft3-emotion-loras", "Anger", "abc", root)
    assert got == root / "moss-va-sft3-emotion-loras" / "Anger" / lora.ADAPTER_FILE
    assert [c[1] for c in calls] == [lora.CONFIG_FILE, lora.ADAPTER_FILE]
    assert calls[0][3] == str(root / "moss-va-sft3-emotion-loras")
    lora.fetch_hf("laion/moss-va-sft3-emotion-loras", "Anger", "abc", root)     # present: no download
    assert len(calls) == 2
    marker = root / "moss-va-sft3-emotion-loras" / ".cache" / "huggingface" / "download" / "x.incomplete"
    marker.parent.mkdir(parents=True)
    marker.write_text("")
    lora.fetch_hf("laion/moss-va-sft3-emotion-loras", "Anger", "abc", root)     # interrupted: resumes
    assert len(calls) == 4
    monkeypatch.setenv("MOSS_TTS_LORAS_DIR", str(tmp_path / "env"))
    assert lora.download_root("/comfy/models") == tmp_path / "env"
    monkeypatch.delenv("MOSS_TTS_LORAS_DIR")
    assert lora.download_root("/comfy/models") == Path("/comfy/models/mosstts_loras")
    assert lora.download_root(None) is None
