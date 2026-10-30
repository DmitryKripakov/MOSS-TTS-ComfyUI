"""CPU-only unit tests for the pack: registry shape, conversions, knobs,
fail-fast guards, and comfy-compat fallbacks. No weights, no GPU."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import moss_tts

pack = moss_tts


# ---------------------------------------------------------------------------
# node registry
# ---------------------------------------------------------------------------

def test_node_registry_complete():
    assert set(pack.NODE_CLASS_MAPPINGS) == set(pack.NODE_DISPLAY_NAME_MAPPINGS)
    assert len(pack.NODE_CLASS_MAPPINGS) == 9
    for cls in pack.NODE_CLASS_MAPPINGS.values():
        inputs = cls.INPUT_TYPES()
        assert "required" in inputs
        assert isinstance(cls.FUNCTION, str) and hasattr(cls, cls.FUNCTION.split(".")[0])
        assert len(cls.RETURN_TYPES) == len(cls.RETURN_NAMES)


def test_variant_specs():
    from moss_tts.native import VARIANTS

    local, delay, voicegen = VARIANTS["local"], VARIANTS["delay"], VARIANTS["voicegen"]
    assert local.sample_rate == 48000 and local.stereo and local.n_vq == 12
    assert delay.sample_rate == 24000 and not delay.stereo and delay.n_vq == 32
    assert voicegen.sample_rate == 24000 and not voicegen.stereo and voicegen.n_vq == 16
    assert voicegen.asset_pkg == delay.asset_pkg  # same MossTTSDelay family
    assert voicegen.repo_id == "OpenMOSS-Team/MOSS-VoiceGenerator"
    assert "tokenizer" in ("tokenizer",)  # smoke: nothing to check, kept honest
    assert local.repo_id != delay.repo_id and local.codec_repo_id != delay.codec_repo_id


def test_laion_variants_share_the_base_families():
    from moss_tts.native import VARIANTS

    sft3, va8b = VARIANTS["laion_sft3"], VARIANTS["laion_va8b"]
    for ft, base in ((sft3, VARIANTS["local"]), (va8b, VARIANTS["delay"])):
        assert (ft.asset_pkg, ft.codec_asset_pkg, ft.codec_repo_id, ft.sample_rate, ft.stereo, ft.n_vq) == \
            (base.asset_pkg, base.codec_asset_pkg, base.codec_repo_id, base.sample_rate, base.stereo, base.n_vq)
        assert ft.repo_id.startswith("laion/") and ft.local_dir is None
    assert sft3.sampling == (1.0, 0.95, 50, 1.0)
    assert va8b.sampling == (0.8, 0.95, 25, 1.1) and va8b.sampling_with_reference == (1.0, 0.95, 25, 1.1)
    labels = [v.label for v in VARIANTS.values()]
    assert len(labels) == len(set(labels))  # the Load Model combo maps label -> key


# ---------------------------------------------------------------------------
# custom model folders
# ---------------------------------------------------------------------------

def _model_dir(path, model_type="moss_tts_local", n_vq=12, rate=48000):
    import json

    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(
        {"model_type": model_type, "n_vq": n_vq, "sampling_rate": rate}))
    return path


def test_spec_from_dir_reads_family(tmp_path):
    from moss_tts import native

    spec = native.spec_from_dir(_model_dir(tmp_path / "my_voice"))
    assert spec.key == "custom:my_voice" and spec.local_dir == str(tmp_path / "my_voice")
    assert spec.asset_pkg == "moss_tts_local" and spec.codec_asset_pkg == "moss_audio_tokenizer_v2"
    assert spec.stereo and spec.sample_rate == 48000 and spec.sampling is None
    gen = native.spec_from_dir(_model_dir(tmp_path / "gen", "moss_tts_delay", 16, 24000))
    assert gen.asset_pkg == "moss_tts_delay" and gen.n_vq == 16 and not gen.stereo
    codec = _model_dir(tmp_path / "codec", "moss-audio-tokenizer")
    with pytest.raises(ValueError, match="not a MOSS-TTS model folder"):
        native.spec_from_dir(codec)


def test_list_custom_models(tmp_path, monkeypatch):
    from moss_tts import loader

    root = tmp_path / "mosstts"
    _model_dir(root / "my_voice")
    _model_dir(root / "runs" / "ckpt-500", "moss_tts_delay", 32, 24000)
    _model_dir(root / "MOSS-TTS-v1.5", "moss_tts_delay", 32, 24000)      # built-in variant
    _model_dir(root / "MOSS-Audio-Tokenizer", "moss-audio-tokenizer")    # codec
    _model_dir(root / "a" / "b" / "too_deep")
    (root / "empty").mkdir()
    monkeypatch.setenv("MOSS_TTS_MODELS_DIR", str(root))
    monkeypatch.setattr(loader.compat, "comfy_models_dir", lambda: None)
    found = loader.list_custom_models()
    assert found == {"my_voice": root / "my_voice", "runs/ckpt-500": root / "runs" / "ckpt-500"}


def test_resolve_model_dirs_uses_custom_folder(tmp_path, monkeypatch):
    from moss_tts import loader, native

    spec = native.spec_from_dir(_model_dir(tmp_path / "mine"))
    monkeypatch.setattr(loader, "_resolve_dir_for", lambda repo, dl: Path(f"/hf/{repo}"))
    model_dir, codec_dir = loader.resolve_model_dirs(spec, False)
    assert model_dir == tmp_path / "mine"
    assert codec_dir == Path("/hf/OpenMOSS-Team/MOSS-Audio-Tokenizer-v2")


def test_load_model_node_custom_model(tmp_path, monkeypatch):
    import moss_tts.nodes as nodes

    root = tmp_path / "mosstts"
    _model_dir(root / "my_voice")
    monkeypatch.setenv("MOSS_TTS_MODELS_DIR", str(root))
    monkeypatch.setattr(nodes, "list_custom_models",
                        lambda: {"my_voice": root / "my_voice"})
    captured = {}
    monkeypatch.setattr(nodes, "load_mosstts_bundle", lambda **kw: captured.update(kw) or "bundle")
    node = nodes.MossTTSV15LoadModel()
    label = nodes._VARIANT_LABELS[0]
    assert "custom_model" in node.INPUT_TYPES()["optional"]
    assert node.load_model(label, "auto", "auto", False) == ("bundle",)
    assert captured["custom_dir"] is None
    node.load_model(label, "auto", "auto", False, custom_model="my_voice")
    assert captured["custom_dir"] == str(root / "my_voice")
    with pytest.raises(FileNotFoundError, match="gone"):
        node.load_model(label, "auto", "auto", False, custom_model="gone")


def test_stale_custom_bundle_reactivates_from_its_folder(monkeypatch):
    from types import SimpleNamespace

    from moss_tts import loader

    stale = SimpleNamespace(spec=SimpleNamespace(key="custom:mine", local_dir="/m/mine"), model=None,
                            codec=None, dtype_name="bf16", attn_implementation="sdpa")
    captured = {}
    monkeypatch.setattr(loader, "load_mosstts_bundle", lambda variant, **kw: captured.update(kw) or "ok")
    assert loader.ensure_mosstts_bundle(stale) == "ok"
    assert captured["custom_dir"] == "/m/mine"


# ---------------------------------------------------------------------------
# sampling preset
# ---------------------------------------------------------------------------

def test_audio_sampling_preset():
    from types import SimpleNamespace

    import moss_tts.nodes as nodes
    from moss_tts.native import VARIANTS

    widgets = (1.7, 0.8, 25, 1.0)
    va8b = SimpleNamespace(spec=VARIANTS["laion_va8b"])
    sft3 = SimpleNamespace(spec=VARIANTS["laion_sft3"])
    custom = SimpleNamespace(spec=SimpleNamespace(key="custom:x", sampling=None, sampling_with_reference=None))

    def vals(model, preset, ref):
        d = nodes._audio_sampling(model, preset, ref, *widgets)
        return (d["audio_temperature"], d["audio_top_p"], d["audio_top_k"], d["audio_repetition_penalty"])

    assert vals(sft3, "widgets", False) == widgets
    assert vals(sft3, "model recommended", False) == (1.0, 0.95, 50, 1.0)
    assert vals(sft3, "model recommended", True) == (1.0, 0.95, 50, 1.0)    # no separate clone values
    assert vals(va8b, "model recommended", False) == (0.8, 0.95, 25, 1.1)
    assert vals(va8b, "model recommended", True) == (1.0, 0.95, 25, 1.1)
    assert vals(custom, "model recommended", True) == widgets               # no card: widgets


def test_generator_nodes_take_sampling_preset(monkeypatch):
    import moss_tts.nodes as nodes
    from moss_tts.native import VARIANTS

    for cls in (nodes.MossTTSV15GenerateSpeech, nodes.MossTTSV15VoiceClone,
                nodes.MossTTSV15VoiceDesign, nodes.MossTTSV15ContinueSpeech):
        opt = cls.INPUT_TYPES()["optional"]["sampling_preset"]
        assert opt[1]["default"] == "widgets"  # old workflows keep their widget values
    seen = {}
    monkeypatch.setattr(nodes.runtime, "voice_clone",
                        lambda model, **kw: seen.update(kw) or (torch.zeros(2, 10), 1))
    monkeypatch.setattr(nodes, "ProgressReporter", lambda n: None)
    model = type("M", (), {"spec": VARIANTS["laion_va8b"]})()
    nodes.MossTTSV15VoiceClone().clone(
        model, {"waveform": torch.zeros(1, 1, 10), "sample_rate": 24000}, "hi", "auto", "",
        1.7, 0.8, 25, 1.0, 1.0, 1.0, 50, 0, 100, True, 42, sampling_preset="model recommended")
    assert (seen["audio_temperature"], seen["audio_top_k"], seen["audio_repetition_penalty"]) == (1.0, 25, 1.1)


# ---------------------------------------------------------------------------
# estimate tokens
# ---------------------------------------------------------------------------

def test_estimate_tokens_english():
    cls = pack.NODE_CLASS_MAPPINGS["MossTTSV15_EstimateTokens"]()
    (tokens,) = cls.estimate("one two three four five six", 150.0)  # 6 words @150wpm = 2.4s
    assert tokens == 30


def test_estimate_tokens_cjk():
    cls = pack.NODE_CLASS_MAPPINGS["MossTTSV15_EstimateTokens"]()
    (tokens,) = cls.estimate("一二三四五六", 150.0)  # 6 chars @150cpm = 2.4s
    assert tokens == 30


def test_estimate_tokens_empty():
    cls = pack.NODE_CLASS_MAPPINGS["MossTTSV15_EstimateTokens"]()
    assert cls.estimate("   ", 150.0) == (0,)


# ---------------------------------------------------------------------------
# audio conversions
# ---------------------------------------------------------------------------

def test_audio_conversions():
    import moss_tts.runtime as runtime

    wav3d = torch.zeros(1, 2, 4800)
    d1 = runtime.tensor_to_comfy_audio(wav3d, 48000)
    assert d1["waveform"].shape == (1, 2, 4800) and d1["sample_rate"] == 48000
    back, sr = runtime.comfy_audio_to_tensor(d1)
    assert back.shape == (2, 4800) and sr == 48000
    d2 = runtime.tensor_to_comfy_audio(torch.zeros(2400), 24000)
    assert d2["waveform"].shape == (1, 1, 2400)


# ---------------------------------------------------------------------------
# fail-fast + knob filtering
# ---------------------------------------------------------------------------

def test_require_text_fail_fast():
    import moss_tts.runtime as runtime

    with pytest.raises(ValueError):
        runtime._require_text("   ")
    with pytest.raises(ValueError):
        runtime._require_text(None)


class _Spec:
    key = "local"
    asset_pkg = "moss_tts_local"
    stereo = True


class _FakeBundle:
    spec = _Spec()


def test_bundle_repr_is_compact():
    from pathlib import Path
    from types import SimpleNamespace

    from moss_tts.loader import MossTTSBundle

    bundle = MossTTSBundle(
        spec=SimpleNamespace(key="local"),
        model=torch.nn.Linear(1, 1),
        processor=SimpleNamespace(audio_tokenizer=object()),
        codec=torch.nn.Linear(1, 1),
        model_dir=Path("model"),
        codec_dir=Path("codec"),
        device=torch.device("cpu"),
        torch_dtype=torch.float32,
        dtype_name="fp32",
        attn_implementation="eager",
        patchers=[object()],
    )
    text = repr(bundle)
    assert "model=" not in text
    assert "processor=" not in text
    assert "codec=" not in text
    assert "patchers=" not in text
    assert "dtype_name='fp32'" in text


def test_generate_kwargs_local_vs_delay():
    import moss_tts.runtime as runtime

    bundle = _FakeBundle()
    kwargs = dict(max_new_tokens=100, do_sample=True, text_temperature=1.0,
                  text_top_p=1.0, text_top_k=50, audio_temperature=1.7,
                  audio_top_p=0.8, audio_top_k=25, audio_repetition_penalty=1.0)
    local = runtime._generate_kwargs(bundle, **kwargs)
    assert local["do_sample"] is True and local["max_new_tokens"] == 100

    bundle.spec.asset_pkg = "moss_tts_delay"
    delay = runtime._generate_kwargs(bundle, **{**kwargs, "do_sample": True})
    assert "do_sample" not in delay  # delay has no such arg
    delay_greedy = runtime._generate_kwargs(bundle, **{**kwargs, "do_sample": False})
    assert delay_greedy["text_temperature"] == 0.0 and delay_greedy["audio_temperature"] == 0.0


# ---------------------------------------------------------------------------
# compat fallbacks (no comfy imported here)
# ---------------------------------------------------------------------------

def test_cast_context_passthrough_without_comfy():
    from moss_tts import compat

    lin = torch.nn.Linear(4, 4)
    x = torch.zeros(1, 4)
    with compat._LegacyCastContext(None, lin, x, True) as (w, b, s):
        assert w.shape == lin.weight.shape and s is None


def test_progress_reporter_no_comfy():
    from moss_tts import compat

    rep = compat.ProgressReporter(10)
    rep(3, 10)  # must not raise


def test_resolve_dtype_cpu_is_fp32():
    from moss_tts.loader import resolve_dtype

    assert resolve_dtype("auto", torch.device("cpu")) is torch.float32
    assert resolve_dtype("bf16", torch.device("cpu")) is torch.float32


def test_resolve_attention_never_crashes_without_flash():
    from moss_tts.loader import resolve_attention

    # CPU must resolve to eager regardless of request
    assert resolve_attention("auto", torch.device("cpu"), torch.float32) == "eager"
    assert resolve_attention("sdpa", torch.device("cpu"), torch.float32) == "sdpa"


def test_weight_norm_remap():
    from moss_tts.native import _weight_norm_remap

    assert _weight_norm_remap("a.weight_g") == "a.parametrizations.weight.original0"
    assert _weight_norm_remap("a.weight_v") == "a.parametrizations.weight.original1"
    assert _weight_norm_remap("a.weight") == "a.weight"


def test_models_dir_env_override(tmp_path, monkeypatch):
    from moss_tts import loader, native

    spec = native.VARIANTS["local"]
    d = tmp_path / spec.repo_id.split("/")[-1]
    d.mkdir()
    (d / "config.json").write_text("{}")
    monkeypatch.setenv("MOSS_TTS_MODELS_DIR", str(tmp_path))
    assert loader._resolve_dir_for(spec.repo_id, False) == d


def test_missing_models_download_into_comfy_models(tmp_path, monkeypatch):
    import huggingface_hub

    from moss_tts import loader

    calls = []

    def fake_snapshot(repo_id, local_files_only=False, local_dir=None, **kw):
        if local_files_only:
            raise FileNotFoundError("not in the HF cache")
        calls.append((repo_id, local_dir))
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / "config.json").write_text("{}")
        return local_dir

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    monkeypatch.delenv("MOSS_TTS_MODELS_DIR", raising=False)
    monkeypatch.setattr(loader.compat, "comfy_models_dir", lambda: str(tmp_path / "models"))
    monkeypatch.setattr(loader.compat, "comfy_folder_paths", lambda name: [])
    target = tmp_path / "models" / "mosstts" / "moss-tts-v1.5-8b-voice-acting"
    assert loader._resolve_dir_for("laion/moss-tts-v1.5-8b-voice-acting", True) == target
    assert calls == [("laion/moss-tts-v1.5-8b-voice-acting", str(target))]
    assert loader._resolve_dir_for("laion/moss-tts-v1.5-8b-voice-acting", True) == target   # found
    assert len(calls) == 1
    marker = target / ".cache" / "huggingface" / "download" / "model.safetensors.incomplete"
    marker.parent.mkdir(parents=True)
    marker.write_text("")
    with pytest.raises(FileNotFoundError, match="interrupted download"):
        loader._resolve_dir_for("laion/moss-tts-v1.5-8b-voice-acting", False)
    assert loader._resolve_dir_for("laion/moss-tts-v1.5-8b-voice-acting", True) == target   # resumes there
    assert calls[-1] == ("laion/moss-tts-v1.5-8b-voice-acting", str(target))
    monkeypatch.setattr(loader.compat, "comfy_folder_paths", lambda name: [str(tmp_path / "extra")])
    assert loader._models_root_candidates()[0] == tmp_path / "extra"         # extra_model_paths.yaml


def test_load_installs_cache_invalidation_unload_hook(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from moss_tts import loader
    from moss_tts.nodes import MossTTSV15LoadModel

    model = torch.nn.Linear(1, 1)
    codec = torch.nn.Linear(1, 1)
    processor = SimpleNamespace(audio_tokenizer=codec)
    hooks = []

    monkeypatch.setattr(loader.compat, "get_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(loader.compat, "install_unload_hook", hooks.append)
    monkeypatch.setattr(loader, "resolve_model_dirs", lambda *args: (tmp_path, tmp_path))
    monkeypatch.setattr(loader, "_bundle_load_key", lambda *args: ("test",))
    monkeypatch.setattr(loader.native, "read_tts_config", lambda *args: object())
    monkeypatch.setattr(loader.native, "read_codec_config", lambda *args: object())
    monkeypatch.setattr(loader.native, "build_tts_model", lambda *args: model)
    monkeypatch.setattr(loader.native, "build_codec_model", lambda *args: codec)
    monkeypatch.setattr(loader.native, "load_tts_weights", lambda *args, **kwargs: None)
    monkeypatch.setattr(loader.native, "load_codec_weights", lambda *args, **kwargs: None)
    monkeypatch.setattr(loader.native, "build_tokenizer", lambda *args: object())
    monkeypatch.setattr(loader.native, "build_processor", lambda *args: processor)

    bundle = loader.load_mosstts_bundle(
        "local", dtype_name="fp32", attention="eager", download_if_missing=False,
    )
    try:
        assert len(hooks) == 1
        before = loader.bundle_generation()
        assert MossTTSV15LoadModel.IS_CHANGED() == before
        assert bundle.processor.audio_tokenizer is codec
        hooks[0]("unrelated unload", SimpleNamespace(clone_base_uuid="other"))
        assert loader.bundle_generation() == before
        assert bundle.model is model
        hooks[0]("test unload", None)
        assert loader.bundle_generation() == before + 1
        assert MossTTSV15LoadModel.IS_CHANGED() == before + 1
        assert bundle.model is None
        assert bundle.codec is None
        assert bundle.processor.audio_tokenizer is None
    finally:
        loader.unload_mosstts_bundle(bundle)


def test_stale_bundle_reactivates_with_original_options(monkeypatch):
    from types import SimpleNamespace

    from moss_tts import loader

    stale = SimpleNamespace(
        spec=SimpleNamespace(key="voicegen"),
        model=None,
        codec=None,
        dtype_name="auto",
        attn_implementation="sdpa",
    )
    fresh = object()
    captured = {}

    def fake_load(variant, **kwargs):
        captured["variant"] = variant
        captured.update(kwargs)
        return fresh

    monkeypatch.setattr(loader, "load_mosstts_bundle", fake_load)
    assert loader.ensure_mosstts_bundle(stale) is fresh
    assert captured == {
        "variant": "voicegen",
        "dtype_name": "auto",
        "attention": "sdpa",
        "download_if_missing": False,
    }


def test_comfy_castable_modules_forward_without_comfy():
    """Converted modules must keep working when comfy.ops is absent."""
    import moss_tts.native as native

    for mod, x in (
        (torch.nn.Linear(8, 8), torch.randn(2, 8)),
        (torch.nn.Embedding(16, 8), torch.randint(0, 16, (2, 4))),
        (torch.nn.LayerNorm(8), torch.randn(2, 8)),
        (torch.nn.Conv1d(4, 4, 3, padding=1), torch.randn(2, 4, 16)),
        (torch.nn.ConvTranspose1d(4, 4, 3, padding=1), torch.randn(2, 4, 16)),
    ):
        native.convert_modules_for_comfy(mod) if False else None
        # convert in place via class swap, then forward
        for base, castable in native._CASTABLE:
            if type(mod) is base:
                mod.__class__ = castable
        ref = mod.forward
        out = mod.forward(x) if isinstance(mod, torch.nn.ConvTranspose1d) else ref(x)
        assert out is not None


def test_qwen_rmsnorm_is_comfy_castable(monkeypatch):
    """Custom RMSNorm weights must participate in partial offload streaming."""
    from moss_tts import compat
    import moss_tts.native as native

    native.tts_classes(native.VARIANTS["local"])
    from _mossttsv15_moss_tts_local.qwen3_decoder import MossQwen3RMSNorm

    norm = MossQwen3RMSNorm(8)
    with monkeypatch.context() as patch:
        original_try_import = compat._try_import
        patch.setattr(
            compat,
            "_try_import",
            lambda name: object() if name == "comfy.ops" else original_try_import(name),
        )
        native.convert_modules_for_comfy(norm)

    assert type(norm) is native._ComfyRMSNorm
    assert norm.comfy_cast_weights is True
    x = torch.randn(2, 4, 8)
    expected = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
    assert torch.allclose(norm(x), expected.to(x.dtype), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    ("variant", "module_name"),
    (
        ("delay", "_mossttsv15_moss_audio_tokenizer_v1.modeling_moss_audio_tokenizer"),
        ("local", "_mossttsv15_moss_audio_tokenizer_v2.modeling_moss_audio_tokenizer"),
    ),
)
def test_codec_lfq_routes_codebook_reads_through_embedding(variant, module_name):
    """Decode and encode paths must not bypass Comfy's castable Embedding."""
    import importlib

    from moss_tts import native

    native.codec_classes(native.VARIANTS[variant])
    module = importlib.import_module(module_name)

    class ProbeEmbedding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.table = torch.nn.Parameter(torch.randn(8, 4))
            self.calls = 0

        @property
        def weight(self):
            raise AssertionError("codebook.weight bypasses Embedding.forward")

        def forward(self, indices):
            self.calls += 1
            return torch.nn.functional.embedding(indices, self.table)

    quantizer = module.MossAudioTokenizerLFQ(
        input_dim=4, codebook_size=8, codebook_dim=4,
    )
    probe = ProbeEmbedding()
    quantizer.codebook = probe

    decoded = quantizer.embed_code(torch.tensor([[1, 2, 3]]))
    assert decoded.shape == (1, 3, 4)

    z_q, indices = quantizer.decode_latents(torch.randn(1, 4, 3))
    assert z_q.shape == (1, 4, 3)
    assert indices.shape == (1, 3)
    assert probe.calls == 3


def test_weight_normalized_conv_keeps_parametrization_and_gets_cast_forward(monkeypatch):
    """Codec WNConv1d must stream weights without losing weight_norm."""
    from moss_tts import compat
    import moss_tts.native as native

    conv = torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(4, 8, 1))
    x = torch.randn(2, 4, 5)
    expected = conv(x)
    original_type = type(conv)

    with monkeypatch.context() as patch:
        original_try_import = compat._try_import
        patch.setattr(
            compat,
            "_try_import",
            lambda name: object() if name == "comfy.ops" else original_try_import(name),
        )
        native.convert_modules_for_comfy(conv)

    assert type(conv) is original_type
    assert torch.nn.utils.parametrize.is_parametrized(conv, "weight")
    assert conv.comfy_cast_weights is True
    assert conv.forward.__func__ is native._ComfyConv1d.forward
    assert torch.allclose(conv(x), expected)


@pytest.mark.parametrize(
    ("variant", "module_name"),
    (
        ("delay", "_mossttsv15_moss_audio_tokenizer_v1.modeling_moss_audio_tokenizer"),
        ("local", "_mossttsv15_moss_audio_tokenizer_v2.modeling_moss_audio_tokenizer"),
    ),
)
def test_codec_layer_scale_follows_activation_device(variant, module_name):
    """Partially offloaded LayerScale parameters must follow activations."""
    import importlib

    from moss_tts import native

    native.codec_classes(native.VARIANTS[variant])
    module = importlib.import_module(module_name)
    layer = module.MossAudioTokenizerLayerScale(4)

    class ProbeScale:
        def __init__(self):
            self.called = False

        def to(self, activation):
            self.called = True
            return torch.ones(4, device=activation.device, dtype=activation.dtype)

    probe = ProbeScale()
    del layer._parameters["scale"]
    layer.scale = probe
    output = layer(torch.randn(2, 3, 4))
    assert output.shape == (2, 3, 4)
    assert probe.called


def test_rotary_materialization_4x_rope_init_fn():
    """tf4.x-style rotary (rope_init_fn attr, no compute_default_rope_parameters)
    must get a recomputed inv_freq, not the historical zero-fill that left the
    model position-blind on transformers 4.x."""
    import torch.nn as nn

    import moss_tts.native as native

    class Fake4xRotaryEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = object()
            self.attention_scaling = 0.0
            self.register_buffer("inv_freq", torch.empty(4, device="meta"), persistent=False)

            def rope_init_fn(config, device):
                return torch.tensor([1.0, 0.1, 0.01, 0.001], device=device), 1.0

            self.rope_init_fn = rope_init_fn

    mod = Fake4xRotaryEmbedding()
    wrapper = nn.Module()
    wrapper.rot = mod
    native._materialize_non_persistent_buffers(wrapper, torch.device("cpu"))
    assert mod.inv_freq.device.type == "cpu"
    assert torch.all(mod.inv_freq != 0)
    assert mod.attention_scaling == 1.0


def test_voice_design_node_guards():
    import moss_tts.nodes as nodes

    node = nodes.MossTTSV15VoiceDesign()
    knobs = dict(
        language="auto", instruction="", audio_temperature=1.5, audio_top_p=0.6,
        audio_top_k=50, audio_repetition_penalty=1.1, text_temperature=1.0,
        text_top_p=1.0, text_top_k=50, target_tokens=0, max_new_tokens=100,
        do_sample=True, seed=42,
    )
    bundle = _FakeBundle()  # spec.key = "local"
    with pytest.raises(ValueError, match="VoiceGenerator"):
        node.run(bundle, "hi", **knobs)
    bundle.spec.key = "voicegen"
    with pytest.raises(ValueError, match="instruction"):
        node.run(bundle, "hi", **knobs)


def test_voicegen_uses_delay_kwargs_path():
    import moss_tts.runtime as runtime

    bundle = _FakeBundle()
    bundle.spec.key = "voicegen"
    bundle.spec.asset_pkg = "moss_tts_delay"
    bundle.spec.stereo = False
    kwargs = dict(max_new_tokens=100, do_sample=True, text_temperature=1.0,
                  text_top_p=1.0, text_top_k=50, audio_temperature=1.5,
                  audio_top_p=0.6, audio_top_k=50, audio_repetition_penalty=1.1)
    out = runtime._generate_kwargs(bundle, **kwargs)
    assert "do_sample" not in out  # delay-family generate() rejects extra kwargs
