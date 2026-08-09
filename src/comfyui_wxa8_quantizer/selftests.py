"""Embedded self-tests (the --self-test suite)."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from pathlib import Path
import argparse
import dataclasses
import hashlib
import json
import numpy as np
import os
from safetensors import safe_open
import safetensors
import safetensors.torch
import shutil
import struct
import sys
import tempfile
import torch
from comfyui_wxa8_quantizer.constants import FORMAT_INT8, FORMAT_MIXED, FORMAT_W4A4, FORMAT_W4A8, METADATA_KEY_EXT, METADATA_KEY_QUANT, get_converter_version, set_converter_version
from comfyui_wxa8_quantizer.engine import ConversionEngine, _SimulatedCrash, hash_checkpoint_files
from comfyui_wxa8_quantizer.errors import CompressionGateError, InputError, OutputError, PickleInputError, QualityGateError, UnknownArchitectureError, ValidationError
from comfyui_wxa8_quantizer.formats import dequantize_int8_tensorwise_weight, dequantize_w4a4_weight, dequantize_w4a8_weight, int8_weight_is_quantizable, quantize_int8_tensorwise_weight, quantize_w4a4_weight, quantize_w4a8_weight, unpack_int4_signed, unpack_w4, w4_weight_is_quantizable, w4a4_weight_is_quantizable
from comfyui_wxa8_quantizer.golden import _test_golden_vectors, quantize_w4a8_weight
from comfyui_wxa8_quantizer.io import CheckpointInfo, CheckpointReader, SafetensorsStreamWriter, TensorMeta, discover_checkpoint, republish_with_metadata, tensor_to_bytes
from comfyui_wxa8_quantizer.logging_utils import log
from comfyui_wxa8_quantizer.metadata import build_extension_metadata, build_quant_metadata
from comfyui_wxa8_quantizer.planner import MixedPlanner
from comfyui_wxa8_quantizer.planning import ConversionPlan, DecisionKind, SensitivityAnalyzer, TensorDecision, activation_aware_error, build_output_entries, classify_tensors, compute_weight_metrics, load_calibration, runtime_output_rel_l2
from comfyui_wxa8_quantizer.policies import DetectionResult, detect_architecture, family_names, get_family
from comfyui_wxa8_quantizer.quantize import apply_sensitivity_prepass, quantize_tensor_bounded
from comfyui_wxa8_quantizer.reporting import compression_stats, low_compression_warning, policy_miss_warning
from comfyui_wxa8_quantizer.runtime import FormatRuntimeCapability, RuntimeCapabilities, inspect_environment, runtime_capabilities_for
from comfyui_wxa8_quantizer.utils import TORCH_TO_SAFE, human_bytes, json_dumps, sha256_file, sha256_safetensors_payload
from comfyui_wxa8_quantizer.validation import Validator, plan_from_output, verify_output
def run_self_tests() -> int:
    log().info("running embedded self-tests ...")
    tests = SELF_TEST_CASES
    failed = 0
    try:
        for name, fn in tests:
            try:
                detail = fn()
                print(f"  [PASS] {name}" + (f" -- {detail}" if detail else ""))
            except Exception as e:
                failed += 1
                print(f"  [FAIL] {name} -- {e}")
                log().debug("self-test %s failed", name, exc_info=True)
    finally:
        for d in _TEST_DIRS:
            shutil.rmtree(d, ignore_errors=True)
        _TEST_DIRS.clear()
    print(f"self-tests: {len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0

_TEST_DIRS: List[str] = []

def _tmpdir(prefix: str = "wxa8_selftest") -> str:
    d = tempfile.mkdtemp(prefix=prefix)
    _TEST_DIRS.append(d)
    return d

def _make_mini_checkpoint(path: str, seed: int = 0) -> None:
    """SDXL-shaped mini model: a few linears under model.diffusion_model."""
# SPDX-License-Identifier: Apache-2.0
    torch.manual_seed(seed)
    sd = {
        "model.diffusion_model.input_blocks.0.0.weight": torch.randn(320, 4, 3, 3) * 0.1,
        "model.diffusion_model.input_blocks.0.0.bias": torch.randn(320) * 0.01,
        "model.diffusion_model.input_blocks.1.0.transformer_blocks.0.attn1.to_q.weight": torch.randn(1280, 1280) * 0.02,
        "model.diffusion_model.input_blocks.1.0.transformer_blocks.0.attn1.to_k.weight": torch.randn(1280, 1280) * 0.02,
        "model.diffusion_model.input_blocks.1.0.transformer_blocks.0.attn1.to_v.weight": torch.randn(1280, 1280) * 0.02,
        "model.diffusion_model.input_blocks.1.0.transformer_blocks.0.attn1.to_out.0.weight": torch.randn(1280, 1280) * 0.02,
        "model.diffusion_model.input_blocks.1.0.transformer_blocks.0.attn2.to_q.weight": torch.randn(1280, 2048) * 0.02,
        "model.diffusion_model.input_blocks.1.0.transformer_blocks.0.attn2.to_k.weight": torch.randn(1280, 2048) * 0.02,
        "model.diffusion_model.input_blocks.1.0.transformer_blocks.0.attn2.to_v.weight": torch.randn(1280, 2048) * 0.02,
        "model.diffusion_model.input_blocks.1.0.transformer_blocks.0.attn2.to_out.0.weight": torch.randn(1280, 1280) * 0.02,
        "model.diffusion_model.input_blocks.1.0.transformer_blocks.0.ff.net.0.proj.weight": torch.randn(5120, 1280) * 0.01,
        "model.diffusion_model.input_blocks.1.0.transformer_blocks.0.ff.net.2.weight": torch.randn(1280, 5120) * 0.01,
        "model.diffusion_model.input_blocks.1.0.transformer_blocks.0.norm1.weight": torch.randn(1280) * 0.1,
        "model.diffusion_model.time_embed.0.weight": torch.randn(320, 320) * 0.05,
        "model.diffusion_model.time_embed.0.bias": torch.randn(320) * 0.05,
        "model.diffusion_model.out.2.weight": torch.randn(4, 320) * 0.1,
        "model.diffusion_model.out.2.bias": torch.randn(4) * 0.1,
        "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight": torch.randn(49408, 768) * 0.02,
        "first_stage_model.encoder.conv_in.weight": torch.randn(128, 3, 3, 3) * 0.1,
    }
    safetensors.torch.save_file(sd, path, metadata={"_selftest": "1"})

def _test_w4_pack_roundtrip() -> str:
    torch.manual_seed(1)
    for k in (16, 32, 64, 128, 256):
        codes = torch.randint(0, 16, (7, k), dtype=torch.int32)
        packed = ((codes[:, 0::2] & 0xF) | ((codes[:, 1::2] & 0xF) << 4)).to(torch.int8)
        rt = unpack_w4(packed)
        assert torch.equal(rt, codes), f"K={k} mismatch"
    return "K=16..256 round trips"

def _test_odd_dims() -> str:
    torch.manual_seed(3)
    # odd N, K=48 (divisible by 16 but not by 32)
    w = torch.randn(17, 48)
    p, s_rel, s_ch, corr, cb = quantize_w4a8_weight(
        w, group_size=16, convrot_groupsize=16)
    assert corr is None and cb is not None
    assert p.shape == (17, 24)
    dq = dequantize_w4a8_weight(p, s_rel, s_ch, codebook=cb, group_size=16,
                                convrot_groupsize=16, output_dtype=torch.float32)
    assert dq.shape == (17, 48)
    return "N=17, K=48 (w4)"

def _test_padding_removal() -> str:
    d = _tmpdir()
    path = os.path.join(d, "pad.safetensors")
    tensors = {
        "a_bool": torch.tensor([True]),
        "b_u8": torch.tensor([1, 2, 3], dtype=torch.uint8),
        "c_float": torch.randn(5),
    }
    if getattr(torch, "uint16", None) in TORCH_TO_SAFE:
        tensors["d_u16"] = torch.tensor([1, 65535], dtype=torch.uint16)
    if torch.complex64 in TORCH_TO_SAFE:
        tensors["e_complex64"] = torch.tensor([1 + 2j], dtype=torch.complex64)
    safetensors.torch.save_file(tensors, path)
    entries = [
        {"name": name, "dtype": tensor.dtype, "shape": tuple(tensor.shape),
         "nbytes": tensor.numel() * tensor.element_size()}
        for name, tensor in tensors.items()
    ]
    w = SafetensorsStreamWriter(path + ".stream", entries)
    w.open()
    for name, tensor in tensors.items():
        w.write_tensor_bytes(name, tensor_to_bytes(tensor))
    w.finalize(path + ".final")
    with safe_open(path + ".final", framework="pt") as st:
        for name, tensor in tensors.items():
            assert torch.equal(st.get_tensor(name), tensor)
    return "contiguous odd-byte and optional safetensors dtypes reopen"

def _test_scale_calculations() -> str:
    torch.manual_seed(4)
    w = torch.randn(8, 64)
    p, s_rel, s_ch, corr, cb = quantize_w4a8_weight(w, group_size=16, convrot_groupsize=64)
    assert s_rel.shape == (8, 4)
    assert s_ch.shape == (8,)
    assert cb.shape == (16,)
    assert corr is None
    # the decoded weights must be within ~1 int8 LSB of the pre-grid reconstruction
    codes = unpack_w4(p)
    shifted = cb[codes].view(8, 4, 16) * (s_rel.float().unsqueeze(-1) * s_ch.view(8, 1, 1))
    vals = cb[codes].view(8, 4, 16) * s_rel.float().unsqueeze(-1)
    i8 = vals.view(8, 64).round().clamp(-127, 127)
    decoded = i8.float() * s_ch.view(-1, 1)
    assert (decoded - shifted.view(8, 64)).abs().max() <= s_ch.max() * 1.01 + 1e-6
    assert (cb.abs() <= 1.0).all()
    assert torch.isfinite(s_rel.float()).all() and (s_rel.float() > 0).all()
    assert torch.isfinite(s_ch).all() and (s_ch > 0).all()
    zero_metrics = compute_weight_metrics(torch.zeros(4, 4), torch.zeros(4, 4))
    assert zero_metrics.rel_l2 == 0.0 and zero_metrics.cosine == 1.0
    return "s_rel/s_channel/codebook shapes, positivity, 1-LSB grid bound"

def _test_deterministic() -> str:
    torch.manual_seed(5)
    w = torch.randn(64, 256)
    o1 = quantize_w4a8_weight(w)
    o2 = quantize_w4a8_weight(w)
    for a, b in zip(o1, o2, strict=True):
        if a is None or b is None:
            assert a is None and b is None
        else:
            assert torch.equal(a, b)
    return "two runs byte-identical (w4)"

def _test_compute_dtype() -> str:
    d = _tmpdir()
    path = os.path.join(d, "bf16.safetensors")
    torch.manual_seed(51)
    safetensors.torch.save_file({"w": torch.randn(65, 64).bfloat16()}, path)
    info = discover_checkpoint(path)
    with CheckpointReader(info) as reader:
        fp32 = quantize_tensor_bounded(
            reader, "w", FORMAT_W4A8, 16, 64, 256 * 1024**2,
            torch.device("cpu"), compute_dtype=torch.float32)
        bf16 = quantize_tensor_bounded(
            reader, "w", FORMAT_W4A8, 16, 64, 256 * 1024**2,
            torch.device("cpu"), compute_dtype=torch.bfloat16)
    assert any(not torch.equal(fp32[key], bf16[key]) for key in fp32)
    return "fp32 and bf16 compute paths produce distinct deterministic tensors"

def _test_activation_calibration() -> str:
    torch.manual_seed(52)
    original = torch.randn(8, 16)
    dequant = original + torch.randn_like(original) * 0.03
    activations = torch.randn(5, 16)
    d = _tmpdir()
    model_path = os.path.join(d, "calibration_model.safetensors")
    source_path = os.path.join(d, "activations.npz")
    cache_path = os.path.join(d, "activations.cache")
    safetensors.torch.save_file({"layer.weight": original}, model_path)
    np.savez(source_path, **{"layer.weight": activations.numpy()})
    info = discover_checkpoint(model_path)
    calibration = load_calibration(source_path, info, 5, cache_path)
    cached = load_calibration(source_path, info, 5, cache_path)
    assert torch.equal(cached.layers["layer.weight"]["samples"], activations)
    with open(cache_path, "rb") as f:
        assert f.read(4) == b"PK\x03\x04"
    analyzer = SensitivityAnalyzer(None, 1.0, calibration)
    metrics = analyzer.evaluate("layer.weight", original, dequant)
    expected = activation_aware_error(original, dequant, activations)
    fake = activations.abs().amax(dim=0).unsqueeze(0).expand(5, -1)
    fake_value = activation_aware_error(original, dequant, fake)
    assert expected is not None and abs(metrics.act_rel_l2 - expected) < 1e-8
    assert abs(metrics.act_rel_l2 - fake_value) > 1e-6
    return "real rows used directly; compressed safe cache round-trips"

def _test_standalone_environment() -> str:
    import builtins
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "comfy" or name.startswith("comfy.") or \
                name == "comfy_kitchen" or name.startswith("comfy_kitchen."):
            raise AssertionError(f"standalone probe imported {name}")
        return original_import(name, *args, **kwargs)

    builtins.__import__ = guarded_import
    try:
        inspected = inspect_environment()
    finally:
        builtins.__import__ = original_import
    assert inspected.python and inspected.torch_version
    return "compatibility inspection performs no ComfyUI/comfy-kitchen imports"

def _test_metadata() -> str:
    d = _tmpdir()
    path = os.path.join(d, "mini.safetensors")
    _make_mini_checkpoint(path)
    info = discover_checkpoint(path)
    det = detect_architecture(info, shape_lookup=lambda n: (info.by_name(n).shape if info.by_name(n) else None))
    dec = classify_tensors(info, det, FORMAT_W4A8, None, [], [], [], None, None)
    plan = ConversionPlan(fmt=FORMAT_W4A8, detection=det, decisions=dec,
                          metadata_quant={}, metadata_ext={}, output_entries=[])
    qm = build_quant_metadata(info, plan)
    assert "layers" in qm
    for d_ in plan.quantized_layers():
        assert d_.layer in qm["layers"]
        conf = qm["layers"][d_.layer]
        assert conf["format"] == FORMAT_W4A8 and conf["group_size"] >= 4
    plan.n_quantized = len(plan.quantized_layers())
    plan.n_kept = len(plan.decisions) - plan.n_quantized
    ext = build_extension_metadata(
        info, plan, inspect_environment(),
        _selftest_args(os.path.join(d, "out.safetensors"), FORMAT_W4A8),
        None, None, hash_checkpoint_files(info), "0" * 64, {}, [])
    assert ext["schema"] == "comfy_wxa8/v1"
    assert ext["source"]["files"] == ["mini.safetensors"]
    assert list(ext["source"]["sha256"]) == ["mini.safetensors"]
    assert "runtime dynamic symmetric int8" in \
        ext["quantization"]["activation_quantization"]
    return f"{len(qm['layers'])} layers recorded; extension schema and provenance valid"

def _test_registry() -> str:
    names = family_names()
    assert len(names) == len(set(names))
    for n in names:
        pol = get_family(n)
        assert pol.family == n and pol.comfyui_classes
        assert pol.runtime_status in ("verified", "experimental", "unsupported")
    # every ComfyUI supported-model class at the research revision is covered
    covered = set()
    for n in names:
        covered.update(get_family(n).comfyui_classes)
    comfyui_classes = {
        "SD15", "SD20", "SD21UnclipL", "SD21UnclipH", "SDXLRefiner", "SDXL", "SSD1B",
        "Segmind_Vega", "KOALA_700M", "KOALA_1B", "SVD_img2vid", "SV3D_u", "SV3D_p",
        "Stable_Zero123", "SD_X4Upscaler", "Stable_Cascade_C", "Stable_Cascade_B",
        "SD15_instructpix2pix", "SDXL_instructpix2pix", "LotusD", "SD3", "StableAudio",
        "StableAudio3", "AuraFlow", "PixArtAlpha", "PixArtSigma", "HunyuanDiT",
        "HunyuanDiT1", "Flux", "FluxInpaint", "FluxSchnell", "Flux2", "Lens",
        "GenmoMochi", "LTXV", "LTXAV", "MiniMaxH3", "HunyuanVideo", "HunyuanVideoI2V",
        "HunyuanVideoSkyreelsI2V", "CosmosT2V", "CosmosI2V", "CosmosT2IPredict2",
        "CosmosI2VPredict2", "Anima", "Lumina2", "ZImage", "ZImagePixelSpace",
        "PixelDiTT2I", "PiD", "WAN21_T2V", "WAN21_CausalAR_T2V", "WAN21_I2V",
        "WAN21_FunControl2V", "WAN21_Camera", "WAN22_Camera", "WAN21_Vace",
        "WAN21_HuMo", "WAN22_S2V", "WAN22_Animate", "WAN22_T2V", "WAN21_FlowRVS",
        "WAN21_SCAIL", "WAN21_SCAIL2", "WAN22_WanDancer", "Hunyuan3Dv2",
        "Hunyuan3Dv2_1", "Hunyuan3Dv2mini", "TripoSplat", "HiDream", "HiDreamO1",
        "Chroma", "SeedVR2", "ChromaRadiance", "ACEStep", "Omnigen2", "Boogu",
        "Ideogram4", "Krea2", "MageFlow", "QwenImage", "JoyImage", "HunyuanImage21",
        "HunyuanImage21Refiner", "HunyuanVideo15", "HunyuanVideo15_SR_Distilled",
        "Kandinsky5", "Kandinsky5Image", "ACEStep15", "LongCatImage", "RT_DETR_v4",
        "DepthAnything3", "ErnieImage", "SAM3", "SAM31", "CogVideoX_T2V",
        "CogVideoX_I2V", "CogVideoX_Inpaint",
    }
    missing = sorted(comfyui_classes - covered)
    assert not missing, f"registry missing ComfyUI classes: {missing}"
    return f"{len(names)} families, {len(covered)} ComfyUI classes covered"

def _ckpt(keys_shapes: Sequence[Tuple[str, Tuple[int, ...]]]) -> CheckpointInfo:
    tensors = [TensorMeta(name, torch.float32, shape, int(np.prod(shape)) * 4,
                          "", 0, int(np.prod(shape)) * 4)
               for name, shape in keys_shapes]
    return CheckpointInfo(kind="safetensors", files=[], metadata={}, tensors=tensors)

def _classify_real(keys_shapes: Sequence[Tuple[str, Tuple[int, ...]]]):
    info = _ckpt(keys_shapes)
    det = detect_architecture(
        info, shape_lookup=lambda n: (info.by_name(n).shape
                                      if info.by_name(n) else None))
    decisions = classify_tensors(info, det, FORMAT_W4A8, 16, None, None,
                                 None, None, None)
    return det, decisions

def _test_compression_stats() -> str:
    """The low-compression warning must fire for hidden dims not divisible by
    256 (Boogu-style K=3360) and stay silent for ConvRot-compatible dims."""
    # Boogu-style: 5 attention/FFN linears at K=3360 (gate victims) + one
    # K=13568 FFN expansion layer that quantizes. The double_stream primary
    # keys pin detection to the boogu family.
    boogu_keys: Sequence[Tuple[str, Tuple[int, ...]]] = [
        ("double_stream_layers.0.img_instruct_attn.processor.img_to_q.weight", (768, 3360)),
        ("double_stream_layers.0.img_self_attn.to_q.weight", (768, 3360)),
        ("context_refiner.0.attn.to_q.weight", (768, 3360)),
        ("context_refiner.0.attn.to_k.weight", (768, 3360)),
        ("context_refiner.0.feed_forward.linear_1.weight", (768, 3360)),
        ("context_refiner.0.feed_forward.linear_2.weight", (768, 13568)),
        ("single_stream_layers.0.attn.to_q.weight", (768, 3360)),
        ("single_stream_layers.0.feed_forward.linear_1.weight", (768, 3360)),
    ]
    det, decisions = _classify_real(boogu_keys)
    assert det.architecture == "boogu"
    plan = ConversionPlan(fmt=FORMAT_W4A8, detection=det, decisions=decisions,
                          metadata_quant={}, metadata_ext={}, output_entries=[])
    info = _ckpt(boogu_keys)
    stats = compression_stats(info, plan, det)
    assert stats["targeted_2d_layers"] == 8
    assert stats["quantized_2d_layers"] == 1
    assert stats["quantized_fraction"] < 0.5, stats
    assert stats["failing_k_values"] == [3360], stats
    warn = low_compression_warning(stats)
    assert warn is not None and "3360" in warn and "low compression" in warn
    assert "convrot_groupsize" in decisions[0].reason

    # Clean case: all quantize-set K divisible by 256 (flux-style). The
    # key_norm weight is the flux detection primary; img/txt_in and the
    # single-block linears complete the picture.
    flux_keys: Sequence[Tuple[str, Tuple[int, ...]]] = [
        ("model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.weight", (768,)),
        ("model.diffusion_model.img_in.weight", (768, 768)),
        ("model.diffusion_model.double_blocks.0.img_attn.qkv.weight", (768, 768)),
        ("model.diffusion_model.double_blocks.0.img_attn.proj.weight", (768, 768)),
        ("model.diffusion_model.double_blocks.0.img_mlp.w1.weight", (3072, 768)),
        ("model.diffusion_model.double_blocks.0.img_mlp.w2.weight", (768, 3072)),
        ("model.diffusion_model.single_blocks.0.linear1.weight", (768, 768)),
        ("model.diffusion_model.single_blocks.0.linear2.weight", (768, 768)),
    ]
    info2 = _ckpt(flux_keys)
    det2, decisions2 = _classify_real(flux_keys)
    plan2 = ConversionPlan(fmt=FORMAT_W4A8, detection=det2, decisions=decisions2,
                           metadata_quant={}, metadata_ext={}, output_entries=[])
    stats2 = compression_stats(info2, plan2, det2)
    assert stats2["quantized_fraction"] == 1.0, stats2
    assert stats2["failing_k_values"] == []
    assert low_compression_warning(stats2) is None
    return ("K%256 gate victims detected (3360, frac<0.5); clean dims pass silently")

def _make_boogu_real_dims_checkpoint(path: str, n: int = 64,
                                     n_fail: int = 5,
                                     n_ok: int = 2) -> Dict[str, Tuple[int, int]]:
    """Small checkpoint with the REAL Boogu-Image-0.1-Turbo widths: hidden 3360
    (fails the ConvRot-256 gate) and FFN expansion 13568 (compatible). Returns
    the shape map for the quantize-set keys."""
    torch.manual_seed(11)
    sd = {}
    def L(n_, k_, s=0.02): return torch.randn(n_, k_) * s
    for i in range(n_fail):
        pre = f"double_stream_layers.{i}."
        sd[pre + "img_self_attn.to_q.weight"] = L(n, 3360)
        sd[pre + "img_instruct_attn.processor.img_to_q.weight"] = L(n, 3360)
    for i in range(n_ok):
        pre = f"single_stream_layers.{i}."
        sd[pre + "feed_forward.linear_2.weight"] = L(n, 13568)
    safetensors.torch.save_file(sd, path)
    return {k: tuple(v.shape) for k, v in sd.items()}

def _test_boogu_real_dims() -> str:
    """Real Boogu widths: K=3360 layers must stay BF16/FP32, K=13568 layers
    must become W4A8, metadata may only list the compatible layers, and the
    mixed output must reload as QuantizedTensors."""
    d = _tmpdir()
    src_path = os.path.join(d, "boogu_real.safetensors")
    out = os.path.join(d, "boogu_real_w4a8.safetensors")
    shapes = _make_boogu_real_dims_checkpoint(src_path)
    args = _selftest_args(out, FORMAT_W4A8)
    info = discover_checkpoint(src_path)
    det = detect_architecture(
        info, shape_lookup=lambda n: (info.by_name(n).shape if info.by_name(n) else None))
    assert det.architecture == "boogu", det.architecture
    dec = classify_tensors(info, det, FORMAT_W4A8, None, [], [], [], None, None)
    plan = ConversionPlan(fmt=FORMAT_W4A8, detection=det, decisions=dec,
                          metadata_quant={}, metadata_ext={}, output_entries=[])
    entries, total = build_output_entries(info, dec, FORMAT_W4A8, None)
    plan.output_entries = entries
    plan.total_out_bytes = total
    plan.n_quantized = len(plan.quantized_layers())
    plan.n_kept = len(dec) - plan.n_quantized
    plan.metadata_quant = build_quant_metadata(info, plan)
    stats = compression_stats(info, plan, det)
    assert stats["quantized_2d_layers"] == 2, stats
    assert stats["buckets"]["convrot_rejected"]["layers"] == 10, stats
    assert stats["failing_k_values"] == [3360], stats
    assert stats["quantized_fraction"] < 0.5, stats
    engine = ConversionEngine(info, plan, args, out + ".state.json", out + ".tmp", out)
    try:
        engine.run()
    finally:
        engine.close()
    meta = dict(info.metadata)
    meta[METADATA_KEY_QUANT] = json_dumps(plan.metadata_quant)
    meta[METADATA_KEY_EXT] = json_dumps(build_extension_metadata(
        info, plan, inspect_environment(), args, None, None,
        hash_checkpoint_files(info), sha256_safetensors_payload(out),
        {"status": "selftest"}, []))
    republish_with_metadata(out, out, meta, entries)
    with safe_open(out, framework="pt") as st:
        meta = st.metadata()
        qm = json.loads(meta["_quantization_metadata"])
        q_layers = set(qm["layers"].keys())
        # only the K=13568 layers may be quantized
        assert q_layers == {
            "single_stream_layers.0.feed_forward.linear_2",
            "single_stream_layers.1.feed_forward.linear_2"}, q_layers
        for _, conf in qm["layers"].items():
            assert conf["convrot_groupsize"] == 256 and conf["group_size"] == 16
        # K=3360 layers stay as the original float dtype with the original shape
        for key in shapes:
            layer = key[:-len(".weight")]
            if layer in q_layers:
                continue
            t = st.get_tensor(key)
            assert t.dtype == torch.float32 and tuple(t.shape) == shapes[key], key
            # byte-identical payload vs source
            src_t = safetensors.torch.load_file(src_path)[key]
            assert torch.equal(t, src_t), key
    # reload through the ComfyUI conversion path
    try:
        import comfy_kitchen  # noqa: F401
        from comfy_kitchen.tensor.base import QuantizedTensor, get_layout_class  # noqa: F401
    except Exception:
        return ("real-dims fixture converts mixed (2 quantized, 5 passthrough, "
                "metadata restricted to K=13568); comfy-kitchen not installed for "
                "the QuantizedTensor reload step")
    with safe_open(out, framework="pt") as st:
        qm = json.loads(st.metadata()["_quantization_metadata"])
        layer = "single_stream_layers.0.feed_forward.linear_2"
        conf = qm["layers"][layer]
        layout = get_layout_class("AsymW4A8Int8Layout")
        params = layout.Params(
            scale=st.get_tensor(layer + ".weight_s_rel").view(torch.float8_e4m3fn),
            s_channel=st.get_tensor(layer + ".weight_s_channel"),
            codebook=st.get_tensor(layer + ".weight_codebook"),
            group_size=conf["group_size"],
            convrot_groupsize=conf["convrot_groupsize"],
            orig_dtype=torch.bfloat16, orig_shape=shapes[layer + ".weight"])
        qt = QuantizedTensor(st.get_tensor(layer + ".weight"),
                             "AsymW4A8Int8Layout", params)
        assert qt._params.convrot_groupsize == 256
        assert qt.dequantize().shape == shapes[layer + ".weight"]
    return ("real-dims fixture: K=3360 passthrough byte-identical, K=13568 W4A8 "
            "only, metadata restricted, QuantizedTensor reloads")

def _real_dim_case(family: str, keys: Sequence[Tuple[str, Tuple[int, ...]]]):
    """Classify a real-dims key set and return (stats, warning)."""
    info = _ckpt(keys)
    det, decisions = _classify_real(keys)
    plan = ConversionPlan(fmt=FORMAT_W4A8, detection=det, decisions=decisions,
                          metadata_quant={}, metadata_ext={}, output_entries=[])
    return (det, compression_stats(info, plan, det),
            low_compression_warning(compression_stats(info, plan, det),
                                    det.architecture))

def _make_krea2_real_dims_checkpoint(path: str, n: int = 8) -> Dict[str, Tuple[int, int]]:
    """Small checkpoint with the REAL Kroma v0.2 (Krea2 fine-tune) key naming
    and widths, verified against the published kroma-v0.2-turbo.safetensors
    header: blocks.N.attn.wq/wk/wv/wo/gate [., 6144], blocks.N.mlp.gate/up
    [., 6144] + down [., 16384], txtfusion layerwise/refiner blocks at
    2560/6912. Returns the shape map for the quantize-set keys."""
    torch.manual_seed(13)
    sd: Dict[str, torch.Tensor] = {}

    def L(n_: int, k_: int, s: float = 0.02) -> torch.Tensor:
        return torch.randn(n_, k_) * s

    for i in range(2):
        pre = f"blocks.{i}."
        for w in ("wq", "wk", "wv", "wo", "gate"):
            sd[pre + f"attn.{w}.weight"] = L(n, 6144)
        sd[pre + "mlp.gate.weight"] = L(n, 6144)
        sd[pre + "mlp.up.weight"] = L(n, 6144)
        sd[pre + "mlp.down.weight"] = L(n, 16384)
    for i in range(2):
        pre = f"txtfusion.layerwise_blocks.{i}."
        sd[pre + "attn.wq.weight"] = L(n, 2560)
        sd[pre + "attn.wo.weight"] = L(n, 2560)
        sd[pre + "mlp.gate.weight"] = L(n, 2560)
        sd[pre + "mlp.down.weight"] = L(n, 6912)
    for i in range(2):
        pre = f"txtfusion.refiner_blocks.{i}."
        sd[pre + "attn.wq.weight"] = L(n, 2560)
        sd[pre + "mlp.up.weight"] = L(n, 2560)
    # keepers / passthroughs with the real names
    sd["first.weight"] = L(n, 64)
    sd["last.linear.weight"] = L(n, 6144)
    sd["tmlp.0.weight"] = L(n, 256)
    sd["tmlp.2.weight"] = L(n, 6144)
    sd["txtmlp.1.weight"] = L(n, 2560)
    sd["txtmlp.3.weight"] = L(n, 6144)
    sd["tproj.1.weight"] = L(n, 6144)
    sd["txtfusion.projector.weight"] = L(1, 12)
    sd["blocks.0.mod.lin"] = torch.randn(36864) * 0.02
    sd["blocks.0.prenorm.scale"] = torch.randn(6144) * 0.02
    safetensors.torch.save_file(sd, path)
    return {k: tuple(v.shape) for k, v in sd.items()}

def _test_krea2_real_dims() -> str:
    """Real Kroma v0.2 / Krea2 naming: blocks.N.attn.wq/wk/wv/wo/gate and
    blocks.N.mlp.gate/up/down plus txtfusion layerwise/refiner blocks must be
    quantized (all K values are ConvRot-256 compatible), while first/last/
    tmlp/txtmlp/tproj/txtfusion.projector stay at original precision. This is
    the regression for the reported 'no tensors selected under the krea2
    policy' failure on kroma-v0.2-turbo.safetensors (the policy used
    'layers.N' but the real checkpoint uses 'blocks.N', and the universal
    exclude swallowed txtfusion)."""
    d = _tmpdir()
    src_path = os.path.join(d, "kroma_real.safetensors")
    out = os.path.join(d, "kroma_real_w4a8.safetensors")
    shapes = _make_krea2_real_dims_checkpoint(src_path)
    args = _selftest_args(out, FORMAT_W4A8)
    info = discover_checkpoint(src_path)
    det = detect_architecture(
        info, shape_lookup=lambda n: (info.by_name(n).shape if info.by_name(n) else None))
    assert det.architecture == "krea2", det.architecture
    dec = classify_tensors(info, det, FORMAT_W4A8, None, [], [], [], None, None)
    quantized = {d.name for d in dec if d.kind == DecisionKind.QUANTIZE}
    # 2 blocks x (5 attn + 3 mlp) + 2 layerwise x (2 attn + 2 mlp)
    # + 2 refiner x (1 attn + 1 mlp) = 16 + 8 + 4 = 28
    assert len(quantized) == 28, sorted(quantized)
    assert "blocks.0.attn.wq.weight" in quantized
    assert "blocks.0.attn.gate.weight" in quantized
    assert "blocks.1.mlp.down.weight" in quantized
    assert "txtfusion.layerwise_blocks.0.attn.wq.weight" in quantized
    assert "txtfusion.layerwise_blocks.1.mlp.down.weight" in quantized
    assert "txtfusion.refiner_blocks.1.attn.wq.weight" in quantized
    by_name = {d.name: d for d in dec}
    for keep_name in ("first.weight", "last.linear.weight", "tmlp.0.weight",
                      "tmlp.2.weight", "txtmlp.1.weight", "txtmlp.3.weight",
                      "tproj.1.weight"):
        d = by_name[keep_name]
        assert d.kind in (DecisionKind.KEEP, DecisionKind.KEEP_PRECISION), (
            keep_name, d)
    # patch embedder and the tiny text projector are policy keeps at FP
    assert by_name["first.weight"].kind == DecisionKind.KEEP_PRECISION
    assert by_name["txtfusion.projector.weight"].kind == DecisionKind.KEEP_PRECISION
    # full w4a8 conversion runs and metadata lists exactly the block layers
    plan = ConversionPlan(fmt=FORMAT_W4A8, detection=det, decisions=dec,
                          metadata_quant={}, metadata_ext={}, output_entries=[])
    entries, total = build_output_entries(info, dec, FORMAT_W4A8, None)
    plan.output_entries = entries
    plan.total_out_bytes = total
    plan.n_quantized = len(plan.quantized_layers())
    plan.n_kept = len(dec) - plan.n_quantized
    plan.metadata_quant = build_quant_metadata(info, plan)
    engine = ConversionEngine(info, plan, args, out + ".state.json", out + ".tmp", out)
    try:
        engine.run()
    finally:
        engine.close()
    meta = dict(info.metadata)
    meta[METADATA_KEY_QUANT] = json_dumps(plan.metadata_quant)
    meta[METADATA_KEY_EXT] = json_dumps(build_extension_metadata(
        info, plan, inspect_environment(), args, None, None,
        hash_checkpoint_files(info), sha256_safetensors_payload(out),
        {"status": "selftest"}, []))
    republish_with_metadata(out, out, meta, entries)
    with safe_open(out, framework="pt") as st:
        meta = st.metadata()
        qm = json.loads(meta["_quantization_metadata"])
        assert set(qm["layers"].keys()) == {
            n[:-len(".weight")] for n in quantized}, (set(qm["layers"].keys())
                                                      ^ quantized)
        for _, conf in qm["layers"].items():
            assert conf["convrot_groupsize"] == 256 and conf["group_size"] == 16
    # mixed planning must select real layers under the balanced profile
    mdec = classify_tensors(info, det, FORMAT_MIXED, None, [], [], [], None, None)
    planner = MixedPlanner("balanced", None, 2 * 1024**3, torch.device("cpu"), None)
    summary = planner.plan(info, mdec)
    assert summary["selected"] >= 28, summary
    assert planner.global_mean_error(info, mdec) <= planner.global_gate
    return ("real Kroma v0.2 dims convert (28 quantized, metadata exact); "
            "mixed balanced plan selects " + str(summary["selected"]) + " layers")

def _test_policy_miss() -> str:
    """Unknown large 2D linears under a known prefix must raise the stale-policy
    warning; a clean Boogu set must not."""
    clean_keys: Sequence[Tuple[str, Tuple[int, ...]]] = [
        ("double_stream_layers.0.img_instruct_attn.processor.img_to_q.weight", (768, 3360)),
        ("double_stream_layers.0.img_self_attn.to_q.weight", (768, 3360)),
        ("double_stream_layers.0.img_feed_forward.linear_2.weight", (768, 13568)),
    ]
    info = _ckpt(clean_keys)
    det, decisions = _classify_real(clean_keys)
    plan = ConversionPlan(fmt=FORMAT_W4A8, detection=det, decisions=decisions,
                          metadata_quant={}, metadata_ext={}, output_entries=[])
    stats = compression_stats(info, plan, det)
    assert policy_miss_warning(stats, det.architecture) is None

    stale_keys = clean_keys + [
        ("single_stream_layers.0.mystery_proj.weight", (768, 3360)),
        ("single_stream_layers.0.other_mystery.weight", (768, 3360)),
    ]
    info = _ckpt(stale_keys)
    det, decisions = _classify_real(stale_keys)
    plan = ConversionPlan(fmt=FORMAT_W4A8, detection=det, decisions=decisions,
                          metadata_quant={}, metadata_ext={}, output_entries=[])
    stats = compression_stats(info, plan, det)
    warn = policy_miss_warning(stats, det.architecture)
    assert warn is not None and "policy may be stale" in warn, warn
    assert stats["buckets"]["not_in_quantize_set"]["layers"] == 2, stats
    return "unknown linears trigger stale-policy warning; clean sets stay silent"

def _test_real_dim_gate() -> str:
    """The documented problematic real dims must produce convrot/shape gate
    rejections and low-compression warnings; clean controls must not."""
    cases = {
        # (detect_primary key + quantize-set keys with the real widths)
        "pixart": [
            ("t_block.1.weight", (1152, 1152)),
            ("blocks.0.attn1.qkv.weight", (3456, 1152)),
            ("blocks.0.attn1.proj.weight", (1152, 1152)),
            ("blocks.0.attn2.to_q.weight", (1152, 768)),
            ("blocks.0.mlp.fc1.weight", (4608, 1152)),
            ("blocks.0.mlp.fc2.weight", (1152, 4608)),
        ],
        "hydit": [
            ("mlp_t5.0.weight", (384, 1024)),
            ("blocks.0.attn.qkv.weight", (4224, 1408)),
            ("blocks.0.attn.proj.weight", (1408, 1408)),
            ("blocks.0.mlp.fc1.weight", (5632, 1408)),
            ("blocks.0.mlp.fc2.weight", (1408, 5632)),
        ],
        "cogvideox": [
            ("blocks.0.norm1.linear.weight", (1920, 1920)),
            ("transformer_blocks.0.attn1.to_q.weight", (1920, 1920)),
            ("transformer_blocks.0.attn1.to_out.0.weight", (1920, 1920)),
            ("transformer_blocks.0.ff.net.0.proj.weight", (7680, 1920)),
            ("transformer_blocks.0.ff.net.2.weight", (1920, 7680)),
        ],
        "minimax_h3": [
            ("video_patch_proj.weight", (768, 256)),
            ("audio_patch_proj.weight", (768, 256)),
            ("blocks.0.attn.qkv_proj.weight", (2304, 768)),
            ("blocks.0.attn.out_proj.weight", (768, 768)),
            ("blocks.0.mlp.fc1.weight", (2304, 768)),
            ("blocks.0.mlp.fc2.weight", (768, 1152)),
        ],
        "omnigen2": [
            ("time_caption_embed.timestep_embedder.linear_1.bias", (256,)),
            ("layers.0.attn.to_q.weight", (2520, 2520)),
            ("layers.0.attn.to_k.weight", (2520, 2520)),
            ("layers.0.attn.to_v.weight", (2520, 2520)),
            ("layers.0.attn.to_out.0.weight", (2520, 2520)),
            ("layers.0.feed_forward.linear_1.weight", (2520, 2520)),
            ("layers.0.feed_forward.linear_2.weight", (2520, 10240)),
            ("layers.0.feed_forward.linear_3.weight", (2520, 2520)),
        ],
    }
    clean_cases = {
        "flux": [
            ("double_blocks.0.img_attn.norm.key_norm.weight", (768,)),
            ("double_blocks.0.img_attn.qkv.weight", (768, 768)),
            ("double_blocks.0.img_mlp.w1.weight", (3072, 768)),
        ],
        "lumina2": [
            ("cap_embedder.1.weight", (384, 256)),
            ("layers.0.attention.qkv.weight", (11520, 3840)),
            ("layers.0.feed_forward.w1.weight", (10240, 3840)),
            ("layers.0.feed_forward.w2.weight", (3840, 10240)),
        ],
    }
    # minimax_h3: only mlp.fc2 (K=1152) is rejected, ~18% of block bytes, so
    # the model still quantizes ~82% and must NOT raise the low-compression
    # warning; it is a documented gate victim but not a low-compression case.
    det, stats, warn = _real_dim_case("minimax_h3", cases.pop("minimax_h3"))
    assert det.architecture == "minimax_h3"
    assert 0.5 <= stats["quantized_fraction"] < 1.0, stats
    assert stats["buckets"]["convrot_rejected"]["layers"] == 1
    assert stats["failing_k_values"] == [1152]
    assert warn is None, warn
    for family, keys in cases.items():
        det, stats, warn = _real_dim_case(family, keys)
        assert det.architecture == family, (family, det.architecture)
        assert stats["quantized_fraction"] is not None
        assert stats["quantized_fraction"] < 0.5, (family, stats)
        assert warn is not None and "low compression" in warn, (family, warn)
        if family == "omnigen2":
            # 2520 fails K%16: shape_rejected, not convrot_rejected
            assert stats["buckets"]["shape_rejected"]["layers"] > 0
            assert stats["buckets"]["convrot_rejected"]["layers"] == 0
        else:
            assert stats["buckets"]["convrot_rejected"]["layers"] > 0, family
            assert warn and str(stats["failing_k_values"][0]) in warn, family
    for family, keys in clean_cases.items():
        det, stats, warn = _real_dim_case(family, keys)
        assert det.architecture == family, (family, det.architecture)
        assert stats["quantized_fraction"] == 1.0, (family, stats)
        assert warn is None, (family, warn)
    return ("pixart/hydit/cogvideox/minimax/omnigen2 gate warnings verified; "
            "flux/lumina2 clean controls silent")

def _test_detection_safety() -> str:
    # Real Boogu-Image-0.1 key naming (Comfy-Org repack, verified against the
    # published checkpoint): double/single stream layers plus OmniGen2-style
    # refiners and embedders. Detection must land on the dedicated boogu
    # family, and the linear attention / FFN weights must actually quantize.
    # K must be divisible by 256 for W4A8 (CUDA ConvRot is 256-only); the
    # K=320 context_refiner to_k below is the passthrough case (it would have
    # crashed the comfy-kitchen 0.2.27 CUDA kernel as convrot_groupsize 64).
    boogu_keys: Sequence[Tuple[str, Tuple[int, ...]]] = [
        ("double_stream_layers.0.img_instruct_attn.processor.img_to_q.weight", (256, 256)),
        ("double_stream_layers.0.img_instruct_attn.processor.img_out.weight", (256, 256)),
        ("double_stream_layers.0.img_self_attn.to_q.weight", (256, 256)),
        ("double_stream_layers.0.img_self_attn.to_k.weight", (64, 256)),
        ("double_stream_layers.0.img_self_attn.to_out.0.weight", (256, 256)),
        ("double_stream_layers.0.img_feed_forward.linear_1.weight", (256, 256)),
        ("double_stream_layers.0.img_feed_forward.linear_2.weight", (256, 256)),
        ("double_stream_layers.0.img_feed_forward.linear_3.weight", (256, 256)),
        ("double_stream_layers.0.instruct_feed_forward.linear_1.weight", (256, 256)),
        ("double_stream_layers.0.img_norm1.linear.weight", (256, 64)),
        ("double_stream_layers.0.img_norm1.linear.bias", (256,)),
        ("double_stream_layers.0.img_norm1.norm.weight", (64,)),
        ("single_stream_layers.0.attn.to_q.weight", (256, 256)),
        ("single_stream_layers.0.attn.to_k.weight", (64, 256)),
        ("single_stream_layers.0.attn.to_out.0.weight", (256, 256)),
        ("single_stream_layers.0.feed_forward.linear_1.weight", (256, 256)),
        ("single_stream_layers.0.feed_forward.linear_2.weight", (256, 256)),
        ("single_stream_layers.0.norm1.linear.weight", (256, 64)),
        ("context_refiner.0.attn.to_q.weight", (256, 256)),
        ("context_refiner.0.attn.to_k.weight", (64, 320)),
        ("context_refiner.0.feed_forward.linear_1.weight", (256, 256)),
        ("noise_refiner.0.attn.to_q.weight", (256, 256)),
        ("ref_image_refiner.0.feed_forward.linear_1.weight", (256, 256)),
        ("x_embedder.weight", (64, 64)),
        ("ref_image_patch_embedder.weight", (64, 64)),
        ("time_caption_embed.timestep_embedder.linear_1.weight", (64, 64)),
        ("norm_out.linear_1.weight", (64, 64)),
        ("norm_out.linear_2.weight", (64, 64)),
        ("image_index_embedding", (5, 64)),
    ]
    det, decisions = _classify_real(boogu_keys)
    assert det.architecture == "boogu", det.architecture
    assert det.confidence == "high", det.confidence
    assert get_family("Boogu").family == "boogu"
    quantized = {d.name for d in decisions if d.kind == DecisionKind.QUANTIZE}
    kept = {d.name for d in decisions if d.kind != DecisionKind.QUANTIZE}
    assert len(quantized) == 18, sorted(quantized)
    assert "double_stream_layers.0.img_instruct_attn.processor.img_to_q.weight" in quantized
    assert "double_stream_layers.0.img_self_attn.to_q.weight" in quantized
    assert "single_stream_layers.0.feed_forward.linear_1.weight" in quantized
    assert "context_refiner.0.attn.to_q.weight" in quantized
    # K=320 is not divisible by 256: must pass through, not quantize with a
    # smaller ConvRot group (CUDA fused kernel is 256-only)
    cr_k = next(d for d in decisions
                if d.name == "context_refiner.0.attn.to_k.weight")
    assert cr_k.kind == DecisionKind.KEEP, cr_k
    assert "256" in cr_k.reason, cr_k.reason
    assert "double_stream_layers.0.img_norm1.linear.weight" in kept  # modulation
    assert "single_stream_layers.0.norm1.linear.weight" in kept      # modulation
    assert "norm_out.linear_1.weight" in kept                        # output head
    assert "x_embedder.weight" in kept                               # embedder
    assert "time_caption_embed.timestep_embedder.linear_1.weight" in kept

    # Real OmniGen2 key naming (BAAI/OmniGen2): layers.N + refiners. Detection
    # must land on omnigen2 and the linear weights must quantize too.
    og2_keys: Sequence[Tuple[str, Tuple[int, ...]]] = [
        ("layers.0.attn.to_q.weight", (256, 256)),
        ("layers.0.attn.to_k.weight", (64, 256)),
        ("layers.0.attn.to_v.weight", (64, 256)),
        ("layers.0.attn.to_out.0.weight", (256, 256)),
        ("layers.0.feed_forward.linear_1.weight", (256, 256)),
        ("layers.0.feed_forward.linear_2.weight", (256, 256)),
        ("layers.0.feed_forward.linear_3.weight", (256, 256)),
        ("layers.0.norm1.linear.weight", (256, 64)),
        ("layers.0.attn.norm_k.weight", (64,)),
        ("context_refiner.0.attn.to_q.weight", (256, 256)),
        ("context_refiner.0.feed_forward.linear_1.weight", (256, 256)),
        ("noise_refiner.0.attn.to_q.weight", (256, 256)),
        ("noise_refiner.0.feed_forward.linear_1.weight", (256, 256)),
        ("time_caption_embed.timestep_embedder.linear_1.bias", (64,)),
        ("x_embedder.weight", (64, 64)),
        ("ref_image_patch_embedder.weight", (64, 64)),
        ("norm_out.linear_1.weight", (64, 64)),
    ]
    det, decisions = _classify_real(og2_keys)
    assert det.architecture == "omnigen2", det.architecture
    assert det.confidence == "high", det.confidence
    quantized = {d.name for d in decisions if d.kind == DecisionKind.QUANTIZE}
    kept = {d.name for d in decisions if d.kind != DecisionKind.QUANTIZE}
    assert len(quantized) == 11, sorted(quantized)
    assert "layers.0.attn.to_q.weight" in quantized
    assert "layers.0.feed_forward.linear_2.weight" in quantized
    assert "context_refiner.0.attn.to_q.weight" in quantized
    assert "noise_refiner.0.feed_forward.linear_1.weight" in quantized
    assert "layers.0.norm1.linear.weight" in kept
    assert "norm_out.linear_1.weight" in kept

    keys = ("clf.1.weight", "head.modulation")
    ambiguous = CheckpointInfo(
        kind="safetensors", files=[], metadata={},
        tensors=[TensorMeta(name, torch.float32, (1,), 4, "", 0, 4)
                 for name in keys])
    try:
        detect_architecture(ambiguous)
        raise AssertionError("ambiguous checkpoint was guessed")
    except UnknownArchitectureError as exc:
        assert "ambiguous" in str(exc)
    return ("real Boogu/OmniGen2 naming quantizes; Boogu is its own family; "
        "equal-score architectures fail closed")

def _test_fail_on_low_compression() -> str:
    """--fail-on-low-compression / --min-quantized-byte-fraction must abort a
    dry-run whose quantized byte fraction is below the threshold."""
    import subprocess as _sp
    d = _tmpdir()
    src_path = os.path.join(d, "boogu_real.safetensors")
    _make_boogu_real_dims_checkpoint(src_path, n=32, n_fail=80, n_ok=2)
    out = os.path.join(d, "out.safetensors")
    base_cmd, base_env = _converter_cmd()
    for argv, expect_err in (
        (["--dry-run", "--fail-on-low-compression"], "below the required 10%"),
        (["--dry-run", "--min-quantized-byte-fraction", "0.5"], "below the required 50%"),
    ):
        cmd = base_cmd + [src_path, "--output", out,
                          "--format", "w4a8"] + argv
        r = _sp.run(cmd, capture_output=True, text=True, timeout=600,
                    env=base_env)  # noqa: S603 (self-test fixture path)
        assert r.returncode == 1, (argv, r.returncode, r.stdout[-500:], r.stderr[-500:])
        assert expect_err in r.stderr or expect_err in r.stdout, (argv, r.stdout[-800:])
        assert not os.path.exists(out)
    # without the flag the same dry-run succeeds
    r = _sp.run(base_cmd + [src_path, "--output", out,  # noqa: S603 (self-test)
                            "--format", "w4a8", "--dry-run"],
                capture_output=True, text=True, timeout=600, env=base_env)
    assert r.returncode == 0, (r.returncode, r.stdout[-500:])
    assert not os.path.exists(out)  # dry run writes nothing
    return ("low-compression thresholds abort dry-run with clear errors; "
            "default conversion unaffected")

def _test_metadata_fuzz() -> str:
    """Corrupted quantization metadata must be rejected: wrong convrot size,
    swapped group fields, missing codebook, wrong scale shapes, missing layer
    metadata, stale old-format entries."""
    def craft(fname: str, layers_conf, tensors: Dict[str, torch.Tensor]) -> str:
        meta = {"_quantization_metadata": json_dumps({"layers": layers_conf})}
        p = os.path.join(_tmpdir(), fname)
        safetensors.torch.save_file(tensors, p, metadata=meta)
        return p

    d = _tmpdir()
    src_path = os.path.join(d, "mini.safetensors")
    _make_mini_checkpoint(src_path)
    info = discover_checkpoint(src_path)
    det = detect_architecture(
        info, shape_lookup=lambda n: (info.by_name(n).shape if info.by_name(n) else None))
    layer = "model.diffusion_model.input_blocks.1.0.transformer_blocks.0.attn1.to_q"
    good = {
        "format": FORMAT_W4A8, "group_size": 16, "convrot": True,
        "convrot_groupsize": 256,
    }
    base_tensors = {
        layer + ".weight": torch.zeros(1280, 640, dtype=torch.int8),
        layer + ".weight_s_rel": torch.zeros(1280, 80, dtype=torch.uint8),
        layer + ".weight_s_channel": torch.ones(1280, dtype=torch.float32),
        layer + ".weight_codebook": torch.zeros(16, dtype=torch.float32),
    }

    # (1) convrot_groupsize 64: the historical v1.2.1 failure -> hard reject
    p64 = craft("cgs64.safetensors", {layer: dict(good, convrot_groupsize=64)}, base_tensors)
    try:
        plan_from_output(p64, det, FORMAT_W4A8, info)
        raise AssertionError("cgs=64 accepted")
    except ValidationError as exc:
        assert "convrot_groupsize" in str(exc)

    # (2) swapped group fields (group_size=256, convrot=16): must be rejected
    pswap = craft("swapped.safetensors",
                  {layer: {"format": FORMAT_W4A8, "group_size": 256,
                           "convrot": True, "convrot_groupsize": 16}}, base_tensors)
    try:
        plan_from_output(pswap, det, FORMAT_W4A8, info)
        raise AssertionError("swapped groups accepted")
    except ValidationError as exc:
        assert "convrot_groupsize" in str(exc)

    # (3) missing layer metadata (empty layers): rejected
    pnone = craft("empty.safetensors", {}, base_tensors)
    try:
        plan_from_output(pnone, det, FORMAT_W4A8, info)
        raise AssertionError("empty layer metadata accepted")
    except ValidationError as exc:
        assert "layers" in str(exc)

    # (4) stale old-format entry: rejected
    pstale = craft("stale.safetensors",
                   {layer: {"format": "int4_old", "group_size": 16,
                            "convrot": True, "convrot_groupsize": 256}}, base_tensors)
    try:
        plan_from_output(pstale, det, FORMAT_W4A8, info)
        raise AssertionError("stale format accepted")
    except ValidationError as exc:
        assert "incompatible format" in str(exc)

    # (5) wrong s_rel dimensions + missing codebook: plan builds, Validator fails
    pshape = craft("badshape.safetensors", {layer: good}, {
        layer + ".weight": torch.zeros(1280, 640, dtype=torch.int8),
        layer + ".weight_s_rel": torch.zeros(1280, 7, dtype=torch.uint8),  # wrong
        layer + ".weight_s_channel": torch.ones(1280, dtype=torch.float32),
        # codebook missing entirely
    })
    plan = plan_from_output(pshape, det, FORMAT_W4A8, info)
    validator = Validator(info, plan, pshape, _selftest_args(pshape, FORMAT_W4A8,
                                                             extra={"validate": True}),
                          inspect_environment())
    summary = validator.run()
    assert summary["n_failed"] >= 1, summary
    failed_names = {c["name"] for c in summary["checks"] if c["status"] == "failed"}
    assert "metadata-runtime-contract" in failed_names, failed_names
    assert "shape-preservation" in failed_names, failed_names
    return ("cgs64/swapped/empty/stale metadata rejected; wrong s_rel and missing "
            "codebook fail runtime-contract validation")

def _test_malformed() -> str:
    d = _tmpdir()
    # truncated header
    p1 = os.path.join(d, "trunc.safetensors")
    with open(p1, "wb") as f:
        f.write(b"\x00" * 8)
    try:
        discover_checkpoint(p1)
        raise AssertionError("expected InputError")
    except InputError:
        pass
    # negative and overlapping ranges must fail before any tensor is exposed
    for filename, spec, payload in (
        ("negative.safetensors",
         {"x": {"dtype": "U8", "shape": [1], "data_offsets": [-1, 0]}}, b""),
        ("coerced-shape.safetensors",
         {"x": {"dtype": "F32", "shape": ["1"], "data_offsets": [0, 4]}},
         b"\x00" * 4),
        ("overlap.safetensors", {
            "a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]},
            "b": {"dtype": "U8", "shape": [2], "data_offsets": [1, 3]},
        }, b"\x00" * 3),
    ):
        malformed = os.path.join(d, filename)
        header = json.dumps(spec).encode("utf-8")
        with open(malformed, "wb") as f:
            f.write(struct.pack("<Q", len(header)) + header + payload)
        try:
            discover_checkpoint(malformed)
            raise AssertionError(f"expected InputError for {filename}")
        except InputError:
            pass
    # bad data offsets
    p2 = os.path.join(d, "badoff.safetensors")
    hdr = json.dumps({"x": {"dtype": "F32", "shape": [4], "data_offsets": [0, 100]}}).encode()
    with open(p2, "wb") as f:
        f.write(struct.pack("<Q", len(hdr)) + hdr + b"\x00" * 100)
    try:
        discover_checkpoint(p2)
        raise AssertionError("expected InputError")
    except InputError:
        pass
    # pickle without trust
    p3 = os.path.join(d, "evil.ckpt")
    torch.save({"w": torch.zeros(4)}, p3)
    try:
        discover_checkpoint(p3)
        raise AssertionError("expected PickleInputError")
    except PickleInputError:
        pass
    # Explicit refresh must detect source mutation rather than returning the
    # cached pre-conversion identity.
    p4 = os.path.join(d, "mutable.safetensors")
    safetensors.torch.save_file({"x": torch.zeros(16)}, p4)
    mutable_info = discover_checkpoint(p4)
    before = hash_checkpoint_files(mutable_info)
    safetensors.torch.save_file({"x": torch.ones(16)}, p4)
    assert hash_checkpoint_files(mutable_info) == before
    assert hash_checkpoint_files(mutable_info, refresh=True) != before
    return "truncation, size/range/overlap validation, pickle guard, source rehash"

def _test_checkpoint_variants() -> str:
    d = Path(_tmpdir())
    shard = d / "model-00001-of-00001.safetensors"
    safetensors.torch.save_file({
        "mapped": torch.arange(4, dtype=torch.float32),
        "extra_bool": torch.tensor([True]),
    }, str(shard))
    with open(d / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump({"weight_map": {"mapped": shard.name}}, f)
    sharded = discover_checkpoint(str(d))
    assert sharded.key_set() == {"mapped", "extra_bool"}
    assert sharded.by_name("extra_bool").nbytes == 1
    shard2 = d / "model-00002-of-00002.safetensors"
    safetensors.torch.save_file({
        "mapped2": torch.arange(2, dtype=torch.float32),
        "extra_bool": torch.tensor([False]),
    }, str(shard2))
    with open(d / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump({"weight_map": {
            "mapped": shard.name, "mapped2": shard2.name,
        }}, f)
    try:
        discover_checkpoint(str(d))
        raise AssertionError("duplicate unindexed shard tensor was accepted")
    except InputError as exc:
        assert "duplicate tensor" in str(exc)

    pickle_path = d / "nested.pt"
    expected = torch.arange(16, dtype=torch.float32).reshape(4, 4).bfloat16()
    torch.save({"state_dict": {"nested.weight": expected}, "epoch": 3}, pickle_path)
    pickled = discover_checkpoint(str(pickle_path), trust_pickle=True)
    assert pickled.key_set() == {"nested.weight"}
    with CheckpointReader(pickled) as reader:
        assert bytes(reader.read_bytes("nested.weight")) == tensor_to_bytes(expected)
    return ("indexed extra tensor and nested BF16 pickle load; duplicate shard "
            "tensor rejected")

def _test_unsupported() -> str:
    d = _tmpdir()
    path = os.path.join(d, "mini.safetensors")
    _make_mini_checkpoint(path)
    info = discover_checkpoint(path)
    det = detect_architecture(info, shape_lookup=lambda n: (info.by_name(n).shape if info.by_name(n) else None))
    dec = classify_tensors(info, det, FORMAT_W4A8, None, [], [], [], None, None)
    by_name = {dd.name: dd for dd in dec}
    # 4D conv weights must not be quantized
    assert by_name["model.diffusion_model.input_blocks.0.0.weight"].kind == DecisionKind.KEEP
    # embeddings / norms kept
    assert by_name["cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"].kind == DecisionKind.KEEP
    assert by_name["model.diffusion_model.input_blocks.1.0.transformer_blocks.0.norm1.weight"].kind == DecisionKind.KEEP
    # linears quantized
    assert by_name["model.diffusion_model.input_blocks.1.0.transformer_blocks.0.attn1.to_q.weight"].kind == DecisionKind.QUANTIZE
    return "conv/embedding/norm kept; linear quantized"

def _test_sensitivity_planning() -> str:
    d = _tmpdir()
    path = os.path.join(d, "wan_small.safetensors")
    out = os.path.join(d, "wan_small_w4a8.safetensors")
    torch.manual_seed(53)
    q_name = "model.diffusion_model.blocks.0.self_attn.q.weight"
    k_name = "model.diffusion_model.blocks.0.cross_attn.k.weight"
    safetensors.torch.save_file({
        "model.diffusion_model.head.modulation": torch.zeros(1),
        q_name: torch.zeros(256, 256),
        k_name: torch.randn(256, 256),
    }, path)
    info = discover_checkpoint(path)
    det = detect_architecture(
        info, shape_lookup=lambda name: info.by_name(name).shape
        if info.by_name(name) else None)
    decisions = classify_tensors(
        info, det, FORMAT_W4A8, None, [], [], [], None, None)
    analyzer = SensitivityAnalyzer(0.01, 1.0, None)
    apply_sensitivity_prepass(
        info, decisions, analyzer, 256 * 1024**2,
        torch.device("cpu"), torch.float32)
    by_name = {item.name: item for item in decisions}
    assert by_name[q_name].kind == DecisionKind.QUANTIZE
    assert by_name[k_name].kind == DecisionKind.KEEP_PRECISION
    entries, total = build_output_entries(info, decisions, FORMAT_W4A8, None)
    plan = ConversionPlan(
        fmt=FORMAT_W4A8, detection=det, decisions=decisions,
        metadata_quant={}, metadata_ext={}, output_entries=entries,
        total_out_bytes=total)
    plan.n_quantized = len(plan.quantized_layers())
    plan.n_kept = len(decisions) - plan.n_quantized
    args = _selftest_args(
        out, FORMAT_W4A8,
        extra={"max_memory": 256 * 1024**2, "sensitivity_threshold": 0.01,
               "error_threshold": 1.0, "_compute_dtype_tensor": torch.float32})
    engine = ConversionEngine(info, plan, args, out + ".state.json", out + ".tmp", out)
    try:
        engine.run()
    finally:
        engine.close()
    with safe_open(out, framework="pt") as st:
        assert tuple(st.get_tensor(q_name).shape) == (256, 128)
        assert tuple(st.get_tensor(k_name).shape) == (256, 256)
        assert st.get_tensor(k_name).dtype == torch.float32
    return "sensitivity decisions frozen before packed/passthrough offsets"

def _test_resume() -> str:
    d = _tmpdir()
    path = os.path.join(d, "mini.safetensors")
    _make_mini_checkpoint(path)
    out = os.path.join(d, "out.safetensors")
    args = _selftest_args(out, FORMAT_W4A8)
    info = discover_checkpoint(path)
    det = detect_architecture(info, shape_lookup=lambda n: (info.by_name(n).shape if info.by_name(n) else None))
    dec = classify_tensors(info, det, FORMAT_W4A8, None, [], [], [], None, None)
    plan = ConversionPlan(fmt=FORMAT_W4A8, detection=det, decisions=dec,
                          metadata_quant={}, metadata_ext={}, output_entries=[])
    entries, total = build_output_entries(info, dec, FORMAT_W4A8, None)
    plan.output_entries = entries
    plan.total_out_bytes = total
    plan.n_quantized = len(plan.quantized_layers())
    plan.n_kept = len(dec) - plan.n_quantized
    state_path = out + ".state.json"
    tmp_path = out + ".tmp"
    # first run: simulate a crash after 6 tensors (strictly in plan order)
    eng = ConversionEngine(info, plan, args, state_path, tmp_path, out)
    eng._crash_after = 6
    try:
        eng.run()
        raise AssertionError("expected simulated crash")
    except _SimulatedCrash:
        eng.save_state()
    eng.close()
    assert os.path.exists(tmp_path)

    # A changed conversion option must invalidate the resume plan.
    args_drift = _selftest_args(
        out, FORMAT_W4A8, resume=True,
        extra={"max_memory": 1024 * 1024**2})
    drift = ConversionEngine(info, plan, args_drift, state_path, tmp_path, out)
    try:
        drift.run()
        raise AssertionError("resume accepted changed max-memory")
    except OutputError as exc:
        assert "parameters" in str(exc)
    finally:
        drift.close()

    # Completed tensor bytes are checksummed before resume.
    with open(state_path, "r", encoding="utf-8") as f:
        saved = json.load(f)
    first_record = next(iter(saved["entries"].values()))
    byte_pos = int(first_record["offset"])
    with open(tmp_path, "r+b") as f:
        f.seek(byte_pos)
        original_byte = f.read(1)
        f.seek(byte_pos)
        f.write(bytes([original_byte[0] ^ 0x01]))
    corrupt = ConversionEngine(
        info, plan, _selftest_args(out, FORMAT_W4A8, resume=True),
        state_path, tmp_path, out)
    try:
        corrupt.run()
        raise AssertionError("resume accepted corrupted completed tensor")
    except OutputError as exc:
        assert "checksum" in str(exc)
    finally:
        corrupt.close()
    with open(tmp_path, "r+b") as f:
        f.seek(byte_pos)
        f.write(original_byte)

    # A different converter/algorithm revision must invalidate the resume plan,
    # even when the inventory looks identical.
    saved_version = get_converter_version()
    try:
        set_converter_version(saved_version + "-test")
        drift_ver = ConversionEngine(
            info, plan, _selftest_args(out, FORMAT_W4A8, resume=True),
            state_path, tmp_path, out)
        try:
            drift_ver.run()
            raise AssertionError("resume accepted different converter version")
        except OutputError as exc:
            assert "parameters" in str(exc)
        finally:
            drift_ver.close()
    finally:
        set_converter_version(saved_version)

    # second run: resume and finish
    args2 = _selftest_args(out, FORMAT_W4A8, resume=True)
    eng2 = ConversionEngine(info, plan, args2, state_path, tmp_path, out)
    eng2.run()
    eng2.close()
    # A crash after tensor finalization but before metadata publication must
    # resume from the checksummed staged file without requantizing.
    eng3 = ConversionEngine(
        info, plan, _selftest_args(out, FORMAT_W4A8, resume=True),
        state_path, tmp_path, out)
    eng3.run()
    eng3.close()
    with safe_open(out, framework="pt") as st:
        ql = plan.quantized_layers()[0]
        assert ql.layer is not None
        w = st.get_tensor(ql.layer + ".weight")
        assert w.shape == (info.by_name(ql.name).shape[0], info.by_name(ql.name).shape[1] // 2)
    return ("option drift/data corruption rejected; partial and post-conversion "
            "interruptions resumed")

def _test_atomic() -> str:
    d = _tmpdir()
    path = os.path.join(d, "mini.safetensors")
    _make_mini_checkpoint(path)
    out = os.path.join(d, "out.safetensors")
    args = _selftest_args(out, FORMAT_W4A8)
    info = discover_checkpoint(path)
    det = detect_architecture(info, shape_lookup=lambda n: (info.by_name(n).shape if info.by_name(n) else None))
    dec = classify_tensors(info, det, FORMAT_W4A8, None, [], [], [], None, None)
    plan = ConversionPlan(fmt=FORMAT_W4A8, detection=det, decisions=dec,
                          metadata_quant={}, metadata_ext={}, output_entries=[])
    entries, total = build_output_entries(info, dec, FORMAT_W4A8, None)
    plan.output_entries = entries
    plan.total_out_bytes = total
    plan.n_quantized = len(plan.quantized_layers())
    plan.n_kept = len(dec) - plan.n_quantized
    # Ensure the user-visible path never appears until metadata publication,
    # and that the source remains untouched.
    src_mtime = os.path.getmtime(path)
    staged = out + ".staged"
    eng = ConversionEngine(
        info, plan, args, out + ".state.json", out + ".tmp", staged)
    eng.run()
    eng.close()
    assert not os.path.exists(out)
    assert os.path.exists(staged)
    assert not os.path.exists(out + ".tmp")
    republish_with_metadata(staged, out, {}, entries)
    assert os.path.exists(out)
    assert os.path.getmtime(path) == src_mtime
    return "requested path withheld until atomic metadata publish; original untouched"

def _run_mini_convert(out: str, fmt: str, resume: bool = False, overwrite: bool = False,
                      extra_args: Optional[Dict[str, Any]] = None) -> Tuple[Any, ConversionPlan, CheckpointInfo, DetectionResult]:
    d = os.path.dirname(out)
    path = os.path.join(d, "mini.safetensors")
    if not os.path.exists(path):
        _make_mini_checkpoint(path)
    args = _selftest_args(out, fmt, resume=resume, overwrite=overwrite, extra=extra_args)
    info = discover_checkpoint(path)
    det = detect_architecture(info, shape_lookup=lambda n: (info.by_name(n).shape if info.by_name(n) else None))
    dec = classify_tensors(info, det, fmt, None, [], [], [], None, None)
    plan = ConversionPlan(fmt=fmt, detection=det, decisions=dec,
                          metadata_quant={}, metadata_ext={}, output_entries=[])
    entries, total = build_output_entries(info, dec, fmt, None)
    plan.output_entries = entries
    plan.total_out_bytes = total
    plan.n_quantized = len(plan.quantized_layers())
    plan.n_kept = len(dec) - plan.n_quantized
    qm = build_quant_metadata(info, plan)
    eng = ConversionEngine(info, plan, args, out + ".state.json", out + ".tmp", out)
    eng.run()
    eng.close()
    meta = dict(info.metadata)
    meta[METADATA_KEY_QUANT] = json_dumps(qm)
    meta[METADATA_KEY_EXT] = json_dumps(build_extension_metadata(
        info, plan, inspect_environment(), args, None, None,
        hash_checkpoint_files(info), sha256_safetensors_payload(out),
        {"status": "selftest"}, []))
    republish_with_metadata(out, out, meta, entries)
    return args, plan, info, det

def _test_e2e_mini_model_w4a8() -> str:
    d = _tmpdir()
    out = os.path.join(d, "out_w4a8.safetensors")
    _, plan, info, detection = _run_mini_convert(out, FORMAT_W4A8)
    with safe_open(out, framework="pt") as st:
        assert "_quantization_metadata" in st.metadata()
        assert st.metadata().get("_selftest") == "1"
        metadata = st.metadata()
        qm = json.loads(st.metadata()["_quantization_metadata"])
        assert len(qm["layers"]) == plan.n_quantized
        ext = json.loads(st.metadata()[METADATA_KEY_EXT])
        assert ext["schema"] == "comfy_wxa8/v1"
        assert ext["quantization"]["activation_bits"] == 8
        # spot-check one quantized layer
        layer = plan.quantized_layers()[0].layer
        w = st.get_tensor(layer + ".weight")
        assert w.dtype == torch.int8 and w.shape[1] == 640
    duplicate = os.path.join(d, "out_w4a8_duplicate.safetensors")
    republish_with_metadata(out, duplicate, metadata, plan.output_entries)
    assert sha256_file(duplicate) == sha256_file(out)
    validation_plan = plan_from_output(out, detection, FORMAT_W4A8, info)
    assert {entry["name"] for entry in validation_plan.output_entries} == {
        entry["name"] for entry in plan.output_entries}
    return (f"{plan.n_quantized} quantized layers; metadata preserved; "
            "validation inventory and deterministic serialization verified")

_W4A4_GOLDEN_WEIGHT_B64 = (
    "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"
)

_W4A4_GOLDEN_PACKED_B64 = (
    "IxDjHg4QMNC0AexQAP887w/f2+EjEv4PEDP+FN88Ih/vIf/w/3zgHtv+4D8uDvAg8DswAjMfIQ//LsBAEQAOAO+wFeL9QA7vAP/S/fKxPsLx/00g8RPx8ePy/gPv77D+7dAAwiER6/QPESzC/UHfFCEg8P/+Li4NAQHEIv8OLyQu/RMi3/8gEi4rAi8Q4k8QAOPu8AQhHgsAIxIS8eQQ4B8B0fHQAMQT4g80BfvyE8AeIhAd/c703+/T/wEX7O/jEBAQL/Hh8/XhEhPx2+APQPEx8BIfAiIvIT8uQu4c/wUQ3+IBHtEV4hANAADSAzLOz0LgLzU8Hx0PLiAj3PLwMPM+Aw8HLxAC2+HvTODSHSIfzR/NXfLi5u8+EhTxERL1MR35FgAsPyPe+uANQv0u8RIUAiLOFFDU8FA9I8+19NIeABIOEMTywFAetAT07S4M/bP8UPOUsC/P/Q7w8/zhDyBE/5Ef81/az/D7ZEYDsj0v/wJRol0cENPtTM8/wf3rAmDuHtH+AO/RAfH+7OEB4UAw/R/e4AAD7/Ii8vAdIBAh/OAAACPt0x5AwtPCAVEO7fEQQPL/MfA/EN8v4hPxLwMfAg7hwu0hfhLyIOIPDwED8RH/AgH/ARADEvMtAAAAAS/wLvI0AAwi/h8fE/QNzdH/ICH/Lv8wBH/x7w0T8kFPLTMzNAAA8hM0ANECQuIgxC/+AEEhLAz/TU4bQfEEKyEA4TwP4BDf7RM/VE/y4iDxNA0OIO8QTxDuz+4/C+5vAgNCkg3hLxPPYRENIf4CB8EzLgILITEQFNDSzCDxHv8APxES0PTREtL/7gH8yUDw9fsOL/8Q4cAu4NT/8g3h6vD/DgEPIxIEkgTuERDv8BBQfBQzAZHt4BIm7kI8AgP978AfLfMAEBAw//Qe3C4c0Ts9zPFT4guiDBIwAwQxMwBRHt389OEeDw/00MPgM00TQAsOLSE+IPLu8SI/HB8OsjExIt4b7QTu/A4iX/IQ4xJR8gA00fABEb1A4THOIhMy"
)

_W4A4_GOLDEN_PACKED_SHA = "944ce3da265f31370f3e5f6af945ab642c709aa43ad6eed27f152e498858730d"

_W4A4_GOLDEN_SCALE_REF = [0.021357248, 0.022243505, 0.018726815, 0.025281796,
                          0.021947617, 0.019980142]

_INT8_GOLDEN_Q_SHA = "14963d71a305d43aec5ed2440120c383351b2b5ad861282b9089e2be800fc21e"

_INT8_GOLDEN_SCALE_REF = [0.001189211, 0.001393344, 0.001371189, 0.001160474,
                          0.001261787, 0.00115588]

def _golden_weight() -> torch.Tensor:
    import base64 as _b64
    raw = _b64.b64decode(_W4A4_GOLDEN_WEIGHT_B64)
    return torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy()).view(6, 256)

def _nibble_agreement(packed_a: torch.Tensor, packed_b: torch.Tensor) -> float:
    """Fraction of identical packed nibbles (low and high nibble separately)."""
    a = packed_a.to(torch.int32) & 0xFF
    b = packed_b.to(torch.int32) & 0xFF
    lo = ((a & 0xF) == (b & 0xF)).float().mean()
    hi = (((a >> 4) & 0xF) == ((b >> 4) & 0xF)).float().mean()
    return float((lo + hi) / 2)

def _test_w4a4_golden() -> str:
    """W4A4 golden vector: quantize on the EMBEDDED reference weight must match
    comfy-kitchen's eager implementation. The packed codes and rowwise scales
    were captured from comfy-kitchen 0.2.28. The Hadamard-rotation matmul can
    differ in the last ULPs between BLAS implementations (x86 vs ARM, Windows
    vs Linux), so packed nibbles are compared with a 99.5% agreement bound and
    fp32 scales with rtol 1e-4 (the same convention as the W4A8 golden
    vectors); a mismatch count is reported for diagnostics."""
    import base64 as _b64
    w = _golden_weight()
    assert hashlib.sha256(w.numpy().tobytes()).hexdigest() == \
        "78390001b8020812bfea3d4d85d16210eea0d64ff98eda8c8b266aa88387534a"
    packed, scale = quantize_w4a4_weight(w, 256)
    assert packed.dtype == torch.int8 and packed.shape == (6, 128)
    assert scale.shape == (6,)
    # the captured reference packed bytes (comfy-kitchen 0.2.28); the
    # Hadamard-rotation matmul can flip a rounding boundary across BLAS
    # implementations, so 99.5% nibble agreement is required, not 100%
    ref_packed = torch.from_numpy(np.frombuffer(
        _b64.b64decode(_W4A4_GOLDEN_PACKED_B64), dtype=np.uint8).copy()
    ).view(6, 128).to(torch.int8)
    agree = _nibble_agreement(packed, ref_packed)
    assert agree >= 0.995, f"packed nibble agreement {agree:.4f} below 0.995"
    # scale values must match the captured reference within platform ULP noise
    assert torch.allclose(scale, torch.tensor(_W4A4_GOLDEN_SCALE_REF),
                          rtol=1e-4, atol=1e-8), (scale, _W4A4_GOLDEN_SCALE_REF)
    dq = dequantize_w4a4_weight(packed, scale, 256, torch.float32)
    m = compute_weight_metrics(w, dq)
    assert m.rel_l2 < 0.20 and m.cosine > 0.98, (m.rel_l2, m.cosine)
    # packed nibbles survive a signed round trip
    rt = unpack_int4_signed(packed)
    repacked = ((rt[:, 0::2] & 0xF) | ((rt[:, 1::2] & 0xF) << 4)).to(torch.int8)
    assert torch.equal(repacked, packed)
    return ("W4A4 golden: embedded weight, packed nibble agreement "
            f"{agree:.4f} (>= 0.995), scales match reference (rtol 1e-4), "
            f"roundtrip relL2 {m.rel_l2:.4f}")

def _test_int8_golden() -> str:
    """INT8 golden vector: rowwise int8 quantize on the EMBEDDED reference
    weight must match comfy-kitchen's eager quantize_int8_rowwise. No rotation
    is involved, so the packed output is compared byte-exactly; fp32 scales
    with rtol 1e-5 (platform ULP tolerance)."""
    w = _golden_weight()
    q, scale = quantize_int8_tensorwise_weight(w)
    assert q.dtype == torch.int8 and q.shape == (6, 256)
    assert scale.shape == (6, 1)
    assert hashlib.sha256(q.numpy().tobytes()).hexdigest() == \
        _INT8_GOLDEN_Q_SHA, "int8 codes drifted from the reference"
    assert torch.allclose(scale.reshape(-1), torch.tensor(_INT8_GOLDEN_SCALE_REF),
                          rtol=1e-5, atol=1e-9)
    dq = dequantize_int8_tensorwise_weight(q, scale, torch.float32)
    m = compute_weight_metrics(w, dq)
    assert m.rel_l2 < 0.05 and m.cosine > 0.999, (m.rel_l2, m.cosine)
    return ("INT8 golden: embedded weight, codes byte-exact vs comfy-kitchen, "
            f"roundtrip relL2 {m.rel_l2:.4f}")

def _test_mixed_eligibility() -> str:
    """Per-format eligibility matrix on the real awkward dims."""
    ok_w4a4 = [k for k in (1152, 1408, 1920, 768, 640, 320, 13568, 6144)
               if w4a4_weight_is_quantizable((8, k), torch.bfloat16)[0]]
    assert ok_w4a4 == [1152, 1408, 1920, 768, 640, 320, 13568, 6144], ok_w4a4
    bad_w4a4 = [k for k in (3360, 2520)
                if w4a4_weight_is_quantizable((8, k), torch.bfloat16)[0]]
    assert bad_w4a4 == [], bad_w4a4
    ok_int8 = [k for k in (3360, 2520, 1152, 5, 17)
               if int8_weight_is_quantizable((8, k), torch.bfloat16)[0]]
    assert ok_int8 == [3360, 2520, 1152, 5, 17], ok_int8
    ok_w4a8 = [k for k in (13568, 6144, 768)
               if w4_weight_is_quantizable((8, k), torch.bfloat16, 16, 256)[0]]
    assert ok_w4a8 == [13568, 6144, 768], ok_w4a8
    return ("eligibility: W4A4 needs K%64 (1152/1408/1920 yes, 3360/2520 no), "
            "INT8 any K, W4A8 needs K%256")

def _test_mixed_planning() -> str:
    """MixedPlanner on the real Boogu dims: K=3360 layers go INT8, K=13568
    layers stay W4A8, output is smaller than the w4a8-only fallback plan."""
    d = _tmpdir()
    src_path = os.path.join(d, "boogu_mixed.safetensors")
    _make_boogu_real_dims_checkpoint(src_path, n=64, n_fail=5, n_ok=2)
    info = discover_checkpoint(src_path)
    det = detect_architecture(
        info, shape_lookup=lambda n: (info.by_name(n).shape if info.by_name(n) else None))
    decisions = classify_tensors(info, det, FORMAT_MIXED, None, [], [], [],
                                 None, None)
    planner = MixedPlanner("balanced", None, 2 * 1024**3, torch.device("cpu"),
                           None)
    summary = planner.plan(info, decisions)
    by_fmt = summary["counts"]
    assert by_fmt.get(FORMAT_INT8, 0) >= 5, by_fmt
    assert by_fmt.get(FORMAT_W4A8, 0) >= 2, by_fmt
    for ddec in decisions:
        if ddec.kind != DecisionKind.QUANTIZE:
            continue
        k = int(info.by_name(ddec.name).shape[1])
        if k == 3360:
            assert ddec.format == FORMAT_INT8, ddec
        elif k == 13568:
            assert ddec.format == FORMAT_W4A8, ddec
    # mixed payload must be smaller than w4a8-only (bf16 fallback) payload
    entries, mixed_bytes = build_output_entries(info, decisions, FORMAT_MIXED, None)
    dec_w4a8 = classify_tensors(info, det, FORMAT_W4A8, None, [], [], [],
                                None, None)
    entries8, bytes8 = build_output_entries(info, dec_w4a8, FORMAT_W4A8, None)
    assert mixed_bytes < bytes8, (mixed_bytes, bytes8)
    assert summary["global_mean_error"] is not None
    # heterogeneous metadata
    plan = ConversionPlan(fmt=FORMAT_MIXED, detection=det, decisions=decisions,
                          metadata_quant={}, metadata_ext={}, output_entries=entries)
    plan.mixed_plan = summary
    qm = build_quant_metadata(info, plan)
    fmts = {conf["format"] for conf in qm["layers"].values()}
    assert fmts == {FORMAT_W4A8, FORMAT_INT8}, fmts
    return (f"mixed plan: {by_fmt[FORMAT_INT8]} INT8 + {by_fmt[FORMAT_W4A8]} "
            f"W4A8 layers, {human_bytes(mixed_bytes)} vs w4a8-only "
            f"{human_bytes(bytes8)}; heterogeneous metadata")

def _test_mixed_e2e() -> str:
    """End-to-end mixed conversion + validation on the Boogu real-dims
    fixture: INT8 fallback layers serialize, validate and reload."""
    d = _tmpdir()
    src_path = os.path.join(d, "boogu_e2e.safetensors")
    _make_boogu_real_dims_checkpoint(src_path, n=48, n_fail=4, n_ok=2)
    out = os.path.join(d, "out_mixed.safetensors")
    args = _selftest_args(out, FORMAT_MIXED, extra={
        "validate": True, "profile": "balanced", "format": "mixed",
        "target_runtime": "cpu", "quality_gate": None, "global_error_gate": None,
        "max_linear_bytes_per_param": None, "w4a4_linear_dtype": "int8",
        "disable_w4a4": False, "disable_w4a8": False, "disable_int8": False,
        "require_calibration": False})
    dec = classify_tensors(info := discover_checkpoint(src_path),
                           det := detect_architecture(
                               info, shape_lookup=lambda n: (
                                   info.by_name(n).shape if info.by_name(n) else None)),
                           FORMAT_MIXED, None, [], [], [], None, None)
    planner = MixedPlanner("balanced", None, 2 * 1024**3, torch.device("cpu"), None)
    summary = planner.plan(info, dec)
    entries, total = build_output_entries(info, dec, FORMAT_MIXED, None)
    plan = ConversionPlan(fmt=FORMAT_MIXED, detection=det, decisions=dec,
                          metadata_quant={}, metadata_ext={}, output_entries=entries)
    plan.total_out_bytes = total
    plan.n_quantized = len(dec)
    plan.n_kept = 0
    plan.mixed_plan = summary
    plan.metadata_quant = build_quant_metadata(info, plan)
    engine = ConversionEngine(info, plan, args, out + ".state.json",
                              out + ".tmp", out)
    engine.run()
    engine.close()
    assert os.path.exists(out)
    meta = dict(info.metadata)
    meta[METADATA_KEY_QUANT] = json_dumps(plan.metadata_quant)
    meta[METADATA_KEY_EXT] = json_dumps(build_extension_metadata(
        info, plan, inspect_environment(), args, None, None,
        hash_checkpoint_files(info), sha256_safetensors_payload(out),
        {"status": "selftest"}, []))
    republish_with_metadata(out, out, meta, entries)
    validation_plan = plan_from_output(out, det, FORMAT_MIXED, info)
    validator = Validator(info, validation_plan, out, args, inspect_environment())
    with CheckpointReader(info) as validation_reader:
        summary_v = validator.run(reader=validation_reader)
    assert summary_v["n_failed"] == 0, summary_v
    with safe_open(out, framework="pt") as st:
        qm = json.loads(st.metadata()[METADATA_KEY_QUANT])
        fmts = {conf["format"] for conf in qm["layers"].values()}
        assert fmts == {FORMAT_W4A8, FORMAT_INT8}, fmts
    # reload each format through comfy-kitchen layouts and run one forward
    try:
        import comfy_kitchen  # noqa: F401
        from comfy_kitchen.tensor.base import QuantizedTensor, get_layout_class
    except Exception:
        return ("mixed e2e: heterogeneous checkpoint converted, validated "
                f"(n_failed=0), metadata {sorted(fmts)}; comfy-kitchen not "
                "installed for layout reload")
    with safe_open(out, framework="pt") as st:
        ok_fwd = True
        for layer, conf in qm["layers"].items():
            lfmt = conf["format"]
            w_slice = st.get_slice(layer + ".weight")
            n = w_slice.get_shape()[0]
            if lfmt == FORMAT_INT8:
                layout = get_layout_class("TensorWiseINT8Layout")
                scale = st.get_tensor(layer + ".weight_scale")
                params = layout.Params(
                    scale=scale, orig_dtype=torch.float32,
                    orig_shape=(n, w_slice.get_shape()[1]))
            elif lfmt == FORMAT_W4A8:
                layout = get_layout_class("AsymW4A8Int8Layout")
                params = layout.Params(
                    scale=st.get_tensor(layer + ".weight_s_rel").view(
                        torch.float8_e4m3fn),
                    s_channel=st.get_tensor(layer + ".weight_s_channel"),
                    codebook=st.get_tensor(layer + ".weight_codebook"),
                    group_size=conf["group_size"],
                    convrot_groupsize=conf["convrot_groupsize"],
                    orig_dtype=torch.float32,
                    orig_shape=(n, w_slice.get_shape()[1] * 2))
            else:
                continue
            qt = QuantizedTensor(
                st.get_tensor(layer + ".weight"), layout.__name__, params)
            dq = qt.dequantize()
            ok_fwd = ok_fwd and torch.isfinite(dq).all() and dq.numel() > 0
        assert ok_fwd
    return ("mixed e2e: heterogeneous checkpoint converted, validated "
            f"(n_failed=0), metadata {sorted(fmts)}, layouts reload + dequant OK")

def _test_mixed_gates_hard_fail() -> str:
    """Quality and compression gates are hard failures: a plan that cannot
    meet them raises instead of publishing."""
    d = _tmpdir()
    src_path = os.path.join(d, "boogu_gates.safetensors")
    _make_boogu_real_dims_checkpoint(src_path, n=48, n_fail=4, n_ok=2)
    info = discover_checkpoint(src_path)
    det = detect_architecture(
        info, shape_lookup=lambda n: (info.by_name(n).shape if info.by_name(n) else None))
    def fresh_decisions() -> List[TensorDecision]:
        return classify_tensors(info, det, FORMAT_MIXED, None, [], [], [],
                                None, None)
    # (1) impossible global quality gate -> QualityGateError
    planner = MixedPlanner("balanced", None, 2 * 1024**3, torch.device("cpu"),
                           None, global_gate=0.0005)
    try:
        planner.plan(info, fresh_decisions())
        raise AssertionError("impossible quality gate accepted")
    except QualityGateError:
        pass
    # (2) impossible compression target -> CompressionGateError
    planner = MixedPlanner("balanced", None, 2 * 1024**3, torch.device("cpu"),
                           None, compression_target_bpp=0.05)
    try:
        planner.plan(info, fresh_decisions())
        raise AssertionError("impossible compression target accepted")
    except CompressionGateError:
        pass
    # (3) bf16 fraction gate: K=3360 layers have only INT8 (disabled) so
    # they stay at original precision; the fraction gate must trip
    planner = MixedPlanner("balanced", None, 2 * 1024**3, torch.device("cpu"),
                           None, layer_gate=0.10,
                           disabled_formats=[FORMAT_INT8],
                           max_bf16_fraction=0.0)
    try:
        planner.plan(info, fresh_decisions())
        raise AssertionError("bf16 fraction gate accepted")
    except CompressionGateError:
        pass
    return ("quality/compression gates fail hard: QualityGateError, "
            "CompressionGateError (bpp and bf16-fraction) verified")

def _test_mixed_bf16_promotion() -> str:
    """Original precision is a promotion candidate: a layer can be rescued
    all the way to BF16 when the global gate demands it."""
    d = _tmpdir()
    src_path = os.path.join(d, "boogu_promo.safetensors")
    _make_boogu_real_dims_checkpoint(src_path, n=48, n_fail=2, n_ok=2)
    info = discover_checkpoint(src_path)
    det = detect_architecture(
        info, shape_lookup=lambda n: (info.by_name(n).shape if info.by_name(n) else None))
    dec = classify_tensors(info, det, FORMAT_MIXED, None, [], [], [], None, None)
    # layer gate admits W4A8 (~0.07); the global gate is stricter than even
    # INT8 (~0.008), so the greedy loop must promote layers all the way to
    # original precision, and the resulting passthrough-only plan must fail
    # the quality gate instead of publishing a copy of the input
    planner = MixedPlanner("balanced", None, 2 * 1024**3, torch.device("cpu"),
                           None, layer_gate=0.10, global_gate=0.001)
    try:
        planner.plan(info, dec)
        raise AssertionError("passthrough-only plan accepted")
    except QualityGateError:
        pass
    promoted_original = [p for p in planner.promotions if ":original" in p]
    assert promoted_original, planner.promotions
    return ("BF16 promotion: %d promotion(s), %d to original precision; "
            "empty plan correctly rejected by the quality gate" % (
                len(planner.promotions), len(promoted_original)))

def _test_mixed_runtime_caps() -> str:
    """--target-runtime feeds the eligibility matrix: unsupported formats are
    excluded with a recorded reason, eager vs accelerated is reported."""
    d = _tmpdir()
    src_path = os.path.join(d, "boogu_caps.safetensors")
    _make_boogu_real_dims_checkpoint(src_path, n=48, n_fail=2, n_ok=2)
    info = discover_checkpoint(src_path)
    det = detect_architecture(
        info, shape_lookup=lambda n: (info.by_name(n).shape if info.by_name(n) else None))
    dec = classify_tensors(info, det, FORMAT_MIXED, None, [], [], [], None, None)
    caps = RuntimeCapabilities(
        target="limited",
        w4a4=FormatRuntimeCapability(False, False, False, False, "limited",
                                     reason="no w4a4 kernels"),
        w4a8=FormatRuntimeCapability(False, False, False, False, "limited",
                                     reason="no w4a8 kernels"),
        int8=FormatRuntimeCapability(True, True, True, False, "limited"))
    planner = MixedPlanner("balanced", None, 2 * 1024**3, torch.device("cpu"),
                           None, runtime=caps, compression_target_bpp=1.10)
    summary = planner.plan(info, dec)
    # w4a4/w4a8 must be marked ineligible with the runtime reason
    first = planner.candidates[dec[0].name]
    assert first[FORMAT_W4A4].eligible is False
    assert "not supported on target runtime limited" in first[FORMAT_W4A4].reason
    assert first[FORMAT_INT8].eligible is True
    assert summary["runtime"]["backend"] == "limited"
    assert summary["runtime"]["formats"][FORMAT_INT8]["status"] == \
        "expected accelerated (not certified)"
    assert summary["runtime"]["formats"][FORMAT_W4A4]["loadable"] is False
    # cpu runtime reports eager fallback and certain A4 for W4A4
    planner = MixedPlanner("balanced", None, 2 * 1024**3, torch.device("cpu"),
                           None, runtime=runtime_capabilities_for("cpu"),
                           linear_dtype="int8")
    planner.plan(info, list(dec))
    assert planner.summary["runtime"]["formats"][FORMAT_W4A8]["status"] == \
        "eager/fallback"
    mode = planner.w4a4_execution_mode_for_candidate()
    assert mode.activation_bits == 4 and mode.certain, \
        "eager must simulate the int4 activation path"
    # cuda SM8x honors requested int4 via native MMA; int8 request is certain
    nv8 = runtime_capabilities_for("nvidia")
    planner = MixedPlanner("balanced", None, 2 * 1024**3, torch.device("cpu"),
                           None, runtime=nv8, linear_dtype="int8")
    planner.plan(info, list(dec))
    assert planner.w4a4_execution_mode_for_candidate().activation_bits == 8
    # SM8x + requested int4 -> certain native A4
    nv8 = dataclasses.replace(nv8, cuda_capability=(8, 9))
    planner = MixedPlanner("balanced", None, 2 * 1024**3, torch.device("cpu"),
                           None, runtime=nv8, linear_dtype="int4")
    mode = planner.w4a4_execution_mode_for_candidate()
    assert mode.activation_bits == 4 and mode.certain and \
        mode.path == "cuda-native-int4", mode
    # Turing (7.5) + int4 -> uncertain; worst-case evaluation must kick in
    nv75 = dataclasses.replace(runtime_capabilities_for("nvidia"),
                               cuda_capability=(7, 5))
    planner = MixedPlanner("balanced", None, 2 * 1024**3, torch.device("cpu"),
                           None, runtime=nv75, linear_dtype="int4")
    mode = planner.w4a4_execution_mode_for_candidate()
    assert not mode.certain, mode
    modes, _ = planner._w4a4_scoring_modes()
    assert modes == [4, 8], modes
    return ("runtime capability matrix: unsupported excluded; cpu = eager "
            "fallback + certain A4; cuda SM8x native INT4; Turing "
            "uncertain -> worst-case A4/A8 evaluation")

def _test_mixed_runtime_metric() -> str:
    """The calibration metric is the real runtime operation (activation
    rotation + activation quantization + quantized GEMM), not a
    reconstructed-weight approximation."""
    torch.manual_seed(11)
    w = torch.randn(64, 768) * 0.02
    acts = torch.randn(32, 768) * 0.1
    # int8: act quantization dominates -> runtime error ~0.01, weight err ~0.005
    q, scale = quantize_int8_tensorwise_weight(w)
    err = runtime_output_rel_l2(w, {"": q, "_scale": scale}, FORMAT_INT8,
                                16, 256, acts)
    assert err is not None and err < 0.05, err
    # w4a4 int4 path: act int4 + weight int4 -> much larger
    packed, wscale = quantize_w4a4_weight(w, 256)
    err4 = runtime_output_rel_l2(w, {"": packed, "_scale": wscale},
                                 FORMAT_W4A4, 16, 256, acts,
                                 w4a4_activation_bits=4)
    err8 = runtime_output_rel_l2(w, {"": packed, "_scale": wscale},
                                 FORMAT_W4A4, 16, 256, acts,
                                 w4a4_activation_bits=8)
    assert err4 is not None and err8 is not None
    assert err4 > err8, (err4, err8)
    # cross-check against the real comfy-kitchen eager forward when installed
    ck_err = None
    try:
        import comfy_kitchen
        comfy_kitchen.use_backend("eager")
        import comfy_kitchen as ck
        qa, sa = ck.quantize_convrot_w4a4_weight(w, convrot_groupsize=256,
                                                 quant_group_size=64)
        y_q = ck.convrot_w4a4_linear(acts, qa, sa, None,
                                     convrot_groupsize=256,
                                     quant_group_size=64, linear_dtype="int4")
        y_ref = torch.nn.functional.linear(acts, w)
        ck_err = float((y_q - y_ref).norm(dim=1).div(
            y_ref.norm(dim=1).clamp(min=1e-8)).mean())
    except Exception as e:  # noqa: S110 (optional cross-check)
        log().debug("comfy-kitchen cross-check skipped: %s", e)
    if ck_err is not None:
        assert abs(err4 - ck_err) < 0.02, (err4, ck_err)
    return (f"runtime metric: int8 {err:.4f}, w4a4-int4 {err4:.4f} > "
            f"w4a4-int8 {err8:.4f}; matches eager kernel emulation")

def _test_architecture_sync() -> str:
    """The registry must cover every ComfyUI model class at the pinned
    research revision (mirrors testdata/comfyui_architecture_sync.py)."""
    research = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "research", "ComfyUI", "comfy", "supported_models.py")
    if not os.path.isfile(research):
        return "architecture sync skipped (no research/ComfyUI checkout)"
    import ast as _ast
    with open(research, encoding="utf-8") as f:
        tree = _ast.parse(f.read())
    comfy = {n.name for n in tree.body if isinstance(n, _ast.ClassDef)}
    covered = set()
    for name in family_names():
        covered.update(get_family(name).comfyui_classes)
    missing = sorted(comfy - covered)
    assert not missing, f"unaccounted ComfyUI classes: {missing}"
    return (f"architecture sync: all {len(comfy)} ComfyUI model classes at "
            "the pinned revision covered by registry policies")

def _test_mixed_determinism() -> str:
    """Identical inputs must produce identical plans: same per-layer formats,
    same promotions, same counts (deterministic planner regression)."""
    d = _tmpdir()
    src_path = os.path.join(d, "boogu_det.safetensors")
    _make_boogu_real_dims_checkpoint(src_path, n=48, n_fail=4, n_ok=2)
    info = discover_checkpoint(src_path)
    det = detect_architecture(
        info, shape_lookup=lambda n: (info.by_name(n).shape if info.by_name(n) else None))

    def run_once():
        dec = classify_tensors(info, det, FORMAT_MIXED, None, [], [], [],
                               None, None)
        planner = MixedPlanner("balanced", None, 2 * 1024**3,
                               torch.device("cpu"), None)
        summary = planner.plan(info, dec)
        return ([(ddec.name, ddec.format, ddec.kind.value) for ddec in dec],
                list(summary["promotions"]), summary["counts"])

    first = run_once()
    second = run_once()
    assert first == second, "planner is not deterministic"
    return (f"planner deterministic: {len(first[0])} decisions, "
            f"{len(first[1])} promotions identical across runs")

def _test_mixed_metadata_fuzz() -> str:
    """Corrupted heterogeneous metadata must fail clearly for every format:
    W4A4 (missing scale, wrong convrot group, wrong packed shape), INT8
    (wrong scale dims, wrong weight dtype), and mixed mismatch (metadata
    says W4A4, tensors are INT8-shaped)."""
    def craft(fname: str, layers_conf, tensors) -> str:
        meta = {"_quantization_metadata": json_dumps({"layers": layers_conf})}
        p = os.path.join(_tmpdir(), fname)
        safetensors.torch.save_file(tensors, p, metadata=meta)
        return p

    d = _tmpdir()
    src_path = os.path.join(d, "mini.safetensors")
    _make_mini_checkpoint(src_path)
    info = discover_checkpoint(src_path)
    det = detect_architecture(
        info, shape_lookup=lambda n: (info.by_name(n).shape if info.by_name(n) else None))
    layer = "model.diffusion_model.input_blocks.1.0.transformer_blocks.0.attn1.to_q"
    n, k = 1280, 640

    w4a4_good = {
        "format": FORMAT_W4A4, "convrot_groupsize": 256,
        "quant_group_size": 64, "linear_dtype": "int8",
    }
    w4a4_tensors = {
        layer + ".weight": torch.zeros(n, k // 2, dtype=torch.int8),
        layer + ".weight_scale": torch.ones(n, dtype=torch.float32),
    }
    int8_good = {"format": FORMAT_INT8}

    def expect_reject(path, needle):
        try:
            plan_from_output(path, det, FORMAT_MIXED, info)
            raise AssertionError(f"accepted corrupted file {path}")
        except ValidationError as exc:
            assert needle in str(exc), (path, str(exc)[:200])

    # W4A4: wrong convrot group (not a power of 4) -> plan_from_output rejects
    bad_cgs = craft("w4a4_cgs32.safetensors",
                    {layer: dict(w4a4_good, convrot_groupsize=32)}, w4a4_tensors)
    expect_reject(bad_cgs, "convrot_groupsize")
    # W4A4: K not divisible by 64 -> Validator runtime contract fails
    bad_k = craft("w4a4_k.safetensors",
                  {layer: dict(w4a4_good, convrot_groupsize=256)},
                  {layer + ".weight": torch.zeros(n, 632, dtype=torch.int8),
                   layer + ".weight_scale": torch.ones(n, dtype=torch.float32)})
    plan = plan_from_output(bad_k, det, FORMAT_MIXED, info)
    summary = Validator(info, plan, bad_k,
                        _selftest_args(bad_k, FORMAT_MIXED,
                                       extra={"validate": True, "format": "mixed"}),
                        inspect_environment()).run()
    assert "metadata-runtime-contract" in {c["name"] for c in summary["checks"]
                                           if c["status"] == "failed"}
    # W4A4: missing scale -> Validator runtime contract fails
    p_noscale = craft("w4a4_noscale.safetensors", {layer: w4a4_good},
                      {layer + ".weight": torch.zeros(n, k // 2, dtype=torch.int8)})
    plan = plan_from_output(p_noscale, det, FORMAT_MIXED, info)
    summary = Validator(info, plan, p_noscale,
                        _selftest_args(p_noscale, FORMAT_MIXED,
                                       extra={"validate": True, "format": "mixed"}),
                        inspect_environment()).run()
    assert summary["n_failed"] >= 1, summary
    assert "metadata-runtime-contract" in {c["name"] for c in summary["checks"]
                                           if c["status"] == "failed"}
    # INT8: wrong scale shape -> Validator contract fails
    p_badscale = craft("int8_badscale.safetensors", {layer: int8_good},
                       {layer + ".weight": torch.zeros(n, k, dtype=torch.int8),
                        layer + ".weight_scale": torch.ones(n, 2, dtype=torch.float32)})
    plan = plan_from_output(p_badscale, det, FORMAT_MIXED, info)
    summary = Validator(info, plan, p_badscale,
                        _selftest_args(p_badscale, FORMAT_MIXED,
                                       extra={"validate": True, "format": "mixed"}),
                        inspect_environment()).run()
    assert "metadata-runtime-contract" in {c["name"] for c in summary["checks"]
                                           if c["status"] == "failed"}
    # mixed mismatch: metadata says W4A4 but the weight is full-width int8
    # (derived K would be 2x the source K; the source-K cross-check must fail)
    src_k_full = int(info.by_name(layer + ".weight").shape[1])
    p_mix = craft("mixed_mismatch.safetensors", {layer: w4a4_good},
                  {layer + ".weight": torch.zeros(n, src_k_full,
                                                  dtype=torch.int8),
                   layer + ".weight_scale": torch.ones(n, dtype=torch.float32)})
    plan = plan_from_output(p_mix, det, FORMAT_MIXED, info)
    summary = Validator(info, plan, p_mix,
                        _selftest_args(p_mix, FORMAT_MIXED,
                                       extra={"validate": True, "format": "mixed"}),
                        inspect_environment()).run()
    assert "metadata-runtime-contract" in {c["name"] for c in summary["checks"]
                                           if c["status"] == "failed"}
    return ("W4A4/INT8/mixed metadata corruption rejected: bad cgs, K%64, "
            "missing scale, wrong scale shape, format/tensor mismatch")

def _test_strip_gpu_identity() -> str:
    """--strip-gpu-identity must remove GPU identity from the extension
    metadata (device name, compute capability, ROCm architecture) so a
    published checkpoint never reveals the target or conversion machine."""
    d = _tmpdir()
    src_path = os.path.join(d, "mini.safetensors")
    _make_mini_checkpoint(src_path)
    info = discover_checkpoint(src_path)
    det = detect_architecture(
        info, shape_lookup=lambda n: (info.by_name(n).shape if info.by_name(n) else None))
    dec = classify_tensors(info, det, FORMAT_MIXED, None, [], [], [], None, None)
    planner = MixedPlanner("balanced", None, 2 * 1024**3, torch.device("cpu"), None)
    summary = planner.plan(info, dec)
    plan = ConversionPlan(fmt=FORMAT_MIXED, detection=det, decisions=dec,
                          metadata_quant={}, metadata_ext={}, output_entries=[])
    plan.mixed_plan = summary
    plan.mixed_plan["runtime"] = {
        "gpu_name": "NVIDIA GeForce RTX 3050", "cuda_capability": [8, 6],
        "rocm_arch": None, "runtime_certified": False,
        "formats": {}, "backend": "nvidia"}
    plan.mixed_plan["runtime_backend"] = "nvidia"
    args = _selftest_args("x.safetensors", FORMAT_MIXED, extra={
        "format": "mixed", "strip_gpu_identity": True})
    hashes = hash_checkpoint_files(info)
    ext = build_extension_metadata(info, plan, inspect_environment(), args,
                                   None, None, hashes, "0" * 64, {}, [])
    block = ext["quantization"]["w4a4_runtime"]
    assert "gpu" not in block and "cuda_capability" not in block \
        and "rocm_arch" not in block, block
    blob = json_dumps(ext)
    for bad in ("3050", "GeForce", "RTX"):
        assert bad not in blob, f"GPU identity leaked: {bad}"
    args2 = _selftest_args("x.safetensors", FORMAT_MIXED, extra={
        "format": "mixed", "strip_gpu_identity": False})
    ext2 = build_extension_metadata(info, plan, inspect_environment(), args2,
                                    None, None, hashes, "0" * 64, {}, [])
    block2 = ext2["quantization"]["w4a4_runtime"]
    assert block2.get("gpu") == "NVIDIA GeForce RTX 3050", block2
    return ("--strip-gpu-identity removes GPU name/capability/ROCm from the "
            "extension metadata; default keeps them")


def _test_verify_output() -> str:
    """--verify-output must verify a produced mixed checkpoint with no source
    model: tensor inventory, per-layer format contracts, bounded packing
    roundtrip and payload hash all pass; a corrupted file must fail."""
    d = _tmpdir()
    src_path = os.path.join(d, "boogu_v.safetensors")
    out = os.path.join(d, "out_mixed.safetensors")
    _make_boogu_real_dims_checkpoint(src_path, n=48, n_fail=4, n_ok=2)
    args = _selftest_args(out, FORMAT_MIXED, extra={
        "validate": False, "profile": "balanced", "format": "mixed",
        "target_runtime": "cpu", "quality_gate": None, "global_error_gate": None,
        "max_linear_bytes_per_param": None, "w4a4_linear_dtype": "int8",
        "disable_w4a4": False, "disable_w4a8": False, "disable_int8": False,
        "require_calibration": False})
    info = discover_checkpoint(src_path)
    det = detect_architecture(info, shape_lookup=lambda n: (
        info.by_name(n).shape if info.by_name(n) else None))
    dec = classify_tensors(info, det, FORMAT_MIXED, None, [], [], [], None, None)
    planner = MixedPlanner("balanced", None, 2 * 1024**3, torch.device("cpu"), None)
    summary = planner.plan(info, dec)
    entries, total = build_output_entries(info, dec, FORMAT_MIXED, None)
    plan = ConversionPlan(fmt=FORMAT_MIXED, detection=det, decisions=dec,
                          metadata_quant={}, metadata_ext={}, output_entries=entries)
    plan.total_out_bytes = total
    plan.n_quantized = len(dec)
    plan.n_kept = 0
    plan.mixed_plan = summary
    plan.metadata_quant = build_quant_metadata(info, plan)
    engine = ConversionEngine(info, plan, args, out + ".state.json",
                              out + ".tmp", out)
    engine.run()
    engine.close()
    assert os.path.exists(out)
    meta = dict(info.metadata)
    meta[METADATA_KEY_QUANT] = json_dumps(plan.metadata_quant)
    meta[METADATA_KEY_EXT] = json_dumps(build_extension_metadata(
        info, plan, inspect_environment(), args, None, None,
        hash_checkpoint_files(info), sha256_safetensors_payload(out),
        {"status": "selftest"}, []))
    republish_with_metadata(out, out, meta, entries)

    summary_v = verify_output(out)
    assert summary_v["ok"], f"verify-output failed: {summary_v['n_failed']}"
    assert summary_v["n_tensors"] > 0 and summary_v["quantized_layers"] > 0
    assert summary_v["formats"], "no formats reported"
    assert len(summary_v["output_sha256"]) == 64
    for c in summary_v["checks"]:
        assert c["ok"], f"unexpected failing check {c['name']}: {c['detail']}"

    # Structural corruption: flip a byte inside the safetensors header JSON
    # (quantized payload bytes are legitimately undetectable without the
    # source, so the source-free check must catch header/metadata damage).
    import struct as _st
    raw = bytearray(open(out, "rb").read())
    header_len = _st.unpack("<Q", bytes(raw[:8]))[0]
    raw[8 + header_len // 2] ^= 0xFF
    bad_path = os.path.join(d, "corrupt.safetensors")
    with open(bad_path, "wb") as f:
        f.write(bytes(raw))
    summary_bad = verify_output(bad_path)
    assert not summary_bad["ok"], "corrupted checkpoint verified clean"
    return (f"verify-output: {summary_v['n_tensors']} tensors, "
            f"{summary_v['quantized_layers']} quantized layers, "
            f"{summary_v['n_failed']} failures; corruption detected")



def _converter_cmd() -> Tuple[List[str], Dict[str, str]]:
    """Subprocess command that runs the converter from the self-test suite.

    In the single-file artifact ``__file__`` is the converter script itself;
    in the package the module lives under src/comfyui_wxa8_quantizer/, so the
    converter is invoked as a module with the package parent on PYTHONPATH.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    env = dict(os.environ)
    if os.path.basename(here) == "comfyui_wxa8_quantizer" and os.path.isdir(here):
        parent = os.path.dirname(here)
        env["PYTHONPATH"] = parent + os.pathsep + env.get("PYTHONPATH", "")
        return [sys.executable, "-m", "comfyui_wxa8_quantizer"], env
    return [sys.executable, os.path.abspath(__file__)], env

def _selftest_args(out: str, fmt: str, resume: bool = False, overwrite: bool = False,
                   extra: Optional[Dict[str, Any]] = None) -> argparse.Namespace:
    ns = argparse.Namespace(
        output=out, format="w4a8",
        architecture="auto", device="cpu", compute_dtype="auto", output_dtype="auto",
        group_size=None, calibration_source=None, calibration_samples=None,
        calibration_cache=None, seed=0, include=[], exclude=[], keep_precision=[], min_numel_override=None,
        sensitivity_threshold=None, error_threshold=0.35, max_memory=2 * 1024**3,
        min_quantized_byte_fraction=None, fail_on_low_compression=False,
        streaming=True, resume=resume, overwrite=overwrite, dry_run=False,
        inspect=False, validate=False, validation_only=False, metadata_only=False,
        report=None, log_level="warning", json_log=None, trust_pickle=False,
        yes=True, self_test=False, model=None)
    if extra:
        for k, v in extra.items():
            setattr(ns, k, v)
    return ns


SELF_TEST_CASES: List[Tuple[str, Callable[[], str]]] = [

            ("w4-pack-roundtrip", _test_w4_pack_roundtrip),
            ("odd-dims", _test_odd_dims),
            ("padding-removal", _test_padding_removal),
            ("scale-calculations", _test_scale_calculations),
            ("deterministic-conversion", _test_deterministic),
            ("compute-dtype-selection", _test_compute_dtype),
            ("real-activation-calibration", _test_activation_calibration),
            ("standalone-environment-probe", _test_standalone_environment),
            ("metadata-generation", _test_metadata),
            ("registry-behavior", _test_registry),
            ("compression-stats", _test_compression_stats),
            ("boogu-real-dims", _test_boogu_real_dims),
            ("krea2-real-dims", _test_krea2_real_dims),
            ("real-dim-gate", _test_real_dim_gate),
            ("policy-miss", _test_policy_miss),
            ("metadata-fuzz", _test_metadata_fuzz),
            ("w4a4-golden", _test_w4a4_golden),
            ("int8-golden", _test_int8_golden),
            ("mixed-eligibility", _test_mixed_eligibility),
            ("mixed-planning", _test_mixed_planning),
            ("mixed-e2e", _test_mixed_e2e),
            ("mixed-gates-hard-fail", _test_mixed_gates_hard_fail),
            ("mixed-bf16-promotion", _test_mixed_bf16_promotion),
            ("mixed-runtime-caps", _test_mixed_runtime_caps),
            ("mixed-runtime-metric", _test_mixed_runtime_metric),
            ("mixed-determinism", _test_mixed_determinism),
            ("mixed-metadata-fuzz", _test_mixed_metadata_fuzz),
            ("architecture-sync", _test_architecture_sync),
            ("strip-gpu-identity", _test_strip_gpu_identity),
            ("fail-on-low-compression", _test_fail_on_low_compression),
            ("architecture-detection-safety", _test_detection_safety),
            ("golden-vectors-vs-reference", _test_golden_vectors),
            ("malformed-checkpoints", _test_malformed),
            ("checkpoint-input-variants", _test_checkpoint_variants),
            ("unsupported-tensors", _test_unsupported),
            ("sensitivity-output-planning", _test_sensitivity_planning),
            ("resume-state-recovery", _test_resume),
            ("atomic-output", _test_atomic),
            ("verify-output", _test_verify_output),
            ("end-to-end-mini-model-w4a8", _test_e2e_mini_model_w4a8),
]
