# Research notes: W4A8 / W3A8 ComfyUI-compatible quantization

## Exact source revisions

| artifact | revision | note |
|---|---|---|
| comfy-kitchen PR #90 "Add optimized w4a8 with int8 codebook" | **MERGED**, merge commit `aa1ab2263dc06225d9de6702dfc087313d4bc971` (2026-08-06); head `b812819a97ac11d01f4a3a16ba47dd38de3b2519` | the reference W4A8 implementation |
| ComfyUI PR #15308 "Support asym w4a8_int" | **OPEN, NOT MERGED** at research time; head `b6578f2ae11ab3dea3156ed68d8724476cda1232`; base `bdcb886a4705a03cf40f4a7226de9fc7c059fc90` (2026-08-06) | the ComfyUI loader support |
| Reference serialized example | `Kijai/MiniMax-H3-experimental/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors` (12.5 GB, header inspected) | produced by the PR author |
| ComfyUI master (research base) | `bdcb886a4705a03cf40f4a7226de9fc7c059fc90` | used for the architecture registry (98 supported-model classes) |

Files studied in comfy-kitchen@aa1ab22:
`tensor/w4a8_int8.py`, `backends/eager/w4a8_int8.py`, `backends/triton/w4a8_int8.py`,
`backends/cuda/ops/w4a8_gemm.cu`, `backends/cuda/__init__.py`,
`tensor/int8_utils.py`, `tensor/base.py`, `tensor/__init__.py`,
`backends/eager/quantization.py`.

Files studied in ComfyUI@bdcb886 (+ PR #15308 diff):
`comfy/ops.py` (`_load_quantized_module`, `pop_scale`, `_quantized_weight_state_dict`,
`mixed_precision_ops`), `comfy/quant_ops.py` (`QUANT_ALGOS`), `comfy/utils.py`
(`convert_old_quants`, `detect_layer_quantization`), `comfy/sd.py`,
`comfy/model_detection.py` (`detect_unet_config`), `comfy/supported_models.py`,
`comfy/ldm/**` (per-family tensor-name policies).

## Verified W4A8 format ("asym_w4a8_int8")

### Quantization (eager reference, bit-exact port)

1. Input: 2D weight `W [N, K]`; constraints `K % 16 == 0`, `K % group_size == 0`,
   `K % convrot_groupsize == 0`, `group_size >= 4`, and
   `(16 % group_size == 0 or group_size % 16 == 0)`. The CUDA dequant kernel requires
   group sizes in `{4, 8, 16}` or multiples of 16 (a 16-wide vector spans at most 4
   groups).
2. ConvRot rotation: `W_rot = W @ (I ⊗ H)^T` where `H` is a normalized regular
   Hadamard built by Kronecker products of `H4 = [[1,1,1,-1],[1,1,-1,1],
   [1,-1,1,1],[-1,1,1,1]]/2` (size must be a power of 4). Runtime applies the same
   rotation online to activations (`x @ (I ⊗ H)`), so `y = x @ W^T` is preserved.
3. Symmetric + codebook path (the default):
   * `group_scale = amax(|W_rot|, per group).clamp(1e-8)`,
     `normalized = W_rot / group_scale`
   * 16-entry Lloyd-Max codebook: `torch.quantile(samples, linspace(0,1,16))` init,
     25 iterations of nearest-assignment + class-mean update; deterministic
     subsample of at most 300 000 elements with `Generator(seed=0)`
   * code assignment = nearest codebook entry; 3 refinement rounds of least-squares
     group scale (`sum(w·q)/sum(q²)`) + reassignment
4. Scales: `s_channel[r] = amax(|cb[q]·gs|, row)/127` (fp32, `[N]`);
   `s_rel = gs / s_channel` (fp32 → **fp8_e4m3fn**, `[N, K/group_size]`).
5. Final code assignment against the runtime int8 decode grid:
   `levels = round(codebook · s_rel).clamp(-127, 127)`; codes = nearest level of
   `W_rot / s_channel`.
6. Packing: unsigned 4-bit codes, even column → low nibble, odd column → high nibble;
   `packed` = int8 `[N, K/2]`.
7. Runtime decode (CUDA / Triton / eager are bit-identical):
   `out8 = round(clamp(cb[code] · s_rel[n, k//G], -127, 127))` (`__float2int_rn`),
   then `W_rot ≈ out8 · s_channel`, then un-rotate (same Hadamard; H is symmetric).

### Serialization (verified against the Kijai example and ComfyUI loader)

Per quantized layer (names are the **full state-dict keys**):

* `{layer}.weight`: int8 `[N, K/2]`
* `{layer}.weight_s_rel`: fp8_e4m3fn `[N, K/group_size]` (native fp8 on disk)
* `{layer}.weight_s_channel`: fp32 `[N]`
* `{layer}.weight_codebook`: fp32 `[16]`

The asymmetric-correction variant (`{layer}.weight_correction [K/gs, N]`) exists in
comfy-kitchen but the ComfyUI loader does not consume it. The converter always uses
the symmetric codebook mode.

Global metadata `__metadata__["_quantization_metadata"]`:
`{"layers": {layer: {"format": "asym_w4a8_int8", "group_size": 16, "convrot": true,
"convrot_groupsize": 256}}}`. ComfyUI's `comfy/utils.py::convert_old_quants` turns
this into per-layer `{layer}.comfy_quant` JSON blobs; `detect_layer_quantization`
enables `mixed_precision_ops`; `comfy/ops.py::_load_quantized_module` (PR #15308)
pops `weight`, `weight_s_rel`, `weight_s_channel`, `weight_codebook` and builds an
`AsymW4A8Int8Layout` QuantizedTensor with `group_size` / `convrot_groupsize` from the
layer config. Non-quantized tensors keep their original names and dtypes.

### Runtime prerequisites (state in metadata and reports)

* comfy-kitchen ≥ `aa1ab2263dc06225d9de6702dfc087313d4bc971` (PR #90, merged) with
  `AsymW4A8Int8Layout` registered; eager works on CPU/CUDA/ROCm; Triton ≥ 3.7 for
  ROCm; the compiled CUDA backend requires PyTorch cu130+ and SM ≥ 8.0.
* ComfyUI ≥ PR #15308 head `b6578f2ae11ab3dea3156ed68d8724476cda1232`.
  **Not merged into ComfyUI master** as of base commit `bdcb886a...`.
* `weight_dtype` (majority-dtype detection) is explicitly bypassed for quantized
  checkpoints (`if model_config.quant_config is not None: weight_dtype = None`), so
  int8-packed weights are safe on disk.

## W3A8: independent extension format (this tool's design)

Not supported by any upstream revision (verified: no `w3a8`/`int3`/3-bit references in
comfy-kitchen@aa1ab22 or ComfyUI@bdcb886). Design:

* format id `asym_w3a8_int8`; same ConvRot framework and int8 decode grid
* 8-entry symmetric Lloyd-Max codebook (levels=8, 25 iterations, seed-0 subsample)
* scales identical in structure: fp8_e4m3fn `s_rel [N, K/gs]`, fp32 `s_channel [N]`
* independent 3-bit packing (8 codes per 3 bytes, little-endian):
  `b0 = c0|c1<<3|c2<<6; b1 = c2>>2|c3<<1|c4<<4|c5<<7; b2 = c5>>1|c6<<2|c7<<5`
* `weight` = int8 `[N, K*3//8]`; `weight_codebook` fp32 `[8]`
* layer config adds `"codebook_size": 8, "packing": "3bit-lsb"`; the extension
  metadata states that loading requires the `--emit-patch` runtime patch
* the emitted patch (verified with `git apply --check` against the reference
  revisions) adds `AsymW3A8Int8Layout` to comfy-kitchen and registers
  `asym_w3a8_int8` in ComfyUI's `QUANT_ALGOS` + loader.

## Architecture registry

42 policy families covering all 98 ComfyUI supported-model classes at
`bdcb886a4705a03cf40f4a7226de9fc7c059fc90` (SD1.5/SD2.x/SDXL(+refiner/SSD1B/Vega/
KOALA), SVD/SV3D, Stable Cascade, SD3 MMDiT, StableAudio, AuraFlow, PixArt Alpha/Sigma,
HunyuanDiT, Flux(+inpaint/schnell/longcat/ovis), Flux2, Chroma(+Radiance), Mochi,
MiniMax H3, LTXV/LTXAV, ACE-Step, Cosmos(+Predict2), Anima, Lumina2/ZImage, PixelDiT/PiD,
Wan 2.1/2.2 (all variants), Hunyuan3D, TripoSplat, HiDream(+O1), SeedVR2, OmniGen2/Boogu,
Ideogram4, Krea2, MageFlow, QwenImage, JoyImage, Kandinsky5, CogVideoX, ErnieImage,
perception models RT-DETR_v4 / DepthAnything3 / SAM3 marked **unsupported** (no
ComfyUI quantized-loading path for them)). Detection signatures were extracted from
`comfy/model_detection.py::detect_unet_config`; per-family quantize/keep/exclude
patterns were derived from the actual module structures in `comfy/ldm/**` and from the
reference MiniMax-H3 W4A8 example (which quantizes exactly
`attn.qkv_proj`, `attn.out_proj`, `mlp.fc1`, `mlp.fc2` per block).

## Executed tests (evidence)

* `--self-test`: 15/15 pass, including golden-vector bit-exactness vs the reference
  implementation (2 configs) and end-to-end mini-model conversions for both formats.
* Side-by-side vs comfy-kitchen@aa1ab22 (eager, CPU): bit-exact quantize and
  dequantize on 9 shape/config combinations (group sizes 16/32/64, convrot 16/64/256).
* Real model: Wan 2.1 T2V 1.3B fp16 (2.64 GiB) → W4A8, 300 layers, output 801 MiB;
  max sampled relL2 **0.0730**, cosine 0.9985, SNR 22.8 dB; all validation checks
  passed; the ComfyUI loader contract (metadata → comfy_quant → weight/s_rel/
  s_channel/codebook names, dtypes, shapes) was verified against the output.
* Small structurally-accurate fixtures for 7 families (sd15, sdxl, flux, wan,
  minimax_h3, hydit, mmdit_sd3): W4A8 and W3A8 conversions all pass standalone
  validation (W4A8 relL2 0.0728–0.0855, W3A8 relL2 0.1492–0.1496), plus:
  sharded-directory input, CLI kill/resume, calibration cache write/read,
  sensitivity-based keep-precision, chunked bounded-memory conversion (2 MiB budget),
  CUDA device path, bf16 compute, fp16 output cast, include/exclude/keep-precision
  filters, overwrite/self-overwrite/pickle/requantize guards, metadata-only,
  validation-only, dry-run, and the emitted W3A8 runtime patch (applies cleanly).
