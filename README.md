# Comfyui w4a8 quantizer

A single-file converter that turns supported generative-model checkpoints into
**W4A8** (`asym_w4a8_int8`) quantized checkpoints for use with compatible ComfyUI /
comfy-kitchen versions.

The script is standalone. It does **not** import, require, or execute any ComfyUI or
comfy-kitchen code at runtime. Every inspection, detection, quantization, packing,
metadata and validation component is reimplemented inside the file from the verified
reference behavior. The reference revisions and the full format specification are
documented in the [Research basis](#research-basis) section below.

## Contents

* [Features](#features)
* [Requirements](#requirements)
* [Installation](#installation)
* [Basic usage](#basic-usage)
* [Supported inputs](#supported-inputs)
* [Main options](#main-options)
* [Examples](#examples)
* [Output format and metadata](#output-format-and-metadata)
* [Architecture support](#architecture-support)
* [Compatibility matrix](#compatibility-matrix)
* [Validation and self-tests](#validation-and-self-tests)
* [Security](#security)
* [Known limitations](#known-limitations)
* [Research basis](#research-basis)
* [patches/](patches/comfyui_w4a8_loader.patch)
* [License](#license)

## Features

* W4A8 weight quantization in the exact `asym_w4a8_int8` layout of comfy-kitchen
  PR #90: ConvRot-rotated int4 weights, a per-tensor 16-entry Lloyd-Max codebook,
  fp8 group scales, and per-channel scales. Packed int4 codes and fp8 scales are
  byte-identical to the reference implementation; fp32 scale fields can differ in
  the last ULPs across platforms (verified with golden vectors and side-by-side
  runs).
* Architecture detection from the checkpoint alone. The embedded registry has 43
  policy families that cover all 98 model classes ComfyUI supported at the research
  revision, each with its own quantize / keep / exclude rules and validation
  thresholds. Unknown or ambiguous models are refused unless `--architecture` is
  given.
* Bounded-memory conversion and validation. Quantized layers and retained tensors
  (including dtype casts) are processed in bounded row/byte chunks, and output is
  written to a temp file and atomically renamed. Interrupted runs resume from a
  checksummed state file.
* Calibration support from local activation data (`.npz` / `.pt` / `.npy` or a
  directory). Calibration is optional and used for sensitivity analysis, which can
  fall back to original precision for sensitive layers. The reference format itself
  is calibration-free.
* Standalone validation that reopens the output, checks inventories, shapes, dtypes,
  metadata, packing round trips and scales, measures reconstruction error against
  the original weights, verifies determinism, and hashes both input and output.
  Volatile timing and peak-memory measurements stay in the report, so identical
  CPU conversions in the same environment can produce byte-identical checkpoints.
* Embedded self-tests (19 checks) covering packing, dtype selection, real activation
  calibration, metadata, safe detection, malformed inputs, sensitivity planning,
  checksummed resume, and atomic-write paths.

## Requirements

* Python >= 3.10
* `torch >= 2.1` (fp8 support)
* `safetensors >= 0.4.3` (fp8 dtype support)
* `numpy`

```bash
pip install torch safetensors numpy
```

No network access is required during conversion.

## Installation

There is nothing to install. The converter is one file:

```bash
cp comfyui_wxa8_quantizer.py /somewhere/on/your/PATH/
```

## Basic usage

```bash
python comfyui_wxa8_quantizer.py ORIGINAL_MODEL \
    --output QUANTIZED_MODEL \
    --format w4a8
```

Only the original model path and the output path are required. `--format` defaults
to `w4a8`. All other parameters have architecture-specific defaults.

## Supported inputs

* a single `.safetensors` checkpoint (ComfyUI-style single files, e.g. SD1.5 / SDXL
  checkpoints with `model.diffusion_model.` keys, MiniMax-H3-style prefix-less files)
* sharded safetensors checkpoints (`model.safetensors.index.json` + shards)
* a model directory (HF-style: `config.json`, `model_index.json`, shard index)
* torch pickle checkpoints (`.ckpt` / `.pt` / `.bin`) **only with `--trust-pickle`**.
  Deserializing pickles executes arbitrary code, so only pass the flag for files you
  trust.

Pickle inputs are always refused without `--trust-pickle`.

## Main options

| option | meaning |
|---|---|
| `--format {w4a8}` | quantization format (default w4a8) |
| `--architecture auto\|NAME` | architecture override (`--list-architectures`) |
| `--device auto\|cpu\|cuda\|rocm` | quantization compute device; `auto` = CPU (deterministic, memory-bounded) |
| `--compute-dtype auto\|fp32\|fp16\|bf16` | precision of the quantization math (default fp32, matches the reference) |
| `--output-dtype auto\|fp16\|bf16` | cast passthrough float tensors (auto = keep original) |
| `--group-size NUMBER` | per-group quantization size (default: architecture policy, 16) |
| `--calibration-source PATH` | local calibration activations (.npz/.pt/.npy or directory) |
| `--calibration-samples N` | cap recorded calibration rows per layer (default 64) |
| `--calibration-cache PATH` | read/write a compressed, safe activation-row cache |
| `--seed N` | reproducibility seed (recorded; codebook subsampling is seed-0 like the reference) |
| `--include/--exclude/--keep-precision PATTERN` | regex filters on tensor names |
| `--sensitivity-threshold N` | keep layers at original precision above an error score |
| `--error-threshold N` | hard relL2 fallback during a sensitivity/calibration prepass (default: architecture policy) |
| `--max-memory SIZE` | per-tensor working-memory budget (larger tensors are chunked) |
| `--streaming` | bounded-memory streaming (default for safetensors) |
| `--resume` | resume an interrupted conversion from its state file |
| `--overwrite` | allow replacing an existing output |
| `--dry-run` | detect, plan and report, write nothing |
| `--inspect` | dump input inventory + detection evidence, exit |
| `--list-architectures` | print the embedded architecture registry |
| `--validate` | full standalone validation after conversion (all quantized layers, hashes, optional static runtime-compat probe) |
| `--validation-only` | validate an existing output against the original model |
| `--metadata-only` | generate metadata + report only |
| `--report PATH` | write human-readable report (+ `PATH.json`) |
| `--log-level`, `--json-log` | logging controls |
| `--trust-pickle` | allow pickle inputs (unsafe for untrusted files) |
| `--yes` | assume yes |
| `--self-test` | run the embedded engineering self-tests |

## Examples

```bash
# Convert a single-file SDXL checkpoint to W4A8 with full validation
python comfyui_wxa8_quantizer.py sd_xl_base_1.0.safetensors \
    --output sd_xl_base_1.0_w4a8.safetensors --format w4a8 --validate

# Convert a sharded/HF-style directory
python comfyui_wxa8_quantizer.py ./model_dir \
    --output model_dir_w4a8.safetensors --format w4a8 --report w4a8_report.txt

# Sensitivity-based fallbacks with local calibration activations
python comfyui_wxa8_quantizer.py model.safetensors \
    --output model_w4a8.safetensors --format w4a8 \
    --calibration-source ./calib_acts.npz --calibration-samples 64 \
    --sensitivity-threshold 0.4

# Resume an interrupted conversion
python comfyui_wxa8_quantizer.py model.safetensors \
    --output model_w4a8.safetensors --format w4a8 --resume
```

## Output format and metadata

Per quantized layer `{layer}` (the full state-dict key, e.g.
`model.diffusion_model.blocks.0.attn.qkv_proj`):

| tensor | dtype | shape | meaning |
|---|---|---|---|
| `{layer}.weight` | int8 | `[N, K/2]` | packed int4 codes (even col = low nibble) |
| `{layer}.weight_s_rel` | fp8_e4m3fn | `[N, K/group_size]` | per-group relative scale |
| `{layer}.weight_s_channel` | fp32 | `[N]` | per-output-channel scale |
| `{layer}.weight_codebook` | fp32 | `[16]` | Lloyd-Max codebook |

All other tensors (biases, norms, embeddings, convs, positionals, heads, buffers)
pass through under their original names and dtypes.

The safetensors header carries two metadata blocks:

* `__metadata__["_quantization_metadata"]` (verified ComfyUI key): a JSON string of
  the form `{"layers": {layer: {"format": "asym_w4a8_int8", "group_size": 16,
  "convrot": true, "convrot_groupsize": 256}}}`. `convrot_groupsize` is always 256:
  the comfy-kitchen CUDA fused kernels implement ConvRot 256 only, so a smaller
  per-layer value cannot run at inference. Layers whose K is not divisible by 256
  are therefore kept at original precision and do not appear in this block. Layer
  names are the full state-dict keys, including any `model.diffusion_model.`
  prefix. ComfyUI's `comfy/utils.py::convert_old_quants` converts this into
  per-layer `comfy_quant` blobs that the PR #15308 loader reads.
* `__metadata__["comfy_wxa8"]` (namespaced extension, never official): converter
  schema `comfy_wxa8/v1`, converter name and version, format revision, detected
  architecture and confidence, source whole-file hashes under portable file labels
  (no machine-specific absolute paths), quantization parameters, runtime activation
  quantization, compute dtype, calibration provenance, sensitivity results,
  reproducibility settings, compatibility requirements (exact revisions), stable
  tensor-payload sha256, validation summary, and warnings. The full output file
  sha256 is written to the report because a file cannot embed its own hash. Per-layer
  metrics and every exclusion/retention reason are written to the JSON report instead
  of expanding the safetensors header.

Other valid string-valued safetensors metadata from the source is preserved; the
converter only replaces its official quantization key and `comfy_wxa8` namespace.

## Architecture support

43 policy families cover all 98 ComfyUI supported-model classes at the research
revision (`bdcb886a4705a03cf40f4a7226de9fc7c059fc90`): SD1.5 / SD2.x / SDXL (+refiner,
SSD-1B, Segmind-Vega, KOALA), SVD/SV3D, Stable Cascade, SD3 MMDiT, StableAudio,
AuraFlow, PixArt Alpha/Sigma, HunyuanDiT, Flux (+inpaint/schnell/longcat/ovis),
Flux2, Chroma (+Radiance), Mochi, MiniMax H3, LTXV/LTXAV, ACE-Step, Cosmos
(+Predict2), Anima, Lumina2/Z-Image, PixelDiT/PiD, Wan 2.1/2.2 (all variants),
Hunyuan3D, TripoSplat, HiDream (+O1), SeedVR2, OmniGen2, Boogu, Ideogram4, Krea2,
MageFlow, QwenImage, JoyImage, Kandinsky5, CogVideoX, ErnieImage.
Boogu and OmniGen2 use the real state-dict naming from their published
checkpoints: Boogu has `double_stream_layers.N.img_self_attn` /
`img_instruct_attn.processor` / `img_feed_forward` and
`single_stream_layers.N.attn` / `feed_forward`, OmniGen2 has `layers.N.attn` /
`feed_forward`, and both share the `context_refiner` / `noise_refiner` /
`ref_image_refiner` blocks and embedders. Use the single-file ComfyUI repack
(Comfy-Org) for Boogu; the raw HF `transformer/` shards are not a single
ComfyUI checkpoint. The perception
models RT-DETR_v4, DepthAnything3 and SAM3 are registered as unsupported, because
ComfyUI has no quantized-loading path for them; conversion is refused unless
`--architecture` forces it.

Detection signatures were extracted from `comfy/model_detection.py::detect_unet_config`.
Per-family quantize / keep / exclude patterns were derived from the actual module
structures in `comfy/ldm/**` and from the reference MiniMax-H3 W4A8 example, which
quantizes exactly `attn.qkv_proj`, `attn.out_proj`, `mlp.fc1`, `mlp.fc2` per block.

Run `python comfyui_wxa8_quantizer.py --list-architectures` to see the full table
with the runtime status of each family.

## Compatibility matrix

Fixture rows were rerun with `comfyui_wxa8_quantizer.py 1.2.2` on torch 2.13.0 /
safetensors 0.8.0 using CPU quantization. Since 1.2.2 the converter only emits
W4A8 layers with `convrot_groupsize = 256` (the comfy-kitchen CUDA fused kernels
implement ConvRot 256 only); 2D linears whose K is not divisible by 256 pass
through at original precision with the reason recorded. The real-model and CUDA
rows are retained from the documented 1.1.x/1.2.1 runs because those checkpoints
and a GPU are not stored in this repository. "Executed"
means the conversion finished and the output passed the standalone validation suite
(reopen, inventory, shapes, dtypes, metadata, pack round trips, scale checks,
reconstruction error vs the original weights, deterministic checks, output hash).
relL2 = per-layer weight reconstruction error (maximum over all quantized layers
when `--validate` was used).

### W4A8 (`asym_w4a8_int8`) conversions

| architecture family | ComfyUI classes | test input | result | max relL2 |
|---|---|---|---|---|
| sd15 | SD15, SD15-inpaint, Zero123 | fixture (8 tensors, 2 quantized) | pass | 0.0729 |
| sdxl | SDXL, SSD-1B, Vega, KOALA, SDXL-inpaint | fixture (28 tensors, 8 quantized) | pass | 0.0729 |
| flux | Flux, FluxInpaint, FluxSchnell, LongCat, Ovis | fixture (23 tensors, 16 quantized) | pass | 0.0729 |
| wan | Wan2.1/2.2 (20 classes) | fixture (31 tensors, 16 quantized) | pass | 0.0727 |
| wan | Wan2.1 T2V 1.3B (real, 2.64 GiB) | 300 layers | pass (full validation) | 0.0730 |
| minimax_h3 | MiniMaxH3 | fixture (30 tensors, prefix-less, 6 quantized) | pass | 0.0727 |
| hydit | HunyuanDiT, HunyuanDiT1 | fixture (12 tensors, 8 quantized) | pass | 0.0728 |
| mmdit_sd3 | SD3 (and SD3.5 family) | fixture (15 tensors, 12 quantized) | pass | 0.0728 |
| lumina2 | Lumina2, ZImage, ZImagePixelSpace | fixture (real naming, 35 tensors, 20 quantized) | pass | 0.0728 |
| lumina2 | Z-Image Turbo (real, sickOllie_zTurbo 11.46 GiB, bf16) | 170 layers, 3.42 GiB out | pass (full validation, cuda) | 0.0730 |
| boogu | Boogu-Image-0.1 (Base/Turbo/Edit naming) | fixture (real naming, 73 layers) | pass | 0.0728 |
| omnigen2 | OmniGen2 | fixture (real naming, 35 layers) | pass | 0.0728 |
| (runtime) | real ComfyUI v0.30.0 + loader patch + comfy-kitchen 0.2.27 load | zimage + minimax_h3 outputs | pass (QuantizedTensor, AsymW4A8Int8Layout) | - |
| (input form) | sharded directory | hydit fixture split in 2 shards | pass | 0.0728 |
| (input form) | bounded-memory chunked path (32 MiB budget) | Wan + MiniMax H3 fixtures | pass | 0.0729 |

### Feature tests (executed)

| feature | result |
|---|---|
| embedded self-tests (`--self-test`) | 19/19 pass |
| golden vectors vs comfy-kitchen reference (packed/fp8 byte-exact, fp32 within 1e-4) | pass (2 configs) |
| side-by-side bit-exactness vs reference (9 shape/config combos, quantize + dequantize) | pass |
| CLI interruption + `--resume` recovery | pass before and after tensor finalization; option drift and completed-tensor corruption rejected |
| calibration (npz), direct activation rows, compressed cache write/read | pass |
| `--sensitivity-threshold` keep-precision | pass; output inventory frozen after prepass |
| `--device cuda` | pass (deterministic-vs-disk documented skip) |
| `--compute-dtype bf16`, `--output-dtype fp16` | pass |
| `--include/--exclude/--keep-precision` filters | pass |
| `--dry-run`, `--inspect`, `--metadata-only`, `--validation-only` | pass |
| overwrite / hardlink self-overwrite / auxiliary-path collision / pickle / requantize guards | pass (refused as designed) |
| odd-byte bool/u8 passthrough tensors | pass |
| nested BF16 pickle state dict and unindexed shard tensor | pass |
| ComfyUI loader contract (names/dtypes/shapes/metadata) on output | pass |

### Not executed or unsupported

* No end-to-end inference was run inside ComfyUI. Standalone validation does not
  claim runtime compatibility; ComfyUI >= v0.31.0 loads W4A8 natively (PR #15308
  merged 2026-08-07).
* Perception models (RT-DETR_v4, DepthAnything3, SAM3/SAM31) refuse conversion
  unless `--architecture` forces them.
* Diffusers-format subfolders (`unet/diffusion_pytorch_model.safetensors`) are
  discovered as files, but their state-dict naming is not converted to ComfyUI
  naming; detection fails safely and requests `--architecture`.
* The remaining families (stable_cascade, stable_audio, aura_flow, mochi, ltxv,
  ace_step, cosmos, cosmos_predict2, anima, pixeldit, hunyuan3d,
  triposplat, hidream, chroma, seedvr2, ideogram4, krea2, mage_flow,
  qwen_image, joyimage, kandinsky5, cogvideox, ernie_image, sd20, sdxl_refiner,
  svd) have explicit policy profiles and detection signatures, but were not
  individually executed with fixture tensors. They share the same verified
  quantization and serialization machinery.

## Validation and self-tests

`--validate` runs the full standalone validation after conversion: output reopen,
tensor inventory, shape and dtype preservation, metadata checks, packing round
trips, scale validation on every layer, reconstruction error (relL2, SNR, cosine)
on every quantized layer, deterministic re-quantization, tensor-payload and full-file
sha256, and an optional static probe of installed comfy-kitchen / ComfyUI source.
The probe never imports either project. Reconstruction and scale checks also use
bounded row chunks when a layer exceeds `--max-memory`. `--validation-only` runs
the same full suite against an existing output.

```bash
python comfyui_wxa8_quantizer.py --self-test
```

The embedded self-tests cover W4 packing round trips, odd dimensions and odd-byte
tensors, scale calculations, compute-dtype selection, real activation rows,
standalone environment inspection, metadata generation, registry behavior (all 98
ComfyUI model classes covered by 43 policy families), fail-closed ambiguity,
golden-vector bit-exactness, malformed checkpoints, input variants, sensitivity
output planning, checksummed resume, atomic output writing, and an end-to-end
mini-model conversion. These are engineering tests, not full-model quality
validation.

## Troubleshooting

**ComfyUI logs `unet unexpected: [...comfy_quant...]` when loading a W4A8
checkpoint.** This is a benign artifact of the load path. The `comfy_quant`
markers are consumed during quantization detection before the weight load; they
may still be listed as unexpected on some ComfyUI builds with dynamic VRAM
loading enabled. The packed weights and scales are not in the list, the model
still runs, and the warning can be ignored. Verified by loading W4A8 outputs of
two structurally different families (Z-Image and MiniMax H3) through the real
ComfyUI v0.30.0 + comfy-kitchen 0.2.27 load path: every quantized layer becomes
an `AsymW4A8Int8Layout` QuantizedTensor and no warning is emitted.

**`RuntimeError: Given normalized_shape=[4096], expected input with shape
[*, 4096], but got input of size [1, 512, 12288]` (or a similar RMSNorm/LayerNorm
shape error) right after sampling starts.** This is a conditioning mismatch, not
a quantization defect. The failing module (for Boogu/OmniGen2 that is
`time_caption_embed.caption_embedder`) is kept at original precision and is
byte-identical in the W4A8 output; the model config it drives (`instruction_feat_dim`,
`hidden_size`) is read from those same kept tensors. The input to that norm comes
from the text encoder, a separate file that the converter never touches. A correct
Boogu workflow (official template, `qwen3vl_8b_fp8_scaled.safetensors` as text
encoder) produces `[B, seq, 4096]` conditioning. If you see a multiple of 4096
(12288 = 3 x 4096), the workflow is concatenating conditioning streams or using a
non-Boogu text encoder; the same workflow fails identically with the bf16
checkpoint. Check the conditioning part of the workflow and update ComfyUI before
suspecting the quantized file.

Community INT4/INT8 ConvRot text encoders (files named `*_int4_convrot.safetensors`
or `*_int8_convrot.safetensors`) are not standard ComfyUI text encoders. Their
distribution repos require a nightly ComfyUI plus a custom INT4 loader
(ComfyUI-INT4-Fast) and must be loaded with that loader's node; the standard
CLIPLoader "will not decode these correctly" and can produce garbage
conditioning that surfaces as the RMSNorm shape error above. Use the official
repack of the same text encoder (for Boogu: `qwen3vl_8b_fp8_scaled.safetensors`
from Comfy-Org) with the standard CLIPLoader.

## Security

The model, metadata, configuration files, paths and calibration data are treated as
untrusted. Pickle loading is opt-in only. Safetensors headers reject duplicate keys,
negative dimensions, negative or overlapping offsets, holes, trailing data and size
mismatches before tensor access. Shard-index traversal is rejected. Output, report,
log, state and cache paths cannot alias source files or each other. Temp files use
exclusive no-follow opens, resume records bind the temp inode and checksum completed
tensors, source and calibration files are hashed around processing, compressed
calibration inputs are expansion-bounded, and publication uses fsync plus atomic
replacement. A new requested output is withheld until validation passes; failed
overwrites leave the previous target intact. No subprocesses or network access are
used.

## Known limitations

1. **ComfyUI runtime support is conditional.** W4A8 loading requires comfy-kitchen ≥
   `aa1ab2263dc06225d9de6702dfc087313d4bc971` (merged) AND ComfyUI PR #15308
   ("Support asym w4a8_int", merged 2026-08-07, shipped in ComfyUI v0.31.0). On
   ComfyUI >= v0.31.0 the loader is native and `patches/comfyui_w4a8_loader.patch`
   is not needed; older builds (v0.30.x) need the patch. Standalone validation
   does not prove runtime compatibility;
   use `--validate` for the optional installed-version probe.
2. **Only 2D linear weights are quantized, and only with `convrot_groupsize =
   256`.** The reference format requires 2D, `K % 16 == 0`, plus group/convrot
   divisibility. The comfy-kitchen CUDA fused kernels (activation ConvRot + int8
   quantize, chunked codebook GEMM, weight rotation) implement ConvRot 256 only;
   a layer whose K is not divisible by 256 cannot run on the CUDA runtime with a
   smaller ConvRot group and raises "convrot fused kernel only supports group_size
   256" at sampling. Such layers pass through at original precision with the reason
   recorded in the report (this is why real checkpoints with K=320 attention
   projections, for example SDXL attn1, keep those weights in bf16/fp16). Layers
   whose K is not divisible by 16 cannot be quantized in this format at all.
3. **Detection is heuristic.** It mirrors ComfyUI's `detect_unet_config` signatures
   at the research revision, but ambiguous or unknown checkpoints are refused unless
   `--architecture` is supplied.
4. **Determinism is backend-scoped.** The in-memory path uses a fixed-seed generator
   on the quantization device; CPU and CUDA can produce different (both valid)
   outputs. `--device auto` = CPU. The bounded path fits its codebook on CPU while
   preserving quantization-group normalization and reference sample order. A reduced
   memory budget can reduce the sample count below 300,000; this is recorded through
   the memory settings and can change bytes while remaining format-compatible (32
   MiB fixture relL2 0.0729 versus 0.0728 in-memory).
5. **Pickle inputs are loaded fully into RAM** (`--trust-pickle` required) and cannot
   be streamed; only convert pickle checkpoints you trust and that fit in memory.
6. **Calibration is optional and used for sensitivity analysis only.** The reference
   format is calibration-free (per-group absmax scales); the converter never claims
   production calibration from synthetic data. Real activation rows, source hashes
   and coverage are recorded in the `comfy_wxa8` extension (or "calibration-free").
7. **Reference drift**: comfy-kitchen/ComfyUI may change the format in the future;
   the converter pins its behavior to the revisions below and records
   `format_revision` in the output metadata.

## Research basis

The format specification was verified against these exact revisions:

| artifact | revision | note |
|---|---|---|
| comfy-kitchen PR #90 "Add optimized w4a8 with int8 codebook" | **MERGED**, merge commit `aa1ab2263dc06225d9de6702dfc087313d4bc971` (2026-08-06); head `b812819a97ac11d01f4a3a16ba47dd38de3b2519` | the reference W4A8 implementation |
| ComfyUI PR #15308 "Support asym w4a8_int" | **MERGED 2026-08-07** as commit `344b43989e` (shipped in ComfyUI v0.31.0); earlier head `8c3a2b27c37bd34e87b58846baf962407c92843c`; base `bdcb886a4705a03cf40f4a7226de9fc7c059fc90` | the ComfyUI loader support |
| comfy-kitchen PR #96 | merge commit `3d16bed29c91a3d9d1abdc0574c87a5ba2b1ef33` | later frozen-LUT and row-chunked producer changes; this converter intentionally retains the PR #90 numerical contract while emitting the same runtime layout |
| Reference serialized example | `Kijai/MiniMax-H3-experimental/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors` (12.5 GB, header inspected) | produced by the PR author |
| ComfyUI master (research base) | `bdcb886a4705a03cf40f4a7226de9fc7c059fc90` | used for the architecture registry (98 supported-model classes) |

Files studied in comfy-kitchen@aa1ab22: `tensor/w4a8_int8.py`,
`backends/eager/w4a8_int8.py`, `backends/triton/w4a8_int8.py`,
`backends/cuda/ops/w4a8_gemm.cu`, `backends/cuda/__init__.py`,
`tensor/int8_utils.py`, `tensor/base.py`, `tensor/__init__.py`,
`backends/eager/quantization.py`.

Files studied in ComfyUI@bdcb886 (+ PR #15308 diff): `comfy/ops.py`
(`_load_quantized_module`, `pop_scale`, `_quantized_weight_state_dict`,
`mixed_precision_ops`), `comfy/quant_ops.py` (`QUANT_ALGOS`), `comfy/utils.py`
(`convert_old_quants`, `detect_layer_quantization`), `comfy/sd.py`,
`comfy/model_detection.py` (`detect_unet_config`), `comfy/supported_models.py`,
`comfy/ldm/**` (per-family tensor-name policies).

### The W4A8 format

1. Input: 2D weight `W [N, K]`; constraints `K % 16 == 0`, `K % group_size == 0`,
   `K % convrot_groupsize == 0`, `group_size >= 4`, and
   `(16 % group_size == 0 or group_size % 16 == 0)`. The CUDA dequant kernel
   requires weight group sizes in `{4, 8, 16}` or multiples of 16, and the CUDA
   fused ConvRot kernels additionally require `convrot_groupsize == 256` (so
   `K % 256 == 0`). The converter therefore emits only `convrot_groupsize = 256`
   and passes through any linear whose K is not divisible by 256.
2. ConvRot rotation: `W_rot = W @ (I ⊗ H)^T` where `H` is a normalized regular
   Hadamard built by Kronecker products of `H4 = [[1,1,1,-1],[1,1,-1,1],
   [1,-1,1,1],[-1,1,1,1]]/2` (size must be a power of 4). Runtime applies the same
   rotation online to activations (`x @ (I ⊗ H)`), so `y = x @ W^T` is preserved.
3. Symmetric codebook path (the default): per-group `group_scale = amax` with a
   16-entry Lloyd-Max codebook (`torch.quantile` init, 25 iterations, deterministic
   seed-0 subsample of at most 300 000 elements), code assignment by nearest entry,
   then 3 refinement rounds of least-squares group scale + reassignment.
4. Scales: `s_channel[r] = amax(row)/127` (fp32, `[N]`);
   `s_rel = gs / s_channel` (fp32 to fp8_e4m3fn, `[N, K/group_size]`).
5. Final code assignment against the runtime int8 decode grid:
   `levels = round(codebook * s_rel).clamp(-127, 127)`.
6. Packing: unsigned 4-bit codes, even column to low nibble, odd column to high
   nibble; `packed` = int8 `[N, K/2]`.
7. Runtime decode (CUDA / Triton / eager are bit-identical):
   `out8 = round(clamp(cb[code] * s_rel[n, k//G], -127, 127))`, then
   `W_rot = out8 * s_channel`, then un-rotate (H is symmetric).
8. Runtime A8 path: ConvRot is applied to each incoming activation row, followed
   by dynamic symmetric int8 quantization with fp32 scale
   `max(abs(row))/127` (minimum `1e-30`), nearest rounding and clamp to
   `[-128, 127]`. The int8 activation and decoded int8 weight accumulate in int32.

The asymmetric-correction variant (`{layer}.weight_correction [K/gs, N]`) exists in
comfy-kitchen but the ComfyUI loader does not consume it. The converter always uses
the symmetric codebook mode.

### Cross-platform determinism

Quantization is deterministic per platform: two runs on the same machine produce
byte-identical output. Across platforms, torch reductions (quantile, amax, mean)
can differ in the last ULPs, so the fp32 `s_channel` and `codebook` fields may vary
in their lowest bits between x86 and ARM or between Windows and Linux. The packed
int8 codes and the fp8 `s_rel` bytes are stable across platforms. The embedded
golden-vector test asserts byte equality for packed/fp8 output and a 1e-4 relative
tolerance for the fp32 fields, and the CI matrix runs it on ubuntu, windows and
macos.

### Runtime prerequisites

* comfy-kitchen ≥ `aa1ab2263dc06225d9de6702dfc087313d4bc971` (PR #90, merged) with
  `AsymW4A8Int8Layout` registered; eager works on CPU/CUDA/ROCm; Triton ≥ 3.7 for
  ROCm; the compiled CUDA backend requires PyTorch cu130+ and SM ≥ 8.0.
* ComfyUI PR #15308 is **merged** (2026-08-07, commit `344b43989e`, shipped in
  ComfyUI v0.31.0), so ComfyUI >= v0.31.0 loads W4A8 natively and no patch is
  needed. For older builds (v0.30.x) the repository ships the exact patch:
  `patches/comfyui_w4a8_loader.patch` (targets ComfyUI v0.30.0, verified with
  `git apply --check` and compile-checked). Apply from the ComfyUI root:

  ```bash
  git apply patches/comfyui_w4a8_loader.patch
  ```

  On Windows with a plain checkout, run the same command from `C:\Comfyui\ComfyUI`
  after copying the patch file there. If your checkout has drifted (forks, extra
  commits), `patch -p1 --fuzz=3 < comfyui_w4a8_loader.patch` handles it. Restart
  ComfyUI afterwards; the startup log should list `asym_w4a8_int8` among the
  native ops.
* `weight_dtype` (majority-dtype detection) is explicitly bypassed for quantized
  checkpoints in ComfyUI, so int8-packed weights are safe on disk.

## License

Apache-2.0 (see [LICENSE](LICENSE)).
