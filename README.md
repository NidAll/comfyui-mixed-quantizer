# ComfyUI mixed precision quantizer

A single-file converter that turns a diffusion model checkpoint into a
quantized checkpoint ComfyUI can load. Detection, quantization, packing,
metadata, and validation are reimplemented in one file against verified
reference behavior; no ComfyUI or comfy-kitchen code is imported at runtime.

Two modes:

* `--format w4a8` (default, stable channel): every quantized layer uses
  `asym_w4a8_int8` (ConvRot 256, group 16). The 16-entry codebook is adaptive
  by default (`--codebook-mode auto`): ordinary Gaussian-after-ConvRot layers
  use a fast fixed table, genuinely heavy-tailed layers get a fitted
  Lloyd-Max table. `--codebook-mode fit` restores the legacy per-layer
  fitting path, which with `--seed 0` stays byte-identical to v1.3.0/1.5.0
  (pinned by golden-vector self-tests).
* `--format mixed` (experimental channel, requires `--experimental`): a
  per-layer optimizer over `convrot_w4a4`, `asym_w4a8_int8`, and
  `int8_tensorwise`. Each layer gets the cheapest format that stays inside a
  quality gate; layers that cannot meet the gate stay at original precision.

Current version: `1.6.0`.

## Versions and changelog

### 1.6.0

Adaptive W4A8 codebooks and text-encoder quantization.

* `--codebook-mode auto|fixed|fit` (default auto). Auto samples deterministic
  evenly spaced rows (2^22 elements) and measures excess kurtosis; layers
  with kurtosis at or below -0.1 use the fixed Gaussian table, the rest get a
  25-iteration fitted Lloyd-Max table. `fixed` never fits, `fit` keeps the
  legacy per-layer path. Auto and fixed run two ALS group-scale refinement
  passes, fit keeps three for the reference path. Code assignment is a binary
  search over the monotonic codebook instead of a 16-way scan, with the old
  lower-index tie rule preserved.
* Text encoders: `--components auto|diffusion|text-encoder|all` selects the
  quantization scope. Auto keeps historical behavior: a diffusion checkpoint
  quantizes only its detected diffusion component, a standalone detected text
  encoder quantizes its text rules. `all` additionally recognizes embedded
  text towers outside an explicit diffusion prefix. Four text policies cover
  the text encoders used by the 35 diffusion families: LLM-style
  (Qwen/Llama/Gemma/Mistral), T5/UMT5, Hugging Face CLIP, and BERT/Jina
  (experimental).
* TE weights join the same planner, gates, metadata, and validator as
  diffusion weights. Embeddings, normalization parameters, heads, text
  projections, and vision/audio towers are never quantized (the ComfyUI TE
  loader supports only fp8/int8 embeddings, so a W4A8/W4A4 record on one
  would fail at load time). Legacy fused OpenCLIP `in_proj_weight`
  parameters are not selected.
* In W4A8 mode a TE linear is quantized only when its width is ConvRot-256
  compatible (K % 256 == 0); mixed mode covers any width through the INT8
  fallback.
* The `comfy_wxa8` extension metadata records the detected component,
  selected components, and codebook mode; `algorithm_identity` bumps
  `quant_algorithm_rev` to adaptive-codebook-searchsorted-r3.
* With `--components auto` the decision scope is unchanged from 1.5.0, and
  with `--codebook-mode fit --seed 0` the W4A8 payload stays on the
  byte-identical reference path.

Self-test suite: 52 checks (was 50).

First stable release. The `-experimental` suffix is gone and the modes are
split into two channels: `--format w4a8` is the stable channel, `--format
mixed` requires `--experimental`.

Fixes:

* Chunked INT8 validation crashed on W4A4-only temporaries; validation now
  produces identical results at any chunk size.
* Runtime certificates are applied before planning, and `runtime_certified`
  is per format instead of a single blanket flag.
* Calibration caches are content-addressed (schema v2) and bound to the
  checkpoint they were built from; a different checkpoint cannot reuse them.
* Source hashes are validated as a label-to-hash mapping, so a shard rename
  with identical content is detected.
* A shard index is authoritative: tensors it does not list are an error
  unless `--allow-extra-shard-tensors` opts in.
* Publication is race-safe: a destination created during conversion is never
  clobbered, and `--overwrite` re-verifies the destination identity.

New behavior:

* Codebook sampling is chunk-invariant: the sample count is canonical
  (300000 elements) and never derived from `--max-memory`. The same input at
  32 MiB and 8 GiB budgets produces the same tensor payload.
* `--seed` controls the sampling indices (default 0 keeps the reference
  path).
* `--nonfinite-policy error|keep` gives NaN/Inf weights an explicit policy;
  calibration rows must be finite.
* Runtime capabilities are tri-state (known, unsupported, unknown). Missing
  environment information is never reported as supported.
* Validation uses independent reference decoders for W4A8, W4A4, and INT8,
  written from the format spec and sharing no code with the production
  dequantizers. A `payload-size-accurate` check confirms the serialized
  payload matches the plan byte for byte.
* Output metadata carries an `algorithm_identity` block with independent
  revisions for the quantizer, formats, planner, validator, and calibration.

Self-test suite: 50 checks (was 40).

### 1.4.0-experimental

Merged the mixed-precision branch into main. Added the per-layer planner
over `convrot_w4a4`, `asym_w4a8_int8`, and `int8_tensorwise`, with quality
and compression gates, runtime capabilities and certificates, calibration
activations, and the v2 extension metadata schema. Mixed mode carried the
`-experimental` suffix on the version.

### 1.3.0

Hardened the W4A8 runtime contract: ConvRot is 256-only, K not divisible by
256 passes through, golden-vector self-tests pinned the packed and fp8
payloads. This is the byte-identical behavior that 1.5.0 still preserves.

## How it works

The converter runs five steps. Nothing is written until the plan passes its
gates.

1. **Inspect.** The safetensors header (names, shapes, dtypes) is read and the
   architecture is matched against an embedded registry of 39 policy families
   (35 diffusion + 4 text encoder) covering all ComfyUI model classes at the
   research revision. Unknown
   architectures fail closed unless `--architecture` is given.
2. **Decide.** `--components` fixes the scope (detected diffusion model,
   standalone text encoder, or embedded text towers). Each family policy
   defines which 2D float linear weights in that scope are quantizable and
   which stay at original precision (patch embedders, time and text embedders,
   output heads, norms, embeddings). In w4a8 mode the further rules are the
   shape gate (K % 256 == 0) and the adaptive codebook (`auto`/`fixed`/`fit`).
   In mixed mode every candidate is quantized
   and dequantized with each eligible format, the error is measured, and the
   cheapest format under the profile's per-layer gate is selected.
3. **Meet the gates.** A parameter-weighted global error gate runs over the
   whole targeted set, so a big FFN counts more than a small projection. If it
   fails, the planner promotes layers greedily, best error reduction per extra
   byte, up to original precision, until the gate passes. The gates are hard:
   a plan that misses them aborts with the measured numbers, the largest
   contributors, and the override that would let it through. A plan that
   quantizes nothing is rejected as passthrough-only.
4. **Write.** Quantized layers are emitted with ComfyUI-native tensor payloads
   and per-layer metadata. The output is written atomically and never
   overwrites the input.
5. **Validate.** `--validate` reopens the output and checks inventory, shapes,
   dtypes, per-format metadata, packing round trips, reconstruction error
   against the source, determinism, and hashes. Reconstruction uses an
   independent reference decoder for each format, written from the format spec
   and sharing no code with the production dequantizers, so a bug in one path
   cannot hide inside the other. A `payload-size-accurate` check confirms the
   serialized tensor payload matches the planned inventory byte for byte.

Without `--calibration-source` the quality gates use weight-only
reconstruction error. With calibration activations they use the simulated
runtime output error: activation rotation, activation quantization, and the
scaled quantized GEMM, always in the ConvRot basis. The simulators are
cross-checked against comfy-kitchen's eager kernels to 1e-4 relative
agreement in `testdata/runtime_equivalence.py`.

## Requirements and install

Python 3.11+, torch, safetensors, numpy. No ComfyUI installation is needed to
convert.

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

`-r requirements-optional.txt` adds comfy-kitchen and packaging, needed only by
the companion tools that import comfy-kitchen (`tools/runtime_certify.py`,
`testdata/runtime_equivalence.py`, `tools/certified_convert.py --certify`).

## Quick start

```bash
# stable W4A8 (adaptive codebook: fixed table for ordinary layers, fitted for heavy-tailed ones)
.venv/bin/python comfyui_wxa8_quantizer.py MODEL.safetensors --output OUT.safetensors --format w4a8 --validate

# fastest W4A8 (no codebook fitting at all)
.venv/bin/python comfyui_wxa8_quantizer.py MODEL.safetensors --output OUT.safetensors --format w4a8 --codebook-mode fixed --validate

# legacy fitted codebooks (reference path, byte-identical to 1.5.0 with seed 0)
.venv/bin/python comfyui_wxa8_quantizer.py MODEL.safetensors --output OUT.safetensors --format w4a8 --codebook-mode fit --seed 0 --validate

# standalone text encoder, W4A8 (auto detects the text component)
.venv/bin/python comfyui_wxa8_quantizer.py t5xxl.safetensors --output t5xxl_w4a8.safetensors --format w4a8 --validate

# standalone text encoder, mixed (INT8 covers widths W4A8 cannot)
.venv/bin/python comfyui_wxa8_quantizer.py t5xxl.safetensors --output t5xxl_mixed.safetensors --format mixed --experimental --profile balanced --target-runtime nvidia --validate

# combined checkpoint with an embedded text tower
.venv/bin/python comfyui_wxa8_quantizer.py combined.safetensors --output combined_w4a8.safetensors --format w4a8 --components all --validate

# mixed, experimental channel (requires --experimental)
.venv/bin/python comfyui_wxa8_quantizer.py MODEL.safetensors --output OUT.safetensors --format mixed --profile auto --experimental --validate

# sanity checks and model inspection
.venv/bin/python comfyui_wxa8_quantizer.py --self-test
.venv/bin/python comfyui_wxa8_quantizer.py --version
.venv/bin/python comfyui_wxa8_quantizer.py --list-architectures
.venv/bin/python comfyui_wxa8_quantizer.py MODEL.safetensors --inspect

# source-free verification of an existing output (no original model needed)
.venv/bin/python comfyui_wxa8_quantizer.py --verify-output OUT.safetensors
```

`--inspect` prints the detected architecture, component, prefix, and evidence,
plus the tensor list. Run it before converting a model you have not seen
before.

## Why mixed precision

W4A8 requires K % 256 == 0 on every quantized layer, because the comfy-kitchen
CUDA fused kernels only implement ConvRot at width 256. Several real
architectures fail that rule:

| Architecture | Problem K | W4A8 | W4A4 | INT8 |
| ------------ | --------: | :--: | :--: | :--: |
| Flux, Qwen, Krea, Z-Image | multiples of 256 | yes | yes | yes |
| SDXL 320/640 blocks | 320, 640 | no | yes | yes |
| PixArt 1152, MiniMax fc2 | 1152 | no | yes | yes |
| HunyuanDiT | 1408 | no | yes | yes |
| CogVideoX-2B | 1920 | no | yes | yes |
| Boogu | 3360 | no | no | yes |
| OmniGen2 | 2520 | no | no | yes |

W4A4 needs K % 64 == 0 (int4 MMA kernel contract). INT8 has no shape
requirement, so it is the universal fallback.

Measured on identical weights (K=768): W4A4 weight error 0.142, W4A8 0.070,
INT8 0.005. W4A4 is about twice as noisy as the W4A8 codebook path, so the
profiles use it only where size matters.

Real conversions: Boogu-Image-0.1-Turbo mixed balanced goes from about 16 GB to
about 9.6 GB (the K=3360 layers become rowwise INT8, the K=13568 layers stay
W4A8). Kroma v0.2 Turbo (Krea2) with `--format w4a8` goes from 25.64 GB to
about 7.7 GB (256 of 430 tensors quantized, 97.5% of parameters).

## Profiles and gates

| Profile | Layer gate | Global mean gate | B/param target | Max original share |
| ------- | ---------: | ---------------: | -------------: | -----------------: |
| balanced | 0.10 | 0.08 | 0.90 | 5% |
| conservative | 0.05 | 0.04 | 1.05 | 2% |
| size-first | 0.15 | 0.10 | 0.75 | 10% |

The defaults sit against the measured W4A8 weight error (~0.073). Override
with `--quality-gate`, `--global-error-gate`,
`--max-linear-bytes-per-param`, and `--max-bf16-fraction` (alias
`--max-original-byte-fraction`). Compression is enforced: a plan over the byte
target or the original-precision share aborts with CompressionGateError.

Without calibration, balanced tends to pick W4A8 everywhere: W4A4 fails the
0.10 gate (error ~0.142) and INT8 costs more bytes than W4A8. W4A4 appears
under size-first (gate 0.15) or when calibration activations lower its
measured output error.

## Main options

| Option | Meaning |
| ------ | ------- |
| `--components auto\|diffusion\|text-encoder\|all` | what to quantize: auto = detected primary component only (default); text-encoder = text rules only; all = diffusion plus embedded text towers outside a prefixed diffusion model |
| `--codebook-mode auto\|fixed\|fit` | W4A8 codebook strategy: auto (default) = fixed table, fitted only for heavy-tailed layers; fixed = never fit; fit = legacy per-layer Lloyd-Max |
| `--format w4a8\|mixed` | quantization mode (default w4a8; mixed requires `--experimental`) |
| `--experimental` | enable experimental features (currently: `--format mixed`) |
| `--profile auto\|balanced\|conservative\|size-first` | gate and compression profile (mixed); auto picks balanced for an accelerator target, conservative for CPU |
| `--target-runtime auto\|nvidia\|amd\|cpu` | runtime used for format eligibility (mixed) |
| `--quality-gate F`, `--global-error-gate F` | error gate overrides |
| `--max-linear-bytes-per-param F`, `--max-bf16-fraction F` | hard compression targets |
| `--calibration-source PATH` | activation rows (.npz/.pt/.npy/dir) for runtime-output gates |
| `--require-calibration` | refuse planning without activation data |
| `--w4a4-linear-dtype int4\|int8` | W4A4 execution variant (default int8) |
| `--disable-w4a4`, `--disable-w4a8`, `--disable-int8` | drop a format from the candidate set |
| `--runtime-certificate PATH`, `--require-runtime-certificate` | observed-behavior override / hard certification |
| `--seed N` | seed for fitted-codebook and kurtosis sampling (default 0; auto-mode row selection is evenly spaced and seed-independent, fixed mode never samples) |
| `--nonfinite-policy error\|keep` | NaN/Inf in quantizable layers: error (default, names the layer) or keep at original precision |
| `--allow-extra-shard-tensors` | tolerate shard tensors the index does not list (default: error; the index is authoritative) |
| `--strip-gpu-identity` | omit GPU name, capability, ROCm from metadata |
| `--device auto\|cpu\|cuda\|rocm` | quantization compute device |
| `--max-memory SIZE` | per-tensor working budget (default 2G) |
| `--validate`, `--validation-only` | full check after conversion / re-check an output |
| `--verify-output PATH` | source-free check of an existing output: structure, metadata, packing, payload hash |
| `--version` | print the converter version and exit |
| `--dry-run`, `--report PATH` | plan and report without writing |
| `--resume`, `--overwrite` | interruption recovery / replace output |
| `--include`, `--exclude`, `--keep-precision` | force or protect layers by regex |
| `--architecture NAME` | override detection |
| `--trust-pickle` | allow pickle inputs |
| `--sensitivity-threshold F` | legacy w4a8-mode keep-precision threshold |

The fitted-codebook sample count is canonical (300000 elements, never derived
from the memory budget), so `--max-memory` changes how much is processed at
once, not the math. A conversion run at 32 MiB and one run at 8 GiB produce
the same tensor payload. Auto mode is chunk-invariant too: its decision rows
are evenly spaced, so the codebook choice does not depend on the chunk
budget. `--seed` controls the fitted-codebook and kurtosis sampling indices;
seed 0 is the reference path.

## Output format and metadata

Per quantized layer the file carries the ComfyUI-native tensors:

* W4A8: `{layer}.weight` int8 [N, K/2], `{layer}.weight_s_rel` fp8 [N, K/16],
  `{layer}.weight_s_channel` fp32 [N], `{layer}.weight_codebook` fp32 [16]
* W4A4: `{layer}.weight` int8 [N, K/2] (packed signed int4, low nibble = even
  column), `{layer}.weight_scale` fp32 [N]
* INT8: `{layer}.weight` int8 [N, K], `{layer}.weight_scale` fp32 [N, 1]

`__metadata__["_quantization_metadata"]` lists every quantized layer with its
format and parameters. ComfyUI reads each entry into its own `.comfy_quant`
record; no custom format names are introduced.

The `comfy_wxa8` extension block is schema-versioned: single-format W4A8
output uses `comfy_wxa8/v1`, mixed output uses `comfy_wxa8/v2` (mode,
per-format contracts, distribution, profile and gates, runtime status,
quality_validation level). Every output carries an `algorithm_identity` block
with independent revisions for the quantization algorithm, the W4A8 format,
the mixed planner, the validator, and the calibration loader, so a converter
version bump is never the only provenance signal. Levels: `unverified`
(weight-only gates), `calibrated` (activation data), `model-verified`
(testdata/model_quality.py passed on the target machine).

## Loading in ComfyUI

* ComfyUI >= v0.31.0 loads all three formats natively (W4A8 support merged
  via PR #15308). Older builds need `patches/comfyui_w4a8_loader.patch`.
* comfy-kitchen >= 0.2.x provides the layout classes: `AsymW4A8Int8Layout`,
  `TensorCoreConvRotW4A4Layout`, `TensorWiseINT8Layout`.
* CUDA kernels need PyTorch cu130+ and SM >= 8.0. ROCm needs triton >= 3.7.
  The eager fallback runs anywhere.

Before conversion, mixed mode checks the installed packages: every selected
format must exist in the ComfyUI quant_ops registry and every layout in
comfy-kitchen, otherwise it aborts with RuntimeCompatibilityError and suggests
the `--disable-*` escape. When neither package is installed it warns and
relies on the runtime-contract validators.

Runtime capabilities are tri-state: loadable, executable, and accelerated are
each known-supported, known-unsupported, or unknown. Missing information is
never reported as supported. A machine without a detected CUDA device gets
`unknown` for the nvidia path, not an optimistic yes, and mixed planning fails
closed on it.

Static existence is not the same as proof on your machine.
`tools/runtime_certify.py` runs tiny real operations for all three formats and
writes a JSON certificate; pass it with `--runtime-certificate` to override
W4A4 dispatch guesses with observed behavior, or add
`--require-runtime-certificate` to refuse conversion unless every selected
format is certified. `tools/certified_convert.py` orchestrates the full chain:
staged conversion, certification, ComfyUI smoke, model-quality gate, then
atomic publish.

## Architecture support

The embedded registry holds 39 policy families (35 diffusion + 4 text
encoder) covering all ComfyUI model classes at the research revision. Each
family defines its quantize set and its
protected layers. Mixed mode does not change detection or protection; it only
changes what happens to a layer once it is a policy candidate. Instead of
passing through when K fails the W4A8 shape rule, the layer is evaluated for
W4A4 and INT8.

## Text encoder quantization

`--components` extends the same machinery to text encoders. TE weights are
2D linears like any other, and the three formats, the planner gates, the
validator, and the metadata all apply directly.

| Selection | Behavior |
| --------- | -------- |
| `auto` (default) | quantizes only the detected primary component: the diffusion scope for a diffusion checkpoint, the text rules for a standalone supported text encoder. Historical diffusion behavior is unchanged |
| `diffusion` | quantizes only the detected diffusion-model scope |
| `text-encoder` (or `text_encoder`) | applies the recognized text-transformer rules and leaves other components untouched |
| `all` | quantizes the detected diffusion scope plus recognized text towers outside an explicit diffusion prefix |

For combined checkpoints the diffusion prefix always wins, so a text pattern
never reclassifies a diffusion layer. Embedded text discovery in `all` mode
is intentionally limited to keys outside the detected diffusion prefix;
review `--inspect` before converting a combined file you have not seen.

Four policies cover the text encoders used by the 35 diffusion families:

| Policy | Status | Recognized linear layers |
| ------ | ------ | ------------------------ |
| `text_encoder_llm` | verified | decoder self-attention `q_proj`/`k_proj`/`v_proj`/`o_proj`/`out_proj`, linear-attention projections, MLP `gate_proj`/`up_proj`/`down_proj`/`fc1`/`fc2`/`w1`/`w2`/`w3` |
| `text_encoder_t5` | verified | encoder/decoder `SelfAttention` and `EncDecAttention` `q`/`k`/`v`/`o`, `DenseReluDense` `wi`/`wi_0`/`wi_1`/`wo` |
| `text_encoder_clip` | verified | Hugging Face CLIP `text_model.encoder.layers.N.self_attn.{q,k,v,out}_proj`, `mlp.{fc1,fc2}` |
| `text_encoder_bert` | experimental | BERT/Jina `attention.self.{query,key,value}`, `attention.output.dense`, `{intermediate,output}.dense` |

Safety exclusions are narrow (text policies do not reuse the diffusion
universal exclusion, which excludes whole `encoder`/`decoder` subtrees):

* Token, word, position, type, and shared embeddings.
* LayerNorm, RMSNorm, input/post-attention and final normalization, and
  Q/K normalization parameters.
* LM heads, classifiers, score heads, poolers, text projections, and
  projection heads.
* Vision towers, image encoders, audio submodels, and multimodal
  projectors.
* Biases, buffers, non-weight entries, non-2D tensors, non-floating tensors,
  and small tensors below the policy threshold.

`--include` can force an eligible linear into the candidate set but never
bypasses component selection, policy exclusions, dtype checks, shape checks,
or the minimum-size check. Embedding tables in particular stay at original
precision because the ComfyUI TE loader supports only fp8/int8 embeddings; a
W4A8/W4A4 record on one would fail at load time.

Format eligibility: in W4A8 mode a TE linear is quantized only when its width
is ConvRot-256 compatible (K % 256 == 0); incompatible layers pass through.
Mixed mode keeps every selected non-empty 2D float TE linear as a candidate
and falls back to INT8 for any width, so it gives broader TE coverage.
Legacy OpenCLIP checkpoints with fused parameters such as
`transformer.resblocks.*.attn.in_proj_weight` are not selected: the native
quant metadata binds configuration to module names derived from keys ending
in `.weight`, and an unattached fused parameter cannot be safely associated
with a runtime linear module.

Real checkpoints: the diffusion file is usually DiT-only (Flux, SD3.5, WAN,
HunyuanVideo, Boogu, Kroma) and the TE ships separately, so converting the TE
file is the common case. SD1.5/SD2.x/SDXL full checkpoints embed their TEs.
The user-facing TE files for the biggest families: `t5xxl_fp8_e4m3fn.safetensors`
(Flux/SD3), `umt5_xxl_fp8_e4m3fn_scaled.safetensors` (WAN),
`llava_llama3_fp8_scaled.safetensors` (HunyuanVideo), `qwen3vl_4b_*` (Krea2 /
Kroma v0.2), `qwen3vl_8b_*` (Boogu), `qwen_2.5_vl_7b_*` (QwenImage,
HunyuanImage 2.1). fp16-to-W4A8 is about a 4x saving on the TE alone; CLIP-L
(246 MB) saves little and is not worth converting.

Loader requirements: ComfyUI >= v0.31.0 loads quantized TE files natively
(`load_clip` runs the same `comfy_quant` conversion as diffusion models); the
v0.30.0 loader patch covers the linear path too.

## Sharded inputs and publication safety

When a directory contains `model.safetensors.index.json`, the index
`weight_map` is authoritative. A tensor in a shard that the index does not
list is an error, a missing or duplicate tensor is an error, and a referenced
shard that does not exist is an error. `--allow-extra-shard-tensors` restores
the old lenient behavior and records a warning. Source hashes are validated as
a label-to-hash mapping, so swapping two shard names with identical content is
detected.

Publication is race-safe. Without `--overwrite` the validated candidate is
published with a no-clobber hard link, so a destination created by another
process during conversion is never replaced. With `--overwrite` the
destination identity (device, inode, non-symlink) is captured before
conversion and re-verified immediately before publication; a concurrent
swap, deletion, or symlink replacement aborts with an error and leaves the
staged file intact.

## Checking the result

* `--self-test`: 52 embedded checks. Golden vectors for W4A8, W4A4, and INT8
  (embedded reference weight, cross-platform safe), the eligibility matrix,
  mixed planning on real Boogu and Kroma dims, binary-search code assignment
  versus the original 16-way nearest-level reference (including exact
  midpoint ties), standalone and embedded text-encoder classification
  (Qwen-style W4A8; T5/CLIP/BERT in mixed mode; `--components all` selects
  an embedded tower beside a prefixed Flux model; embeddings, norms, and
  vision-tower linears are never quantized), hard
  gate failures, BF16 promotion, runtime capability matrix, planner
  determinism, corrupted metadata rejection for all three formats,
  chunk-invariant payloads, chunked vs full validation equality,
  reference-decoder consensus, source-free `--verify-output`, shard-hash
  identity, publication races, channel gating, and architecture sync.
* `--validate`: reopens the output and checks inventory, shapes, dtypes,
  per-format metadata and runtime contract, scales, packing round trips,
  reconstruction error bounds (W4A8 policy bound, W4A4 0.20, INT8 0.05) via
  the independent reference decoders, determinism, payload size, and hashes.
* `testdata/cuda_smoke.py`: CUDA regression, 10/10 on an RTX 3050: fused W4A8
  kernels, INT8 at K=3360, W4A4 at K=1152 in both `linear_dtype` variants, a
  full mixed checkpoint through the kernels.
* `testdata/runtime_equivalence.py`: simulators vs comfy-kitchen eager
  kernels, exact agreement to 1e-4 (measured 0 to 5e-8) across the awkward K
  matrix.
* `testdata/comfyui_smoke.py`: real ComfyUI load path; asserts every
  quantized layer's layout matches its metadata format, then runs one forward.
  `--require-format` forces the checkpoint to actually contain the listed
  formats.
* `testdata/model_quality.py`: BF16-relative denoiser comparison on the target
  machine (rel L2, cosine, SNR per timestep against a threshold). A passing
  run earns the model-verified label.
* `testdata/comfyui_patch_smoke.py`: LoRA, offload, and low-VRAM integration
  smoke for real mixed checkpoints.

CI workflows are temporarily removed and live in git history (`ci.yml`,
`cuda-smoke.yml`, `nightly-sync.yml`, `release-compat.yml`). Until they are
restored, run the self-tests, fixture conversions, and the sync scripts
locally before merging or releasing.

## Companion tools

* `tools/hf_mixed_quantize.py`: Colab-ready flow that downloads a model from
  Hugging Face, converts it for CUDA inference, strips GPU identity, and
  uploads the result to a new repo in the user's HF account.
* `tools/hf_mixed_quantize_optimized.py`: same flow, faster (hf_transfer
  downloads, opt-in `--validate`, disk preflight, phase timing). Use this one
  for real conversions.
* `tools/runtime_certify.py` and `tools/certified_convert.py`: the certificate
  generator and the staged convert-certify-publish chain.
* `testdata/make_fixtures.py`: builds the fixture checkpoints used in
  verification.

## Known limitations

* Mixed mode is experimental. Without calibration the quality gates are
  weight-based; runtime-output gates need `--calibration-source`.
* Pure W4A8 still requires ConvRot-256-compatible widths (K % 256 == 0).
  Incompatible layers remain at original precision; mixed mode allows an
  INT8 fallback.
* Embedded text discovery is conservative. In `all` mode, text towers are
  discovered outside an explicit diffusion prefix; a prefixless combined
  checkpoint may need a separate text-encoder conversion.
* Only recognized naming conventions are selected. Custom model wrappers may
  need a new family policy or carefully reviewed `--include` patterns.
* Legacy fused OpenCLIP attention parameters are not quantized; use split
  projection weights.
* The BERT/Jina text policy is experimental; validate model quality and
  runtime loading before distribution.
* Auto codebook selection is a heuristic. `fixed` and `fit` exist for
  controlled comparisons.
* Text embeddings and output heads intentionally remain unquantized. This is
  a safety/quality choice, not a coverage bug.
* On the eager backend W4A4 always runs the int4 activation path, which is
  noisier than the CUDA int8 path. The planner simulates the variant you
  selected, so a CPU conversion is evaluated on the int4 path.
* A full forward with all three layouts in one model has only been exercised
  at kernel level; the end-to-end check needs a real checkpoint on ComfyUI >=
  v0.31.0.
* W4A4 at ConvRot 16/64 uses the generic activation rotation path, slower
  than the fused 256-wide kernels.
* The int4 activation variant (`--w4a4-linear-dtype int4`) shows noticeably
  worse error than int8 on synthetic data (0.24 vs 0.16 total); benchmark it
  per model.
* W4A4 checkpoints combined with LoRA or dynamic offload are untested.
* A stale calibration cache from an older converter (no v2 fingerprint) is
  accepted with a warning; rebuild it to bind it to the checkpoint.

## Research basis

* W4A8: comfy-kitchen PR #90 (`AsymW4A8Int8Layout`, merge
  `aa1ab2263dc06225d9de6702dfc087313d4bc971`); ComfyUI PR #15308 (merged
  2026-08-07, ComfyUI >= v0.31.0).
* W4A4: comfy-kitchen `TensorCoreConvRotW4A4Layout` and the eager
  `quantize_convrot_w4a4_weight`: regular-Hadamard rotation, rowwise signed
  int4, scale = absmax / 7, emission range [-7, 7], packed low nibble = even
  column. The int4 MMA kernels pin `quant_group_size = 64`.
* INT8: ComfyUI `TensorWiseINT8Layout`, the same rowwise contract as the
  Comfy-Org int8_convrot checkpoint family. The scale is stored [N, 1] so the
  eager and CUDA backends both broadcast.
* Golden vectors for the two new formats were generated from comfy-kitchen
  0.2.28 and embedded in the self-tests.

## License

Apache License 2.0 (see LICENSE).
