# ComfyUI mixed precision quantizer

A single-file converter that turns a diffusion model checkpoint into a quantized
checkpoint ComfyUI can load. It runs standalone: detection, quantization,
packing, metadata, and validation are reimplemented in one file against
verified reference behavior, and no ComfyUI or comfy-kitchen code is imported
at runtime.

Two modes:

* `--format w4a8` (default): every quantized layer uses `asym_w4a8_int8`
  (ConvRot 256, group 16, Lloyd-Max codebook). This path is byte-identical to
  v1.3.0, guarded by the golden-vector self-tests.
* `--format mixed`: a per-layer optimizer over `convrot_w4a4`,
  `asym_w4a8_int8`, and `int8_tensorwise`. Each layer gets the cheapest format
  that stays inside a quality gate; layers that cannot meet the gate stay at
  original precision.

Current version: `1.4.0-experimental`.

## How it works

The converter runs five steps. Nothing is written until the plan passes its
gates.

1. **Inspect.** The safetensors header (names, shapes, dtypes) is read and the
   architecture is matched against an embedded registry of 43 policy families
   covering all 98 ComfyUI model classes. Unknown architectures fail closed
   unless `--architecture` is given.
2. **Decide.** Each family policy defines which 2D float linear weights are
   quantizable and which stay at original precision (patch embedders, time and
   text embedders, output heads, norms). In w4a8 mode the only further rule is
   the shape gate (K % 256 == 0). In mixed mode every candidate is quantized
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
   against the source, determinism, and hashes.

Without `--calibration-source` the quality gates use weight-only reconstruction
error. With calibration activations they use the simulated runtime output
error: activation rotation, activation quantization, and the scaled quantized
GEMM, always in the ConvRot basis. The simulators are cross-checked against
comfy-kitchen's eager kernels to 1e-4 relative agreement in
`testdata/runtime_equivalence.py`.

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
# stable W4A8
.venv/bin/python comfyui_wxa8_quantizer.py MODEL.safetensors --output OUT.safetensors --format w4a8 --validate

# mixed, profile picked from the machine (NVIDIA/AMD: balanced, CPU: conservative)
.venv/bin/python comfyui_wxa8_quantizer.py MODEL.safetensors --output OUT.safetensors --format mixed --profile auto --validate

# sanity checks and model inspection
.venv/bin/python comfyui_wxa8_quantizer.py --self-test
.venv/bin/python comfyui_wxa8_quantizer.py --list-architectures
.venv/bin/python comfyui_wxa8_quantizer.py MODEL.safetensors --inspect
```

`--inspect` prints the detected architecture, the policy that applies, and the
tensor list. Run it before converting a model you have not seen before.

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

Real conversions: Boogu-Image-0.1-Turbo mixed balanced goes from ~16 GB to
~9.6 GB (the K=3360 layers become rowwise INT8, the K=13568 layers stay W4A8).
Kroma v0.2 Turbo (Krea2) with `--format w4a8` goes from 25.64 GB to ~7.7 GB
(256 of 430 tensors quantized, 97.5% of parameters).

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
| `--format w4a8\|mixed` | quantization mode (default w4a8) |
| `--profile auto\|balanced\|conservative\|size-first` | gate and compression profile (mixed) |
| `--target-runtime auto\|nvidia\|amd\|cpu` | runtime used for format eligibility (mixed) |
| `--quality-gate F`, `--global-error-gate F` | error gate overrides |
| `--max-linear-bytes-per-param F`, `--max-bf16-fraction F` | hard compression targets |
| `--calibration-source PATH` | activation rows (.npz/.pt/.npy/dir) for runtime-output gates |
| `--require-calibration` | refuse planning without activation data |
| `--w4a4-linear-dtype int4\|int8` | W4A4 execution variant (default int8) |
| `--disable-w4a4`, `--disable-w4a8`, `--disable-int8` | drop a format from the candidate set |
| `--runtime-certificate PATH`, `--require-runtime-certificate` | observed-behavior override / hard certification |
| `--strip-gpu-identity` | omit GPU name, capability, ROCm from metadata |
| `--device auto\|cpu\|cuda\|rocm` | quantization compute device |
| `--max-memory SIZE` | per-tensor working budget (default 2G) |
| `--validate`, `--validation-only` | full check after conversion / re-check an output |
| `--dry-run`, `--report PATH` | plan and report without writing |
| `--resume`, `--overwrite` | interruption recovery / replace output |
| `--include`, `--exclude`, `--keep-precision` | force or protect layers by regex |
| `--architecture NAME` | override detection |
| `--trust-pickle` | allow pickle inputs |
| `--sensitivity-threshold F` | legacy w4a8-mode keep-precision threshold |

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
quality_validation level). Levels: `unverified` (weight-only gates),
`calibrated` (activation data), `model-verified` (testdata/model_quality.py
passed on the target machine).

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

Static existence is not the same as proof on your machine.
`tools/runtime_certify.py` runs tiny real operations for all three formats and
writes a JSON certificate; pass it with `--runtime-certificate` to override
W4A4 dispatch guesses with observed behavior, or add
`--require-runtime-certificate` to refuse conversion unless every selected
format is certified. `tools/certified_convert.py` orchestrates the full chain:
staged conversion, certification, ComfyUI smoke, model-quality gate, then
atomic publish.

## Architecture support

The embedded registry holds 43 policy families covering all 98 ComfyUI model
classes at the research revision. Each family defines its quantize set and its
protected layers. Mixed mode does not change detection or protection; it only
changes what happens to a layer once it is a policy candidate. Instead of
passing through when K fails the W4A8 shape rule, the layer is evaluated for
W4A4 and INT8.

## Checking the result

* `--self-test`: 39 embedded checks. Golden vectors for W4A8, W4A4, and INT8
  (embedded reference weight, cross-platform safe), the eligibility matrix,
  mixed planning on real Boogu and Kroma dims, hard gate failures, BF16
  promotion, runtime capability matrix, planner determinism, corrupted
  metadata rejection for all three formats, and architecture sync.
* `--validate`: reopens the output and checks inventory, shapes, dtypes,
  per-format metadata and runtime contract, scales, packing round trips,
  reconstruction error bounds (W4A8 policy bound, W4A4 0.20, INT8 0.05),
  determinism, and hashes.
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

MIT (see LICENSE).
