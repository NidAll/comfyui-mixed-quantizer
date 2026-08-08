# AGENTS.md (experimental/mixed-precision branch)

## Project

`comfyui_wxa8_quantizer.py` is a standalone, single-file converter that turns
generative-model checkpoints into ComfyUI-native quantized checkpoints. It
does not import ComfyUI or comfy-kitchen at runtime.

This branch (`experimental/mixed-precision`) adds `--format mixed`: a
per-layer optimizer over the ComfyUI-native formats `convrot_w4a4`,
`asym_w4a8_int8`, and `int8_tensorwise`. It is experimental and not merged to
main. `--format w4a8` remains the default and is byte-identical to main
v1.3.0 (golden vectors and the 35/35 self-test suite prove it).

- Local path: `/home/nidall/projects/testdeepseek/quantizationscripts_w4a8_w3a8`
- Repo (public): `https://github.com/NidAll/comfyui-w4a8-quantizer`
- Branch: `experimental/mixed-precision` (do not merge to main without review)
- Script version: `1.4.0-experimental` (`CONVERTER_VERSION`)
- Audit status: P0 items closed (hard quality/compression gates, BF16
  promotion candidate, runtime-output calibration metric, per-format runtime
  compatibility probe). Remaining certification gaps are documented in the
  README limitations (real-model three-layout forward, LoRA/offload).

## Format facts (verified, do not guess)

### W4A8 (unchanged from main)

Per quantized layer: `{layer}.weight` int8 [N, K/2] (even column low nibble),
`{layer}.weight_s_rel` fp8_e4m3fn [N, K/16], `{layer}.weight_s_channel` fp32
[N], `{layer}.weight_codebook` fp32 [16]. Metadata: `format=asym_w4a8_int8`,
`group_size=16`, `convrot=true`, `convrot_groupsize=256`. K % 256 == 0 is
required (CUDA fused kernels are ConvRot-256-only).

### W4A4 (new, convrot_w4a4)

- `{layer}.weight` int8 [N, K/2], packed signed int4, low nibble = even column
  (matches comfy-kitchen `_pack_int4_row_major`)
- `{layer}.weight_scale` fp32 [N], rowwise absmax / 7, emission range [-7, 7]
  (matches eager `quantize_convrot_w4a4_weight`; golden-vector byte digests in
  `_test_w4a4_golden`)
- Metadata: `format=convrot_w4a4`, `convrot_groupsize` (largest power of 4 in
  {16, 64, 256} dividing K), `quant_group_size=64` (kernel contract), and
  `linear_dtype` (execution variant only: int4 or int8, default int8; never a
  quality fallback)
- Eligibility: K % 64 == 0 and K % cgs == 0. Boogu 3360 and OmniGen2 2520 are
  NOT eligible (3360 % 64 = 32, 2520 % 64 = 24). PixArt 1152, HunyuanDiT 1408,
  CogVideoX-2B 1920, SDXL 320/640, MiniMax fc2 1152 ARE eligible.
- Measured weight error ~0.142, about 2x the W4A8 codebook path (0.070).
  Dequant requires the signed nibble interpretation (`unpack_int4_signed`,
  NOT `unpack_w4` which is unsigned).

### INT8 (new, int8_tensorwise)

- `{layer}.weight` int8 [N, K], `{layer}.weight_scale` fp32 [N, 1]
  (rowwise absmax / 127, clamp min 1e-30; [N, 1] so eager and CUDA backends
  both broadcast; matches eager `quantize_int8_rowwise`, golden digests in
  `_test_int8_golden`)
- Metadata: `{"format": "int8_tensorwise"}` only. No ConvRot, no K constraint.
  This is the universal fallback tier and the fix for Boogu/OmniGen2.
- Measured weight error ~0.005. Comfy-Org ships the same rowwise scheme in
  its `*_int8_convrot` checkpoints (those add convrot=true; we do not).

## Mixed mode design

`MixedPlanner` (in the converter): per-layer candidate evaluation (quantize +
dequantize each eligible format; with calibration activations the error is
the RUNTIME OUTPUT error, an exact eager-path emulation: activation rotation,
dynamic rowwise activation quantization (int8, or int4 for the W4A4 int4
variant), and the scaled quantized GEMM), cheapest-acceptable selection under
the profile's per-layer gate, then a greedy promotion loop (best error
reduction per extra byte, original precision included as the final rescue)
until the PARAM-WEIGHTED global mean gate passes. Selection and promotion
mutate `TensorDecision.format` / `.convrot_groupsize` and must run BEFORE
`build_output_entries` (output shapes and offsets depend on per-layer
formats).

`--target-runtime` feeds `RuntimeCapabilities` (`runtime_capabilities_for`)
into eligibility: nvidia = all three formats accelerated (verified on the RTX
3050), amd = HIP/triton paths (not certified here), cpu = eager fallback for
all three. Unsupported formats get `eligible=False` with the reason recorded.
Before conversion `_check_runtime_compatibility` fails closed: every selected
format must exist in the installed ComfyUI quant_ops registry (when comfy is
installed) and every required layout must exist in comfy-kitchen (when
installed); neither installed -> warning only.

Profiles (`MIXED_PROFILE_DEFAULTS`):

| Profile | layer gate | global gate | B/param target | max original share |
| ------- | ---------: | ----------: | -------------: | -----------------: |
| balanced | 0.10 | 0.08 | 0.90 | 0.05 |
| conservative | 0.05 | 0.04 | 1.05 | 0.02 |
| size-first | 0.15 | 0.10 | 0.75 | 0.10 |

Gate defaults are anchored to the measured W4A8 weight error (~0.073). Do not
"improve" them without re-measuring on real dims. `--quality-gate`,
`--global-error-gate`, `--max-linear-bytes-per-param`, and
`--max-bf16-fraction` override; without calibration the quality gates use
weight-only error and the planner warns.

Gates are HARD (audit P0 fixes): `_enforce_gates` raises QualityGateError
(weighted mean error above the global gate, or a plan that quantizes nothing)
and CompressionGateError (effective targeted bpp above target, or
original-precision share above the limit). The global mean is PARAM-WEIGHTED
by layer params, not a plain average. BF16/FP16 is a real promotion candidate
(original bytes, zero error): a layer can be promoted INT8 -> original, and
an all-original plan is rejected, never published.

## Environment

Same as main: uv venv, no model checkpoints stored locally. GPU is an RTX
3050; `--device cuda` works for quantization and CUDA smoke tests run for
real here. `research/ComfyUI` is v0.30.0 + loader patch (gitignored, do not
reset). `.venv` has torch 2.13.0+cu130 and comfy-kitchen 0.2.28.

## Common commands

```bash
.venv/bin/python comfyui_wxa8_quantizer.py --self-test          # 35/35 required
.venv/bin/python comfyui_wxa8_quantizer.py --list-architectures
.venv/bin/python comfyui_wxa8_quantizer.py MODEL.safetensors --inspect

# w4a8 (stable path, unchanged)
.venv/bin/python comfyui_wxa8_quantizer.py MODEL.safetensors --output OUT.safetensors --format w4a8 --validate

# mixed (experimental)
.venv/bin/python comfyui_wxa8_quantizer.py MODEL.safetensors --output OUT.safetensors --format mixed --profile auto --validate
.venv/bin/python comfyui_wxa8_quantizer.py MODEL.safetensors --output OUT.safetensors --format mixed --profile size-first --validate

# fixtures
.venv/bin/python testdata/make_fixtures.py testdata/wan_fixture.safetensors
.venv/bin/python testdata/make_fixtures.py testdata/boogu_real_fixture.safetensors
.venv/bin/python comfyui_wxa8_quantizer.py testdata/boogu_real_fixture.safetensors     --output testdata/boogu_real_fixture_mixed.safetensors --format mixed --profile balanced --validate

# CUDA regression (real GPU; 9/9 on the RTX 3050)
.venv/bin/python testdata/cuda_smoke.py
```

Expected results on the fixtures: boogu_real mixed = 19 W4A8 + 134 INT8
layers, 346.69 MiB vs 1.19 GiB input, 328/328 validation checks pass; wan
mixed balanced = 16 W4A8 + 4 INT8; wan mixed size-first = 10 W4A8 + 9 W4A4 +
1 INT8 (9.75 MiB).

## Verification before claiming success

1. `--self-test` must pass 35/35 (includes W4A4/INT8 golden byte digests,
   eligibility matrix, mixed planning on real Boogu dims, mixed e2e with
   comfy-kitchen layout reload, hard gate failures, BF16 promotion, runtime
   capability matrix, runtime-output metric vs eager kernels, architecture
   sync).
2. All fixture families must pass `--validate` in BOTH `--format w4a8`
   (regression; max relL2 about 0.073) and `--format mixed --profile balanced`
   (0 failed).
3. `testdata/cuda_smoke.py` must pass 9/9 on a CUDA machine.
4. CI matrix (ubuntu/windows/macos) runs self-tests, w4a8 fixture conversions,
   and mixed fixture conversions including a `--validation-only` re-check of a
   mixed output.
5. For loader questions, reproduce with the real ComfyUI path:
   `PYTHONPATH=research/ComfyUI .venv/bin/python`, flow
   `load_torch_file -> convert_old_quants -> load_diffusion_model_state_dict`,
   then check weights are `QuantizedTensor` with the expected layout classes
   (`TensorCoreConvRotW4A4Layout`, `AsymW4A8Int8Layout`, `TensorWiseINT8Layout`).
   `testdata/comfyui_smoke.py` now asserts the layout of every quantized layer
   against its metadata format for both w4a8 and mixed checkpoints.
6. `testdata/comfyui_architecture_sync.py` must pass against the pinned
   ComfyUI revision (CI tarball mode) and against the local
   `research/ComfyUI` checkout when present. The nightly workflow checks
   ComfyUI main and fails naming any new unaccounted class.

## Known behavior, do not "fix" it

- `unpack_w4` is the unsigned W4A8 codec; W4A4 dequant uses
  `unpack_int4_signed` (signed nibbles [-8, 7]). They are different on purpose.
- INT8 `weight_scale` is [N, 1], not [N]. [N] breaks the eager backend
  (`q.float() * scale` fails to broadcast) and the [N, 1] form works on eager
  and CUDA. The runtime-contract validator accepts both shapes for
  backwards-compatible reading.
- The w4a8-only mode must stay byte-identical to main. Any refactor of shared
  code (quantize dispatch, output entries, engine writers, metadata builders,
  plan_from_output, Validator) must keep `--format w4a8` outputs unchanged;
  the golden-vector and deterministic-conversion self-tests guard this.
- Extension metadata is schema-versioned: w4a8 outputs use `comfy_wxa8/v1`
  (W4A8-global fields), mixed outputs use `comfy_wxa8/v2` (`mode: mixed`,
  per-format contracts, distribution, gates, runtime status). Never write
  W4A8-global fields (fp8 scales, codebook packing) into a v2 block.
- The global quality metric is the PARAM-WEIGHTED mean over quantized layers.
  `global_mean_error(info, decisions)` needs `info` now; callers must pass it.
- The calibration metric simulates the runtime operation
  (`runtime_output_rel_l2` / `_simulate_quantized_chunk`): activation
  rotation, activation quantization, scaled GEMM. It is NOT
  `(dequant(W)-W)X`. The self-test cross-checks the W4A4-int4 simulation
  against comfy-kitchen's eager `convrot_w4a4_linear` (must agree within
  0.02).
- ConvRot for W4A8 is 256-only. W4A4 may use cgs 16/64/256 per layer.
- Only 2D linear weights are quantized. Convolutions, embeddings, norms,
  heads, and modulations pass through with the reason recorded.
- Fixtures are gitignored (`testdata/*.safetensors`); reports and the
  `boogu_real_fixture_mixed` golden output artifacts follow the repo rules:
  keep the tree clean of large artifacts, commit reports under
  `testdata/reports/`.
- Pickle inputs require `--trust-pickle`. Outputs never overwrite inputs.

## Editing and docs conventions

- One converter file. Edit with exact-string edits and recompile.
- README.md is the only markdown file. Humanizer-clean: no em dashes, no AI
  vocabulary, no rule-of-three padding, plain technical prose.
- Commit messages summarize behavior and evidence. This branch is
  experimental; do not push to main.
- The ComfyUI W4A4/INT8 loader contracts were verified against the installed
  comfy-kitchen 0.2.28 (eager implementations) and ComfyUI v0.30.0+patch
  (`comfy/ops.py` layer_conf parsing: convrot_w4a4 reads convrot_groupsize,
  hard-codes quant_group_size 64, reads linear_dtype; int8_tensorwise reads
  weight_scale plus optional convrot fields).

## User context

The user runs ComfyUI on Windows at `C:\Comfyui\ComfyUI` (v0.30.0 plus 23
commits) with the loader patch. ComfyUI >= v0.31.0 loads all three formats
natively. The real Boogu-Image-0.1-Turbo (hidden 3360, FFN 13568) is the
primary target: mixed balanced turns the 364 previously-BF16 layers into
INT8, bringing the output from ~16 GB to ~9.6 GB while keeping the 54
K=13568 layers at W4A8.
