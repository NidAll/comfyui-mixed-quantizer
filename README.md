# ComfyUI mixed precision quantizer (experimental)

A single-file converter that turns supported generative-model checkpoints into
quantized checkpoints for ComfyUI. It supports two modes:

* `--format w4a8`: the stable single-format path. Every quantized layer uses
  `asym_w4a8_int8` (ConvRot 256, group 16, Lloyd-Max codebook).
* `--format mixed`: the experimental per-layer optimizer. Each layer gets the
  cheapest ComfyUI-native format that stays inside a quality gate, chosen from
  `convrot_w4a4`, `asym_w4a8_int8`, and `int8_tensorwise`. Layers that cannot
  meet the gate stay at original precision.

The script is standalone. It does not import, require, or execute any ComfyUI
or comfy-kitchen code at runtime. Inspection, detection, quantization, packing,
metadata, and validation are reimplemented inside the file from verified
reference behavior. See [Research basis](#research-basis).

The mixed mode lives on the experimental branch `experimental/mixed-precision`
and is not yet merged to main. The w4a8 path on this branch is byte-identical
to main v1.3.0 (golden vectors prove it).

## Contents

* [Why mixed precision](#why-mixed-precision)
* [Requirements](#requirements)
* [Installation](#installation)
* [Basic usage](#basic-usage)
* [Mixed mode design](#mixed-mode-design)
* [Profiles and gates](#profiles-and-gates)
* [Main options](#main-options)
* [Output format and metadata](#output-format-and-metadata)
* [Architecture support](#architecture-support)
* [Validation](#validation)
* [Known limitations](#known-limitations)
* [Research basis](#research-basis)
* [License](#license)

## Why mixed precision

The W4A8 format requires `K % 256 == 0` for every quantized layer (the
comfy-kitchen CUDA fused kernels only implement ConvRot at 256). Several real
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

W4A4 needs `K % 64 == 0` (int4 MMA kernel contract) and Boogu's 3360 and
OmniGen2's 2520 fail even that. INT8 has no shape requirement at all, so it
closes the coverage gap. In mixed mode, a Boogu checkpoint that used to keep
364 of 418 layers in BF16 keeps none of them at full precision: the K=3360
layers become rowwise INT8 and only policy-protected layers stay BF16.

Measured on identical weights (K=768): W4A4 weight error 0.142, W4A8 0.070,
INT8 0.005. W4A4 is about twice as noisy as the W4A8 codebook path, so the
profiles treat it as a size-first tier, W4A8 as the balanced workhorse, and
INT8 as the quality and coverage tier.

## Requirements

* Python 3.11+
* torch (any recent version), safetensors, numpy
* No ComfyUI or comfy-kitchen import at converter runtime

## Installation

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
# optional, for the ComfyUI loader reproduction (research/ComfyUI):
uv pip install --python .venv/bin/python comfy-kitchen comfy-aimdo pillow \
    tqdm torchaudio opencv-python transformers psutil av einops requests
```

## Basic usage

```bash
# stable W4A8 (unchanged behavior)
.venv/bin/python comfyui_wxa8_quantizer.py MODEL.safetensors \
    --output OUT.safetensors --format w4a8 --validate

# experimental mixed mode, automatic GPU/architecture detection
.venv/bin/python comfyui_wxa8_quantizer.py MODEL.safetensors \
    --output OUT_mixed.safetensors --format mixed --profile auto --validate
```

`--profile auto` probes the machine (NVIDIA CUDA or AMD ROCm selects
`balanced`, CPU selects `conservative`) and prints the choice. The detected
architecture comes from the same registry used by the w4a8 path.

Quick checks:

```bash
.venv/bin/python comfyui_wxa8_quantizer.py --self-test
.venv/bin/python comfyui_wxa8_quantizer.py --list-architectures
.venv/bin/python comfyui_wxa8_quantizer.py MODEL.safetensors --inspect
```

## Mixed mode design

The planner replaces the binary quantize-or-keep decision with per-layer
candidate evaluation. For every policy-targeted 2D linear weight it evaluates
each eligible format:

1. Quantize and dequantize the layer with that format (bounded memory, row
   chunks for the rowwise formats, the pre-fit codebook path for W4A8).
2. Measure the error against the original weight, and against recorded
   calibration activations when `--calibration-source` is given.
3. Estimate the exact serialized byte count.

Selection is "cheapest acceptable": the smallest candidate whose error is
below the profile's per-layer gate wins. If no candidate passes, the layer
stays at original precision with the reason recorded.

After local selection, a global gate runs on the parameter-weighted mean
error over all selected layers (a huge FFN counts more than a small
projection). If it fails, the planner promotes greedily: it repeatedly
upgrades the layer with the best error reduction per extra byte (W4A4 to
W4A8, W4A8 to INT8, and INT8 all the way to original precision when that is
the cheapest way to buy quality) until the gate passes. Promotions are listed
in the report and in the console warnings.

The gates are hard. A plan that cannot meet its gates is not published:
QualityGateError and CompressionGateError abort the conversion with the
measured numbers and the override that would let it through. A plan that
quantizes nothing is rejected as a passthrough-only checkpoint. The
compression gates enforce the profile's effective bytes/parameter target and
its original-precision share of the targeted linear payload, both reported in
the console summary and the report.

Eligibility rules per format:

* `convrot_w4a4`: K % 64 == 0 and K % cgs == 0, cgs picked per layer as the
  largest power of 4 in {16, 64, 256} dividing K.
* `asym_w4a8_int8`: K % 16 == 0 and K % 256 == 0 (group 16, ConvRot 256).
* `int8_tensorwise`: any K, any 2D float weight.

`--disable-w4a4`, `--disable-w4a8`, `--disable-int8` remove formats from the
candidate set, which is useful for A/B experiments.

The `linear_dtype` field of `convrot_w4a4` is an execution property, not a
quality fallback. It selects the int4 or int8 activation path in ComfyUI and
never changes the stored 4-bit weights. The default is `int8`. One caveat:
the comfy-kitchen eager backend accepts both values but always executes the
int4 activation path; `linear_dtype=int8` only changes the CUDA kernels. The
planner's runtime metric simulates the selected variant, so a CPU-only
conversion evaluates the int4 path.

`--target-runtime` feeds a real capability matrix into the planner.
Eligibility is per format and per backend: nvidia reports all three formats
accelerated, amd reports the HIP/triton paths, cpu reports eager fallback for
every format. A format the target runtime cannot run is excluded from the
candidates with the reason recorded, and the report lists the support status
per format (accelerated vs eager fallback vs unsupported).

## Profiles and gates

| Profile | Layer gate | Global mean gate | B/param target | Max original share |
| ------- | ---------: | ---------------: | -------------: | -----------------: |
| balanced | 0.10 | 0.08 | 0.90 | 5% |
| conservative | 0.05 | 0.04 | 1.05 | 2% |
| size-first | 0.15 | 0.10 | 0.75 | 10% |

The defaults are set against measured W4A8 behavior: the codebook path has
weight-only relL2 around 0.073, so the balanced global gate sits just above it
and the conservative gate sits below it. `--quality-gate F` and
`--global-error-gate F` override both. Without calibration the gates use
weight-only reconstruction error and the planner prints a warning;
`--calibration-source` switches them to runtime-output error.

The calibration metric is the real quantized operation, not a
reconstructed-weight approximation. For every candidate the planner emulates
the eager runtime path exactly: activation rotation (ConvRot formats),
dynamic rowwise activation quantization (int8, or int4 for the W4A4 int4
variant), and the scaled quantized GEMM. The error is the mean over samples
of ||Y_quant - Y_bf16|| / ||Y_bf16||. The emulation is cross-checked against
comfy-kitchen's eager kernels in the self-tests (int8 error 0.011, W4A4-int4
0.204, W4A4-int8 0.143 on the golden activation set).

Compression is enforced, not advisory. `--max-linear-bytes-per-param F`
replaces the profile's effective bytes/parameter target and
`--max-bf16-fraction F` replaces the original-precision share limit; a plan
that misses either aborts with CompressionGateError.

## Main options

```text
--format w4a8|mixed            quantization mode (default w4a8)
--profile auto|balanced|conservative|size-first
--target-runtime auto|nvidia|amd|cpu
--quality-gate F               per-layer error gate override
--global-error-gate F          global mean error gate override
--max-linear-bytes-per-param F hard bytes/parameter target (profile default)
--max-bf16-fraction F          hard original-precision share limit
--w4a4-linear-dtype int4|int8  convrot_w4a4 execution variant (default int8)
--disable-w4a4 / --disable-w4a8 / --disable-int8
--require-calibration          refuse planning without activation data
--calibration-source PATH      local activation rows (.npz/.pt/.npy/dir)
--sensitivity-threshold F      legacy w4a8-mode keep-precision threshold
--max-memory SIZE              per-tensor working budget (default 2G)
--device auto|cpu|cuda|rocm    quantization compute device
--validate                     full standalone validation after conversion
--validation-only              validate an existing output
--dry-run                      plan and report without writing
--report PATH                  human-readable report
--resume / --overwrite         interruption recovery / replace output
```

All w4a8-mode options keep their previous meaning. `--format w4a8` is the
default and behaves exactly as in v1.3.0.

## Output format and metadata

Per quantized layer the output file carries the ComfyUI-native tensors:

* W4A8: `{layer}.weight` int8 [N, K/2], `{layer}.weight_s_rel` fp8 [N, K/16],
  `{layer}.weight_s_channel` fp32 [N], `{layer}.weight_codebook` fp32 [16]
* W4A4: `{layer}.weight` int8 [N, K/2] (packed signed int4, low nibble = even
  column), `{layer}.weight_scale` fp32 [N] (rowwise absmax / 7)
* INT8: `{layer}.weight` int8 [N, K], `{layer}.weight_scale` fp32 [N, 1]
  (rowwise absmax / 127)

`__metadata__["_quantization_metadata"]` lists every quantized layer with its
own format and parameters:

```json
{
  "layers": {
    "blocks.0.attn.qkv": {
      "format": "asym_w4a8_int8",
      "group_size": 16,
      "convrot": true,
      "convrot_groupsize": 256
    },
    "blocks.2.attn.qkv": {
      "format": "convrot_w4a4",
      "convrot_groupsize": 256,
      "quant_group_size": 64,
      "linear_dtype": "int8"
    },
    "blocks.4.attn.qkv": {
      "format": "int8_tensorwise"
    }
  }
}
```

ComfyUI reads each entry into its own `.comfy_quant` record. No custom format
name is introduced.

The `comfy_wxa8` extension block is schema-versioned. Single-format W4A8
outputs use `comfy_wxa8/v1` with the W4A8-global fields (fp8 scales, codebook
packing, format revision). Mixed outputs use `comfy_wxa8/v2`: a `mode:
"mixed"` marker, per-format contract details (weight bits, scale dtype,
packing, ConvRot), the distribution (layer counts, params, bytes per format,
kept payload, effective bytes/parameter, original-precision share, weighted
global error, promotions), the profile and gates, and the runtime target with
per-format support status. The v2 revision is `mixed-r1`, never the W4A8
revision.

## Runtime compatibility

Before conversion, the mixed mode probes the machine statically (installed
package source, never a runtime import). Every selected format must exist in
the installed ComfyUI `quant_ops.py` registry and every required layout
(`AsymW4A8Int8Layout`, `TensorCoreConvRotW4A4Layout`, `TensorWiseINT8Layout`)
must exist in the installed comfy-kitchen. A missing format or layout aborts
the conversion with RuntimeCompatibilityError and suggests the `--disable-*`
escape. When neither package is installed the probe is unavailable and the
per-format runtime-contract validators remain the verification layer, with a
warning. `--validate` re-checks the same per-format requirements after the
conversion.

## Architecture support

The embedded registry keeps its 43 policy families covering all 98 ComfyUI
model classes at the research revision. Mixed mode does not change detection
or protection. It only changes what happens to a layer once it is a policy
candidate: instead of passing through when K fails the W4A8 shape rule, the
layer is evaluated for the other formats. Unknown architectures still fail
closed unless `--architecture` is given.

## Validation

* `--self-test`: 35 embedded checks, including golden vectors for W4A4 and
  INT8 (byte-identical to comfy-kitchen's eager implementations), the
  eligibility matrix, mixed planning on real Boogu dims, hard-failure tests
  for every gate, BF16 promotion, the runtime capability matrix, the
  runtime-output metric cross-checked against comfy-kitchen's eager kernels,
  an end-to-end mixed conversion with layout reload, and an architecture
  coverage sync against the pinned ComfyUI revision.
* `--validate`: reopens the output, checks the inventory, shapes, dtypes,
  per-format metadata and runtime contract (each format has its own shape and
  K rules), scales, packing round trips, reconstruction error against the
  source (per-format bounds: W4A8 policy bound, W4A4 0.20, INT8 0.05),
  determinism, and hashes.
* `testdata/cuda_smoke.py`: CUDA regression for the fused W4A8 kernels, the
  INT8 non-ConvRot path at K=3360, W4A4 at K=1152 with cgs=16 in both
  `linear_dtype` variants, and a full mixed checkpoint through the kernels.
  9/9 checks pass on an RTX 3050. The workflow runs on release tags, pushes
  to the default branch, and PRs touching the converter, in addition to
  manual dispatch.
* `testdata/comfyui_smoke.py`: real ComfyUI load path for W4A8 and mixed
  checkpoints. It reads the per-layer metadata and asserts each quantized
  module's layout matches its format (TensorCoreConvRotW4A4Layout /
  AsymW4A8Int8Layout / TensorWiseINT8Layout), then runs one diffusion-model
  forward. A full three-layout forward still needs the real checkpoint on a
  ComfyUI >= v0.31.0 machine.
* `testdata/comfyui_architecture_sync.py`: compares the embedded registry
  (43 families) with ComfyUI's `supported_models.py` class set. CI runs it
  against the pinned research revision; a nightly workflow runs it against
  ComfyUI main and fails naming any newly added, unaccounted class.

## Known limitations

* Mixed mode is experimental. The quality gates are weight-based without
  calibration; runtime-output gates need `--calibration-source`. The gates
  are hard either way: an unmet gate aborts the conversion instead of
  publishing a checkpoint that misses its targets.
* The W4A4/W4A8/INT8 quality characteristics assume the CUDA kernels for
  `linear_dtype=int8`. On the eager backend, W4A4 always runs the int4
  activation path, which is noisier; the planner simulates the variant you
  selected, so a CPU conversion is evaluated on the int4 path.
* The ComfyUI loader accepts the three formats individually and mixed
  checkpoints load them per layer, but a full one-step model forward with all
  three layouts in one model has only been exercised at kernel level. The
  end-to-end check needs the real checkpoint on a ComfyUI >= v0.31.0 machine.
* W4A4 at ConvRot 16/64 uses the generic activation rotation path, which is
  slower than the fused 256-wide kernels. Speed should be measured per GPU;
  ComfyUI issue #14824 shows native INT8 ConvRot can be slower than FP8 on
  some hardware.
* The int4 activation variant (`--w4a4-linear-dtype int4`) has noticeably
  worse error than the int8 variant on synthetic data (0.24 vs 0.16 total)
  and should be benchmarked per model.
* Checkpoints that mix W4A4 with LoRA or dynamic offload have not been
  tested. Issue #14642 (INT8 ConvRot requantized as plain tensorwise on LoRA
  offload) was a ComfyUI-side bug and is closed, but it is a reminder that
  quantized weights interact with the patching machinery.

## Research basis

* W4A8 format: comfy-kitchen PR #90 (merge `aa1ab2263dc06225d9de6702dfc087313d4bc971`).
  ComfyUI loader: PR #15308, merged 2026-08-07 (ComfyUI >= v0.31.0 loads W4A8
  natively; older builds need `patches/comfyui_w4a8_loader.patch`).
* W4A4: comfy-kitchen `TensorCoreConvRotW4A4Layout` and the eager
  `quantize_convrot_w4a4_weight` (regular-Hadamard rotation, rowwise signed
  int4, scale = absmax / 7, emission range [-7, 7], packed low nibble = even
  column). The int4 MMA kernels pin `quant_group_size = 64`.
* INT8: ComfyUI `TensorWiseINT8Layout`, the same rowwise contract as the
  Comfy-Org `int8_convrot` checkpoint family. Scale stored [N, 1] so eager and
  CUDA backends both broadcast.
* Golden vectors for the two new formats were generated from comfy-kitchen
  0.2.28 and are embedded in the self-tests as byte digests.
* `research/ComfyUI`: checkout at v0.30.0 with the loader patch applied
  (working tree only, do not reset). `research/comfy-kitchen`: checkout at
  `aa1ab22`.

## License

MIT (see LICENSE).
