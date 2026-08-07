# comfyui_wxa8_quantizer.py

A single-file converter that turns supported generative-model checkpoints into
**W4A8** (`asym_w4a8_int8`) quantized checkpoints for use with compatible ComfyUI /
comfy-kitchen versions.

The script is standalone. It does **not** import, require, or execute any ComfyUI or
comfy-kitchen code at runtime. Every inspection, detection, quantization, packing,
metadata and validation component is reimplemented inside the file from the verified
reference behavior. See [RESEARCH_NOTES.md](RESEARCH_NOTES.md) for the exact source
revisions.

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

Only the original model path and the output path are required (`--format` defaults
to `w4a8`). All other parameters have architecture-specific defaults.

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
| `--calibration-samples N` | cap calibration rows per layer |
| `--calibration-cache PATH` | read/write calibration statistics cache |
| `--seed N` | reproducibility seed (recorded; codebook subsampling is seed-0 like the reference) |
| `--include/--exclude/--keep-precision PATTERN` | regex filters on tensor names |
| `--sensitivity-threshold N` | keep layers at original precision above an error score |
| `--error-threshold N` | per-layer reconstruction relL2 bound (fallback to keep-precision) |
| `--max-memory SIZE` | per-tensor working-memory budget (larger tensors are chunked) |
| `--streaming` | bounded-memory streaming (default for safetensors) |
| `--resume` | resume an interrupted conversion from its state file |
| `--overwrite` | allow replacing an existing output |
| `--dry-run` | detect, plan and report, write nothing |
| `--inspect` | dump input inventory + detection evidence, exit |
| `--list-architectures` | print the embedded architecture registry |
| `--validate` | full standalone validation after conversion (sampled layers, hashes, optional runtime-compat probe) |
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

## Embedded self-tests

```bash
python comfyui_wxa8_quantizer.py --self-test
```

Covers W4 packing round trips, odd dimensions, scale calculations, deterministic
conversion, metadata generation, registry behavior (all 98 ComfyUI model classes
covered by 42 policy families), golden-vector bit-exactness against the reference
implementation, malformed checkpoints, unsupported tensors, resume-state recovery,
atomic output writing, and an end-to-end mini-model conversion. These are
engineering tests, not full-model quality validation.

## Output format

Per quantized layer `{layer}` (the full state-dict key, e.g.
`model.diffusion_model.blocks.0.attn.qkv_proj`):

| tensor | dtype | shape |
|---|---|---|
| `{layer}.weight` | int8 (packed int4, even col = low nibble) | `[N, K/2]` |
| `{layer}.weight_s_rel` | fp8_e4m3fn (per-group relative scale) | `[N, K/group_size]` |
| `{layer}.weight_s_channel` | fp32 (per-output-channel scale) | `[N]` |
| `{layer}.weight_codebook` | fp32 (16-entry Lloyd-Max codebook) | `[16]` |

Plus `__metadata__["_quantization_metadata"]` with per-layer config
(`{"format": "asym_w4a8_int8", "group_size": 16, "convrot": true,
"convrot_groupsize": 256}`) and a namespaced `comfy_wxa8` extension-metadata block.
Non-quantized tensors pass through unchanged (or cast by `--output-dtype`).

See [METADATA_SPEC.md](METADATA_SPEC.md) for the full specification and
[RESEARCH_NOTES.md](RESEARCH_NOTES.md) for the verified reference behavior and the
runtime prerequisites. ComfyUI PR #15308 is **not merged** into ComfyUI master at
the research revision; see the compatibility section there.

## Security

The model, metadata, configuration files, paths and calibration data are treated as
untrusted. Pickle loading is opt-in only, safetensors headers are size- and
offset-validated before any data access, output paths are checked against input
files, outputs are written to a temp file and atomically renamed, and no subprocesses
or network access are used during conversion.
