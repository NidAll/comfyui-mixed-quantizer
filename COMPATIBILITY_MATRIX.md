# Architecture compatibility matrix (executed tests)

All runs below used `comfyui_wxa8_quantizer.py 1.0.0` on torch 2.13.0+cu130 /
safetensors 0.8.0 (CPU quantization; the CUDA path was also exercised). "Executed"
means the conversion finished and the output passed the standalone validation suite
(reopen, inventory, shapes, dtypes, metadata, pack round trips, scale checks,
reconstruction error vs the original weights, deterministic checks, output hash).
relL2 = per-layer weight reconstruction error (max over sampled layers).

## W4A8 (`asym_w4a8_int8`)

| architecture family | ComfyUI classes | test input | result | max relL2 |
|---|---|---|---|---|
| sd15 | SD15, SD15-inpaint, Zero123 | fixture (8 tensors) | pass | 0.0729 |
| sdxl | SDXL, SSD-1B, Vega, KOALA, SDXL-inpaint | fixture (28 tensors) | pass | 0.0730 |
| flux | Flux, FluxInpaint, FluxSchnell, LongCat, Ovis | fixture (23 tensors) | pass | 0.0730 |
| wan | Wan2.1/2.2 (20 classes) | fixture (31 tensors) | pass | 0.0730 |
| wan | Wan2.1 T2V 1.3B (real, 2.64 GiB) | 300 layers | pass (full validation) | 0.0730 |
| minimax_h3 | MiniMaxH3 | fixture (28 tensors, prefix-less) | pass | 0.0728 |
| hydit | HunyuanDiT, HunyuanDiT1 | fixture (12 tensors) | pass | 0.0729 |
| mmdit_sd3 | SD3 (and SD3.5 family) | fixture (15 tensors) | pass | 0.0731 |
| (input form) | sharded directory | hydit fixture split in 2 shards | pass | 0.0729 |
| (input form) | bounded-memory chunked path (2 MiB budget) | minimax_h3 fixture | pass | 0.0855 |

## W3A8 (`asym_w3a8_int8`, extension format)

| architecture family | test input | result | max relL2 |
|---|---|---|---|
| sd15 | fixture | pass | 0.1492 |
| sdxl | fixture | pass | 0.1496 |
| flux | fixture | pass | 0.1492 |
| wan | fixture | pass | 0.1495 |
| minimax_h3 | fixture | pass | 0.1492 |
| hydit | fixture | pass | 0.1493 |
| mmdit_sd3 | fixture | pass | 0.1495 |

## Feature tests (executed)

| feature | result |
|---|---|
| embedded self-tests (`--self-test`) | 15/15 pass |
| golden-vector bit-exactness vs comfy-kitchen reference | pass (2 configs) |
| side-by-side bit-exactness vs reference (9 shape/config combos, quantize + dequantize) | pass |
| CLI kill + `--resume` recovery | pass (deterministic-vs-disk verified) |
| calibration (npz), `--calibration-samples`, cache write/read | pass |
| `--sensitivity-threshold` keep-precision | pass |
| `--device cuda` | pass (deterministic-vs-disk documented skip) |
| `--compute-dtype bf16`, `--output-dtype fp16` | pass |
| `--include/--exclude/--keep-precision` filters | pass |
| `--dry-run`, `--inspect`, `--metadata-only`, `--validation-only` | pass |
| `--emit-patch` (W3A8 runtime patch) | pass; `git apply --check` OK on both reference trees |
| overwrite / self-overwrite / pickle / requantize guards | pass (refused as designed) |
| ComfyUI loader contract (names/dtypes/shapes/metadata) on output | pass |

## Not executed / unsupported

* No end-to-end inference was run inside ComfyUI. The runtime loader requires the
  unmerged ComfyUI PR #15308, and standalone validation does not claim runtime
  compatibility.
* Perception models (RT-DETR_v4, DepthAnything3, SAM3/SAM31) are registered as
  **unsupported** (no ComfyUI quantized-loading path) and refuse conversion unless
  `--architecture` forces them.
* Diffusers-format subfolders (`unet/diffusion_pytorch_model.safetensors`) are
  discovered as files, but their state-dict naming is not converted to ComfyUI
  naming; detection fails safely and requests `--architecture` (documented
  limitation).
* The remaining families (stable_cascade, stable_audio, aura_flow, mochi, ltxv,
  ace_step, cosmos, cosmos_predict2, anima, lumina2, pixeldit, hunyuan3d,
  triposplat, hidream, chroma, seedvr2, omnigen2, ideogram4, krea2, mage_flow,
  qwen_image, joyimage, kandinsky5, cogvideox, ernie_image, sd20, sdxl_refiner,
  svd) have explicit policy profiles and detection signatures, but were not
  individually executed with fixture tensors. They share the same verified
  quantization and serialization machinery.
