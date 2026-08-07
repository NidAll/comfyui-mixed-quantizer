# Metadata specification: comfyui_wxa8_quantizer outputs

## Official (verified) metadata

### `__metadata__["_quantization_metadata"]`: verified ComfyUI key

JSON string, shape:

```json
{
  "layers": {
    "model.diffusion_model.blocks.0.attn.qkv_proj": {
      "format": "asym_w4a8_int8",
      "group_size": 16,
      "convrot": true,
      "convrot_groupsize": 256
    }
  }
}
```

* Layer names are the **full state-dict keys** (including any `model.diffusion_model.`
  prefix; prefix-less models such as MiniMax H3 use plain `blocks.0.attn.qkv_proj`).
* This is the exact shape produced by the reference tooling and consumed by ComfyUI's
  `comfy/utils.py::convert_old_quants`, which converts it into per-layer
  `{layer}.comfy_quant` JSON blobs that `comfy/ops.py::_load_quantized_module`
  (PR #15308) reads. For W3A8 the same structure is used with
  `"format": "asym_w3a8_int8"` plus `"codebook_size": 8` and `"packing": "3bit-lsb"`.

### Per-layer tensors (verified names)

| key | dtype | shape | meaning |
|---|---|---|---|
| `{layer}.weight` | int8 | `[N, K/2]` (w4) / `[N, K*3//8]` (w3) | packed codes |
| `{layer}.weight_s_rel` | fp8_e4m3fn | `[N, K/group_size]` | per-group relative scale |
| `{layer}.weight_s_channel` | fp32 | `[N]` | per-output-channel scale |
| `{layer}.weight_codebook` | fp32 | `[16]` (w4) / `[8]` (w3) | Lloyd-Max codebook |

All other tensors (biases, norms, embeddings, convs, positionals, heads, buffers)
pass through under their original names and dtypes.

## Namespaced extension metadata (never official)

### `__metadata__["comfy_wxa8"]`: extension schema

```json
{
  "converter": "comfyui_wxa8_quantizer",
  "converter_version": "1.0.0",
  "format": "asym_w4a8_int8",
  "format_revision": "asym-w4a8-int8-r1",
  "architecture": "wan",
  "detection_confidence": "medium",
  "unet_prefix": "model.diffusion_model.",
  "source": {
    "kind": "safetensors",
    "files": ["..."],
    "total_bytes": 0,
    "sha256": {"file": "hex"}
  },
  "quantization": {
    "weight_bits": 4,
    "activation_bits": 8,
    "weight_quantization": "per-group asymmetric codebook (Lloyd-Max, symmetric)",
    "group_size": 16,
    "convrot": true,
    "convrot_groupsize": 256,
    "scale_dtype": "fp8_e4m3fn",
    "packing": "int4-nibble-lsb",
    "symmetric": true,
    "n_quantized_layers": 300,
    "n_kept_tensors": 525
  },
  "calibration": {
    "source": null,
    "method": "calibration-free (reference format)",
    "synthetic": false
  },
  "sensitivity": {
    "enabled": false,
    "threshold": null,
    "error_threshold": 0.35,
    "layers_kept": []
  },
  "reproducibility": {
    "seed": 0,
    "device": "cpu",
    "torch_version": "2.13.0",
    "deterministic": true,
    "codebook_subsample_seed": 0
  },
  "compatibility": {
    "comfy_kitchen": {
      "required_revision": "aa1ab2263dc06225d9de6702dfc087313d4bc971",
      "pr": 90,
      "merged": true,
      "layout": "AsymW4A8Int8Layout"
    },
    "comfyui": {
      "required_pr": 15308,
      "required_head": "b6578f2ae11ab3dea3156ed68d8724476cda1232",
      "merged": false,
      "note": "ComfyUI PR #15308 is NOT merged into master as of bdcb886a..."
    },
    "w3a8_runtime": "requires the revision-aware runtime patch emitted by --emit-patch...",
    "cuda_backend": {"requires": "PyTorch cu130+, SM >= 8.0", "min_sm": [8, 0]},
    "triton_backend": {"requires": "triton >= 3.7 (ROCm)"}
  },
  "output": {"sha256": "hex", "bytes": 0, "entries": 0},
  "validation": {"checks": [...], "n_passed": 0, "n_failed": 0, "output_sha256": "hex"},
  "elapsed_seconds": 0.0,
  "peak_rss_bytes": 0,
  "warnings": []
}
```

* All `comfy_wxa8` fields are extension data. They are never presented as official
  ComfyUI metadata.
* Large per-tensor information (per-layer metrics, exclusion reasons) is not stored
  in safetensors metadata; it lives in the JSON report (`--report PATH.json`) to
  keep headers small.
* `output.sha256` is filled by the full validation pass (`--validate`); without it,
  the report carries the hash instead.
