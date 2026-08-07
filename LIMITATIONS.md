# Known limitations

1. **ComfyUI runtime support is conditional.** W4A8 loading requires comfy-kitchen ≥
   `aa1ab2263dc06225d9de6702dfc087313d4bc971` (merged) AND ComfyUI PR #15308 head
   `b6578f2ae11ab3dea3156ed68d8724476cda1232` (**not merged** into ComfyUI master at
   the research revision `bdcb886a...`). Standalone validation does not prove runtime
   compatibility; use `--validate` for the optional installed-version probe.
2. **W3A8 has no upstream runtime.** It is an independent extension format; loading
   requires the `--emit-patch` runtime patch (revision-aware, verified to apply).
   W3A8 is not a renamed W4A8: different codebook size (8), different packing
   (3-bit, 8 codes / 3 bytes), different bit rate (about 0.375 B/elem vs 0.5 B/elem).
3. **Only 2D linear weights are quantized.** The reference format requires 2D,
   `K % 16 == 0`, plus group/convrot divisibility. Convolutions, embeddings, norms,
   positionals, heads, and modulations pass through at original precision, with the
   reason recorded in the report. Layers whose K is not divisible by 16 cannot be
   quantized in this format at all.
4. **Detection is heuristic.** It mirrors ComfyUI's `detect_unet_config` signatures
   at the research revision, but ambiguous or unknown checkpoints are refused unless
   `--architecture` is supplied. Models from other frameworks (diffusers subfolder
   naming, non-ComfyUI prefixes) are discovered but not renamed; detection fails
   safely.
5. **Determinism is device-scoped.** The codebook subsample uses a fixed-seed
   generator on the quantization device; CPU and CUDA produce different (both valid)
   outputs. `--device auto` = CPU. The chunked bounded-memory path draws its
   subsample with the same seed but over chunk boundaries, so it can differ slightly
   from the in-memory path (validated: relL2 0.0855 vs 0.0728 on the same fixture);
   both are recorded as valid.
6. **Pickle inputs are loaded fully into RAM** (`--trust-pickle` required) and cannot
   be streamed; only convert pickle checkpoints you trust and that fit in memory.
7. **Calibration is optional and used for sensitivity analysis only.** The reference
   format is calibration-free (per-group absmax scales); the converter never claims
   production calibration from synthetic data, and `_quantization_metadata` records
   calibration provenance (or "calibration-free").
8. **Header/metadata size**: per-layer quantization config is stored compactly in
   `_quantization_metadata`; per-tensor metrics and exclusion reasons live in the
   JSON report, not in safetensors metadata.
9. **No network access** during conversion; nothing is downloaded automatically.
10. **Reference drift**: comfy-kitchen/ComfyUI may change the format in the future;
    the converter pins its behavior to the revisions in RESEARCH_NOTES.md and records
    `format_revision` in the output metadata.
