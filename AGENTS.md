# AGENTS.md

## Project

`comfyui_wxa8_quantizer.py` is a standalone, single-file converter that turns
generative-model checkpoints into the ComfyUI W4A8 format (`asym_w4a8_int8`):
4-bit weights packed as int8, an fp8 group scale, an fp32 channel scale, and a
16-entry Lloyd-Max codebook. It does not import ComfyUI or comfy-kitchen at
runtime. W3A8 was removed in v1.1.0 and must not be reintroduced; `--format`
accepts only `w4a8` and defaults to it.

- Local path: `/home/nidall/projects/testdeepseek/quantizationscripts_w4a8_w3a8`
- Repo (public): `https://github.com/NidAll/comfyui-w4a8-quantizer`
  (renamed from `comfyui-wxa8-quantizer`; old URL redirects)
- Branch: `main`, SSH remote `git@github.com:NidAll/comfyui-w4a8-quantizer.git`
- Script version: `1.2.1` (`CONVERTER_VERSION` in the script)

## Format facts (verified, do not guess)

The format spec comes from comfy-kitchen PR #90 (merged,
`aa1ab2263dc06225d9de6702dfc087313d4bc971`). The ComfyUI loader is PR #15308
(open, head `8c3a2b27c37bd34e87b58846baf962407c92843c`), shipped as
`patches/comfyui_w4a8_loader.patch` (verified against ComfyUI v0.30.0). Without
that patch ComfyUI fails with `KeyError: 'asym_w4a8_int8'`.

Per quantized layer, the output file has:

- `{layer}.weight` int8 `[N, K/2]`, even column in the low nibble, odd in the high
- `{layer}.weight_s_rel` fp8_e4m3fn `[N, K/group_size]`
- `{layer}.weight_s_channel` fp32 `[N]`
- `{layer}.weight_codebook` fp32 `[16]`
- `__metadata__["_quantization_metadata"]` = `{"layers": {layer: {"format":
  "asym_w4a8_int8", "group_size": 16, "convrot": true,
  "convrot_groupsize": 256}}}` (full state-dict keys, prefix included)
- `__metadata__["comfy_wxa8"]` extension block (never described as official)

Shape rules: 2D only, `K % 16 == 0`, `K % group_size == 0`, group_size >= 4 and
`(16 % group_size == 0 or group_size % 16 == 0)`, `K % convrot_groupsize == 0`.
The convrot group is picked per layer as the largest power of 4 up to 256 that
divides K. The CUDA kernel requires group sizes in {4, 8, 16} or multiples of 16.

## Architecture registry

The script embeds 43 policy families covering all 98 ComfyUI model classes at
the research revision `bdcb886a4705a03cf40f4a7226de9fc7c059fc90`. Detection
signatures mirror ComfyUI's `detect_unet_config`. Each family has its own
quantize / keep / exclude patterns and validation thresholds. Generic fallback
only after explicit `--architecture`.

Z-Image / Lumina2 real naming (used by `sickOllie_zTurbo`): `attention.qkv` and
`attention.out`, `feed_forward.w1/w2/w3`, `context_refiner`, `noise_refiner`,
`adaLN_modulation` (capitalization matters; universal exclude has both cases).
Do not rename these patterns back to `attn.qkv` / `mlp.w1`; that broke real
checkpoints in v1.1.1.

Boogu / OmniGen2 real naming (verified against the published checkpoints,
v1.2.1):
- Boogu-Image-0.1 (Base/Turbo/Edit, Comfy-Org repack, prefix-less):
  `double_stream_layers.N.img_self_attn.to_q`, `.img_instruct_attn.processor.img_to_q`,
  `.img_feed_forward.linear_1/2/3`, `.instruct_feed_forward.linear_1/2/3`,
  `single_stream_layers.N.attn.to_q`, `single_stream_layers.N.feed_forward.linear_1/2/3`,
  plus `context_refiner` / `noise_refiner` / `ref_image_refiner` with
  `attn.to_q` / `feed_forward.linear_1/2/3`. Modulation linears
  (`img_normN.linear`, `instruct_normN.linear`, `single_stream_layers.N.norm1.linear`,
  `noise_refiner.N.norm1.linear`, `ref_image_refiner.N.norm1.linear`) and
  `norm_out.linear_1/2` are kept at original precision.
- OmniGen2 (BAAI/OmniGen2): `layers.N.attn.to_q/to_k/to_v/to_out.0`,
  `layers.N.feed_forward.linear_1/2/3`, same refiners. Its hidden dim is 2520,
  which fails the W4A8 `K % 16 == 0` rule; only `feed_forward.linear_2`
  (K=10240) layers quantize, the rest pass through with a recorded reason.
Boogu is its own family (not an omnigen2 alias); detection is disambiguated by
`img_self_attn` / `img_feed_forward` / `processor` keys before the omnigen2
`layers.0.attn.to_q` check.

## Environment

The working tree is kept clean of large artifacts. No `.venv` and no model
checkpoints are stored locally; both are recreated on demand.

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
# optional, only for ComfyUI loader reproduction (research/ComfyUI):
uv pip install --python .venv/bin/python comfy-kitchen comfy-aimdo pillow \
    tqdm torchaudio opencv-python transformers psutil av einops requests
```

Expected versions: torch 2.13.0+cu130, safetensors 0.8.0, comfy-kitchen 0.2.27,
comfy-aimdo 0.4.13. There is no pip module inside the venv; install with `uv
pip install --python .venv/bin/python PKG`.

- GPU: quantization defaults to CPU
  (`--device auto`), which is deterministic and matches the golden vectors.
  `--device cuda` is faster; the codebook subsample is device-dependent, so
  CUDA output differs from CPU output (both valid).
- `research/comfy-kitchen`: checkout at `aa1ab22` (the PR #90 merge commit).
- `research/ComfyUI`: checkout at `v0.30.0` with the loader patch applied in the
  working tree (do not reset unless you re-apply the patch). Used for loader
  reproduction, not by the converter.

## Common commands

```bash
.venv/bin/python comfyui_wxa8_quantizer.py --self-test          # 13/13 required
.venv/bin/python comfyui_wxa8_quantizer.py --list-architectures
.venv/bin/python comfyui_wxa8_quantizer.py MODEL.safetensors --inspect
.venv/bin/python comfyui_wxa8_quantizer.py MODEL.safetensors \
    --output OUT.safetensors --format w4a8 --dry-run
.venv/bin/python comfyui_wxa8_quantizer.py MODEL.safetensors \
    --output OUT.safetensors --format w4a8 --device cuda --validate \
    --report report.txt
```

Fixture generation and conversion (small, fast, no real models):

```bash
.venv/bin/python testdata/make_fixtures.py testdata/wan_fixture.safetensors
.venv/bin/python comfyui_wxa8_quantizer.py testdata/wan_fixture.safetensors \
    --output testdata/wan_fixture_w4a8.safetensors --format w4a8 --validate
```

Families: `sdxl, sd15, flux, wan, minimax_h3, hydit, mmdit_sd3, zimage, boogu, omnigen2` (boogu and omnigen2 use the real state-dict naming).
`zimage` uses the real Lumina2/Z-Image naming. Reports live in
`testdata/reports/` (path-sanitized).

Reference results from the real checkpoint (not kept on disk): the user's
`sickOllie_zTurbo.safetensors` (11.46 GiB bf16) converts to
`sickOllie_zTurbo_w4a8.safetensors` (3.42 GiB, 170 layers, max relL2 0.0730,
~4 minutes, relL2 validated). Root-level `*.safetensors` is
gitignored; do not commit model files. Use `testdata/make_fixtures.py` for
small regeneration tests instead of real models.

## Verification before claiming success

1. `--self-test` must pass 13/13.
2. Convert affected families from `testdata/make_fixtures.py` with `--validate`;
   max relL2 should be about 0.073 (chunked path about 0.085).
3. For converter changes, the CI matrix runs self-tests and a fixture
   conversion on ubuntu / windows / macos on every push.
4. For loader questions, reproduce with the real ComfyUI path:
   `PYTHONPATH=research/ComfyUI .venv/bin/python` and the flow
   `load_torch_file -> convert_old_quants -> model_config_from_unet ->
   get_model -> load_model_weights`, then check that weights are
   `QuantizedTensor` with `layout=AsymW4A8Int8Layout`.

## Known behavior, do not "fix" it

- The golden-vector test compares packed int8 and fp8 scales byte-exactly but
  fp32 `s_channel` / `codebook` with `rtol=1e-4`: torch reductions differ in the
  last ULPs across platforms (x86 vs ARM, Windows vs Linux). Byte-exact fp32
  comparison fails CI on macOS/Windows.
- `--device cuda` and the chunked path (`--max-memory`) produce different but
  valid codebooks; `deterministic-vs-disk` is skipped for those with a reason.
- ComfyUI may log `unet unexpected: [...comfy_quant...]` on builds with dynamic
  VRAM loading. The markers are consumed during detection; the warning is
  benign and documented in the README. It is a ComfyUI-side artifact, not a
  checkpoint problem.
- Only 2D linear weights are quantized. Convolutions, embeddings, norms, heads,
  and modulations pass through at original precision with the reason recorded.
- Pickle inputs require `--trust-pickle`. Outputs never overwrite inputs.

## Editing and docs conventions

- The converter is one file. There is no `build/` part structure anymore; edit
  `comfyui_wxa8_quantizer.py` directly with exact-string edits and recompile.
- Docs: README.md is the only markdown file (research notes, metadata spec,
  compatibility matrix, limitations, troubleshooting are folded into it).
  Keep it humanizer-clean: no em dashes, no AI vocabulary, no rule-of-three
  padding, plain technical prose. Other markdown files were deleted on purpose.
- Commit messages summarize behavior and evidence. Push with
  `git push origin main`; CI runs automatically.
- The loader patch is generated from ComfyUI PR #15308 head `8c3a2b27`; when the
  PR merges upstream, the README prerequisites and the patch file need updating.

## User context

The user runs ComfyUI on Windows at `C:\Comfyui\ComfyUI` (v0.30.0 plus 23
commits, `g0ab8332bfa`, comfy-kitchen 0.2.27) with the loader patch applied.
The patch must be re-applied after every ComfyUI update until PR #15308 merges.
The user tests with `sickOllie_zTurbo.safetensors` there; expect questions about
load warnings and VRAM behavior.
