# Production-readiness fix plan (prepared)

Grounds the 34-phase enhancement plan in the actual `1.4.0-experimental`
codebase. Every claim below was verified against the current source with
line anchors. Status legend: CONFIRMED (bug exists) / PARTIAL (partly
handled) / ALREADY-OK (plan claim already satisfied) / N-A (not applicable).

## Part A - Audit: plan claims vs code reality

| # | Plan claim | Verdict | Evidence |
| - | ---------- | ------- | -------- |
| P0-1 | Chunked INT8 validation crashes: `del packed, scale, dq, rt, repacked` references W4A4-only vars | **CONFIRMED** | line 8366 in `Validator.run` chunked branch. `rt`/`repacked` defined only inside `if d.format == FORMAT_W4A4:` (8368-8375). INT8 chunked validation raises `NameError`. Latent: requires layer work bytes > `--max-memory` (default 2G). W4A8 branch (8347) is correct. |
| P0-2 | Certificate applied after planning; `runtime_certified` is not per-format | **CONFIRMED** | main() lines 11460-11496: `MixedPlanner(...)` + `plan()` run with pre-cert caps; only then `dataclasses.replace(runtime_caps, runtime_certified=True, **certed)` + `mixed_planner.runtime = runtime_caps` (11495). Line 6195: `runtime_certified=(self.runtime_certificate is not None)` - not per-format. Planner summary is frozen pre-cert while metadata is built post-cert. |
| P0-3 | Calibration cache accepted on name+width match only | **CONFIRMED** | `load_calibration` (5201-5270): acceptance = provenance is dict + `key in layer_keys` + `tensor.shape[1] == meta.shape[1]`. No checkpoint/calibration-file/revision fingerprint in `__provenance__`. |
| P0-4 | Source-hash compared as sorted values, not mapping | **CONFIRMED** | lines 8083-8084: `sorted(recorded_source_hashes.values()) == sorted(input_hashes.values())`. Shard identity not part of provenance. |
| P0-5 | Shard index not authoritative: unexpected tensors silently appended | **CONFIRMED** | lines 2621-2625: tensors in shards not in `weight_map` are appended without error. Missing indexed shard and duplicate tensor already error (2564, 2571); indexed-tensor-missing already errors via `RawSafetensorsFile.get` (2350-2353). Need `--allow-extra-shard-tensors`. |
| P0-6 | Publication `os.replace` unconditional; no TOCTOU protection | **CONFIRMED** | line 11724 `os.replace(validation_path, out_path)`; no destination identity captured at start, no identity re-check before replace, no no-clobber primitive. |
| P1-1 | Codebook sampling size depends on `--max-memory` | **CONFIRMED** | `_codebook_sample_size(max_mem, total)` (5608-5611) = `min(300000, total, max(4096, max_mem//128))`; used at 5736, 5868, 6113, 7051. Chunked gather (5640+) draws seed-0 indices; differs from full path when budget < ~38MB. |
| P1-2 | `--seed` exists but does not control sampling | **CONFIRMED** | CLI at 11063 (`--seed`, default 0); `_fit_codebook` (3086-3097) and `_gather_codebook_samples` (5640+) hard-code `manual_seed(0)`. |
| P1-3 | Output-size accounting | PARTIAL | `build_output_entries` uses `tensor_nbytes(dt, meta.shape)` with the actual output dtype for passthrough (5125-5126); planner uses `quantized_format_bytes(n,k,fmt)` (3316, used at 6192). The two estimators are separate code paths; consistency check (Phase 13's "one authoritative cost function") not implemented. |
| P1-4 | Non-finite handling | PARTIAL | `_flag_quantized_input`/readers validate structure; no explicit `torch.isfinite` policy before quantization (quantile/codebook on NaN). |
| P1-5 | Independent reference decoder in validation | **CONFIRMED** | Validator calls the same `dequantize_w4a8_weight` / `dequantize_weight_by_format` used in production (8347-8367). Shared-code agreement risk is real. |
| P2+ | `safetensors.torch._TYPES` private API | **CONFIRMED** | line 2083 `getattr(safetensors.torch, "_TYPES", {})`. |
| P2+ | Full/chunk determinism invariant test | N-A (absent) | No test quantizes the same tensor at multiple chunk sizes and compares hashes. |
| P2+ | Fuzz/adversarial safetensors fixtures | N-A (absent) | `_test_malformed` covers truncation/overlap/basic cases only. |
| P2+ | Exit-code contract | PARTIAL | `main` returns 0/1/2; no stable per-failure codes. |
| P2+ | Resume binds algorithm revisions | PARTIAL | State file binds temp identity + input hashes; no algorithm-revision or CLI-config hash binding. |

## Part B - P0 fix package (execution order, TDD: regression test first)

### F1. INT8 chunked validation (P0-1) - anchor 8366
- Move W4A4-only cleanup inside the W4A4 conditional; drop manual `del` of
  shared names (scope is already function-local).
- Regression tests (self-test, e.g. `chunked-validation-matrix`): convert
  INT8 / W4A4 / W4A8 fixtures, then `--validation-only` at `--max-memory`
  1 MiB (forces bounded path) and full memory; assert identical
  pass/fail and identical metrics. Must fail on current code with
  `NameError: rt` for INT8.
- Note: self-test count changes 40 -> 41; README + AGENTS.md must be
  updated in the same commit (docs convention).

### F2. Certificate sequencing (P0-2) - anchors 11460-11496, 6195
- Reorder in `main()`: load certificate -> `_check_runtime_certificate`
  -> build FINAL `RuntimeCapabilities` (certified flags per format) ->
  construct `MixedPlanner` with final caps -> `plan()` -> metadata.
- Delete the post-plan mutation (`mixed_planner.runtime = runtime_caps`).
- In `MixedPlanner` candidate construction (6195): set
  `runtime_certified = runtime.capability(fmt).certified` (needs a
  capability lookup helper on `RuntimeCapabilities`), fall back to
  `self.runtime_certificate is not None` only where no capability exists.
- Regression test: with a v1/v2 certificate covering only W4A8, plan a
  mixed conversion; assert W4A8 candidate `runtime_certified=True`,
  W4A4/INT8 `False`; assert planner summary and extension metadata agree
  (no post-plan runtime mutation visible in summary).

### F3. Content-addressed calibration cache (P0-3) - anchors 5201-5270, writer ~5398
- New provenance schema `comfy-wxa8-calibration/v2` with:
  `checkpoint_fingerprint` (per-input-file sha256 mapping),
  `calibration_files` hashes, `preprocessing_revision`,
  `max_samples`, `compute_precision`, `architecture_fingerprint`.
- On load: if fingerprint fields present and != expected -> reject cache
  (log + fall through to fresh load; hard error only under
  `--require-calibration-cache-fingerprint` in production profile).
- Back-compat: v1 caches without fingerprints load with a warning
  (they stay acceptable until the profile requires v2).
- Regression test: write cache from model A, load with model B of same
  layer names/widths -> must not be accepted (stats empty, fresh load
  happens); cache round-trips for model A.

### F4. Source-hash mapping comparison (P0-4) - anchor 8083-8084
- Compare `recorded_source_hashes == input_hashes` (dict equality,
  keys = canonical resolved paths). If key forms differ between record
  and runtime, normalize once at record time.
- Regression test: sharded fixture, swap two shard paths while keeping
  byte-identical content -> mapping comparison must fail
  (sorted-values comparison passes today).

### F5. Authoritative shard index (P0-5) - anchor 2621-2625
- Default: extra tensor not in `weight_map` -> `InputError` listing the
  tensor and its shard (fail before any conversion work).
- Add `--allow-extra-shard-tensors` to restore append behavior, recorded
  in extension metadata + report warnings.
- Regression test: shard containing an extra tensor -> error without
  flag; converted with flag and warning.

### F6. TOCTOU-safe publication (P0-6) - anchor 11724 + main start
- Before conversion, when `--overwrite`: capture destination identity
  via `os.lstat` (dev, ino, and assert not a symlink). Immediately
  before `os.replace`: re-lstat; if missing/changed/symlink -> fail
  `destination changed concurrently; refusing overwrite`.
- Without `--overwrite`: no-clobber publish via `os.link(validation_path,
  out_path)` + unlink (same-dir files => same filesystem; raises
  FileExistsError on race). Fall back to error if `os.link` unsupported.
- Regression test: after staging completes, create the destination path,
  then attempt publication -> must refuse, staged file retained.

## Part C - P1 package (after P0)

- F7: Canonical codebook sampling: fixed `CODEBOOK_SAMPLE_COUNT` (keep
  300000) plus deterministic per-tensor index generator seeded from
  `checkpoint_fingerprint + tensor_name + quant_algorithm_revision +
  user_seed`; drop `_codebook_sample_size`; chunk-invariance test
  (`--max-memory 8M / 64M / 8G` -> identical payload sha256).
- F8: Make `--seed` control all stochastic sampling (codebook indices,
  calibration split) or remove it; pick the first option.
- F9: Single authoritative output-size estimator used by planner,
  `build_output_entries`, and post-serialization check (est. vs actual
  payload delta -> validator failure above tolerance).
- F10: Explicit non-finite policy: `--nonfinite-policy error|keep`
  (default error) applied per candidate before quantization; report
  detections; calibration activation finiteness check.
- F11: Runtime capability three-state model (supported/unsupported/
  unknown x accelerated/fallback/unknown) - no optimistic True.

## Part D - Sequencing and constraints

1. All fixes land in the single file with exact-string edits; recompile
   after each change (project convention).
2. Self-test count is a documented number (40/40): every new test bumps
   it, and README + AGENTS.md are updated in the same commit.
3. `--format w4a8` outputs must stay byte-identical to current main:
   golden-vector + deterministic-conversion self-tests guard F2/F7.
4. w4a8-only mode unaffected by mixed-mode fixes (F2, F5 only touch
   mixed/shard paths; F5 also touches w4a8 sharded input - verify with
   existing fixture conversions).
5. CI is absent locally; run `--self-test`, fixture conversions
   (w4a8 + mixed + `--validation-only`), `cuda_smoke.py` where relevant.
6. Phases 2+ (descriptor-anchored IO, pickle redesign, runtime probe v3,
   modular source) are separate change sets; P0 + P1 + independent
   validation are the prerequisite for a stable W4A8 release candidate,
   matching the plan's own release sequence.

## Part E - Verification matrix (before claiming any fix)

| Fix | Must pass |
| --- | --------- |
| F1 | chunked INT8 validation succeeds; results identical to full-memory; pre-fix NameError reproduced first (red-green) |
| F2 | mixed plan with partial certificate: per-format runtime_certified correct; summary == metadata |
| F3 | cache from another checkpoint rejected; same-checkpoint cache accepted |
| F4 | shard-swap fixture fails; normal conversions still pass |
| F5 | extra-tensor shard errors; `--allow-extra-shard-tensors` works |
| F6 | concurrent-destination fixture refuses publication; normal overwrite works |
| all | `--self-test` N/N (updated count), w4a8 fixture byte-identical regression, mixed fixture conversion + validation, README/AGENTS sync |
