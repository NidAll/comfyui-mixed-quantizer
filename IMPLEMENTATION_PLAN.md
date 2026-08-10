# Remove-Experimental Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use executing-plans to implement
> this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Release the converter as stable 1.5.0 (no `-experimental` suffix): P0 +
P1 fixes from PRODUCTION_PLAN.md, independent validation, and stable/experimental
channel separation (w4a8 + int8 stable, mixed behind `--experimental`).

**Architecture:** All changes land in the single file `comfyui_wxa8_quantizer.py`
with exact-string edits. Every fix is TDD: embedded self-test first (watched
fail), then minimal fix, then full self-test + fixture regression. Algorithm
revisions are introduced as constants (Phase 0) so each fix bumps its revision
and metadata records it.

**Tech Stack:** Python 3.11, torch 2.13.0+cu130, safetensors 0.8.0,
comfy-kitchen 0.2.28 (tests only), RTX 3050 (CUDA).

## Global Constraints

- Single file is the source of truth: `comfyui_wxa8_quantizer.py`.
- `--format w4a8` outputs must stay byte-identical to current main (golden +
  deterministic self-tests guard this). Baseline: boogu_real fixture payload
  `tensor_data_sha256` is frozen (see testdata/reports/BASELINE.txt).
- Self-test count is documented (40/40): every new case updates the count in
  README.md and AGENTS.md in the SAME commit.
- w4a8-only mode unaffected by mixed-mode fixes (verify with fixture
  conversions: w4a8 + mixed + `--validation-only`).
- No CI: run `--self-test`, fixture conversions, `cuda_smoke.py` locally.
- README.md sync is mandatory per commit that changes user-visible behavior.
- AGENTS.md is gitignored locally but must stay in sync (it is the local
  runbook; not committed to the repo).
- Fixtures are gitignored (`testdata/*.safetensors`); commit reports under
  `testdata/reports/` (text, not .json, which is gitignored).

---

### Task 1: Baseline freeze manifest

**Files:**
- Create: `testdata/reports/BASELINE.txt` (committed)
- Test: none (recording task)

**Interfaces:**
- Produces: `testdata/reports/BASELINE.txt` — machine-readable text block:
  converter version, self-test count, algorithm revisions, fixture input
  sha256, per-mode output sha256 + `tensor_data_sha256` (payload), validation
  counts.

- [ ] **Step 1:** Verify mixed baseline finished: `tail /tmp/baseline_mixed.log`
  (expect 328/328 checks, output ~346 MiB, sha256 recorded).
- [ ] **Step 2:** Extract `tensor_data_sha256` from both baseline outputs and
  compare with the pre-existing artifacts (payload identity).
- [ ] **Step 3:** Write `testdata/reports/BASELINE.txt` with the frozen values.
- [ ] **Step 4:** Commit: `chore: freeze 1.4.0-experimental behavior baseline`.

---

### Task 2: Algorithm revision constants

**Files:**
- Modify: `comfyui_wxa8_quantizer.py` near line 92 (`CONVERTER_VERSION`)
- Test: extend self-test `_test_metadata` (or new `algorithm-identity` case)

**Interfaces:**
- Produces: module constants `QUANT_ALGORITHM_REVISION`,
  `W4A8_FORMAT_REVISION`, `MIXED_PLANNER_REVISION`, `VALIDATION_REVISION`,
  `CALIBRATION_REVISION` (strings, e.g. `"lloydmax-codebook-r2"`), plus
  `get_algorithm_identity()` returning a dict; embedded in extension metadata
  under `algorithm_identity` and asserted by the self-test.

- [ ] **Step 1:** Write failing self-test `algorithm-identity`:
  `get_algorithm_identity()` returns non-empty revisions and
  `build_extension_metadata` embeds them.
- [ ] **Step 2:** Run `--self-test`, confirm the new case fails.
- [ ] **Step 3:** Add the five constants + `get_algorithm_identity()`; write
  revisions into extension metadata (both v1 and v2 paths).
- [ ] **Step 4:** Run `--self-test`, confirm N/41 passes.
- [ ] **Step 5:** Commit: `feat: algorithm revision identity (Phase 0)`.

---

### Task 3: F1 - INT8 chunked validation fix

**Files:**
- Modify: `comfyui_wxa8_quantizer.py:8366` (chunked validator cleanup)
- Test: new self-test `chunked-validation-matrix`

**Interfaces:**
- Consumes: existing `_quant_work_bytes`, `--max-memory`, `--validation-only`.
- Produces: validator behavior identical across chunk sizes.

- [ ] **Step 1:** Write failing self-test `chunked-validation-matrix`: build a
  mini checkpoint with one INT8, one W4A4, one W4A8 layer (mixed convert with
  `--max-memory` 1 MiB to force the bounded path), then run
  `--validation-only` at 1 MiB and at default memory; assert both succeed and
  report identical metrics. On current code the INT8 layer raises
  `NameError: rt`.
- [ ] **Step 2:** Run `--self-test`, confirm the INT8 leg fails with NameError.
- [ ] **Step 3:** Fix line 8366: scope cleanup per format; drop shared `del`:

```python
                            else:
                                scale_slice = st.get_slice(d.layer + ".weight_scale")
                                for r0 in range(0, n, chunk_rows):
                                    r1 = min(n, r0 + chunk_rows)
                                    packed = packed_slice[r0:r1]
                                    scale = scale_slice[r0:r1]
                                    dq = dequantize_weight_by_format(
                                        {"": packed, "_scale": scale},
                                        d.format, d.group_size,
                                        d.convrot_groupsize, torch.float32)
                                    acc.update(original_view[r0:r1], dq, None)
                                    if d.format == FORMAT_W4A4:
                                        rt = unpack_int4_signed(packed)
                                        repacked = (
                                            (rt[:, 0::2] & 0xF)
                                            | ((rt[:, 1::2] & 0xF) << 4)
                                        ).to(torch.int8)
                                        pack_ok = pack_ok and bool(torch.equal(repacked, packed))
                                        del rt, repacked
                                    del packed, scale, dq
```

- [ ] **Step 4:** Run `--self-test` (N/42), then the w4a8 boogu fixture
  conversion regression (payload hash must match BASELINE.txt).
- [ ] **Step 5:** Commit: `fix: INT8 chunked validation NameError; add chunked validation matrix`.

---

### Task 4: F2 - certificate sequencing + per-format runtime_certified

**Files:**
- Modify: `comfyui_wxa8_quantizer.py` lines 11460-11496 (main), line 6195
  (candidate construction), `RuntimeCapabilities` (~4775)
- Test: extend self-test `mixed-runtime-caps` (or new `mixed-cert-seq` case)

**Interfaces:**
- Consumes: existing `load_runtime_certificate`, `_check_runtime_certificate`.
- Produces: exactly one authoritative `RuntimeCapabilities` object; candidates
  carry per-format `runtime_certified`.

- [ ] **Step 1:** Write failing self-test `mixed-cert-seq`: craft a v1
  certificate covering only W4A8 (schema `comfy-wxa8-runtime-cert/v1`),
  run mixed planning with `--runtime-certificate`; assert W4A8 candidate
  `runtime_certified is True`, W4A4/INT8 candidates `False`, and that the
  planner summary equals the metadata runtime block (no post-plan mutation).
  Current code fails: all candidates report `True` (line 6195) and the
  summary was frozen before certification.
- [ ] **Step 2:** Confirm the new case fails.
- [ ] **Step 3:** Reorder main(): certificate load -> `_check_runtime_certificate`
  -> build final caps (`dataclasses.replace(runtime_caps, runtime_certified=True,
  **certed)`) -> `MixedPlanner(runtime=final_caps)` -> `plan()`; delete the
  `mixed_planner.runtime = runtime_caps` mutation. Add
  `RuntimeCapabilities.capability(fmt)` lookup; at 6195 set
  `runtime_certified=(self.runtime.capability(fmt).certified
  if fmt in MIXED_FORMATS else False)`.
- [ ] **Step 4:** Run `--self-test`, then boogu mixed fixture conversion
  (payload hash must match BASELINE.txt; metadata runtime block consistent).
- [ ] **Step 5:** Commit: `fix: apply runtime certificate before planning; per-format certified flags`.

---

### Task 5: F3 - content-addressed calibration cache

**Files:**
- Modify: `comfyui_wxa8_quantizer.py` `load_calibration` (5201-5270) and the
  cache writer (~5398)
- Test: extend self-test `real-activation-calibration`

**Interfaces:**
- Consumes: `hash_checkpoint_files(info)` for the checkpoint fingerprint.
- Produces: provenance schema `comfy-wxa8-calibration/v2` with
  `checkpoint_fingerprint`, `calibration_files`, `preprocessing_revision`,
  `max_samples`, `compute_precision`.

- [ ] **Step 1:** Write failing self-test: write a v2 cache for checkpoint A;
  load it for checkpoint B with identical layer names/widths; assert the cache
  is rejected (stats empty, fresh load from source). Current code accepts B's
  cache (name+width match only).
- [ ] **Step 2:** Confirm failure.
- [ ] **Step 3:** On write: embed fingerprint = `hash_checkpoint_files(info)`
  mapping + calibration file sha256s + revision + max_samples + precision.
  On load: if v2 fields present and mismatch -> log warning, reject cache,
  fall through to fresh load; v1 caches (no fingerprint) load with a warning.
- [ ] **Step 4:** Run `--self-test`, then a mixed conversion with
  `--calibration-source` + `--calibration-cache` twice (second run must load
  the cache; a changed checkpoint must not).
- [ ] **Step 5:** Commit: `fix: content-addressed calibration cache (v2 fingerprint)`.

---

### Task 6: F4 - source-hash mapping equality

**Files:**
- Modify: `comfyui_wxa8_quantizer.py:8083-8084` (validator source-hash check)
- Test: extend self-test `malformed-checkpoints` or new `shard-hash-identity`

**Interfaces:**
- Consumes: `recorded_source_hashes` and `input_hashes` (both dict
  path->sha256, canonical resolved paths).
- Produces: strict mapping equality.

- [ ] **Step 1:** Write failing self-test: build a 2-shard fixture, convert,
  then re-run validation with the shard paths swapped in the recorded hashes
  (same values, different keys); assert validation FAILS on the hash check.
  Current code passes (sorted values compare equal).
- [ ] **Step 2:** Confirm failure.
- [ ] **Step 3:** Replace the comparison with dict equality; normalize key
  forms once at record time (`str(Path(p).resolve())`).
- [ ] **Step 4:** Run `--self-test` + sharded fixture conversion regression.
- [ ] **Step 5:** Commit: `fix: source-hash identity is path-mapping equality`.

---

### Task 7: F5 - authoritative shard index

**Files:**
- Modify: `comfyui_wxa8_quantizer.py:2621-2625` (extra-tensor append),
  `build_arg_parser` (~11063)
- Test: extend self-test `checkpoint-input-variants`

**Interfaces:**
- Consumes: `shard_index["weight_map"]`.
- Produces: CLI `--allow-extra-shard-tensors`; InputError on extras by default.

- [ ] **Step 1:** Write failing self-test: fixture whose shard contains a
  tensor absent from `weight_map`; assert `InputError` without the flag and
  success + warning with it. Current code silently appends.
- [ ] **Step 2:** Confirm failure.
- [ ] **Step 3:** Default: collect extras, raise `InputError` listing them;
  with the flag, keep the append path and add a warning; record the flag in
  report warnings.
- [ ] **Step 4:** Run `--self-test` + boogu fixture conversions (single-file
  paths unaffected).
- [ ] **Step 5:** Commit: `fix: shard index is authoritative; --allow-extra-shard-tensors opt-in`.

---

### Task 8: F6 - TOCTOU-safe publication

**Files:**
- Modify: `comfyui_wxa8_quantizer.py` main() publication (~11724) and
  destination capture near output-path checks (~11375)
- Test: new self-test `publication-race`

**Interfaces:**
- Consumes: existing `--overwrite`, staged/validation path lifecycle.
- Produces: no-clobber publish without `--overwrite`; identity re-check with
  `--overwrite`.

- [ ] **Step 1:** Write failing self-test: stage a conversion (via the
  internal engine on a small fixture), create the destination file after
  staging, then publish; assert publication REFUSES (no clobber) and the
  staged file remains. Current code: `os.replace` silently overwrites.
- [ ] **Step 2:** Confirm failure.
- [ ] **Step 3:** Without `--overwrite`: `os.link(validation_path, out_path)`
  (same-dir => same fs; FileExistsError on race), then unlink validation.
  With `--overwrite`: capture `os.lstat(out_path)` (dev, ino, not symlink)
  before conversion; before replace, re-lstat and verify identity; on
  mismatch raise `destination changed concurrently; refusing overwrite`.
- [ ] **Step 4:** Run `--self-test`, `--overwrite` boogu w4a8 regression.
- [ ] **Step 5:** Commit: `fix: race-free publication (no-clobber default, overwrite identity check)`.

---

### Task 9: F7 - canonical chunk-invariant codebook sampling

**Files:**
- Modify: `comfyui_wxa8_quantizer.py` `_codebook_sample_size` (5608),
  `_fit_codebook` (3086), `_gather_codebook_samples` (5640)
- Test: new self-test `chunk-invariant-payload`

**Interfaces:**
- Consumes: `--max-memory`, `--seed`.
- Produces: identical packed payload for any `--max-memory`; deterministic
  per-tensor sampler seeded from `checkpoint_fingerprint + tensor_name +
  QUANT_ALGORITHM_REVISION + seed`.

- [ ] **Step 1:** Write failing self-test: convert the same w4a8 fixture at
  `--max-memory 8M`, `64M`, `8G`; assert identical output payload sha256.
  (With a >300000-element tensor so budgeted sampling differs.) Current code:
  8M budget samples ~65536 vs 300000 -> different codebooks -> different
  payloads.
- [ ] **Step 2:** Confirm failure.
- [ ] **Step 3:** Remove `_codebook_sample_size`; fixed
  `CODEBOOK_SAMPLE_COUNT = 300000`; sampler = stateless generator over
  `(checkpoint fingerprint, tensor name, revision, seed)`; both full and
  chunked paths use the same index set.
- [ ] **Step 4:** Run `--self-test` + w4a8 boogu payload regression (must
  equal BASELINE.txt since default path is unchanged for tensors under the
  sample cap).
- [ ] **Step 5:** Commit: `feat: chunk-invariant canonical codebook sampling`.

---

### Task 10: F8 - --seed controls sampling

**Files:**
- Modify: `comfyui_wxa8_quantizer.py` sampler plumbing (Task 9 output),
  `build_arg_parser` (11063)
- Test: extend `chunk-invariant-payload` with a `--seed` leg

- [ ] **Step 1:** Extend the Task 9 self-test: same input, `--seed 1` vs
  `--seed 2` produce different payloads (on a tensor above the sample cap);
  same seed reproduces byte-identical payloads. Current code: seed is
  metadata-only (fails).
- [ ] **Step 2:** Confirm failure.
- [ ] **Step 3:** Thread `args.seed` into the sampler seed derivation;
  record the derived seed in metadata (already has `reproducibility.seed`).
- [ ] **Step 4:** Run `--self-test`; commit: `feat: --seed controls codebook sampling`.

---

### Task 11: F9 - single authoritative size estimator

**Files:**
- Modify: `comfyui_wxa8_quantizer.py` `quantized_format_bytes` (3316),
  `build_output_entries` (5080), planner candidate bytes (6192), report
- Test: extend self-test `compression-stats`

**Interfaces:**
- Produces: `estimated_output_bytes(decision, out_dtype)` used by planner,
  entries builder and report; post-serialization delta check
  (est. vs actual payload bytes) as validator info check.

- [ ] **Step 1:** Failing test: for a mixed plan, assert planner
  `estimated_bytes` equals the sum of `build_output_entries` nbytes for the
  same decisions (current code has two estimators; any mismatch fails).
- [ ] **Step 2:** Confirm failure.
- [ ] **Step 3:** One function; planner + entries + report call it; add
  est-vs-actual payload delta to the validator (tolerance: metadata
  overhead excluded, compare tensor payload bytes only).
- [ ] **Step 4:** Run `--self-test` + boogu mixed fixture (report numbers
  must match BASELINE.txt).
- [ ] **Step 5:** Commit: `feat: single output-size estimator with serialization delta check`.

---

### Task 12: F10 - explicit non-finite policy

**Files:**
- Modify: `comfyui_wxa8_quantizer.py` quantization dispatch (3395) and CLI
- Test: new self-test `nonfinite-policy`

**Interfaces:**
- Produces: `--nonfinite-policy error|keep` (default `error`); report records
  detections; calibration activation finiteness check.

- [ ] **Step 1:** Failing test: checkpoint with a NaN weight in a quantizable
  layer -> default conversion fails with a clear error naming the layer;
  with `keep` the layer passes through at original precision; calibration
  stats with NaN rows rejected.
- [ ] **Step 2:** Confirm failure.
- [ ] **Step 3:** Before quantizing each candidate: `torch.isfinite` check;
  `error` -> `InputError`/`QualityGateError`-style abort; `keep` -> decision
  to passthrough with warning; report section; calibration load validation.
- [ ] **Step 4:** Run `--self-test`; commit: `feat: non-finite weight policy (error|keep)`.

---

### Task 13: F11 - three-state runtime capabilities

**Files:**
- Modify: `comfyui_wxa8_quantizer.py` `FormatRuntimeCapability` (4740),
  `runtime_capabilities_for` (4876)
- Test: extend `mixed-runtime-caps`

**Interfaces:**
- Produces: per-format states `supported/unsupported/unknown` and
  `accelerated/fallback/unknown` (no optimistic True when information is
  missing); `describe()` distinguishes unknown.

- [ ] **Step 1:** Failing test: force an unknown compute capability (env stub)
  -> capabilities report `unknown` for loadability/execution/acceleration,
  and mixed planning refuses to treat them as accelerated.
- [ ] **Step 2:** Confirm failure.
- [ ] **Step 3:** Three-state model; unknown capability -> candidate
  `eligible` only via eager/fallback semantics with `runtime_certain=False`;
  metadata reflects states.
- [ ] **Step 4:** Run `--self-test` + CUDA smoke (`testdata/cuda_smoke.py`)
  to prove the RTX 3050 path still resolves supported/accelerated.
- [ ] **Step 5:** Commit: `feat: three-state runtime capability model`.

---

### Task 14: F-ref - independent reference decoder

**Files:**
- Modify: `comfyui_wxa8_quantizer.py` add `reference_decode_w4a8`,
  `reference_decode_w4a4`, `reference_decode_int8`, `reference_unpack_*`;
  Validator uses them for reconstruction checks (Task 3 chunked path too)
- Test: new self-test `reference-decoder-consensus`

**Interfaces:**
- Produces: slow, simple, independent decoders (no Hadamard fast paths, no
  shared helpers with production); validator error checks computed with them.

- [ ] **Step 1:** Failing test: `reference_decode_*` implemented from the
  format spec (nibble unpacking, scale multiply, ConvRot inverse via
  H.T@H@x identity, rowwise int8) agree with production dequant within 1e-4
  on golden vectors, and validator uses them (validator output changes).
- [ ] **Step 2:** Confirm failure.
- [ ] **Step 3:** Implement reference decoders from the README spec text
  only; switch Validator reconstruction/roundtrip checks to them.
- [ ] **Step 4:** Run `--self-test` + runtime_equivalence.py (eager kernels
  still agree to 1e-4) + w4a8/mixed fixture validations.
- [ ] **Step 5:** Commit: `feat: independent reference decoders in validation`.

---

### Task 15: Stable/experimental channels + 1.5.0

**Files:**
- Modify: `comfyui_wxa8_quantizer.py` `CONVERTER_VERSION` (92), CLI, main()
- Test: update `--version` expectations; new self-test `channel-gating`

**Interfaces:**
- Produces: `--format mixed` requires `--experimental` (without it:
  `UsageError` pointing at the flag); default channel `w4a8` (+ int8 via
  mixed disabled? no - int8 stays inside mixed; stable w4a8 unchanged);
  version `1.5.0`.

- [ ] **Step 1:** Failing test: `--format mixed` without `--experimental`
  raises UsageError; with the flag it plans; `--version` prints `1.5.0`.
- [ ] **Step 2:** Confirm failure.
- [ ] **Step 3:** Set `CONVERTER_VERSION = "1.5.0"` (drop `-experimental`);
  add `--experimental`; gate mixed format on it; README rewrite of the
  channel story; AGENTS.md sync; self-test count final.
- [ ] **Step 4:** Full verification battery: `--self-test`, boogu w4a8 +
  mixed fixture conversions + `--validation-only`, `cuda_smoke.py`,
  `runtime_equivalence.py --seeds 3`, payload hashes vs BASELINE.txt.
- [ ] **Step 5:** Commit: `release: 1.5.0 stable; mixed behind --experimental`.

---

## Self-review

- Spec coverage: P0 F1-F6 -> Tasks 3-8; P1 F7-F11 -> Tasks 9-13; independent
  validation (plan Part C/P1-5) -> Task 14; Phase 0 identity -> Tasks 1-2;
  Phase 22 channels + version -> Task 15. Release gates from Phase 34 that are
  locally runnable are in Task 15 Step 4. Phases 2-5, 9-12, 16-33 (descriptor
  IO, pickle redesign, probe v3, holdout, planner r2, fuzzing, pytest
  migration, exit codes, modular source) are explicitly out of scope for the
  1.5.0 cut and remain documented in PRODUCTION_PLAN.md as the mixed-stable
  qualification cycle (1.6.0).
- Placeholder scan: all steps carry concrete anchors or code.
- Type consistency: `get_algorithm_identity()` (Task 2) is consumed by the
  metadata builder and the Task 2 test; `runtime.capability(fmt)` (Task 4) is
  used by Task 4's candidate code; `estimated_output_bytes` (Task 11) is used
  by planner, entries and report; reference decoders (Task 14) replace
  dequant calls inside Validator only.

## Execution handoff

Plan saved as `IMPLEMENTATION_PLAN.md`. Execution options:
1. **Inline execution** (this session): start at Task 1, batch with
   checkpoints after each task (self-test + fixture regression).
2. **Subagent-driven**: fresh agent per task with review gates.

Which approach?
