# Distilling an already-short-schedule model — scoping plan

Status: **scoped, not started.**

## The research question (this is the actual contribution, not a footnote)

> Does few-step distillation theory — built and validated on 25-50-step teachers — hold when the
> starting point is already a short, production-optimized schedule? If it breaks down, what needs
> to change?

This came out of a real, converging pattern this session, not a guess:

1. **Three independent inference-time step-skipping techniques** (CFG difference-caching,
   AB-Cache, VDE — see `CLAUDE.md`) all failed to reach a usable quality/speed tradeoff on
   TRELLIS.2's compressed 12-step production schedule, each for a version of the same underlying
   reason: not enough redundancy left across steps to exploit, regardless of the specific
   mathematical technique.
2. **A literature-wide gap on the distillation side, confirmed by reading the actual papers, not
   assumed from abstracts:**

| method | published result | starting schedule | tested on already-short (≤12-step) teacher? |
|---|---|---|---|
| MDT-dist (arXiv:2509.04406) | 9.0x @ 1 step / 6.5x @ 2 steps, original TRELLIS | 25 steps | **No** — no evidence either way |
| SCFM ("Shortcutting...", arXiv:2510.17858) | 32→3-4 steps, Flux/SD3.5 | 32 steps | **No** — paper explicitly states it targets 32→3-4, not further compression of an 8-12 step schedule |
| VDE (caching, not distillation) | 2.0-2.5x, Flux/Qwen/Wan2.1 | 50 steps | **No** — own paper flags <15-step schedules as reduced-gain |
| HiCache / TaylorSeer (caching) | up to 5.5x, FLUX/HunyuanVideo | 25-50 steps | **No** |

Five independent papers, two different research families (training-free caching, distillation),
zero coverage of the regime TRELLIS.2 actually ships in. Not a coincidence worth ignoring.

3. **Real, adjacent theory predicts this should be hard, not just unattempted**: distilling an
   already-compressed model is documented (in LLM-distillation and progressive-distillation
   literature) to suffer **compounding error from covariate shift** (train/inference intermediate
   distributions diverge more when there's less schedule to smooth over) and **entropy
   collapse** (structural decisions get locked in almost immediately, less room left to correct).
   This is mechanistically the same failure mode already measured breaking VDE/AB-Cache/CFG-cache
   on this exact model.

This reframes the project from "port an existing technique to a new model" (incremental) to
"characterize a real, timely, increasingly-relevant regime nobody has tested, using TRELLIS.2 as
the concrete case study" — a materially stronger basis for a CVPR submission, and one where a
negative result (if it turns out this regime really is fundamentally harder) is *itself* a
publishable finding, not a failed side quest.

## Two candidate distillation methods — deliberately testing both, cheapest first

### SCFM (arXiv:2510.17858) — the pilot-first candidate

- **Mechanism** (verified against the actual paper, not the abstract): hybrid loss —
  `ℒ = (1/N)[Σ_{i≤k} (V_θ(x_t,t) − V_θ*(x_t,t))² + Σ_{i>k} (V_θ(x_t,t) − V_θ⁻(x_t,t))²]`, where
  `V_θ*` is the frozen real teacher and `V_θ⁻` is an EMA-updated stopgrad copy of the student
  itself (`θ⁻ = 0.999·θ⁻ + 0.001·θ`), `k/N = 0.4`. Velocity targets are a **weighted interpolation
  across a window of time intervals** (real teacher steps blended together), not a single-point
  finite-difference derivative.
- **Why this might suit a short schedule better than MDT-dist**: MDT-dist's VM loss leans on a
  fine finite-difference approximation (`Δt=1e-2`) of the local derivative — implicitly assuming a
  reasonably fine, locally-smooth time grid. SCFM's window-interpolation target is explicitly built
  to summarize information across *multiple* real steps at once, which may degrade more gracefully
  when there are fewer, more coarsely-and-unevenly-spaced steps to begin with (TRELLIS.2's real
  `rescale_t` warp makes the 12 steps non-uniform in time, closer to SCFM's own multi-step-window
  assumption than to a fine local derivative). This is a hypothesis to test, not a settled fact.
- **Real, verified cost**: 32-step Flux teacher → 3-step student in **under 24 A100 GPU hours**,
  batch size 16, lr 2e-5 (AdamW), ~1000-2000 iterations. An order of magnitude cheaper than
  MDT-dist's implied scale (batch=256, thousands of iterations) — genuinely tractable on a 3-GPU
  DGX pod, where MDT-dist's scale probably isn't without a much longer run.
- **No code released** (checked: project page link in the paper is a placeholder) — this would be
  a from-scratch reimplementation from the paper's equations, not a port of reference code (unlike
  VDE, where the real reference implementation was available and read directly).

### MDT-dist (arXiv:2509.04406) — the fallback / comparison candidate

- Real mechanism and hyperparameters already documented in this repo's `CLAUDE.md`. Kept as a
  second method to compare against if SCFM's pilot result is inconclusive, or if compute allows a
  second, larger run — not because it's expected to be worse, but because having two independently-
  motivated methods land on the same conclusion (or disagree in an explainable way) is stronger
  evidence than one method alone, especially for the "does short-schedule distillation
  fundamentally break down" question.

## What's already real and available in this repo

Checked directly, not assumed:

| have | what |
|---|---|
| ✅ | `train.py` — distributed training entrypoint, checkpoint load/resume |
| ✅ | `trellis2/trainers/flow_matching/` — base flow-matching trainer + CFG mixin (`mixins/classifier_free_guidance.py`) |
| ✅ | `data_toolkit/` — full pipeline for Objaverse-XL / ABO / HSSD / TexVerse |
| ✅ | `configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16.json` — real training config scoped to `shape_slat_flow_model_512` |
| ✅ | `dgx_connect.sh` — access to the user's DGX pod, 3 GPUs |
| ❌ | any distillation-specific loss/trainer code for either method — the actual new work |

## Real, specific risks — why this isn't a drop-in port, for either method

1. **12-step teacher, not 25-32.** The central open question this whole plan exists to answer —
   see above. Both methods' own theoretical guarantees (MDT-dist's Theorem 1 error bound; SCFM's
   implicit smoothness assumption in its window-interpolation target) are untested at this length.
2. **Sparse O-Voxel tokens, not dense.** Both methods were built for dense image/video latents or
   original TRELLIS's structure. TRELLIS.2's `shape_slat_flow_model_512` operates on
   `SparseTensor` — variable token count per sample, no fixed spatial grid. Any finite-difference,
   re-noising, or EMA-averaging step needs to be well-defined per-token (or per-sample aggregated)
   — a real adaptation for either method, not a shape reinterpretation.
3. **Real training data is required for both.** Not training-free. Need actual 3D asset data run
   through the real `data_toolkit/` encode pipeline to produce real SLat training targets — full
   pipeline validation is real work before any distillation-specific code runs, regardless of
   which method is implemented first.
4. **Compute scale, still real even for the cheaper method.** SCFM's 24 A100-hours is for Flux on
   presumably full A100s; this project's DGX pod specifics (GPU model, how much of the 3 GPUs are
   actually available given it's shared) need to be checked against that before assuming parity.
   Scoped as a **pilot** either way, not a claim of matching either paper's full result.

## Phased plan

### Phase 0 — data pipeline validation (no distillation code yet)

**Status: complete, including at the real production resolution (512).** Metadata, download,
mesh-dump, PBR-dump, O-Voxel-conversion, and latent-encoding stages all verified working on a
real 196-object ObjaverseXL(sketchfab) pilot slice at `/tmp/pilot_test`, first at a fast
resolution-64 pilot pass and then re-run in full at resolution 512 (the actual resolution
`slat_flow_img2shape_dit_1_3B_512` expects) to close the scope gap flagged after the first pass.

Real, unplanned finding along the way: this repo's `data_toolkit/` (and the identical upstream
microsoft/TRELLIS.2 repo, confirmed via GitHub API) documents a `datasets.<SUBSET>` plugin
interface (`build_metadata.py`/`download.py`/`dump_mesh.py`/`dump_pbr.py` all depend on it) but
never ships the actual per-dataset connector modules — they only exist in microsoft/TRELLIS v1's
`dataset_toolkits/datasets/`. Not a local-checkout issue; a real gap in Microsoft's own public
release.

Concrete progress, in order:
- ✅ Ported `data_toolkit/datasets/ObjaverseXL.py` from v1 (attributed), fixed a real interface
  mismatch (v1's `download()` used a kwarg named `output_dir`; TRELLIS.2's `download.py` calls it
  via `**opt` with key `download_root`).
- ✅ Metadata fetch: real, works immediately (168,307 real ObjaverseXL-sketchfab records via a
  public HF CSV).
- ✅ Download: real, works, with real friction — `objaverse.xl`'s underlying downloader hit five
  separate transient network `IncompleteRead` interruptions fetching a 196-object slice (not a
  bug; real network flakiness on large sequential downloads). Recovered by reconstructing the
  sha256/local_path mapping directly from files already on disk (Sketchfab's on-disk filenames are
  its own object UIDs, extractable from `metadata.csv`'s `file_identifier` URLs) rather than
  continuing to retry a fresh full run each time. 196/210 objects, 0 mismatches.
- ✅ Mesh dump: found and fixed two more real bugs before this worked. (1) `dump_mesh.py`/
  `dump_pbr.py` hardcode a **Linux-only** Blender fetch (`apt-get` + a Linux tarball), unmodified
  from upstream — fixed to detect macOS and use the real, already-installed `/Applications/
  Blender.app` instead. (2) The ported `foreach_instance()` called `func(file, sha256)` (matching
  v1's own convention), but TRELLIS.2's own `_dump_mesh`/`_dump_pbr` (not written by this project)
  both do `metadatum['sha256']` — expecting the *full row dict*, not the bare string. First real
  run failed on all 196/196 objects with `TypeError: string indices must be integers, not 'str'`
  before this was found and fixed. Real result after both fixes: 196/196 real `.pickle` mesh dumps
  produced, ~18s total, exit code 0.
- ✅ PBR dump: found and fixed a third real bug. `install_pillow.py`'s pip install falls back to a
  user-site install on macOS (Blender.app's bundled site-packages isn't writable) —
  `~/.local/lib/python3.13/site-packages`. Pip itself reported "Requirement already satisfied,"
  yet `blender -b -P dump_pbr.py` still raised `ModuleNotFoundError: No module named 'PIL'`.
  Setting `PYTHONPATH` in the subprocess environment did **not** fix it — confirmed directly that
  Blender's embedded Python ignores `PYTHONPATH`. Real fix: inject the path via `sys.path` from
  *inside* the Blender-side script itself, before the `PIL` import. Real result: **168/196 (86%)**
  succeeded, exit code 0. The 28 failures are genuine, expected content limitations, not a bug —
  error messages list unsupported Blender shader-node combinations (`Emission`, `Transparent BSDF`,
  `Mix Shader`, `Color Attribute`) this baking pipeline doesn't handle; real-world 3D assets have
  material complexity this toolkit's PBR extraction wasn't built to cover. Not chased further —
  86% is a solid, real pilot yield, not a target to push to 100%.
- ✅ O-Voxel conversion (`dual_grid.py`/`voxelize_pbr.py`): the deepest gap found so far — the
  vendored `o-voxel` C++ extension (`o_voxel._C`) had never been built at all in this checkout, so
  `o_voxel.io`/`o_voxel.serialize` were completely non-functional (`import` itself failed). Root
  cause was mechanical, not algorithmic: `setup.py`'s own CPU-build branch requires
  `src/ext_cpu.cpp`, which simply didn't exist, even though the real CPU geometry implementations
  (`flexible_dual_grid.cpp`, `volumetic_attr.cpp` — 775/872 lines, complete, not stubs) already do.
  Wrote `ext_cpu.cpp` (`ext.cpp`, the CUDA-era entry point, trimmed to only the CPU-available
  bindings). That surfaced two more real, mechanical compiler-strictness gaps (this vendored code
  was presumably only ever validated on Linux/gcc or nvcc): 8 fatal `-Wc++11-narrowing` errors
  (`size_t`→`long long`/`int` in brace-init lists) across `filter_parent.cpp`, `svo.cpp`,
  `filter_neighbor.cpp`, `flexible_dual_grid.cpp` — fixed with explicit casts; and 2 `invalid
  suffix 'd' on floating constant` errors (`1e-6d`, `0.0d` — not valid C++) — fixed by dropping the
  suffix. Also found the `o-voxel/third_party/eigen` submodule was declared but never initialized
  in this checkout (empty directory) — `git submodule update --init` fixed it (no commit needed,
  the parent repo's gitlink already pointed at the right commit). One more real gap: the z-order/
  Hilbert (de)serialize CPU functions live in `.cu` files alongside CUDA-only `__global__` kernels
  (can't compile without nvcc, and the shared headers can't even be included in a CPU build) — wrote
  `serialize_cpu.cpp`, the CPU logic ported verbatim with CUDA decorators stripped. Caught a real
  bug of my own here before shipping it: my first draft stored `.contiguous().data_ptr()` from a
  temporary tensor in a separate statement, so for any non-contiguous input (exactly what
  `serialize.py`'s `coords[:, i]` column-slicing always produces) the pointer was already dangling
  before use — a trivial all-zero round-trip test decoded to garbage, which is what caught it, not
  the build succeeding. Fixed by holding the contiguous copies as named locals spanning the whole
  function; re-verified the z_order/Hilbert round-trip explicitly against a genuinely
  non-contiguous strided tensor view. Also added `no_file` support to `ObjaverseXL.py`'s
  `foreach_instance` (TRELLIS.2's own `dual_grid.py`/`voxelize_pbr.py` call it with
  `no_file=True` since those stages read mesh_dumps/pbr_dumps by sha256, not from a raw downloaded
  file — the ported v1 `foreach_instance` had no such mode). Real result once everything built:
  **196/196** dual-grid conversions (resolution 64) and **168/168** PBR voxelizations (resolution
  64, i.e. all PBR-dumped objects), both exit code 0, zero errors.
- ✅ Latent encoding (`encode_shape_latent.py`/`encode_ss_latent.py`): the real, separate unknown
  this time was device code, not the connector-plugin pattern — both scripts (unmodified from
  upstream) hardcode `.cuda()`/`torch.cuda.synchronize()`/`torch.cuda.empty_cache()` throughout.
  Ported to this project's established MPS convention (`trellis2/backends.py`'s `HAS_MPS`
  pattern, `torch.mps.synchronize()`/`empty_cache()`, already used elsewhere in this repo) via a
  `DEVICE` module constant, falling back to real CUDA if ever run on a CUDA machine. Also needed
  two encoder checkpoints neither present locally nor in `trellis2/model_revisions.py`'s
  `MODEL_FILES` manifest — `ss_enc_conv3d_16l8_fp16` (`microsoft/TRELLIS-image-large`) and
  `shape_enc_next_dc_f16c32_fp16` (`microsoft/TRELLIS.2-4B`), neither needed by the inference CLI
  (decode-only). Confirmed both real via the HF API before downloading (119MB/709MB), added to
  the manifest at the user's explicit direction so `download_weights.py --offline` keeps covering
  them. Real result: single-instance sanity test then full-batch, both scripts, both **196/196**,
  zero errors — real, finite output (shape latent: `feats` `[N,32]` float32, `coords` `[N,3]`
  uint8; ss latent: `z` `[8,16,16,16]` float32).
  Verified the channel dimension matches `slat_flow_img2shape_dit_1_3B_512`'s real training
  config exactly (`in_channels: 32` == our shape-latent `feats` width), confirming correct
  encoder/model pairing — but at this point the pilot had used dual-grid resolution 64 for speed
  (shape-latent coords range `[0,3]`, not the real `[0,31]` the config's `resolution: 32` implies
  for a true 512-resolution input) — flagged as a real scope gap, not yet closed.
- ✅ **Resolution-512 re-run, closing the scope gap above.** Re-ran the full O-Voxel-conversion +
  latent-encoding chain at the real production resolution: `dual_grid.py --resolution 512`
  (**196/196**, ~52s), `voxelize_pbr.py --resolution 512` (**168/168**, ~40s),
  `encode_shape_latent.py --resolution 512` (**196/196**, ~2m9s) — real output file size scales
  as expected (dual-grid `.vxz`: 17KB → 922KB per object, ~54x, consistent with a finer surface
  mesh at 8x linear resolution). Shape-latent coords now range **`[0,31]`**, exactly matching the
  config's `resolution: 32` — the scope gap is closed, confirmed with real data, not just
  arithmetic. Then `encode_ss_latent.py --shape_latent_name shape_enc_next_dc_f16c32_fp16_512`
  (**196/196**) — real, finite `z` `[8,16,16,16]` float32 for every object.
  **Real finding along the way, not a bug:** `encode_ss_latent.py`'s own output path is keyed
  only by `--resolution` (its fixed sparse-structure grid size, 64) — not by which
  `--shape_latent_name` fed it. Re-running it against the resolution-512 shape latents therefore
  silently collided with the earlier resolution-64-derived output in the same
  `ss_latents/ss_enc_conv3d_16l8_fp16_64/` directory, and its own "already processed" check (a
  plain file-existence scan plus that directory's own stale `metadata.csv`) treated all 196 as
  already done, so the first re-run attempt processed **0** objects. Not a bug in this project's
  code — TRELLIS.2's own upstream script has no concept of "regenerate because the input
  changed." Fixed by moving the stale `.npz` outputs and the stale per-stage `metadata.csv` aside
  (kept, not deleted, at `ss_latents/ss_enc_conv3d_16l8_fp16_64_from_res64_stale/`) before
  re-running clean. Worth remembering for any future re-run of this stage against a different
  shape-latent source at the same `--resolution`.
- Exit criterion: **fully met, including at production scale.** A real, small,
  correctly-encoded training set exists on disk (196 objects, shape latents + ss latents at the
  real resolution-512 scale) — shape/dtype/channel-dimension correctness verified directly
  against `shape_slat_flow_model_512`'s/`slat_flow_img2shape_dit_1_3B_512`'s real training
  config, not just asserted.

### Phase 1 — implement SCFM's loss first (cheaper, and no reference code to lean on means doing this carefully matters more)

**Status: implemented, not yet verified (that's Phase 2).** `trellis2/trainers/flow_matching/
scfm_distill.py` (`SCFMDistillTrainer`), implementing the paper's vanilla Algorithm 1 (single
stopgrad EMA — Appendix E's dual fast/slow EMA "final version" is a documented TODO, not silently
skipped: the paper itself frames it as a training-*speed* optimization with "negligible impact on
final converged results").

Read the actual PDF directly (downloaded, not a secondary web summary) before writing any code —
a first WebFetch-based pass materially mis-described what index `i` means in the paper's Eq (13)
(said it indexed timesteps; the real paper states plainly "`N` is the total batch size," i.e. `i`
indexes batch elements). Re-reading the real page images caught this before it became a wrong
implementation — exactly the kind of error this project's "read the actual source, don't guess"
discipline exists to prevent, applied here to a paper instead of code.

Real adaptations made, not blind ports:
- **`n=12`** for the windowing base discretization grid, matching `shape_slat_flow_model_512`'s
  actual real production step count (`pipeline.json`), not an arbitrary finer virtual grid the
  paper's own Flux/SD3.5 experiments don't specify a value for. This ties the implementation
  directly to the project's real research question (does distillation hold up starting from an
  already-short, already-calibrated schedule) — querying the teacher at invented finer virtual
  timesteps it was never calibrated near would quietly dodge the exact risk this project exists to
  test. Real, deliberate consequence: the paper's `{2,4,...,n/4}` coarse-skip set collapses to just
  `{2}` at `n=12`.
- Confirmed the paper's Eq (17) timestep shift (`S_s(t) = st/(1+(s-1)t)`) is **algebraically
  identical** to this project's own `FlowEulerSampler`'s `rescale_t` mechanism — same formula,
  different name, not a coincidence worth re-deriving from scratch. The real production
  `rescale_t=3.0` already falls inside the paper's own default `s ∈ [2.5,4.5]` sampling range.
- Adapted the windowed-interpolation target (Eq 12) and the batch's teacher/self-distill split for
  `SparseTensor.feats` (variable per-sample token count) via real sub-batch `SparseTensor`
  indexing + `sparse_cat`, not a flattened boolean mask (would not respect per-sample token-count
  variability).
- Targets are built using the model's **real production CFG** behavior (guidance_strength=7.5,
  guidance_rescale=0.5, guidance_interval=(0.6,1.0), read from the real `pipeline.json`), not a
  bare unguided forward pass — reimplemented per-batch-element rather than reusing the existing
  sampler mixins, since those assume one shared scalar `t` for the whole batch (true during
  iterative sampling), incompatible with SCFM training's need for a different `t` per element
  within one forward pass.

Also fixed a real, separate gap found while sanity-checking that the new file even imports:
`BasicTrainer` (used by every trainer in this repo) unconditionally imports `tensorboard`, never
listed in `requirements_macos_core.txt` — nothing had exercised the training path on macOS until
this phase. Added, pinned to the installed version.

Verified only at the "imports cleanly, class MRO resolves, shift-grid formula is numerically
correct" level so far — real correctness verification (gradient flow, finite loss, a real
forward+backward pass on Phase 0's real encoded pilot data) is Phase 2, deliberately not done yet.

### Phase 2 — correctness verification (this project's established methodology)
- Synthetic check: gradient flow to every trainable parameter, finite loss, no NaNs.
- Real-data check: one real forward+backward pass on Phase 0's real encoded data.

### Phase 3 — small-scale pilot training on the DGX pod
- Explicitly a **feasibility/correctness pilot**: reduced batch (gradient accumulation or a
  smaller real batch), fewer iterations than the paper's ~2000, on Phase 0's data slice.
- Target 2-4 student steps first, not the extreme case, given the untested short-teacher risk.
- **Directly test the theoretical prediction, not just the headline metric**: track a diversity/
  entropy proxy across training (not just loss/error), since covariate-shift and entropy-collapse
  theory specifically predicts *early structural commitment* as the failure signature — if that
  shows up, it's evidence for the hypothesis, not just a quality regression to note in passing.
- Exit criterion: loss decreases meaningfully over training, and a real sample at the reduced step
  count is structurally coherent — not yet a rigorous quality claim.

### Phase 4 — MDT-dist as the comparison method (only after Phase 3 gives a real signal)
- Same phases (0 already shared, 1-3 repeated) for MDT-dist's VM loss, using the same pilot data
  and evaluation protocol, so the two methods' results are directly comparable.

### Phase 5 — real evaluation against this project's established baselines
- Real DINOv3 conditioning, real full-model wall-clock timing, real quality comparison against the
  T_run1 baseline (voxel/mesh-level, not just latent rel_error — the CFG-caching work's own lesson
  that rel_error alone doesn't capture downstream quality).
- Report honestly against the 10x target and against the "does it work at all from a short
  teacher" question regardless of outcome — a clean negative result with the entropy/covariate-
  shift evidence from Phase 3 is a legitimate, publishable answer to the research question above,
  not just a failed pilot.

## What "novel enough for CVPR" actually rests on

Not "ran existing code on a new model" for either method. The real contribution: (a) the first
test, on any model, of whether either MDT-dist-style or SCFM-style few-step distillation holds up
starting from an already-short (12-step) production teacher — a literature-wide gap, not a
TRELLIS.2-specific footnote; (b) a real technical adaptation of either loss formulation to a
sparse, variable-token-count representation (O-Voxel); (c) the comparative framing across this
session's caching negative results AND the distillation pilot(s) — "training-free step-skipping
provably fails here, here's whether/why retraining-based compression does or doesn't, and what
that implies for the growing number of production models that ship pre-compressed like TRELLIS.2
does" — a general, timely narrative, not a narrow one-model result.
