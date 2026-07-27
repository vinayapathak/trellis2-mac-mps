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
- Run `data_toolkit/build_metadata.py` + `download.py` for a **small slice** of Objaverse-XL (low
  hundreds of objects) — enough for a real pilot, not a full reproduction.
- Run the encode pipeline (`encode_shape_latent.py`, `encode_ss_latent.py`) end to end on that
  slice, confirm the output SLat encodings are shaped/typed correctly for
  `shape_slat_flow_model_512`'s real training config.
- Exit criterion: a real, small, correctly-encoded training set exists on disk, verified — not
  assumed to work because the tooling exists.

### Phase 1 — implement SCFM's loss first (cheaper, and no reference code to lean on means doing this carefully matters more)
- New file: `trellis2/trainers/flow_matching/scfm_distill.py`. Teacher = frozen pretrained
  `shape_slat_flow_model_512`; student = trainable copy initialized from the same weights; maintain
  an EMA stopgrad copy of the student per the paper's `θ⁻` update rule.
- Implement the hybrid loss and the windowed-interpolation velocity target, adapted for
  `SparseTensor.feats` — the real, novel engineering piece, done carefully since there's no
  reference implementation to check against here (unlike VDE).
- Reimplement from the equations, cross-check every design decision against the paper's own text
  rather than guessing at gaps the abstract-level read didn't cover — same discipline used for
  every other technique ported this session.

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
