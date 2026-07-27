# MDT-dist for TRELLIS.2's shape_slat_flow_model_512 — scoping plan

Status: **scoped, not started.** Written after three independent inference-time step-skipping
techniques (CFG difference-caching, AB-Cache, VDE — see `CLAUDE.md`) all failed to reach a usable
quality/speed tradeoff on TRELLIS.2's compressed 12-step production schedule. Distillation is a
structurally different lever — it changes how many steps the model *needs*, rather than trying to
skip steps it still nominally has — and MDT-dist (arXiv:2509.04406, tested on original TRELLIS, 9.0x
at 1 step / 6.5x at 2 steps) is the only technique found with real published evidence anywhere near
the user's stated 10x target. Nobody has applied it to TRELLIS.2 or its sparse O-Voxel SLat models
— confirmed by reading the paper directly and checking for prior art (see `CLAUDE.md`'s MDT-dist
sections).

Target for a CVPR-track writeup, so this plan is built around producing a defensible, honestly
scoped result, not the fastest possible hack.

## What's already real and available in this repo

Checked directly, not assumed:

| have | what |
|---|---|
| ✅ | `train.py` — distributed training entrypoint, checkpoint load/resume |
| ✅ | `trellis2/trainers/flow_matching/` — base flow-matching trainer + CFG mixin (`mixins/classifier_free_guidance.py`) |
| ✅ | `data_toolkit/` — full pipeline for Objaverse-XL / ABO / HSSD / TexVerse (same dataset families MDT-dist trained on) |
| ✅ | `configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16.json` — real training config scoped to `shape_slat_flow_model_512` |
| ✅ | `dgx_connect.sh` — access to the user's DGX pod, 3 GPUs |
| ❌ | any distillation-specific loss/trainer code — the actual new work |

## The real algorithm (from the paper + reference-adjacent read, not re-derived)

Two losses, trained jointly (`ℒ = ℒ_VM + λ·ℒ_VD`, `λ=1.0`):

- **Velocity Matching (VM)**: `ℒ_VM = E[‖u_θ(x_t,t) − v_pretrain(x_t,t)‖²]`, where
  `u_θ ≈ φ_θ(x_t,t) + t·dφ_θ(x_t,t)/dt`, the derivative discretely approximated
  (`Δt=1e-2`) with the **gradient of the derivative term detached** — this is the load-bearing
  trick that makes it trainable at all. Forces the student's one-step prediction to imply the
  correct *instantaneous* velocity at every noise level, not just the right final answer.
- **Velocity Distillation (VD)**: score-distillation style — the student generates a sample, that
  sample is re-noised, and the teacher's velocity there becomes the target. Exposes the student to
  its own generation distribution during training. Paper's own ablation: VM alone gets most of the
  benefit (FD_incep 18.42 vs. 18.09 with VD added) — smaller, complementary gain, not the main
  driver.
- Paper's real hyperparameters (original TRELLIS, for reference — **not assumed to transfer
  unchanged**, see risks below): SLat transformer 4k iterations, batch 256, lr 1e-8, AdamW,
  student initialized from the pretrained teacher weights, CFG scale 40 for VM / 100 for VD
  (much higher than TRELLIS.2's own production `guidance_strength=7.5` — presumably to sharpen
  distillation targets, a real detail worth preserving, not guessing differently).

## Real, specific risks — why this isn't a drop-in port

1. **12-step teacher, not 25.** MDT-dist's own numbers are for a 25→1-2 step compression. TRELLIS.2's
   real production teacher already runs at 12 steps. Whether VM/VD's error bound (Theorem 1 in the
   paper, bounding the primary transport objective's error by the VM loss's error) still gives a
   *useful* bound at this much shorter starting schedule is a genuinely open, untested question —
   the paper gives no evidence either way. This is also exactly the axis that broke the three
   caching techniques already tried, so it's the single most important thing to check early, not
   assume.
2. **Sparse O-Voxel tokens, not dense.** MDT-dist was built for original TRELLIS's structure.
   TRELLIS.2's `shape_slat_flow_model_512` operates on `SparseTensor` — variable token count per
   sample, no fixed spatial grid. The VM loss's finite-difference derivative and the VD loss's
   re-noising step both need to be well-defined per-token (or per-sample, aggregated) rather than
   per-fixed-pixel-position — a real adaptation, not just a shape reinterpretation.
3. **Real training data is required.** This is not training-free. Need actual 3D asset data (image
   + shape) run through the real `data_toolkit/` encode pipeline to produce real SLat training
   targets — a full pipeline validation is real work before any distillation-specific code even runs.
4. **Compute scale mismatch.** Paper's batch=256 over thousands of iterations on a 1.1B model very
   likely used substantially more than 3 GPUs. This plan is explicitly scoped as a **pilot**, not
   a claim of matching the paper's full result — same honesty discipline already used in this
   project for the from-scratch DINOv3 work (a real, smaller, defensible claim beats an inflated one).

## Phased plan

### Phase 0 — data pipeline validation (no distillation code yet)
- Run `data_toolkit/build_metadata.py` + `download.py` for a **small slice** of Objaverse-XL (low
  hundreds of objects, not the full 500K) — enough for a real pilot, not a full reproduction.
- Run the encode pipeline (`encode_shape_latent.py`, `encode_ss_latent.py`) end to end on that
  slice, confirm the output SLat encodings are shaped/typed correctly for
  `shape_slat_flow_model_512`'s real training config.
- Exit criterion: a real, small, correctly-encoded training set exists on disk. No modeling
  work started until this is true and verified, not assumed.

### Phase 1 — VM loss only (skip VD initially, matching the paper's own ablation priority)
- New file: `trellis2/trainers/flow_matching/mdt_distill.py`, subclassing/composing with the
  existing base flow-matching trainer (non-destructive — new file, existing trainers untouched,
  same convention as every sampler added this session).
- Teacher: frozen copy of the real pretrained `shape_slat_flow_model_512` weights.
- Student: trainable copy, initialized from the same weights (per the paper — no new architecture).
- Implement the discrete-derivative VM loss with the detached-gradient trick, adapted to operate
  over `SparseTensor.feats` (per-token, or aggregated per-sample — needs a real design decision,
  made and documented once actually implemented, not guessed here).

### Phase 2 — correctness verification (this project's own established methodology)
- Synthetic check: gradient flow to every trainable parameter (mirrors `dinov3_from_scratch/verify.py`'s
  pattern), finite loss, no NaNs, across a few real-shaped synthetic batches.
- Real-data check: one real forward+backward pass on Phase 0's real encoded data, confirm the loss
  is finite and gradients are sane (not just synthetic-shape correctness — this project's own
  established lesson that synthetic checks alone aren't sufficient).

### Phase 3 — small-scale pilot training on the DGX pod
- Explicitly scoped as a **feasibility/correctness pilot**, not a full reproduction: reduced batch
  size (gradient accumulation to approximate effective batch, or accept a smaller real batch),
  far fewer iterations than the paper's 4k, on the Phase 0 data slice.
- Target the paper's safer end first: distill toward **2-4 student steps**, not the extreme 1-step
  case, given TRELLIS.2's already-short 12-step teacher (risk #1 above).
- Exit criterion: VM loss decreases meaningfully over training (not just "runs without crashing"),
  and a real sample generated at the reduced step count is at least structurally coherent (a voxel
  count / gross-shape sanity check, not yet a rigorous quality claim).

### Phase 4 — add VD (only if Phase 3's VM-only result needs it)
- Mirrors the paper's own finding that VD is a secondary refinement, not the primary driver — don't
  add complexity before confirming it's needed.

### Phase 5 — real evaluation against this project's established baselines
- Same methodology as every other investigation this session: real DINOv3 conditioning, real
  full-model wall-clock timing (not isolated), and a real quality comparison against the T_run1
  baseline (voxel/mesh-level, not just latent rel_error — the CFG-caching work's own lesson that
  rel_error alone doesn't fully capture downstream quality).
- Report honestly against the 10x target regardless of outcome — if the pilot only reaches, say,
  3-4x at this small scale, that's still a real, reportable number and a legitimate basis for
  scoping a larger training run, not a result to inflate.

## What "novel enough for CVPR" actually rests on

Not just "ran existing code on a new model." The real contribution, if the risks above are
navigated: (a) the first evidence of whether Marginal-Data-Transport distillation holds up when
the teacher schedule is already short (12 steps, not 25) — a genuinely untested regime with a
plausible, non-obvious failure mode given what this project's own caching experiments already
found; (b) the first adaptation of the VM/VD loss formulation to a sparse, variable-token-count
representation (O-Voxel), a real technical extension, not a reshape; (c) if the comparative
caching-vs-distillation finding in `CLAUDE.md` is included, a genuine "here's what doesn't work
and a principled reason why, plus what does" narrative — stronger framing than either half alone.
