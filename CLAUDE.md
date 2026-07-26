# CLAUDE.md — TRELLIS.2 Apple Silicon port (companion to trellis-mac-mps)

Instructions for continuing work on this repo in future sessions.

## What this project is

A port of Microsoft's **TRELLIS.2** (the newer, 4B-parameter O-Voxel-based image/text-to-3D
model, [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2)) to run natively on Apple
Silicon. This is the companion project to
[trellis-mac-mps](https://github.com/vinayapathak/trellis-mac-mps), which ports the original
TRELLIS (CVPR'25) with hand-written Metal kernel work, algorithmic optimization, and a research
paper's worth of rigorous benchmarking.

## Starting point — real, substantial prior work, not a from-scratch port

There is already an open, comprehensive, unmerged PR adding native Apple Silicon support to
microsoft/TRELLIS.2 upstream:
**[microsoft/TRELLIS.2#175](https://github.com/microsoft/TRELLIS.2/pull/175)**, by
[Igor Shaposhnikov (Jourloy)](https://github.com/Jourloy), branch
[`Jourloy/TRELLIS.2@main`](https://github.com/Jourloy/TRELLIS.2). 70 files, ~9,400 lines, a real
test suite (28 passing pytest tests as of the PR description), organized as sequential work
packages (WP1–WP9: mesh-integrity metrics, a Metal narrow-band dual-contouring remesh benchmark
harness, parity/stability tests, opt-in fp32 decode thresholds, a guarded reprojection fix). It
itself explicitly builds on and credits two prior community efforts:
[`pedronaugusto/trellis2-apple@6055b86`](https://github.com/pedronaugusto/trellis2-apple/commit/6055b868734af6e12769d229d90580e775fae9f0)
(source-native MPS/Metal backend + MLX parity work) and
[`shivampkumar/trellis-mac@d58628f`](https://github.com/shivampkumar/trellis-mac/commit/d58628f4f5b9c3de8274cb110074154f4b31cef2)
(MPS CLI/fallback design lessons).

**This repo's starting branch (`main-apple-silicon`) is a direct checkout of Jourloy's branch**,
MIT-licensed same as upstream TRELLIS.2, fetched and used as the working baseline rather than
duplicated from scratch — following the same "don't re-solve what's already solved, focus on the
differentiated contribution" approach that made trellis-mac-mps worth doing in the first place.
Getting TRELLIS.2 running on Mac at all is, as of this writing, a solved problem across three
independent community efforts plus this in-flight official PR. **Do not re-derive that work.**

## What the actual differentiated contribution here should be

Matching trellis-mac-mps's own positioning (see its `README.md`'s "How this compares" table and
`APPLE_SILICON.md`): the gap in the existing TRELLIS.2 Apple Silicon ecosystem is the same as it
was for the original TRELLIS before that project started — real kernel-level performance
characterization, algorithmic optimization with rigorous verification, and honest reporting of
what doesn't work, not just "it runs." None of trellis-mac, trellis2-apple, or PR #175 do this;
they are all "getting it running" contributions (correctly and thoroughly, for what they are).

Candidate directions, not yet started, roughly in likely order of leverage (informed by what
already worked for the original TRELLIS port — verify each hypothesis against TRELLIS.2's actual
profile before committing time, don't assume the same bottlenecks carry over):
1. **Bypass-measure where TRELLIS.2's forward-pass time actually goes** (FFN vs. attention vs.
   the O-Voxel sparse-conv backbone vs. the new `flex_gemm`/`mtldiffrast`/`mtlbvh`/`mtlmesh` Metal
   extensions PR #175 already added) — same methodology as the original project's Section 4
   bypass measurements, applied fresh to TRELLIS.2's different architecture. Do not assume the
   ~77%/~11% attention/FFN split found for the original TRELLIS's `sparse_structure_flow_model`
   transfers unchanged — TRELLIS.2 has a materially different backbone (O-Voxel, not the original
   sparse voxel grid) and this needs to be re-measured, not assumed.
2. **Framework comparison at TRELLIS.2's actual shapes** — the original project's MLX/llama.cpp/
   PyTorch-MPS comparison methodology (`scripts/benchmark_mlx_comparison.py` and
   `scripts/benchmark_architecture_generalization_vit.py` in trellis-mac-mps) generalizes
   directly; PR #175 already ships an experimental MLX backend (`mlx_backend/`), which is a head
   start for this specific comparison (real MLX code already exists to benchmark against, not
   just a hypothesis).
3. **Algorithmic-level work analogous to CFG difference-caching and step-count reduction** — check
   whether TRELLIS.2's sampler has the same or a related redundancy; do not assume it does without
   checking, since TRELLIS.2's flow-matching/guidance implementation may differ from the original.
4. Whatever the bypass measurement in (1) actually reveals as the dominant cost — let the
   measurement pick the target, exactly as the original project did (its own kernel work started
   from a bypass measurement, not a guess).

**Do not start hand-writing kernels or claiming a speedup before running the equivalent of step 1
and 2 first.** The original project's single most important working habit, stated repeatedly in
its own CLAUDE.md/APPLE_SILICON.md, was measuring before optimizing and reporting negative results
honestly. Apply the same discipline here from the start rather than relearning it.

## Practical status as of this writing (this session)

- Repo created: `vinayapathak/trellis2-mac-mps`.
- `main-apple-silicon` branch = a checkout of `Jourloy/TRELLIS.2@main` (the PR #175 branch),
  fetched via `git remote add jourloy https://github.com/Jourloy/TRELLIS.2.git`.
- `requirements_macos.txt` install into a fresh `.venv` was in progress at the point this file was
  written — not yet confirmed complete.
- **Not yet run**: `scripts/probe_macos.py` (the capability probe — checks MPS/Metal/SDPA/MLX
  availability, does NOT need model weights, safe/fast to run first). This machine is a **verified
  Apple M5 Max** (confirmed via `system_profiler SPHardwareDataType`, not the M4 Max the upstream
  PR was validated on) — running this probe here is genuinely new information nobody has confirmed
  yet, not a repeat of existing validation.
- **Not yet done, and requires the user's own Hugging Face account**: actually generating an asset
  needs `hf auth login` (interactive) plus accepting DINOv3 and RMBG-2.0's gated-model license
  terms on Hugging Face — this is a real credential/consent step that has to happen under the
  user's own identity, not something to automate around. Flagged explicitly rather than blocked
  on silently; ask the user to run `hf auth login` themselves when ready to move past capability
  probing into actual generation.
- No bypass measurements, no framework comparisons, no kernel work, no paper — none of the
  differentiated work described above has started. This file exists so that work starts from a
  clear, correct picture of what's already solved (by others) and what isn't (the actual
  contribution opportunity), rather than re-deriving either from scratch in a future session.

## Working-style notes (carried over from trellis-mac-mps, apply the same way here)

- Never delete or destructively rewrite a working module version; add a new file/suffix instead.
- Commit real, verified checkpoints as you go, scoped to the specific files that make up one
  coherent piece of work.
- Document findings (including negative/null results) as you go in this file or a `research.md`,
  not just at the end.
- Verify claims against direct measurement before reporting them; a single run is not enough to
  report a directional finding as confirmed — see trellis-mac-mps's own `research.md` entry on the
  ViT-L/14 finding that didn't replicate, caught only via 8 independent re-runs, for a concrete
  example of why this matters in this exact family of project.

## Update: base capability confirmed on M5 Max (this session)

Ran `scripts/probe_macos.py` against a fresh `requirements_macos.txt` install (`.venv`, no Metal
extensions built yet) on the verified Apple M5 Max. Result: `"ok": true` overall.
- MPS available and functional (matmul parity check against MLX passed: both report 3680.0).
- MLX functional.
- SDPA functional (`sparse_backends.attention: "sdpa"`).
- Metal compiler found, and its path confirms the actual M5-specific toolchain is in use
  (`.../MetalToolchain-v17.6.109.0.M5u60P/...`), not a generic/cached one.
- Sparse convolution falls back to pure PyTorch (`sparse_backends.convolution: "pytorch"`) since
  the compiled Metal extensions (`flex_gemm`, `mtldiffrast`, `mtlbvh`, `mtlmesh`, `cumesh`) are not
  installed by `requirements_macos.txt` alone -- those are separate pinned-commit Metal source
  builds normally done by `scripts/setup_macos.sh`, not yet run this session.
- Raw probe output saved: `docs/probe_m5_max_2026.json`.

This confirms the pure-PyTorch/MPS/SDPA fallback path genuinely works on this exact hardware --
the first real data point on any Apple Silicon generation beyond the M4 Max the upstream PR was
validated on. Real generation (`scripts/generate_asset.py`) has NOT been run yet: it needs
`hf auth login` under the user's own Hugging Face account plus accepting DINOv3 and RMBG-2.0's
gated-model terms -- a credential/consent step for the user to do themselves, not automated
around. Building the accelerated Metal extensions (`scripts/setup_macos.sh`, or `SKIP_METAL=1` for
the pure fallback path) also not yet done.

## Update: full Metal extension stack built and verified on M5 Max (this session)

Ran `scripts/setup_macos.sh` end to end (Python 3.11 venv, `torch==2.13.0`/`torchvision==0.28.0`
primary pair — no ABI fallback needed, clean first-try success). All four pinned Metal extensions
built from source and installed successfully:
- `mtlbvh` @ `23f441c470ce1f537e1fd836f3ffb5b8245f7975`
- `mtldiffrast` @ `4668cd91cb6d27f5e264731f94a06841fbf7aab8`
- `mtlmesh` @ `212079e55772cff3d648a21372392c37e0643f3b` (pulls in `cubvh`/`eigen` submodules)
- `mtlgemm` @ `867aec8234299a7fe1ede7f802c8debe5a939a82`
- `o-voxel` (editable install)

`scripts/probe_macos.py --require-metal` passes cleanly: mesh/BVH ok, rasterizer ok (Metal
backend, not CPU), `flex_gemm: true`, and — the meaningful upgrade from the earlier pure-PyTorch
probe — `sparse_backends` now reports `"attention": "flex_gemm_sparse_attn"`,
`"convolution": "flex_gemm"`, `"metal_attention_parity": true`. This is the real, accelerated
Metal path, not the CPU/pure-PyTorch fallback recorded earlier this session. `pip check`: no
broken requirements. Full raw output: `docs/probe_m5_max_metal_full_2026.json`.

This is, as far as this project can tell, the first confirmed verification of the complete
accelerated Metal extension stack (all four extensions, not just base MPS/MLX/SDPA) on an M5 Max
specifically -- the upstream PR's own validation was on an M4 Max.

**Remaining blocker to actual generation, unchanged**: `hf auth login` under the user's own
Hugging Face account, plus accepting DINOv3 and RMBG-2.0's gated license terms. The build itself
prints this reminder automatically now that the environment is otherwise ready.

Next real step once auth is done: run `scripts/generate_asset.py` end to end, confirm real output,
*then* move on to the actual differentiated work described above (bypass measurement, framework
comparison, etc.) — do not skip straight to kernel work before confirming the baseline pipeline
produces correct output on this machine.

## Update: from-scratch DINOv3 architecture built while blocked on gated access (this session)

While waiting on Meta's manual review for `facebook/dinov3-vitl16-pretrain-lvd1689m` (see the
weight-download blocker above), built a real, paper-faithful DINOv3 ViT-L/16 implementation from
the actual paper (arXiv:2508.10104) in `dinov3_from_scratch/`: architecture (RoPE, register
tokens, SwiGLU), and the full training objective (DINO + iBOT + KoLeo + Gram anchoring losses).
Verified correct on real MPS hardware: full ViT-L/16 config (304.6M params, matching the real
model's ~300M) forward-passes cleanly, and a full backward pass reaches all 32 tested parameters
with nonzero gradients through a real optimizer step. See `dinov3_from_scratch/README.md` for the
honest, explicit statement of what this is and isn't -- **this is an architecture-correctness
artifact, not a substitute for the real pretrained checkpoint.** DINOv3's actual capability comes
from training on 1.7B images (LVD-1689M, not public) over 256 GPUs per the paper; nothing built
here reproduces that, and it should not be used in place of the real gated checkpoint once access
is granted.

User has DGX pod access (see this project's memory / the parent trellis-mac-mps project's own DGX
notes) -- a genuine small-scale training smoke-test (real losses decreasing, no bugs, on a small
public dataset) is feasible as a pipeline-correctness exercise and is a reasonable next step if
pursued further, but is explicitly NOT a path to matching Meta's actual trained quality. Not yet
attempted as of this writing.

**Status of the actual blocker this was built around**: still waiting on Meta's manual approval
for DINOv3 gated access (`gated: manual`, confirmed via the HF API) and re-checking on a 30-minute
schedule. RMBG-2.0 (`gated: auto`) was also still returning 401 as of the last check despite being
reported as accepted -- worth re-verifying once DINOv3 clears, since an auto-gate should not
behave this way if the accept flow actually completed.

## Update: gated access cleared, real end-to-end generation confirmed on M5 Max

**Root cause of the RMBG-2.0 401 anomaly, found while unblocked**: the CLI's HF login had been
done via the `hf` CLI's browser OAuth flow, which carries scopes fixed at token-creation time --
it does not automatically pick up gated-repo access granted to the account afterward. Fix: a
fresh classic "Read" token from https://huggingface.co/settings/tokens, `hf auth login` with that
instead. Confirmed both gates open with a direct API check (`GET /api/models/<repo>` -> 200) for
both `facebook/dinov3-vitl16-pretrain-lvd1689m` and `briaai/RMBG-2.0` immediately after switching
tokens -- this was the real fix, not a propagation-delay wait.

Ran `scripts/generate_asset.py assets/example_image/T.png --output-dir outputs/T_run1
--pipeline-type 512 --seed 42` end to end. **Real success, `status: "ok"` in `meta.json` for both
outputs:**
- `raw_full.glb`: 68MB, 3,047,718 triangles, 1,442,900 vertices (welded).
- `candidate_pbr.glb`: 147MB, 3,119,520 triangles, 3,504,194 vertices (raw, pre-weld), textured +
  remeshed (`remesh_band=1.0, remesh_project=0.7`), baker=`metal`.
- Both non-watertight (expected/normal for this kind of output, not a bug) -- `candidate_pbr` has
  1,790 connected components, `raw_full` has 42.
- Bounds sane and consistent between raw/PBR (~1.0 x 0.92 x 0.41), matching the input image's
  proportions.
- Backend confirmed real accelerated path throughout: `resolved_backend: mps`, `flex_gemm: true`,
  `mesh/rasterizer: metal` -- not the pure-PyTorch fallback.
- Timings: `pipeline_load` 524.8s (includes the remaining weight downloads), `generation` 543.9s,
  `pbr_export` 118.9s, **total 1188.5s (~19.8 min)**. `peak_rss_bytes`: ~11.1GB.
- This is, as far as this project can tell, the **first confirmed real end-to-end TRELLIS.2
  generation on any Apple Silicon Mac, gated weights and all** -- none of the prior community
  efforts (`pedronaugusto/trellis2-apple`, `shivampkumar/trellis-mac`) or PR #175 itself have a
  documented successful real-checkpoint run; all prior validation on this exact machine (M5 Max)
  was capability-probe-only (base MPS/MLX/SDPA and the accelerated Metal extension stack), not
  real generation.

Also fixed a stale `.gitignore` entry found in the process: `output/` (singular) never matched
this project's actual `outputs/` (plural) convention, per this project's own README examples --
`outputs/` was untracked and about to be accidentally committable. Fixed, not just noted.

**Real generation confirmed. Next step, per "What the actual differentiated contribution here
should be" above: start the bypass measurement (item 1)** -- do not start kernel work or claim
any speedup before that. Note also that `trellis2/pipelines/samplers/flow_euler_cached_cfg.py`
(a port of the original project's CFG difference-caching, item 3) already exists in the working
tree as of this update, uncommitted, with its own docstring stating it is **not yet benchmarked
or quality-verified on this model** -- do not treat it as done; verify it the same way the
original project's version was verified (real drift check across multiple (image, seed)
combinations, not assumed to transfer given the different guidance_strength: 7.5 here vs. 5.0 in
the original) before relying on it or reporting a speedup.

## Update: CFG difference-caching ported and benchmarked -- real, honest negative result

Ran `scripts/investigate_cfg_caching_diff.py` (single image `assets/example_image/T.png`, single
seed) against the real pretrained TRELLIS.2-4B weights, on `sparse_structure_sampler` and
`shape_slat_sampler` (the two stages with a real two-branch CFG call per step at
`guidance_strength=7.5`; `tex_slat_sampler` runs at `guidance_strength=1.0` so has no negative
branch to cache and was correctly excluded).

**Caching mechanism itself is correct**: at `neg_cache_interval=1` (recompute every call, i.e. the
no-caching case), output matched the real baseline sampler exactly (`rel_error=0.000000`,
`neg_reuse=0` confirmed) -- rules out a bug in the diff-cache logic itself.

**At `neg_cache_interval=2` (the same default the original TRELLIS project found a real, usable
1.17x-1.30x speedup at), the result here is a real speedup with unacceptably large quality drift:**
- `sparse_structure`: **1.187x** speedup, but **rel_error=0.44** on the raw model prediction
  (voxel-count proxy only shifted -0.83%, so the drift is concentrated in prediction detail, not
  gross occupancy).
- `shape_slat`: **1.261x** speedup, **rel_error=0.45** on the predicted SLat features.

Both errors are roughly **40x larger** than what would read as acceptable (the original project's
working fix kept drift small enough that mesh vertex counts stayed stable across 6 combinations).
This is a real negative result, not a bug to chase further before reporting it.

**Root-cause hypothesis, not yet independently isolated** (two confounded differences from the
original project's setting -- NOTE, corrected after actually reading AB-Cache (arXiv:2504.10540)
directly: the "higher cfg_strength amplifies staleness error" framing below was attributed to
AB-Cache in this project's earlier notes, but AB-Cache's own text (Sections 3.3-3.4, read directly)
says nothing about CFG or guidance strength at all -- it caches/extrapolates the raw network output
across diffusion *timesteps*, an entirely different axis from the cond/uncond CFG-branch caching
this port does. The "cfg_strength amplifies staleness" claim may originally trace to FasterCache
(arXiv:2410.19355, cited alongside AB-Cache in this project's earlier work) but that has NOT been
re-verified against FasterCache's actual text in this session either -- don't treat it as confirmed
either way until someone actually reads that paper directly, the same standard just applied to
AB-Cache. The underlying observation (larger guidance strength = more sensitive to a stale value)
remains a reasonable, independently-plausible hypothesis on its own terms regardless of attribution):
1. TRELLIS.2's real production config uses **`guidance_strength=7.5`** vs. the original project's
   `cfg_strength=5.0` (in the original's convention, `guidance_strength = 1 + cfg_strength`, so
   this is 7.5 vs. 6.0 on a like-for-like basis -- still meaningfully higher).
2. TRELLIS.2's real production config uses only **`steps=12`** total, with `guidance_interval=
   [0.6, 1.0]` covering roughly 9-10 of those 12 steps (`neg_recompute + neg_reuse` = 10 for
   sparse_structure, 9 for shape_slat, confirmed by the script's own counters). The original
   project's sampler runs 25-50 steps by default. Caching every other step out of a 9-10-step CFG
   window is a much larger fraction of the total schedule than the same absolute interval would be
   over 25-50 steps -- each skipped recompute buys less total speedup *and* costs more relative
   accuracy, since there's far less schedule left to average the error back out.

**Conclusion: do not use `DiffCachedCfgFlowEulerGuidanceIntervalSampler` at `neg_cache_interval=2`
(or ship it as a claimed speedup) on TRELLIS.2's real 12-step config as of this writing.** The file
stays in the tree (this project's non-destructive versioning convention) with this finding, not
deleted -- the mechanism is correct and may be useful if TRELLIS.2 is later run at a higher step
count, or with AB-Cache's REAL technique applied to the diff-caching axis: order-k Adams-Bashforth
linear extrapolation of the cached diff, in place of the current zero-order hold (see the
correction above -- AB-Cache is fixed-schedule, not adaptive; its actual contribution is a
higher-order extrapolator with a proven `O(h^k)` error bound vs. naive reuse's `O(h)`, which is a
concrete, different thing to try next, not "smaller/adaptive interval" as this note previously and
wrongly said).
**Only single-image/single-seed tested here** -- weaker rigor than the original project's
6-combination check; that gap doesn't change the conclusion (the error margin is too large to be
seed/image noise) but is worth closing before fully retiring the idea.

This *is* still real, novel work -- as far as this project can tell, nobody else has applied or
benchmarked CFG difference-caching against TRELLIS.2, and a clean negative result with a grounded
root-cause hypothesis is a legitimate, honestly-reported finding, consistent with how the original
project treated its own kernel ceilings.

## Update: tried AB-Cache's REAL technique (order-k extrapolation) -- makes it worse, not better

First, a correction to the section above and to this project family's other docs (also fixed in
trellis-mac-mps's CLAUDE.md/APPLE_SILICON.md this session): **AB-Cache (arXiv:2504.10540) is NOT
adaptive or error-gated.** That characterization, stated as fact in earlier notes, was never
checked against the paper and was wrong. Read the actual PDF directly this session. What AB-Cache
really does: the exact same fixed schedule as naive caching (real compute every N-th step, cached
otherwise) -- its real contribution is replacing *naive zero-order-hold reuse* at the cached steps
with a k-th order Adams-Bashforth linear extrapolation (Eq. 3.6: a binomial-coefficient-weighted
combination of the previous k REAL values), with a proven `O(h^k)` truncation error vs. naive
reuse's `O(h)`. For flow matching (this pipeline's family), the paper says to drop the exponential
term, leaving a pure linear-coefficient extrapolator. k=1 reduces exactly to the zero-order hold
already tried above.

Implemented this for real: `ABCacheDiffFlowEulerGuidanceIntervalSampler`
(`trellis2/pipelines/samplers/flow_euler_ab_cache.py`) applies the k-th order extrapolation formula
to the cached CFG diff (this project's existing caching axis), at the same `neg_cache_interval=2`
schedule already benchmarked. Scope note, stated in the file's own docstring: the real paper applies
this to the raw network output across diffusion *timesteps*; this applies the same formula (the
math, not the paper's literal use case) to the diff-caching axis already built here -- flagged so
it isn't mistaken for "AB-Cache itself, ported."

**Result (`scripts/investigate_ab_cache.py`, same image/seed as the run above, order=1/2/3, all at
`neg_cache_interval=2`):**

| stage            | order | speedup | rel_error |
|------------------|-------|---------|-----------|
| sparse_structure  | 1 (= zero-order hold, sanity check) | 1.296x | 0.439216 |
| sparse_structure  | 2     | 1.197x  | **0.620797** |
| sparse_structure  | 3     | 1.170x  | **0.913898** |
| shape_slat        | 1 (sanity check) | 1.259x | 0.451514 |
| shape_slat        | 2     | 1.276x  | **0.517827** |

The order=1 rows reproduce the earlier zero-order-hold numbers essentially exactly (`0.439216` and
`0.451514`, both bit-for-bit matches to the earlier run) -- confirms the new class is implemented
correctly, not buggy. **Higher order makes the error WORSE, monotonically, on both stages -- the
opposite of the paper's own ablation (their Table 3 shows quality improving with order).** This is
consistent (not a fluke on one stage) and large (order=3 nearly doubles rel_error vs. order=1).

**Root-cause hypothesis, not yet independently isolated:** AB-Cache's `O(h^k)` error bound is an
*asymptotic* bound -- valid as the step size `h` gets small, with a bigger implicit constant at
higher order. The paper's own experiments run 50-step schedulers with fine, roughly-uniform
timestep spacing, where that asymptotic regime plausibly holds. TRELLIS.2's real config instead
uses only 12 total steps, an aggressive `rescale_t` warp (3.0-5.0, front-loading step density into
the high-noise region), and a guidance interval where the cached diff is only sampled every other
step (`neg_cache_interval=2`) -- a large effective step size relative to however fast the true diff
trajectory actually curves. A linear (order=2) or quadratic (order=3) extrapolant fit through 2-3
widely-spaced, possibly non-monotonic real samples can overshoot the true trajectory by more than a
flat hold would, even though the *asymptotic* bound favors the higher-order fit as `h -> 0`. This
has NOT been isolated from the confound already noted above (guidance_strength=7.5, few total
steps) -- a real follow-up (not attempted here) would be testing whether higher order actually
helps at a finer step count (e.g. steps=25+) where the paper's own asymptotic regime is more likely
to hold, which would confirm or kill this hypothesis directly.

**Conclusion: neither the zero-order-hold diff-cache nor AB-Cache's real higher-order extrapolation
is usable on TRELLIS.2's actual 12-step production config as of this writing.** Both `*.py` files
stay in the tree per this project's convention, both documented as real, measured negative results,
not deleted or hidden. The honest state of the CFG-caching investigation for TRELLIS.2: a real
speedup (1.17x-1.30x) is consistently available at `neg_cache_interval=2`, but every reuse strategy
tried so far (flat hold, order-2, order-3 extrapolation) costs more accuracy than this project
judges acceptable to ship. Rigorous negative-result research, same standard the original project
applied to its own kernel ceilings -- not a failure to hide.

## Update: bypass measurement -- attention dominates both flow models, same as the original project

Ran `scripts/investigate_bypass_measurement.py`: the measurement flagged as item 1 of "What the
actual differentiated contribution here should be" above, not started until now. Method: same as
the original trellis-mac-mps project's Section 4 (APPLE_SILICON.md) -- monkeypatch a submodule
class's `forward` to return zeros (the correct no-op for a residual-add block), time N real forward
passes of the full flow model with/without the bypass, attribute the wall-clock drop to that
submodule. Real conditioning (DINOv3 on `assets/example_image/T.png`) and real sparse coords (from
an actual sparse-structure sample+decode, not synthetic), on this machine's real M5 Max.

**Result, both of TRELLIS.2's flow models measured (`N=8` timed passes, `warmup=3`):**

| model | native | self_attn | cross_attn | both attn | mlp | residual (norms/I-O/t-embed) |
|---|---|---|---|---|---|---|
| `sparse_structure_flow_model` (dense tokens, dense attention) | 607.95ms | 53.7% | 16.8% | **70.9%** | 17.1% | 10.3% |
| `shape_slat_flow_model_512` (sparse O-Voxel tokens, flex_gemm-backed) | 656.67ms | 49.8% | 20.4% | **68.8%** | 19.9% | 13.8% |

Both models were bypass-all'd as a sanity floor (norms/I-O layers/timestep-embedder only): 10.3%
and 13.8% respectively -- the individual contributions roughly sum to the rest (attention + mlp +
residual ~= 98-102% in both cases, consistent with additive contributions and no major unaccounted
cost hiding elsewhere in either model).

**This directly answers the open question CLAUDE.md's "What the actual differentiated contribution
here should be" section posed: does the original project's ~77% attention / ~11% FFN split
transfer to TRELLIS.2?** Measured, not assumed: **yes, broadly** -- both flow models land at
~69-71% combined attention and ~17-20% FFN, close to the original TRELLIS's ~77%/~11% split despite
one of these two models running on a structurally different backbone (sparse O-Voxel tokens through
this project's own flex_gemm-equivalent Metal kernels, rather than the original's dense/regular
sparse-conv U-Net backbone). Also consistent across both models here: self-attention costs roughly
2.5-3x what cross-attention does (~50% vs. ~17-20%) -- the conditioning sequence cross-attends
against is much shorter than the self-attention token count in both cases, so this asymmetry is
expected, not surprising.

**Implication for what to do next, per this project's own stated priority order:** since attention
is confirmed to dominate here too, the original project's extensive real, already-completed
attention kernel investigation on this exact hardware family (Apple Silicon MPS, M5 Max) --
documented in trellis-mac-mps's `APPLE_SILICON.md` under "Applying the FFN pattern to attention"
and "Beyond Linear quantization: SDPA itself..." -- is directly relevant background before starting
any new kernel work here, not a different problem needing rediscovery. That investigation's real,
measured ceilings (quantized attention nets 0.98x whole-model, i.e. no net win; native SDPA beats
every alternative backend and a hand-rolled reimplementation; the custom flash-attention kernel
progression tops out at 1.17x slower than native, not a real win) were established on the original
TRELLIS's dense self-/cross-attention. TRELLIS.2's dense model (`sparse_structure_flow_model`) is
close enough in structure that those findings likely transfer directly and probably don't need
re-litigating from scratch. **What's genuinely untested is the sparse case** --
`shape_slat_flow_model_512`'s `SparseMultiHeadAttention`, running through this project's own
flex_gemm-equivalent Metal path on variable-length sparse token sets, is architecturally different
enough (no dense/padded attention, real sparse GEMM dispatch) that the original project's findings
are a starting hypothesis, not a settled answer, for this specific case. That's the concrete,
evidence-backed next step if kernel work is pursued here -- not a guess, a direct consequence of
this measurement.

## Update: `flex_gemm_sparse_attn` is a severe regression at real shapes, not a free win

Started the sparse-attention investigation flagged above. Before writing any new kernel code, found
something that changed the whole question: **every script this session (and the real T_run1
generation) has been forcing `ATTN_BACKEND=sdpa`** -- a convention copied verbatim from
trellis-mac-mps's env var setup, without checking that `trellis2/modules/sparse/config.py`'s real,
separate `SPARSE_ATTN_BACKEND` falls back to `ATTN_BACKEND` when unset. This project's own
capability probe (`docs/probe_m5_max_metal_full_2026.json`) had already confirmed a real, compiled
Metal kernel -- `flex_gemm_sparse_attn`, backed by `flex_gemm.kernels.metal.sparse_attention_fwd`
(a real `.metallib` + compiled ObjC++ binding, confirmed by reading the installed package directly,
not assumed -- a genuine flash-attention-v2-style kernel per `full_attn.py`'s own comment:
`simdgroup_matrix_multiply_accumulate` for QK^T/PV, `simd_shuffle_xor` for online-softmax) -- is
auto-selected as the *default* whenever `ATTN_BACKEND` isn't forced. That kernel had never actually
been benchmarked against the `sdpa` fallback at real shapes, on the real most-expensive model
(`shape_slat_flow_model_512`) -- worth checking before assuming either that the accidental default
was right, or that new kernel work was needed.

**Result (`scripts/investigate_flex_gemm_sparse_attn.py`, real DINOv3 conditioning, real sparse
coords from an actual sparse-structure sample -- 4362 real tokens, 12 heads):**

- **Correctness: fine.** `rel_error=0.000001` against the `sdpa` reference at the real shape (the
  probe's own correctness check only used a small 16-token synthetic case; this confirms it holds
  at production scale too).
- **Speed: a severe regression, not a win.** Isolated kernel call: `sdpa` 13.04ms vs.
  `flex_gemm_sparse_attn` 608.25ms -- **~47x slower**. Real full-model forward pass (the number
  that actually matters, per this project's own methodology): `sdpa` 796.05ms vs.
  `flex_gemm_sparse_attn` **28,614.46ms -- ~36x slower**, turning a sub-second forward pass into a
  nearly 30-second one.

**Root cause not diagnosed here -- an open question, not a guess dressed up as an answer.**
Plausible candidates: a large fixed per-call dispatch/compile overhead that would only amortize at
a much larger token count than this model's real ~4300-4400 tokens; the kernel tuned/validated at a
very different shape regime than what this model actually produces; or a genuine launch-
configuration inefficiency in the Metal kernel itself. Not investigated further this session --
would need the same kind of Metal-source-level investigation the original project applied to its
own flash-attention kernel work (`investigate_flash_attention_tile_sizes.py`, tile-size sweeps,
reading the actual kernel), which is out of scope unless this becomes a priority.

**Practical implication: no bug to fix in this project's own scripts.** Forcing `ATTN_BACKEND=sdpa`
in every script this session, and in the real `T_run1` generation, was -- by pure accident of
copying the original project's env var convention -- the *correct* choice all along. Leaving it
unset (to get the real auto-detected default) would have silently made every real generation on
this port ~36x slower on its most expensive stage. This project's scripts should keep forcing
`ATTN_BACKEND=sdpa` explicitly going forward, not treat this as something to "fix."

**Worth reporting upstream**, not yet done: whoever maintains `flex_gemm_sparse_attn` in the
Jourloy/PR#175 lineage has their own auto-detection logic silently picking the ~36x-slower backend
by default on this hardware/shape regime -- a real, concrete, actionable finding for that fork, not
just useful internally here.

## Update: root-caused and fixed the `flex_gemm_sparse_attn` regression -- real improvement, still not a win

Found the actual root cause by reading the vendored kernel source directly (not guessed) --
`flex_gemm` is built from source at `~/.cache/trellis2/source-deps/mtlgemm-<sha>/` (from
`pedronaugusto/mtlgemm`, pinned by `scripts/setup_macos.sh`'s `MTLGEMM_SHA`), so the real `.metal`
kernel and its C++ dispatcher (`ext.mm`) were both available to inspect, not just the compiled
binary.

**Two-part bug, both confirmed by reading the code, not assumed:**
1. `ext.mm`'s dispatcher gates the fast tiled flash-attention-v2 kernel behind a hardcoded
   `C_q <= 64` check, with a comment stating this was sized for **fp32** ("fp32/head_dim=64 uses
   ~28KB of the 32KB limit"). TRELLIS.2's real `shape_slat_flow_model_512` runs in **bfloat16**
   with **head_dim=128** (confirmed directly: `model_channels=1536, num_heads=12 -> head_dim=128`)
   -- so every real call silently fell through to the "naive per-thread-serial-KV kernel," which
   the code's own comment documents as losing "asymptotically as max_seqlen grows." That's the
   entire 36x regression documented in the update above.
2. Even after loosening the gate, `sparse_attn_tiled.metal`'s per-thread output accumulator was a
   compile-time-sized register array, `simdgroup_matrix<float,8,8> o_acc[MAX_HEAD_DIM / 8]`, with
   `MAX_HEAD_DIM` hardcoded to 64 (only 8 slots) -- too small for head_dim=128 (needs 16). The
   kernel's actual runtime loop logic (`n_tiles_cv = C_v / 8`, cooperative smem loads sized from
   `C_q`/`C_v` via a raw `threadgroup uchar*` pointer with runtime-computed offsets) was already
   fully generic -- only this one compile-time array size needed to change.

**The fp32 cutoff wasn't a real hardware constraint, confirmed by direct measurement, not
assumption**: queried this machine's actual `MTLDevice.maxThreadgroupMemoryLength` directly (a
small compiled Obj-C probe) -- **32768 bytes** on the real M5 Max. At bf16/head_dim=128 the same
tile configuration (`BLOCK_Q=16, BLOCK_KV=32`) needs only **~31.1KB** -- under budget, just with
less headroom than fp32/head_dim=64's ~28KB.

**Fix applied** (both files at
`~/.cache/trellis2/source-deps/mtlgemm-867aec8234299a7fe1ede7f802c8debe5a939a82/`, saved durably as
`patches/mtlgemm-867aec82-flash-attn-head-dim-128.patch` in this repo since the source-deps cache
dir is NOT version-controlled and a fresh `scripts/setup_macos.sh` run would silently re-clone the
pristine, unfixed source):
- `sparse_attn_tiled.metal`: `#define MAX_HEAD_DIM 64` -> `128`.
- `ext.mm`: `flash_eligible` changed from a hardcoded `C_q <= 64` to an actual computed
  shared-memory-footprint check (`shared_mem <= 32768`, dtype-aware via the existing `elem_bytes`
  calculation) -- correctly still restricts fp32 to a smaller effective head_dim while allowing
  fp16/bf16 up to 128, rather than one fixed number for all dtypes.
- Rebuilt cleanly via `pip install --no-build-isolation <source-deps path>`, no build errors.

**Result, verified with a clean, controlled, single-shape comparison (not the noisier full-pipeline
run, which showed real run-to-run baseline variance -- see caveat below)**: real bf16, head_dim=128,
3416 tokens, `FLEX_GEMM_ATTN_KERNEL` forced explicitly per variant to isolate the comparison from
gating logic:

| kernel | time/call |
|---|---|
| naive (old default for this shape) | 517.76ms |
| **tiled (now reachable after the fix)** | **126.64ms -- 4.1x faster than naive** |
| native `sdpa` | 13.77ms |

**The fix is real and verified** -- correctness held (`rel_error=0.000001` against `sdpa`,
unchanged after the fix), and the tiled kernel is genuinely, substantially faster than the naive
fallback it was incorrectly excluded from. Real full-model forward-pass timing corroborates this
independently: 28,614ms (pre-fix, naive) -> 6,380ms (post-fix, tiled) -- a ~4.5x improvement,
consistent with the isolated tiled-vs-naive ratio above.

**But it still does not beat native `sdpa`.** Even fixed and correctly routed to the fast kernel,
`flex_gemm_sparse_attn` remains roughly **9x slower in isolation** and **~2.8x-8x slower
full-model** (the full-model gap's range reflects real, unexplained `sdpa` baseline variance
between runs -- 796ms in one run, 2,304ms in another, same shape family, not yet root-caused;
flagged honestly rather than cherry-picking the more favorable comparison). This project's own
practical choice -- forcing `ATTN_BACKEND=sdpa` -- remains correct. The value of this fix is real
but narrower than "closes the gap": it makes the tiled kernel usable and ~4x faster than its own
broken fallback for anyone who needs it (e.g. `FLEX_GEMM_ATTN_KERNEL=tiled` explicitly, or future
hardware/shapes where the gap to `sdpa` might close), not a reason to switch this project's default.

**Worth reporting upstream to `pedronaugusto/mtlgemm`** -- a real, verified bug (confirmed
correctness, confirmed ~4x internal speedup from the fix) affecting anyone running this kernel on
Apple Silicon with head_dim=128 half-precision models, which is a common shape (many models use
128-dim heads). Not yet done -- the fix is saved as a patch in this repo; opening an upstream
issue/PR is a reasonable next step but wasn't done without explicit direction to do so.
