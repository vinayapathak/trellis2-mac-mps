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
