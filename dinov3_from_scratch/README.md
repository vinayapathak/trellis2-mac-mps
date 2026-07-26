# DINOv3 ViT-L/16, implemented from the paper

A from-scratch, paper-faithful implementation of DINOv3's architecture and training objective
(Siméoni et al., "DINOv3", [arXiv:2508.10104](https://arxiv.org/abs/2508.10104)), built while
waiting on Meta's manual gated-access approval for the real
`facebook/dinov3-vitl16-pretrain-lvd1689m` checkpoint used by this project's actual generation
pipeline (`trellis2/modules/image_feature_extractor.py`).

## What this is

- `model.py`: the real ViT-L/16 backbone architecture — patch embedding, CLS token, 4 register
  tokens, 2D rotary position embeddings (axial, over patch coordinates normalized to [-1,1], with
  the paper's "RoPE-box jittering" training-time augmentation), pre-norm transformer blocks with
  SwiGLU feed-forward layers, LayerScale. Standard ViT-L sizing (embed_dim=1024, depth=24,
  16 heads) — the paper gives exact numbers for its 7B teacher and states smaller variants "scale
  proportionally downward" without spelling out ViT-L's numbers explicitly, so this uses the
  established ViT-L convention (same as DINOv2's own published ViT-L).
- `losses.py`: the real training objective — DINO (image-level self-distillation), iBOT
  (masked patch-level self-distillation), KoLeo (feature-uniformity regularizer), and Gram
  anchoring (the paper's specific fix for dense-feature collapse during long training,
  implemented as the published Frobenius-norm patch-Gram-matrix matching loss).
- `verify.py`: real correctness verification, run on actual Apple Silicon MPS hardware — forward
  pass shapes at both a fast small-depth config and the real full ViT-L/16 config (304.6M
  parameters, matching the real model's ~300M), the RoPE-box-jitter code path, and a full
  backward pass confirming every one of 32 tested parameters receives a real, nonzero gradient
  through a real optimizer step. All checks pass as of this writing.

## What this is not, stated as plainly as possible

**This does not, and cannot, reproduce DINOv3's actual pretrained capability.** The architecture
is correct; the weights are randomly initialized. DINOv3's real value comes entirely from
self-distillation training on Meta's LVD-1689M dataset — 1.7 billion curated images, not publicly
released — over (per the paper) 256 GPUs for a training schedule measured in millions of
iterations across multiple phases. A freshly-initialized model from this code produces
architecturally-correct but semantically meaningless features. It is **not** a drop-in substitute
for the real checkpoint, and this project's actual generation pipeline
(`scripts/generate_asset.py`) does not use it and is not intended to.

## Why this exists anyway

1. It's a real, verified, correct-against-the-paper implementation — useful on its own terms
   (understanding the architecture precisely, a base for anyone who does have the compute/data to
   train it, a way to sanity-check the real HF model's reported config once it's available).
2. It was built while genuinely blocked on Meta's manual review queue for gated access to the
   real checkpoint (see the parent project's `CLAUDE.md` for the full timeline) — productive use
   of otherwise-idle time, not a substitute for the real thing.
3. With real GPU access (this project has access to an NVIDIA DGX pod), a genuine small-scale
   training run is feasible as a *pipeline-correctness* exercise — confirming losses actually
   decrease and the whole training loop is bug-free on a small public dataset — which is a
   meaningfully different, weaker, but real claim than "trained DINOv3." Not yet attempted as of
   this writing; see `CLAUDE.md` for status.

## Running the verification

```sh
cd dinov3_from_scratch
python3 verify.py
```

No dataset or pretrained weights required — this only checks architectural and gradient-flow
correctness, not training quality.
