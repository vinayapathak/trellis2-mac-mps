#!/usr/bin/env python3
"""
Checks whether the original trellis-mac-mps project's dense-attention findings (native SDPA beats
a hand-rolled 3-op reimplementation by 1.43x; all torch.nn.attention.sdpa_kernel backend overrides
resolve identically on MPS) transfer to TRELLIS.2's sparse_structure_flow_model, per this project's
CLAUDE.md item 1 follow-up -- do not assume, check.

Does NOT attempt to port the original project's quantized-attention (matmul2d int8) kernel here:
confirmed first, by reading the real model config directly, that TRELLIS.2's
sparse_structure_flow_model uses use_rope=True/pe_mode='rope', whereas the original TRELLIS's
model used use_rope=False -- and the original project's QuantizedMultiHeadAttentionMatmul2D
explicitly raises NotImplementedError for use_rope=True (a deliberate, non-silent guard, not an
oversight). Porting that kernel to support RoPE is real, un-started engineering work, not a
same-day check -- out of scope here. This script covers the two checks that ARE directly portable
without new kernel infrastructure.
"""
import os
import sys
import time

os.environ.setdefault("SPARSE_BACKEND", "pytorch")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("FLEX_GEMM_QUIET", "1")

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trellis2.model_revisions import TRELLIS_REPO, TRELLIS_REVISION
from trellis2.pipelines import Trellis2ImageTo3DPipeline

CACHE_DIR = os.path.expanduser("~/.cache/trellis2/huggingface")
IMAGE_PATH = "assets/example_image/T.png"
N_TIMED = 15
N_WARMUP = 4


def load_pipeline():
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
        TRELLIS_REPO, revision=TRELLIS_REVISION, cache_dir=CACHE_DIR, local_files_only=True,
    )
    pipeline.low_vram = False
    pipeline.to(torch.device("mps"))
    return pipeline


def timed(fn, n=N_TIMED, warmup=N_WARMUP):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.mps.synchronize()
        t0 = time.time()
        for _ in range(n):
            fn()
        torch.mps.synchronize()
        return (time.time() - t0) / n


def hand_rolled_attention(q, k, v, scale):
    # Direct 3-op reimplementation: QK^T -> softmax -> @V. No fused SDPA kernel involved.
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = attn.softmax(dim=-1)
    return torch.matmul(attn, v)


def main():
    print("Loading pipeline (offline, cached weights)...")
    pipeline = load_pipeline()
    print(f"  loaded. device={pipeline.device}")

    m = pipeline.models['sparse_structure_flow_model']
    b0 = m.blocks[0]
    print(f"  config: model_channels={m.model_channels}, num_heads={m.num_heads}, "
          f"head_dim={b0.self_attn.head_dim}, use_rope={b0.self_attn.use_rope}, "
          f"qk_rms_norm={b0.self_attn.qk_rms_norm}, dtype={m.dtype}, num_blocks={m.num_blocks}")

    image = Image.open(IMAGE_PATH)
    proc = pipeline.preprocess_image(image)
    cond = pipeline.get_cond([proc], 512)
    print("  real DINOv3 cond computed.")

    reso, in_ch = m.resolution, m.in_channels
    x = torch.randn(1, in_ch, reso, reso, reso, device=pipeline.device)
    t = torch.full((1,), 500.0, device=pipeline.device)

    # -------------------------------------------------------------------
    # Check 1: hand-rolled 3-op attention vs native SDPA, at this model's
    # REAL self-attention shape (post-RoPE Q/K, since that's what actually
    # feeds the attention op in the real forward pass) -- captured directly
    # from a real forward pass by monkeypatching the actual attention
    # function this module calls, rather than hand-replaying its internal
    # to_qkv/rms_norm/rope pipeline (fragile, already got this wrong once).
    # -------------------------------------------------------------------
    print("\n=== 1. Hand-rolled 3-op attention vs native SDPA (real self-attn shape, post-RoPE) ===")
    # modules.py does `from .full_attn import scaled_dot_product_attention`, which creates its own
    # local binding -- patching full_attn's namespace doesn't affect modules.py's calls. Patch the
    # name where it's actually looked up.
    import trellis2.modules.attention.modules as dense_modules
    orig_sdpa_fn = dense_modules.scaled_dot_product_attention
    captured = {}

    def capturing_sdpa(*args, **kwargs):
        if 'args' not in captured:
            captured['args'] = args
        return orig_sdpa_fn(*args, **kwargs)

    dense_modules.scaled_dot_product_attention = capturing_sdpa
    try:
        with torch.no_grad():
            m(x, t, cond['cond'])
    finally:
        dense_modules.scaled_dot_product_attention = orig_sdpa_fn

    q, k, v = captured['args']  # first block's self-attn call: (q, k, v), already post-RMSNorm+RoPE
    print(f"  real post-RoPE Q/K/V shape: {tuple(q.shape)}, dtype={q.dtype}")

    scale = 1.0 / (b0.self_attn.head_dim ** 0.5)
    q_t = q.permute(0, 2, 1, 3).contiguous()
    k_t = k.permute(0, 2, 1, 3).contiguous()
    v_t = v.permute(0, 2, 1, 3).contiguous()

    t_sdpa = timed(lambda: F.scaled_dot_product_attention(q_t, k_t, v_t))
    t_hand = timed(lambda: hand_rolled_attention(q_t, k_t, v_t, scale))
    out_sdpa = F.scaled_dot_product_attention(q_t, k_t, v_t)
    out_hand = hand_rolled_attention(q_t, k_t, v_t, scale)
    rel_err = ((out_sdpa.float() - out_hand.float()).norm() / out_sdpa.float().norm().clamp(min=1e-8)).item()
    print(f"  native sdpa:  {t_sdpa*1000:.3f}ms/call")
    print(f"  hand-rolled:  {t_hand*1000:.3f}ms/call")
    print(f"  speedup (sdpa vs hand-rolled): {t_hand/t_sdpa:.3f}x")
    print(f"  rel_error(sdpa vs hand-rolled output): {rel_err:.6f}")

    # -------------------------------------------------------------------
    # Check 2: sdpa_kernel backend override sweep, at the real full-model
    # level (the number that actually matters, per this project's own
    # "isolated vs real-model" lesson).
    # -------------------------------------------------------------------
    print("\n=== 2. sdpa_kernel backend override sweep (real full-model forward, sparse_structure_flow_model) ===")
    backends = {
        'default (no override)': None,
        'MATH': SDPBackend.MATH,
        'EFFICIENT_ATTENTION': SDPBackend.EFFICIENT_ATTENTION,
        'FLASH_ATTENTION': SDPBackend.FLASH_ATTENTION,
        'CUDNN_ATTENTION': SDPBackend.CUDNN_ATTENTION,
        'OVERRIDEABLE': SDPBackend.OVERRIDEABLE,
    }
    fn = lambda: m(x, t, cond['cond'])
    for name, backend in backends.items():
        try:
            if backend is None:
                t_b = timed(fn)
            else:
                with sdpa_kernel(backend):
                    t_b = timed(fn)
            print(f"  {name:24s}: {t_b*1000:.2f}ms/forward")
        except Exception as e:
            print(f"  {name:24s}: FAILED ({type(e).__name__}: {e})")

    print("\n=== SUMMARY ===")
    print("Check 1 tests whether native SDPA still beats a hand-rolled reimplementation on this")
    print("model's real (RoPE-included) attention shape -- the original project found 1.43x.")
    print("Check 2 tests whether SDPA backend overrides still resolve identically -- the original")
    print("project found they all converge to the same MPS implementation.")


if __name__ == "__main__":
    main()
