#!/usr/bin/env python3
"""
Real finding this investigation started from: every script in this session (and the real T_run1
generation) forced ATTN_BACKEND=sdpa -- a convention copied from the original trellis-mac-mps
project's env var setup, WITHOUT checking whether it also silently overrides
trellis2/modules/sparse/config.py's SPARSE_ATTN_BACKEND (it falls back to ATTN_BACKEND when unset).
This project's own capability probe (docs/probe_m5_max_metal_full_2026.json) already confirmed a
real, compiled Metal kernel is available and auto-selected as the default when ATTN_BACKEND isn't
forced: "flex_gemm_sparse_attn" (flex_gemm.kernels.metal.sparse_attention_fwd, a real
flash-attention-v2-style Metal kernel -- confirmed by reading trellis2/modules/sparse/attention/
full_attn.py directly, not assumed). This script checks, for the first time this session, whether
that already-built kernel is actually faster than the sdpa-with-padding fallback we've been using
by accident -- BEFORE writing any new kernel code, per this project's own "measure before
optimizing" discipline.

sdpa's real cost here isn't just SDPA itself: for variable-length sparse token sequences, the sdpa
path (full_attn.py) pads into a dense [N, max_len, H, C] tensor with a float attention mask, calls
native SDPA, then unpads -- real, measurable overhead flex_gemm_sparse_attn's ragged/cu_seqlens
kernel avoids entirely. Correctness of flex_gemm_sparse_attn was already confirmed by the capability
probe itself (torch.allclose vs. SDPA reference at a small synthetic shape, rtol/atol 2e-2,
"metal_attention_parity": true) -- this script focuses on the real-shape speed question, which has
never been checked, plus a real-shape finite-output sanity check (small synthetic shapes aren't
always representative -- this project's own established lesson).
"""
import os
import sys
import time

os.environ.setdefault("SPARSE_BACKEND", "pytorch")
os.environ.setdefault("ATTN_BACKEND", "sdpa")  # dense model still uses this; sparse model's ATTN is overridden live below
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("FLEX_GEMM_QUIET", "1")

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trellis2.model_revisions import TRELLIS_REPO, TRELLIS_REVISION
from trellis2.pipelines import Trellis2ImageTo3DPipeline
import trellis2.modules.sparse.config as sparse_config
from trellis2.modules.sparse import SparseTensor
from trellis2.modules.sparse.attention.modules import SparseMultiHeadAttention
from trellis2.modules.sparse.attention.full_attn import sparse_scaled_dot_product_attention

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


def main():
    print("Loading pipeline (offline, cached weights)...")
    pipeline = load_pipeline()
    print(f"  loaded. device={pipeline.device}")

    image = Image.open(IMAGE_PATH)
    proc = pipeline.preprocess_image(image)
    cond = pipeline.get_cond([proc], 512)
    print("  real DINOv3 cond computed.")

    ss_model = pipeline.models['sparse_structure_flow_model']
    reso, in_ch = ss_model.resolution, ss_model.in_channels
    x_ss = torch.randn(1, in_ch, reso, reso, reso, device=pipeline.device)
    print("  sampling real sparse structure for real coords...")
    ss_out = pipeline.sparse_structure_sampler.sample(
        ss_model, x_ss, **cond, **pipeline.sparse_structure_sampler_params, verbose=False,
    )
    decoder = pipeline.models['sparse_structure_decoder']
    with torch.no_grad():
        decoded = decoder(ss_out.samples) > 0
    SS_RES = 32
    if SS_RES != decoded.shape[2]:
        ratio = decoded.shape[2] // SS_RES
        decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
    coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int().to(pipeline.device)
    n_tokens = coords.shape[0]
    print(f"  {n_tokens} real sparse coords/tokens")

    shape_model = pipeline.models['shape_slat_flow_model_512']
    feats = torch.randn(n_tokens, shape_model.in_channels, device=pipeline.device)
    x_shape = SparseTensor(feats=feats, coords=coords)
    t_shape = torch.full((1,), 500.0, device=pipeline.device)

    print(f"\n=== 1. Isolated kernel-level benchmark (real shape: {n_tokens} tokens, "
          f"{shape_model.num_heads} heads, this model's real head_dim) ===")
    real_block = shape_model.blocks[0]
    head_dim = real_block.self_attn.head_dim
    num_heads = real_block.self_attn.num_heads
    gen = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn(n_tokens, num_heads, head_dim, dtype=torch.float32, generator=gen).to(pipeline.device)
    k = torch.randn(n_tokens, num_heads, head_dim, dtype=torch.float32, generator=gen).to(pipeline.device)
    v = torch.randn(n_tokens, num_heads, head_dim, dtype=torch.float32, generator=gen).to(pipeline.device)
    qv = SparseTensor(feats=q, coords=coords)
    kv_ = SparseTensor(feats=k, coords=coords)
    vv = SparseTensor(feats=v, coords=coords)

    for backend in ("sdpa", "flex_gemm_sparse_attn"):
        sparse_config.set_attn_backend(backend)
        fn = lambda: sparse_scaled_dot_product_attention(qv, kv_, vv)
        t = timed(fn)
        out = fn()
        finite = torch.isfinite(out.feats).all().item()
        print(f"  {backend:22s}: {t*1000:.3f}ms/call, finite={finite}")

    print(f"\n=== 2. Correctness check at REAL shape (not the probe's small 16-token synthetic case) ===")
    sparse_config.set_attn_backend('sdpa')
    out_sdpa = sparse_scaled_dot_product_attention(qv, kv_, vv)
    sparse_config.set_attn_backend('flex_gemm_sparse_attn')
    out_flex = sparse_scaled_dot_product_attention(qv, kv_, vv)
    diff = (out_sdpa.feats.float() - out_flex.feats.float())
    rel_err = (diff.norm() / out_sdpa.feats.float().norm().clamp(min=1e-8)).item()
    max_abs = diff.abs().max().item()
    print(f"  rel_error={rel_err:.6f}, max_abs_diff={max_abs:.6f} "
          f"(probe's own synthetic check used rtol/atol=2e-2 -- compare against that bar)")

    print(f"\n=== 3. Real full-model forward-pass benchmark (shape_slat_flow_model_512, "
          f"{n_tokens} real tokens) ===")
    for backend in ("sdpa", "flex_gemm_sparse_attn"):
        sparse_config.set_attn_backend(backend)
        fn = lambda: shape_model(x_shape, t_shape, cond['cond'])
        t = timed(fn, n=8, warmup=3)
        print(f"  {backend:22s}: {t*1000:.2f}ms/forward")

    print("\n=== SUMMARY ===")
    print("See above for isolated-kernel, correctness, and real full-model numbers.")
    print("Per this project's established methodology: only the full-model number in section 3 "
          "is the one that matters for a speedup claim.")


if __name__ == "__main__":
    main()
