#!/usr/bin/env python3
"""
Root-causes why native scaled_dot_product_attention loses to a naive hand-rolled reimplementation
by ~3.3x on TRELLIS.2's sparse_structure_flow_model real self-attention shape
((1, 4096, 12, 128), bf16, post-RoPE) -- see CLAUDE.md's "dense-attention findings do NOT simply
transfer" update for the original finding this investigates.

Isolates one variable at a time against synthetic tensors (not the real model, so each sweep point
is cheap and independently controlled), matching this project's established methodology (dtype
sweeps, tile-size sweeps in the original trellis-mac-mps project). Candidate hypotheses tested, in
order of cheapest-to-rule-out first:
  1. Contiguity: are the REAL post-RoPE q/k tensors non-contiguous, and does that alone explain it?
  2. Is the gap present even with plain random (non-RoPE) tensors at the same shape/dtype? (If yes,
     RoPE itself isn't the cause -- it's the shape/dtype combination.)
  3. head_dim sweep at fixed seq_len=4096, bf16, contiguous.
  4. seq_len sweep at fixed head_dim=128, bf16, contiguous.
  5. dtype sweep (fp32/fp16/bf16) at the exact real shape.
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

DEVICE = "mps"
N_TIMED = 20
N_WARMUP = 5


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


def hand_rolled(q, k, v, scale):
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = attn.softmax(dim=-1)
    return torch.matmul(attn, v)


def make_qkv(B, H, L, D, dtype, contiguous=True, seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(B, H, L, D, generator=gen, dtype=torch.float32).to(dtype).to(DEVICE)
    k = torch.randn(B, H, L, D, generator=gen, dtype=torch.float32).to(dtype).to(DEVICE)
    v = torch.randn(B, H, L, D, generator=gen, dtype=torch.float32).to(dtype).to(DEVICE)
    if not contiguous:
        # Simulate a non-contiguous layout the way RoPE's rotate-half + cat/view pattern can leave
        # behind: allocate a wider buffer and take a strided slice.
        def stride_it(t):
            buf = torch.randn(B, H, L, D * 2, generator=gen, dtype=torch.float32).to(dtype).to(DEVICE)
            return buf[..., :D]
        q, k, v = stride_it(q), stride_it(k), stride_it(v)
    return q, k, v


def bench_pair(q, k, v, label):
    scale = 1.0 / (q.shape[-1] ** 0.5)
    t_sdpa = timed(lambda: F.scaled_dot_product_attention(q, k, v))
    t_hand = timed(lambda: hand_rolled(q, k, v, scale))
    ratio = t_sdpa / t_hand
    print(f"  {label:48s}: sdpa={t_sdpa*1000:8.3f}ms  hand={t_hand*1000:8.3f}ms  "
          f"sdpa/hand={ratio:.3f}x  q.is_contiguous={q.is_contiguous()}")
    return t_sdpa, t_hand


def main():
    print("=== 0. Reproduce the real captured shape/dtype with SYNTHETIC contiguous random tensors ===")
    print("    (isolates: is this about RoPE specifically, or just the shape+dtype?)")
    q, k, v = make_qkv(1, 12, 4096, 128, torch.bfloat16, contiguous=True)
    bench_pair(q, k, v, "synthetic random, contiguous, bf16, (1,12,4096,128)")

    print("\n=== 1. Non-contiguous q/k/v at the same shape (simulating RoPE's strided output) ===")
    q2, k2, v2 = make_qkv(1, 12, 4096, 128, torch.bfloat16, contiguous=False)
    bench_pair(q2, k2, v2, "synthetic random, NON-contiguous, bf16, (1,12,4096,128)")
    q2c, k2c, v2c = q2.contiguous(), k2.contiguous(), v2.contiguous()
    bench_pair(q2c, k2c, v2c, "same tensors, .contiguous()'d first")

    print("\n=== 2. head_dim sweep (fixed seq_len=4096, heads scaled to keep total channels ~1536, bf16) ===")
    for head_dim in (32, 48, 64, 80, 96, 112, 128, 160, 192, 256):
        heads = max(1, 1536 // head_dim)
        q, k, v = make_qkv(1, heads, 4096, head_dim, torch.bfloat16)
        bench_pair(q, k, v, f"head_dim={head_dim:4d}, heads={heads:3d}, seq_len=4096")

    print("\n=== 3. seq_len sweep (fixed head_dim=128, heads=12, bf16) ===")
    for seq_len in (256, 512, 1024, 2048, 3072, 4096, 6144, 8192):
        q, k, v = make_qkv(1, 12, seq_len, 128, torch.bfloat16)
        bench_pair(q, k, v, f"seq_len={seq_len:5d}, head_dim=128, heads=12")

    print("\n=== 4. dtype sweep (exact real shape: heads=12, seq_len=4096, head_dim=128) ===")
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        q, k, v = make_qkv(1, 12, 4096, 128, dtype)
        bench_pair(q, k, v, f"dtype={str(dtype):20s}")

    print("\n=== SUMMARY ===")
    print("Look for where sdpa/hand crosses 1.0x (sdpa loses above 1.0, wins below 1.0) in each")
    print("sweep to localize which variable(s) actually drive the regression.")


if __name__ == "__main__":
    main()
