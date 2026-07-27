#!/usr/bin/env python3
"""
Follow-up to investigate_sdpa_regression_root_cause.py: that sweep found clean SYNTHETIC tensors
at the exact real shape ((1,12,4096,128) bf16) only show SDPA losing by ~1.3x, not the ~3.3x seen
with the REAL model's captured post-RoPE tensors in investigate_dense_attn_transfer.py. The
head_dim/seq_len sweeps also showed noisy, non-monotonic spikes (head_dim=48,112,160,192,256 all
jumping to ~3.2x while neighbors stayed ~1.0-1.5x) that look more like measurement noise/thermal
state than a real algorithmic cliff.

This script tests that directly: captures REAL q/k/v from the real model (as before), then
IMMEDIATELY afterward, in the SAME process/same thermal-and-memory state, benchmarks synthetic
random tensors at the identical shape/dtype/device -- isolating "is the 3.3x about the real
trained-model VALUES specifically, or just about system state that differs between separate
script runs?" Also repeats the real-vs-hand comparison multiple times within one run to directly
measure run-to-run variance.
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
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trellis2.model_revisions import TRELLIS_REPO, TRELLIS_REVISION
from trellis2.pipelines import Trellis2ImageTo3DPipeline

CACHE_DIR = os.path.expanduser("~/.cache/trellis2/huggingface")
IMAGE_PATH = "assets/example_image/T.png"
N_TIMED = 20
N_WARMUP = 5


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


def hand_rolled(q, k, v, scale):
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = attn.softmax(dim=-1)
    return torch.matmul(attn, v)


def bench_pair(q, k, v, label, n=N_TIMED):
    scale = 1.0 / (q.shape[-1] ** 0.5)
    t_sdpa = timed(lambda: F.scaled_dot_product_attention(q, k, v), n=n)
    t_hand = timed(lambda: hand_rolled(q, k, v, scale), n=n)
    print(f"  {label:50s}: sdpa={t_sdpa*1000:8.3f}ms  hand={t_hand*1000:8.3f}ms  "
          f"sdpa/hand={t_sdpa/t_hand:.3f}x")
    return t_sdpa, t_hand


def main():
    print("Loading pipeline (offline, cached weights)...")
    pipeline = load_pipeline()
    print(f"  loaded. device={pipeline.device}")

    m = pipeline.models['sparse_structure_flow_model']
    b0 = m.blocks[0]

    image = Image.open(IMAGE_PATH)
    proc = pipeline.preprocess_image(image)
    cond = pipeline.get_cond([proc], 512)
    print("  real DINOv3 cond computed.")

    reso, in_ch = m.resolution, m.in_channels
    x = torch.randn(1, in_ch, reso, reso, reso, device=pipeline.device)
    t = torch.full((1,), 500.0, device=pipeline.device)

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

    q_real, k_real, v_real = captured['args']
    q_real_t = q_real.permute(0, 2, 1, 3).contiguous()
    k_real_t = k_real.permute(0, 2, 1, 3).contiguous()
    v_real_t = v_real.permute(0, 2, 1, 3).contiguous()
    shape = tuple(q_real_t.shape)
    dtype = q_real_t.dtype
    print(f"  real captured shape={shape}, dtype={dtype}")

    print("\n=== A. REAL trained-model tensors, repeated 3x in this process (checks run-to-run variance) ===")
    for i in range(3):
        bench_pair(q_real_t, k_real_t, v_real_t, f"real tensors, trial {i+1}/3")

    print("\n=== B. SYNTHETIC random tensors, SAME shape/dtype/device, SAME process/thermal state ===")
    gen = torch.Generator(device="cpu").manual_seed(0)
    q_syn = torch.randn(*shape, generator=gen, dtype=torch.float32).to(dtype).to(pipeline.device)
    k_syn = torch.randn(*shape, generator=gen, dtype=torch.float32).to(dtype).to(pipeline.device)
    v_syn = torch.randn(*shape, generator=gen, dtype=torch.float32).to(dtype).to(pipeline.device)
    for i in range(3):
        bench_pair(q_syn, k_syn, v_syn, f"synthetic tensors, trial {i+1}/3")

    print("\n=== C. Real Q/K, but reshuffled/detached from their true statistics (shuffle along last dim) ===")
    # If the gap is about VALUE STATISTICS specifically (not just "trained vs random"), shuffling
    # elements within each real tensor destroys structure while keeping the exact same marginal
    # value distribution -- a finer-grained test than "real vs pure random."
    def shuffle_last_dim(t):
        idx = torch.randperm(t.shape[-1], device=t.device)
        return t[..., idx].contiguous()
    q_shuf = shuffle_last_dim(q_real_t)
    k_shuf = shuffle_last_dim(k_real_t)
    v_shuf = shuffle_last_dim(v_real_t)
    bench_pair(q_shuf, k_shuf, v_shuf, "real tensors, shuffled along head_dim")

    print("\n=== D. Value-range comparison: real vs synthetic ===")
    for name, tens in [("real q", q_real_t), ("synthetic q", q_syn)]:
        tf = tens.float()
        print(f"  {name:15s}: mean={tf.mean().item():+.4f} std={tf.std().item():.4f} "
              f"min={tf.min().item():+.4f} max={tf.max().item():+.4f} "
              f"contiguous={tens.is_contiguous()} stride={tens.stride()}")

    print("\n=== SUMMARY ===")
    print("If A's real-tensor sdpa/hand ratio varies a lot trial-to-trial, or if B's synthetic")
    print("ratio (same process/thermal state as A) is close to A's, that points to measurement")
    print("noise/system state rather than a real value-dependent SDPA regression. If A stays high")
    print("and consistent while B stays low and consistent IN THE SAME PROCESS, that's real signal")
    print("that the trained model's actual activation values specifically trigger a slow path.")


if __name__ == "__main__":
    main()
