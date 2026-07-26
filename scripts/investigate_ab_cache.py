#!/usr/bin/env python3
"""
Benchmark ABCacheDiffFlowEulerGuidanceIntervalSampler (order-k Adams-Bashforth extrapolation on
the CFG diff, replacing the zero-order hold in DiffCachedCfgFlowEulerGuidanceIntervalSampler) at
ab_order=1 (sanity check: must reproduce the zero-order-hold numbers from
investigate_cfg_caching_diff.py), ab_order=2, and ab_order=3, all at neg_cache_interval=2 -- same
setting that gave 1.19-1.26x speedup / ~0.44-0.45 rel_error with zero-order hold. See this
project's CLAUDE.md for the AB-Cache correction (fixed schedule + higher-order extrapolation, not
adaptive/error-gated) this script's design is based on.
"""
import os
import sys
import time

os.environ.setdefault("SPARSE_BACKEND", "pytorch")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("FLEX_GEMM_QUIET", "1")

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trellis2.model_revisions import TRELLIS_REPO, TRELLIS_REVISION
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.pipelines.samplers import ABCacheDiffFlowEulerGuidanceIntervalSampler

CACHE_DIR = os.path.expanduser("~/.cache/trellis2/huggingface")
IMAGE_PATH = "assets/example_image/T.png"


def load_pipeline():
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
        TRELLIS_REPO, revision=TRELLIS_REVISION, cache_dir=CACHE_DIR, local_files_only=True,
    )
    pipeline.low_vram = False
    pipeline.to(torch.device("mps"))
    return pipeline


def make_ab_sampler(base_sampler, neg_cache_interval, ab_order):
    return ABCacheDiffFlowEulerGuidanceIntervalSampler(
        sigma_min=base_sampler.sigma_min,
        neg_cache_interval=neg_cache_interval,
        ab_order=ab_order,
    )


def run_sparse_structure(pipeline, cond, sampler, seed):
    flow_model = pipeline.models['sparse_structure_flow_model']
    reso, in_channels = flow_model.resolution, flow_model.in_channels
    gen = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(1, in_channels, reso, reso, reso, generator=gen).to(pipeline.device)
    params = dict(pipeline.sparse_structure_sampler_params)
    t0 = time.time()
    out = sampler.sample(flow_model, noise, **cond, **params, verbose=False)
    torch.mps.synchronize()
    elapsed = time.time() - t0
    return out.samples, elapsed


def run_shape_slat(pipeline, cond, coords, sampler, seed):
    from trellis2.modules.sparse import SparseTensor
    flow_model = pipeline.models['shape_slat_flow_model_512']
    gen = torch.Generator(device="cpu").manual_seed(seed)
    noise_feats = torch.randn(coords.shape[0], flow_model.in_channels, generator=gen).to(pipeline.device)
    noise = SparseTensor(feats=noise_feats, coords=coords)
    params = dict(pipeline.shape_slat_sampler_params)
    t0 = time.time()
    out = sampler.sample(flow_model, noise, **cond, **params, verbose=False)
    torch.mps.synchronize()
    elapsed = time.time() - t0
    return out.samples, elapsed


def rel_error(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp(min=1e-8)).item()


def main():
    print("Loading pipeline (offline, cached weights)...")
    pipeline = load_pipeline()
    print(f"  loaded. device={pipeline.device}")

    image = Image.open(IMAGE_PATH)
    print(f"Preprocessing {IMAGE_PATH} and computing real DINOv3 conditioning...")
    proc = pipeline.preprocess_image(image)
    cond = pipeline.get_cond([proc], 512)
    print("  cond computed.")

    base_ss_sampler = pipeline.sparse_structure_sampler
    base_shape_sampler = pipeline.shape_slat_sampler

    print("\n=== Baseline (sparse_structure) ===")
    z_base, t_base = run_sparse_structure(pipeline, cond, base_ss_sampler, seed=0)
    print(f"  baseline: {t_base:.2f}s")

    results = {}
    for order in (1, 2, 3):
        print(f"\n=== sparse_structure: ABCache order={order}, neg_cache_interval=2 ===")
        sampler = make_ab_sampler(base_ss_sampler, neg_cache_interval=2, ab_order=order)
        z, t = run_sparse_structure(pipeline, cond, sampler, seed=0)
        err = rel_error(z, z_base)
        speedup = t_base / t if t > 0 else float('nan')
        print(f"  time={t:.2f}s speedup={speedup:.3f}x rel_error={err:.6f}")
        print(f"  neg_recompute={sampler.neg_recompute_count} "
              f"reuse_ab={sampler.neg_reuse_ab_count} reuse_hold={sampler.neg_reuse_hold_count}")
        results[('sparse_structure', order)] = (speedup, err)

    print("\n=== sparse_structure SUMMARY (order=1 should ~match prior zero-order-hold rel_error=0.439216) ===")
    for order in (1, 2, 3):
        speedup, err = results[('sparse_structure', order)]
        print(f"  order={order}: speedup={speedup:.3f}x rel_error={err:.6f}")

    # Only take the expensive shape_slat stage to the order that looks most promising (or order=2
    # as the paper's own most commonly used default) -- running all 3 orders on shape_slat would be
    # ~4 x (baseline + 3 orders) x ~15s/step x 12 steps, too slow to justify before sparse_structure
    # already tells us whether higher order actually reduces error here.
    best_order = min((2, 3), key=lambda o: results[('sparse_structure', o)][1])
    print(f"\nBest higher order on sparse_structure by rel_error: order={best_order}. "
          f"Running shape_slat (expensive stage) at order=1 (sanity) and order={best_order}.")

    decoder = pipeline.models['sparse_structure_decoder']
    SS_RES = 32
    with torch.no_grad():
        decoded_base = decoder(z_base) > 0
    if SS_RES != decoded_base.shape[2]:
        ratio = decoded_base.shape[2] // SS_RES
        decoded_base = torch.nn.functional.max_pool3d(decoded_base.float(), ratio, ratio, 0) > 0.5
    coords = torch.argwhere(decoded_base)[:, [0, 2, 3, 4]].int().to(pipeline.device)
    print(f"  using {coords.shape[0]} coords derived from the baseline sparse-structure decode")

    slat_base, t_slat_base = run_shape_slat(pipeline, cond, coords, base_shape_sampler, seed=1)
    print(f"\n=== shape_slat baseline: {t_slat_base:.2f}s ===")

    for order in (1, best_order):
        print(f"\n=== shape_slat: ABCache order={order}, neg_cache_interval=2 ===")
        sampler = make_ab_sampler(base_shape_sampler, neg_cache_interval=2, ab_order=order)
        slat, t = run_shape_slat(pipeline, cond, coords, sampler, seed=1)
        err = rel_error(slat.feats, slat_base.feats)
        speedup = t_slat_base / t if t > 0 else float('nan')
        print(f"  time={t:.2f}s speedup={speedup:.3f}x rel_error(feats)={err:.6f}")
        print(f"  neg_recompute={sampler.neg_recompute_count} "
              f"reuse_ab={sampler.neg_reuse_ab_count} reuse_hold={sampler.neg_reuse_hold_count}")
        results[('shape_slat', order)] = (speedup, err)

    print("\n=== FINAL SUMMARY ===")
    for (stage, order), (speedup, err) in sorted(results.items()):
        print(f"  {stage} order={order}: speedup={speedup:.3f}x rel_error={err:.6f}")
    print("\nCompare to zero-order-hold DiffCachedCfgFlowEulerGuidanceIntervalSampler at "
          "neg_cache_interval=2: sparse_structure 1.187x/0.439216, shape_slat 1.261x/0.451514.")


if __name__ == "__main__":
    main()
