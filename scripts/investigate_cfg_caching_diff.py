#!/usr/bin/env python3
"""
Verify + benchmark DiffCachedCfgFlowEulerGuidanceIntervalSampler
(trellis2/pipelines/samplers/flow_euler_cached_cfg.py) against the real, production
FlowEulerGuidanceIntervalSampler, on the real pretrained TRELLIS.2-4B model with real image
conditioning. Mirrors trellis-mac-mps's scripts/investigate_cfg_caching_diff.py methodology,
applied fresh to TRELLIS.2's sparse_structure_sampler and shape_slat_sampler (the two stages with
a real two-branch CFG call per step, per the real pipeline.json: guidance_strength=7.5,
guidance_interval=[0.6, 1.0]; tex_slat_sampler runs at guidance_strength=1.0 and never has a
negative branch to cache, so it's excluded).

Steps, in order:
  1. Correctness: neg_cache_interval=1 (recomputes every call -- functionally the no-caching case)
     against the real baseline sampler, on sparse_structure only (cheap, ~12 steps / ~15s) --
     should be numerically identical, since interval=1 never actually reuses a cached value.
  2. Real wall-clock + relative-error comparison at neg_cache_interval=2 vs baseline, on BOTH
     sparse_structure (cheap sanity data point) and shape_slat (the real payoff stage: ~30s/step
     baseline, 12 steps).
  3. A cheap structural proxy for output quality: decode both sparse-structure results to voxel
     coordinates and compare occupied-voxel counts (full mesh/PBR bake, as in T_run1, takes ~15
     more minutes and is not repeated here for every config -- see this script's printed caveats).
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
from trellis2.pipelines.samplers import DiffCachedCfgFlowEulerGuidanceIntervalSampler

CACHE_DIR = os.path.expanduser("~/.cache/trellis2/huggingface")
IMAGE_PATH = "assets/example_image/T.png"


def load_pipeline():
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
        TRELLIS_REPO, revision=TRELLIS_REVISION, cache_dir=CACHE_DIR, local_files_only=True,
    )
    # Real pipeline.json has no "low_vram" key -> defaults to True (models stay on CPU, moved
    # per-call inside sample_sparse_structure/sample_shape_slat). This script calls
    # sampler.sample() directly, bypassing those per-call moves, so force low_vram=False and let
    # Pipeline.to() move (and keep) every model resident on mps -- correct for a benchmarking
    # script where we don't care about VRAM footprint, and it keeps host<->device transfer
    # overhead out of the timing comparison for both configs equally anyway.
    pipeline.low_vram = False
    pipeline.to(torch.device("mps"))
    return pipeline


def make_cached_sampler(base_sampler, neg_cache_interval):
    return DiffCachedCfgFlowEulerGuidanceIntervalSampler(
        sigma_min=base_sampler.sigma_min, neg_cache_interval=neg_cache_interval,
    )


def run_sparse_structure(pipeline, cond, sampler, seed):
    flow_model = pipeline.models['sparse_structure_flow_model']
    reso, in_channels = flow_model.resolution, flow_model.in_channels
    gen = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(1, in_channels, reso, reso, reso, generator=gen).to(pipeline.device)
    params = dict(pipeline.sparse_structure_sampler_params)
    t0 = time.time()
    out = sampler.sample(
        flow_model, noise, **cond, **params, verbose=False,
    )
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
    out = sampler.sample(
        flow_model, noise, **cond, **params, verbose=False,
    )
    torch.mps.synchronize()
    elapsed = time.time() - t0
    return out.samples, elapsed


def rel_error(a, b):
    a = a.float()
    b = b.float()
    return ((a - b).norm() / b.norm().clamp(min=1e-8)).item()


def main():
    print("Loading pipeline (offline, cached weights)...")
    pipeline = load_pipeline()
    print(f"  loaded. device={pipeline.device}")

    image = Image.open(IMAGE_PATH)
    print(f"Preprocessing {IMAGE_PATH} and computing real DINOv3 conditioning...")
    proc = pipeline.preprocess_image(image)
    cond = pipeline.get_cond([proc], 512)  # matches pipeline.run()'s cond_512 = self.get_cond([image], 512)
    print("  cond computed.")

    base_ss_sampler = pipeline.sparse_structure_sampler
    base_shape_sampler = pipeline.shape_slat_sampler

    print("\n=== 1. Correctness check: neg_cache_interval=1 vs baseline (sparse_structure) ===")
    z_base, t_base = run_sparse_structure(pipeline, cond, base_ss_sampler, seed=0)
    cached1 = make_cached_sampler(base_ss_sampler, neg_cache_interval=1)
    z_c1, t_c1 = run_sparse_structure(pipeline, cond, cached1, seed=0)
    err1 = rel_error(z_c1, z_base)
    print(f"  baseline: {t_base:.2f}s | cached(interval=1): {t_c1:.2f}s | rel_error={err1:.6f}")
    print(f"  neg_recompute={cached1.neg_recompute_count} neg_reuse={cached1.neg_reuse_count} (expect reuse=0)")
    assert cached1.neg_reuse_count == 0, "interval=1 should never reuse -- caching logic bug"
    assert err1 < 1e-4, f"interval=1 should be numerically identical to baseline, got rel_error={err1}"
    print("  PASS: interval=1 matches baseline bit-for-bit (within fp tolerance), reuse count is 0.")

    print("\n=== 2a. Real speed + drift: neg_cache_interval=2 vs baseline (sparse_structure) ===")
    cached2 = make_cached_sampler(base_ss_sampler, neg_cache_interval=2)
    z_c2, t_c2 = run_sparse_structure(pipeline, cond, cached2, seed=0)
    err2 = rel_error(z_c2, z_base)
    speedup_ss = t_base / t_c2 if t_c2 > 0 else float('nan')
    print(f"  baseline: {t_base:.2f}s | cached(interval=2): {t_c2:.2f}s | speedup={speedup_ss:.3f}x | rel_error={err2:.6f}")
    print(f"  neg_recompute={cached2.neg_recompute_count} neg_reuse={cached2.neg_reuse_count}")

    # Structural proxy: decode both to occupied-voxel coords, compare counts. Mirrors
    # sample_sparse_structure's own decode path exactly: ss_res=32 for pipeline_type '512'.
    SS_RES = 32
    decoder = pipeline.models['sparse_structure_decoder']
    with torch.no_grad():
        decoded_base = decoder(z_base) > 0
        decoded_c2 = decoder(z_c2) > 0
    if SS_RES != decoded_base.shape[2]:
        ratio = decoded_base.shape[2] // SS_RES
        decoded_base = torch.nn.functional.max_pool3d(decoded_base.float(), ratio, ratio, 0) > 0.5
        decoded_c2 = torch.nn.functional.max_pool3d(decoded_c2.float(), ratio, ratio, 0) > 0.5
    n_base = int(decoded_base.sum().item())
    n_c2 = int(decoded_c2.sum().item())
    print(f"  occupied voxels: baseline={n_base} cached(interval=2)={n_c2} "
          f"(delta={100*(n_c2-n_base)/max(n_base,1):.2f}%)")

    print("\n=== 2b. Real speed + drift: neg_cache_interval=2 vs baseline (shape_slat, the expensive stage) ===")
    coords = torch.argwhere(decoded_base)[:, [0, 2, 3, 4]].int().to(pipeline.device)
    print(f"  using {coords.shape[0]} coords derived from the baseline sparse-structure decode "
          f"(shared across both runs below for a fair comparison)")

    slat_base, t_slat_base = run_shape_slat(pipeline, cond, coords, base_shape_sampler, seed=1)
    cached_slat2 = make_cached_sampler(base_shape_sampler, neg_cache_interval=2)
    slat_c2, t_slat_c2 = run_shape_slat(pipeline, cond, coords, cached_slat2, seed=1)
    err_slat = rel_error(slat_c2.feats, slat_base.feats)
    speedup_slat = t_slat_base / t_slat_c2 if t_slat_c2 > 0 else float('nan')
    print(f"  baseline: {t_slat_base:.2f}s | cached(interval=2): {t_slat_c2:.2f}s | "
          f"speedup={speedup_slat:.3f}x | rel_error(feats)={err_slat:.6f}")
    print(f"  neg_recompute={cached_slat2.neg_recompute_count} neg_reuse={cached_slat2.neg_reuse_count}")

    print("\n=== SUMMARY ===")
    print(f"sparse_structure: {speedup_ss:.3f}x, rel_error={err2:.6f}, voxel-count delta="
          f"{100*(n_c2-n_base)/max(n_base,1):.2f}%")
    print(f"shape_slat:       {speedup_slat:.3f}x, rel_error={err_slat:.6f}")
    print("\nNOTE: this checks numerical drift in the raw model prediction / decoded voxel count,")
    print("not a full rendered/mesh comparison (that requires the ~15-minute remesh+bake pipeline")
    print("stage, not repeated here per-config). Treat this as a first, real, honest signal --")
    print("not the same rigor as trellis-mac-mps's 6-combination multi-image verification.")


if __name__ == "__main__":
    main()
