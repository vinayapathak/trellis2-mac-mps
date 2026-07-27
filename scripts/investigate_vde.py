#!/usr/bin/env python3
"""
Benchmarks VDEFlowEulerGuidanceIntervalSampler (trellis2/pipelines/samplers/flow_euler_vde.py --
ported from Tan-Junwen/VDE's real reference implementation, CVPR 2026) on shape_slat_flow_model_512,
the confirmed bottleneck stage (~65% of real generation time, per this project's bypass measurement
and T_run1 timing).

Protocol, same rigor as this project's other sampler investigations:
  1. Correctness: interval=1 (never estimates) must match the real baseline sampler exactly.
  2. Real speed + quality sweep across stable_step/interval configs -- VDE skips BOTH cond and
     uncond model calls on estimate steps (unlike this project's own CFG-diff-cache, which always
     keeps the positive branch fresh), so this is tested as a materially different, bigger lever,
     not assumed to behave like the earlier (failed) fixed-interval/AB-Cache attempts.
  3. Quality proxy: relative error on the predicted SLat features vs. the real baseline, same
     metric used throughout this project's CFG-caching investigations for direct comparability.

Real DINOv3 conditioning, real sparse coords from an actual sparse-structure sample -- not
synthetic stand-ins.
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
from trellis2.pipelines.samplers import VDEFlowEulerGuidanceIntervalSampler

CACHE_DIR = os.path.expanduser("~/.cache/trellis2/huggingface")
IMAGE_PATH = "assets/example_image/T.png"


def load_pipeline():
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
        TRELLIS_REPO, revision=TRELLIS_REVISION, cache_dir=CACHE_DIR, local_files_only=True,
    )
    pipeline.low_vram = False
    pipeline.to(torch.device("mps"))
    return pipeline


def make_vde_sampler(base_sampler, stable_step, interval):
    return VDEFlowEulerGuidanceIntervalSampler(
        sigma_min=base_sampler.sigma_min, stable_step=stable_step, interval=interval,
    )


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
    proc = pipeline.preprocess_image(image)
    cond = pipeline.get_cond([proc], 512)
    print("  real DINOv3 cond computed.")

    print("  sampling real sparse structure for real coords...")
    ss_model = pipeline.models['sparse_structure_flow_model']
    reso, in_ch = ss_model.resolution, ss_model.in_channels
    x_ss = torch.randn(1, in_ch, reso, reso, reso, device=pipeline.device)
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
    print(f"  {coords.shape[0]} real sparse coords")

    base_sampler = pipeline.shape_slat_sampler
    print(f"\n  real guidance_interval={pipeline.shape_slat_sampler_params.get('guidance_interval')}, "
          f"steps={pipeline.shape_slat_sampler_params.get('steps')}, "
          f"guidance_strength={pipeline.shape_slat_sampler_params.get('guidance_strength')}")

    print("\n=== Baseline ===")
    slat_base, t_base = run_shape_slat(pipeline, cond, coords, base_sampler, seed=1)
    print(f"  baseline: {t_base:.2f}s")

    print("\n=== 1. Correctness: VDE stable_step=99, interval=1 (never estimates) ===")
    sanity = make_vde_sampler(base_sampler, stable_step=99, interval=1)
    slat_sanity, t_sanity = run_shape_slat(pipeline, cond, coords, sanity, seed=1)
    err_sanity = rel_error(slat_sanity.feats, slat_base.feats)
    print(f"  time={t_sanity:.2f}s rel_error={err_sanity:.6f} "
          f"real_calls={sanity.real_calls} estimate_calls={sanity.estimate_calls} (expect estimate=0)")
    assert sanity.estimate_calls == 0, "interval=1 should never estimate -- sampler bug"
    assert err_sanity < 1e-3, f"interval=1 should match baseline closely, got rel_error={err_sanity}"
    print("  PASS")

    print("\n=== 2. Real speed + quality sweep ===")
    configs = [
        (1, 2), (2, 2), (1, 3), (2, 3), (1, 4), (0, 2),
    ]
    results = {}
    for stable_step, interval in configs:
        sampler = make_vde_sampler(base_sampler, stable_step=stable_step, interval=interval)
        slat, t = run_shape_slat(pipeline, cond, coords, sampler, seed=1)
        err = rel_error(slat.feats, slat_base.feats)
        speedup = t_base / t if t > 0 else float('nan')
        print(f"  stable_step={stable_step} interval={interval}: time={t:.2f}s speedup={speedup:.3f}x "
              f"rel_error={err:.6f} real_calls={sampler.real_calls} estimate_calls={sampler.estimate_calls}")
        results[(stable_step, interval)] = (speedup, err, sampler.real_calls, sampler.estimate_calls)

    print("\n=== SUMMARY ===")
    print(f"baseline: {t_base:.2f}s")
    for (ss, iv), (speedup, err, real_c, est_c) in results.items():
        print(f"  stable_step={ss} interval={iv}: speedup={speedup:.3f}x rel_error={err:.6f} "
              f"({real_c} real + {est_c} estimated calls)")


if __name__ == "__main__":
    main()
