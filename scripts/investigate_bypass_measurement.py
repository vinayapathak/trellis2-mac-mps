#!/usr/bin/env python3
"""
Bypass-measure where TRELLIS.2's forward-pass time actually goes, on the two flow models --
same methodology as the original trellis-mac-mps project's Section 4 (APPLE_SILICON.md: "The
ceiling story here is very different from FFN's..."), applied fresh here per this project's own
CLAUDE.md item 1: do NOT assume the original project's ~77% attention / ~11% FFN split transfers.

Method: monkeypatch a submodule class's forward to return zeros (the correct no-op for a
residual-add block: h_bypassed = 0 -> x_out = x_in + 0 = x_in), time N repeated real forward
passes of the full flow model with and without the bypass, and attribute the DROP in wall-clock
time to that submodule's true contribution. This captures real dispatch/pipelining costs that an
isolated per-module microbenchmark would miss -- the same reason the original project used this
method instead of just timing each submodule alone.

Two architecturally different models, confirmed by reading the actual model code (not assumed):
  - sparse_structure_flow_model (SparseStructureFlowModel): DENSE tokens, plain nn.Linear I/O,
    MultiHeadAttention (trellis2/modules/attention/modules.py) + FeedForwardNet
    (trellis2/modules/transformer/blocks.py) -- structurally close to the original TRELLIS's own
    sparse_structure_flow_model, so its split MIGHT resemble the original project's ~77/11 finding,
    but that's a hypothesis to check here, not assumed.
  - shape_slat_flow_model_512 (SLatFlowModel): SPARSE tokens (sp.SparseTensor), sp.SparseLinear
    I/O, SparseMultiHeadAttention (trellis2/modules/sparse/attention/modules.py) +
    SparseFeedForwardNet (trellis2/modules/sparse/transformer/blocks.py) -- the real O-Voxel/
    flex_gemm-backed backbone, and (per this project's own real T_run1 generation timing) by far
    the most expensive sampling stage (~30s/step vs. sparse_structure's ~1.1s/step and texture's
    ~15s/step) -- the single highest-value target for this measurement.
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

CACHE_DIR = os.path.expanduser("~/.cache/trellis2/huggingface")
IMAGE_PATH = "assets/example_image/T.png"
N_TIMED = 8
N_WARMUP = 3


def load_pipeline():
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
        TRELLIS_REPO, revision=TRELLIS_REVISION, cache_dir=CACHE_DIR, local_files_only=True,
    )
    pipeline.low_vram = False
    pipeline.to(torch.device("mps"))
    return pipeline


def timed_forward(fn, n=N_TIMED, warmup=N_WARMUP):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.mps.synchronize()
        t0 = time.time()
        for _ in range(n):
            fn()
        torch.mps.synchronize()
        return (time.time() - t0) / n


def bench_dense_model(model, x, t, cond):
    from trellis2.modules.attention.modules import MultiHeadAttention
    from trellis2.modules.transformer.blocks import FeedForwardNet

    orig_attn_forward = MultiHeadAttention.forward
    orig_mlp_forward = FeedForwardNet.forward

    def zero_attn_forward(target_types):
        def _forward(self, x, context=None, phases=None):
            if self._type in target_types:
                return torch.zeros_like(x)
            return orig_attn_forward(self, x, context, phases)
        return _forward

    def zero_mlp_forward(self, x):
        return torch.zeros_like(x)

    results = {}
    native_fn = lambda: model(x, t, cond)
    results['native'] = timed_forward(native_fn)
    print(f"  native:                {results['native']*1000:.2f}ms")

    for label, target in [('self_attn', {'self'}), ('cross_attn', {'cross'}), ('both_attn', {'self', 'cross'})]:
        MultiHeadAttention.forward = zero_attn_forward(target)
        try:
            results[f'bypass_{label}'] = timed_forward(native_fn)
        finally:
            MultiHeadAttention.forward = orig_attn_forward
        delta = results['native'] - results[f'bypass_{label}']
        print(f"  bypass {label:12s}:  {results[f'bypass_{label}']*1000:.2f}ms  "
              f"(implied contribution: {delta*1000:.2f}ms, {100*delta/results['native']:.1f}%)")

    FeedForwardNet.forward = zero_mlp_forward
    try:
        results['bypass_mlp'] = timed_forward(native_fn)
    finally:
        FeedForwardNet.forward = orig_mlp_forward
    delta = results['native'] - results['bypass_mlp']
    print(f"  bypass {'mlp':12s}:  {results['bypass_mlp']*1000:.2f}ms  "
          f"(implied contribution: {delta*1000:.2f}ms, {100*delta/results['native']:.1f}%)")

    MultiHeadAttention.forward = zero_attn_forward({'self', 'cross'})
    FeedForwardNet.forward = zero_mlp_forward
    try:
        results['bypass_all'] = timed_forward(native_fn)
    finally:
        MultiHeadAttention.forward = orig_attn_forward
        FeedForwardNet.forward = orig_mlp_forward
    print(f"  bypass {'all':12s}:  {results['bypass_all']*1000:.2f}ms  (residual floor: norms/I/O layers/t-embed)")

    return results


def bench_sparse_model(model, x, t, cond):
    from trellis2.modules.sparse.attention.modules import SparseMultiHeadAttention
    from trellis2.modules.sparse.transformer.blocks import SparseFeedForwardNet

    orig_attn_forward = SparseMultiHeadAttention.forward
    orig_mlp_forward = SparseFeedForwardNet.forward

    def zero_attn_forward(target_types):
        def _forward(self, x, context=None):
            if self._type in target_types:
                return x.replace(torch.zeros_like(x.feats))
            return orig_attn_forward(self, x, context)
        return _forward

    def zero_mlp_forward(self, x):
        return x.replace(torch.zeros_like(x.feats))

    results = {}
    native_fn = lambda: model(x, t, cond)
    results['native'] = timed_forward(native_fn)
    print(f"  native:                {results['native']*1000:.2f}ms")

    for label, target in [('self_attn', {'self'}), ('cross_attn', {'cross'}), ('both_attn', {'self', 'cross'})]:
        SparseMultiHeadAttention.forward = zero_attn_forward(target)
        try:
            results[f'bypass_{label}'] = timed_forward(native_fn)
        finally:
            SparseMultiHeadAttention.forward = orig_attn_forward
        delta = results['native'] - results[f'bypass_{label}']
        print(f"  bypass {label:12s}:  {results[f'bypass_{label}']*1000:.2f}ms  "
              f"(implied contribution: {delta*1000:.2f}ms, {100*delta/results['native']:.1f}%)")

    SparseFeedForwardNet.forward = zero_mlp_forward
    try:
        results['bypass_mlp'] = timed_forward(native_fn)
    finally:
        SparseFeedForwardNet.forward = orig_mlp_forward
    delta = results['native'] - results['bypass_mlp']
    print(f"  bypass {'mlp':12s}:  {results['bypass_mlp']*1000:.2f}ms  "
          f"(implied contribution: {delta*1000:.2f}ms, {100*delta/results['native']:.1f}%)")

    SparseMultiHeadAttention.forward = zero_attn_forward({'self', 'cross'})
    SparseFeedForwardNet.forward = zero_mlp_forward
    try:
        results['bypass_all'] = timed_forward(native_fn)
    finally:
        SparseMultiHeadAttention.forward = orig_attn_forward
        SparseFeedForwardNet.forward = orig_mlp_forward
    print(f"  bypass {'all':12s}:  {results['bypass_all']*1000:.2f}ms  (residual floor: norms/I/O layers/t-embed)")

    return results


def main():
    print("Loading pipeline (offline, cached weights)...")
    pipeline = load_pipeline()
    print(f"  loaded. device={pipeline.device}")

    image = Image.open(IMAGE_PATH)
    proc = pipeline.preprocess_image(image)
    cond = pipeline.get_cond([proc], 512)
    print("  real DINOv3 cond computed.")

    print("\n" + "=" * 70)
    print("MODEL 1: sparse_structure_flow_model (dense tokens, dense attention)")
    print("=" * 70)
    ss_model = pipeline.models['sparse_structure_flow_model']
    reso, in_ch = ss_model.resolution, ss_model.in_channels
    x_ss = torch.randn(1, in_ch, reso, reso, reso, device=pipeline.device)
    t_ss = torch.full((1,), 500.0, device=pipeline.device)
    n_params_ss = sum(p.numel() for p in ss_model.parameters())
    print(f"  params: {n_params_ss/1e6:.1f}M, resolution={reso}, in_channels={in_ch}")
    ss_results = bench_dense_model(ss_model, x_ss, t_ss, cond['cond'])

    print("\n" + "=" * 70)
    print("MODEL 2: shape_slat_flow_model_512 (sparse O-Voxel tokens, flex_gemm-backed)")
    print("=" * 70)
    from trellis2.modules.sparse import SparseTensor
    shape_model = pipeline.models['shape_slat_flow_model_512']
    n_params_shape = sum(p.numel() for p in shape_model.parameters())
    # Real coords from an actual sparse-structure sample+decode, not a synthetic guess -- matches
    # this project's own established methodology (real conditioning/shapes, not torch.randn stand-ins).
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
    print(f"  params: {n_params_shape/1e6:.1f}M, {coords.shape[0]} real sparse coords")

    feats = torch.randn(coords.shape[0], shape_model.in_channels, device=pipeline.device)
    x_shape = SparseTensor(feats=feats, coords=coords)
    t_shape = torch.full((1,), 500.0, device=pipeline.device)
    shape_results = bench_sparse_model(shape_model, x_shape, t_shape, cond['cond'])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, results in [('sparse_structure_flow_model', ss_results), ('shape_slat_flow_model_512', shape_results)]:
        native = results['native']
        print(f"\n{name} (native: {native*1000:.2f}ms/forward):")
        for key in ('self_attn', 'cross_attn', 'both_attn', 'mlp'):
            bk = f'bypass_{key}'
            if bk in results:
                delta = native - results[bk]
                print(f"  {key:12s}: {100*delta/native:5.1f}% of forward time")
        residual = results['bypass_all']
        print(f"  residual (norms/I-O/t-embed): {100*residual/native:5.1f}% of forward time")


if __name__ == "__main__":
    main()
