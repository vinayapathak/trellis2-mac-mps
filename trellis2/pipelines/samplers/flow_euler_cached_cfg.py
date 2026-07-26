from typing import *
from .flow_euler import FlowEulerSampler, FlowEulerGuidanceIntervalSampler

"""
Ports trellis-mac-mps's CFG difference-caching fix (see that project's
trellis/pipelines/samplers/flow_euler_cached_cfg.py and APPLE_SILICON.md's "Fixed CFG caching via
difference-caching" section) to TRELLIS.2's FlowEulerGuidanceIntervalSampler.

Ported, not assumed to transfer unchanged -- verified first that TRELLIS.2's sampler has the same
structural redundancy the original technique targets:
  - Same class hierarchy shape: GuidanceIntervalSamplerMixin + ClassifierFreeGuidanceSamplerMixin
    + FlowEulerSampler -> FlowEulerGuidanceIntervalSampler (trellis2/pipelines/samplers/flow_euler.py).
  - Same two-model-call-per-step structure inside the guidance interval (cond + neg_cond).
  - Same underlying math, different surface convention: TRELLIS.2's
    ClassifierFreeGuidanceSamplerMixin combines branches as
    `guidance_strength * pred_pos + (1 - guidance_strength) * pred_neg`, which is algebraically
    identical to the original project's `(1 + cfg_strength) * pred - cfg_strength * neg_pred` form
    with `guidance_strength = 1 + cfg_strength` (both reduce to `neg + w * (pos - neg)`).
  - One real difference from the original project, confirmed by reading the actual pretrained
    pipeline.json (~/.cache/trellis2/huggingface/models--microsoft--TRELLIS.2-4B): TRELLIS.2 also
    ships an optional post-hoc `guidance_rescale` correction (STD-matching between the CFG'd and
    positive-only x0 predictions) that the original TRELLIS sampler mixin does not have. Preserved
    here, applied to the *combined* prediction exactly as ClassifierFreeGuidanceSamplerMixin does,
    since it's a correction on top of the branch combination, not part of what's being cached.
  - Real pipeline.json values (all three stages use FlowEulerGuidanceIntervalSampler, steps=12):
    sparse_structure_sampler: guidance_strength=7.5, guidance_interval=[0.6, 1.0], guidance_rescale=0.7
    shape_slat_sampler:       guidance_strength=7.5, guidance_interval=[0.6, 1.0], guidance_rescale=0.5
    tex_slat_sampler:         guidance_strength=1.0, guidance_interval=[0.6, 0.9], guidance_rescale=0.0
    tex_slat_sampler's guidance_strength=1.0 means it NEVER runs the negative branch at all (both
    mixins short-circuit to a single positive-only call whenever guidance_strength==1) -- there is
    nothing to cache there. Only sparse_structure_sampler and shape_slat_sampler have a real
    negative branch to cache, and this needs benchmarking on those two, not assumed from the
    original project's own numbers (different model, different backbone, different guidance
    strength: 7.5 here vs the original's 5.0 -- AB-Cache's own finding was that stronger cfg_strength
    amplifies staleness error, so this needs its own real drift check, not a transplanted result).

Not yet benchmarked or quality-verified on this model as of this writing -- see this project's
CLAUDE.md for status. Subclasses FlowEulerGuidanceIntervalSampler (the real, production sampler
class) rather than editing it in place, per the shared versioning convention across both ports in
this project family -- swap this class in for pipeline.sparse_structure_sampler /
pipeline.shape_slat_sampler to test, leaving the original completely untouched for anything that
doesn't opt in.
"""


class DiffCachedCfgFlowEulerGuidanceIntervalSampler(FlowEulerGuidanceIntervalSampler):
    def __init__(self, *args, neg_cache_interval: int = 2, **kwargs):
        """
        Args:
            neg_cache_interval: recompute the negative-branch prediction every N calls to
                _inference_model that fall inside the guidance interval; reuse a diff-derived
                estimate on the N-1 calls in between. 1 = no caching (identical behavior to the
                base FlowEulerGuidanceIntervalSampler). Higher = more caching, more speed, more
                risk of quality drift.
        """
        super().__init__(*args, **kwargs)
        self.neg_cache_interval = neg_cache_interval

    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        guidance_strength: float = 3.0,
        guidance_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs,
    ):
        # Reset cache state at the start of every sample() call -- sampler instances are reused
        # across pipeline.run() calls, and stale cache state from a previous sample must not leak
        # into the next one.
        self._diff_cache = None
        self._interval_call_count = 0
        self.neg_recompute_count = 0  # exposed for benchmark/investigation scripts
        self.neg_reuse_count = 0
        return FlowEulerSampler.sample(
            self, model, noise, cond, steps, rescale_t, verbose,
            neg_cond=neg_cond, guidance_strength=guidance_strength,
            guidance_interval=guidance_interval, **kwargs,
        )

    def _inference_model(
        self, model, x_t, t, cond, neg_cond, guidance_strength, guidance_interval,
        guidance_rescale: float = 0.0, **kwargs,
    ):
        in_interval = guidance_interval[0] <= t <= guidance_interval[1]
        if not in_interval or guidance_strength == 1:
            return FlowEulerSampler._inference_model(self, model, x_t, t, cond, **kwargs)
        if guidance_strength == 0:
            return FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)

        pred_pos = FlowEulerSampler._inference_model(self, model, x_t, t, cond, **kwargs)
        if self._diff_cache is None or self._interval_call_count % self.neg_cache_interval == 0:
            pred_neg = FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)
            self._diff_cache = pred_pos - pred_neg
            self.neg_recompute_count += 1
        else:
            pred_neg = pred_pos - self._diff_cache
            self.neg_reuse_count += 1
        self._interval_call_count += 1

        pred = guidance_strength * pred_pos + (1 - guidance_strength) * pred_neg

        if guidance_rescale > 0:
            x_0_pos = self._pred_to_xstart(x_t, t, pred_pos)
            x_0_cfg = self._pred_to_xstart(x_t, t, pred)
            std_pos = x_0_pos.std(dim=list(range(1, x_0_pos.ndim)), keepdim=True)
            std_cfg = x_0_cfg.std(dim=list(range(1, x_0_cfg.ndim)), keepdim=True)
            x_0_rescaled = x_0_cfg * (std_pos / std_cfg)
            x_0 = guidance_rescale * x_0_rescaled + (1 - guidance_rescale) * x_0_cfg
            pred = self._xstart_to_pred(x_t, t, x_0)

        return pred
