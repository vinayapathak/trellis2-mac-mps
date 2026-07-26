import math
from collections import deque
from typing import *
from .flow_euler import FlowEulerSampler, FlowEulerGuidanceIntervalSampler

"""
Applies AB-Cache's REAL mathematical technique (arXiv:2504.10540, read directly via PDF fetch --
see this project's CLAUDE.md and trellis-mac-mps's APPLE_SILICON.md for the correction this
required) to the diff-caching axis this project already built
(flow_euler_cached_cfg.DiffCachedCfgFlowEulerGuidanceIntervalSampler).

What AB-Cache actually is (confirmed from the paper, not assumed): the SAME fixed schedule as
naive caching (real compute every N-th call, cached/estimated otherwise) -- there is no adaptive or
error-gated trigger, despite an earlier, now-corrected note in this project family's CLAUDE.md/
APPLE_SILICON.md claiming otherwise. What's actually different from naive reuse is what happens AT
the cached steps: instead of a zero-order hold (assume no change), AB-Cache fits a k-th order
Adams-Bashforth extrapolation through the previous k REAL (non-extrapolated) values and projects it
forward, with a proven O(h^k) truncation error vs. naive reuse's O(h) (paper Eq. 3.6):

    v_n ~= sum_{i=1}^{k} (-1)^(i+1) * C(k,i) * v_{n-i}          (exponential term dropped for
                                                                   flow matching, per the paper)

k=1 reduces to v_n ~= v_{n-1}, i.e. exactly the existing zero-order-hold diff-cache -- so this class
is a strict generalization, not a different technique; ab_order=1 should reproduce
DiffCachedCfgFlowEulerGuidanceIntervalSampler's behavior and numbers as a sanity check.

Important scope note: the real paper applies this to the raw network output ε_θ(x_t, t) across
consecutive DIFFUSION TIMESTEPS (skipping whole forward passes in the sampling loop). This class
applies the same k-th order extrapolation formula to the CFG diff (pred_pos - pred_neg) across
consecutive calls to the negative branch within the guidance interval -- a different caching axis
(this project's existing one, inherited from the FasterCache-inspired diff-cache), not a literal
reimplementation of AB-Cache's own use case. The math (the extrapolation formula and its provable
error-order improvement over zero-order hold) is what's being reused here, not the specific
application the paper demonstrates it on -- flagged explicitly so this isn't mistaken for "AB-Cache
itself, ported."
"""


class ABCacheDiffFlowEulerGuidanceIntervalSampler(FlowEulerGuidanceIntervalSampler):
    def __init__(self, *args, neg_cache_interval: int = 2, ab_order: int = 2, **kwargs):
        """
        Args:
            neg_cache_interval: same fixed schedule as DiffCachedCfgFlowEulerGuidanceIntervalSampler
                -- recompute the negative branch every N calls inside the guidance interval, reuse
                (now via AB extrapolation instead of a zero-order hold) on the N-1 calls between.
            ab_order: order k of the Adams-Bashforth extrapolation applied to the cached diff at
                reuse steps. k=1 is exactly the existing zero-order-hold behavior (sanity-check
                equivalence). k=2/3 match the orders the real paper evaluates (Table 3: quality
                improves monotonically with order in their own ablation).
        """
        super().__init__(*args, **kwargs)
        self.neg_cache_interval = neg_cache_interval
        self.ab_order = ab_order

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
        self._diff_history = deque(maxlen=self.ab_order)
        self._interval_call_count = 0
        self.neg_recompute_count = 0
        self.neg_reuse_ab_count = 0     # reused via real order-k>=2 extrapolation
        self.neg_reuse_hold_count = 0   # reused via zero-order hold (insufficient history yet, or ab_order=1)
        return FlowEulerSampler.sample(
            self, model, noise, cond, steps, rescale_t, verbose,
            neg_cond=neg_cond, guidance_strength=guidance_strength,
            guidance_interval=guidance_interval, **kwargs,
        )

    def _ab_extrapolate(self):
        # self._diff_history: oldest -> newest, i.e. hist[-1] = diff_{n-1}, hist[-2] = diff_{n-2}, ...
        hist = list(self._diff_history)
        k = len(hist)
        est = None
        for i in range(1, k + 1):
            coeff = ((-1) ** (i + 1)) * math.comb(k, i)
            term = coeff * hist[-i]
            est = term if est is None else est + term
        return est

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
        do_recompute = (
            len(self._diff_history) == 0
            or self._interval_call_count % self.neg_cache_interval == 0
        )
        if do_recompute:
            pred_neg = FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)
            diff = pred_pos - pred_neg
            self._diff_history.append(diff)
            self.neg_recompute_count += 1
        else:
            if len(self._diff_history) >= self.ab_order and self.ab_order >= 2:
                diff = self._ab_extrapolate()
                self.neg_reuse_ab_count += 1
            else:
                diff = self._diff_history[-1]
                self.neg_reuse_hold_count += 1
            pred_neg = pred_pos - diff
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
