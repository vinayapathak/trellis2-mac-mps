import torch
from typing import *
from .flow_euler import FlowEulerSampler, FlowEulerGuidanceIntervalSampler

"""
Ports VDE (Velocity Decomposition and Estimation, arXiv:2605.23381, CVPR 2026, Tan et al.) to
TRELLIS.2's shape_slat_flow_model_512. Read the paper directly (methodology + results tables) and
cloned the authors' own reference implementation (Tan-Junwen/VDE, VDE4FLUX/inference_flux1.py) to
port the REAL algorithm, not a re-derivation from the abstract. As of this port, the authors'
VDE4Trellis2/ directory in their own repo contains a single placeholder file ("测试文件.txt",
literally "test file.txt", empty) -- confirmed nobody has actually implemented or published a
TRELLIS/TRELLIS.2 result for VDE. This is a real gap, not assumed.

What VDE actually does (confirmed from the reference code, simpler and more aggressive than the
paper's prose suggests): treats the ENTIRE latent as one flattened vector (no per-token/per-pixel
decomposition), decomposes the model's predicted velocity into a component parallel to the current
latent (scalar alpha) and a component orthogonal to it (scalar beta times a unit direction vector).
At "anchor" steps, alpha/beta/direction are computed for real from a real model call. At
"estimate" steps, alpha and beta are LINEARLY EXTRAPOLATED from the two most recent anchors
(torch.lerp with an unclamped weight -- true extrapolation, not just interpolation), while the
orthogonal direction is held flat from the last anchor (zero-order hold) -- and, critically, NO
model call happens at all on an estimate step (unlike this project's own CFG-diff-cache, which
always recomputes the positive/conditional branch fresh every step and only skips the negative
branch). This is a strictly bigger lever: it can skip both the cond and uncond branches together.

Adaptation notes for TRELLIS.2 (none of these are in the reference FLUX code, which has no CFG loop
at all -- FLUX uses embedded/distilled guidance, a single model call per step):
  - Decomposition is applied to the FINAL, post-CFG-combination (and post-guidance-rescale)
    velocity `pred` -- the actual quantity that feeds the Euler step -- not to pred_pos/pred_neg
    individually. This matches the reference's own scope: VDE skips transformer forward passes
    entirely, and TRELLIS.2's guidance-interval CFG needs TWO forward passes per real step, so an
    "estimate" step here skips both at once.
  - Operates on `x_t.feats`/`pred.feats` (SparseTensor), global norm/dot-product over the full
    flattened feature tensor -- direct analogue of FLUX's [B,C,H,W] latent treated as one vector.
  - Scope: only applied inside the guidance interval where guidance_strength requires two real
    branches (matching this project's other custom samplers' scope) -- outside it, or when
    guidance_strength is 0 or 1, falls back to a single real call every time, untouched.
  - `stable_step`/`interval` are counted against calls WITHIN the guidance interval
    (`self._interval_call_count`), not raw sampler step index, since that's the region VDE's
    savings actually apply to here.

Not yet benchmarked or quality-verified on TRELLIS.2 as of writing this file -- see this project's
CLAUDE.md for the real, measured result.
"""


def _decompose_velocity(latents_feats: torch.Tensor, pred_feats: torch.Tensor):
    lat_norm = torch.norm(latents_feats)
    lat_unit = latents_feats / (lat_norm + 1e-6)

    vel_tangential_norm = torch.sum(pred_feats * lat_unit)
    vel_tangential = vel_tangential_norm * lat_unit
    vel_normal = pred_feats - vel_tangential
    vel_normal_norm = torch.norm(vel_normal)
    vel_normal_unit = vel_normal / (vel_normal_norm + 1e-6)

    alpha = vel_tangential_norm / (lat_norm + 1e-6)
    beta = vel_normal_norm / (lat_norm + 1e-6)
    return alpha, beta, vel_normal_unit


def _predict_alpha_beta(anchor_ts, anchor_alpha, anchor_beta, current_t):
    if not anchor_ts:
        raise ValueError("VDE needs at least one anchor step before prediction.")
    if len(anchor_ts) < 2:
        return anchor_alpha[0], anchor_beta[0]

    t1, t2 = anchor_ts[-2], anchor_ts[-1]
    alpha1, alpha2 = anchor_alpha[-2], anchor_alpha[-1]
    beta1, beta2 = anchor_beta[-2], anchor_beta[-1]

    dt = t2 - t1
    if dt == 0:
        return alpha2, beta2

    weight = (current_t - t1) / dt  # deliberately unclamped -- true extrapolation, matching upstream
    weight_t = torch.tensor(weight, dtype=alpha1.dtype, device=alpha1.device)
    return torch.lerp(alpha1, alpha2, weight_t), torch.lerp(beta1, beta2, weight_t)


class VDEFlowEulerGuidanceIntervalSampler(FlowEulerGuidanceIntervalSampler):
    def __init__(self, *args, stable_step: int = 1, interval: int = 2, **kwargs):
        """
        Args:
            stable_step: number of real (anchor) calls at the start of the guidance interval
                before estimation is allowed to begin (upstream FLUX default: 6 out of 50 steps,
                ~14% warmup -- TRELLIS.2's real interval only has ~9-10 calls total, so this
                defaults much smaller here; needs >=1 for a single anchor, >=2 real anchors before
                linear extrapolation (vs. flat hold of the first anchor) actually kicks in).
            interval: after warmup, run a real (anchor) call every `interval`-th call; estimate
                (zero model calls) on the calls in between. interval=1 disables estimation
                entirely (sanity-check equivalence to the baseline sampler).
        """
        super().__init__(*args, **kwargs)
        self.stable_step = stable_step
        self.interval = interval

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
        self._anchor_ts = []
        self._anchor_alpha = []
        self._anchor_beta = []
        self._vel_normal_unit = None
        self._interval_call_count = 0
        self.real_calls = 0     # exposed for benchmark/investigation scripts
        self.estimate_calls = 0
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

        # Note: unlike the FLUX reference (which forces the diffusion process's absolute final
        # step to be real, since nothing corrects for staleness afterward), there's no equivalent
        # "final call in the guidance interval" special-case here -- t is confirmed inside the
        # interval at this point, so a check like `t <= guidance_interval[0]` would be dead code.
        # Cadence is controlled purely by stable_step/interval.
        should_run_full = (
            self._interval_call_count <= self.stable_step
            or (self.interval <= 1)
            or ((self._interval_call_count - self.stable_step) % self.interval == 0)
        )

        if should_run_full:
            pred_pos = FlowEulerSampler._inference_model(self, model, x_t, t, cond, **kwargs)
            pred_neg = FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)
            pred = guidance_strength * pred_pos + (1 - guidance_strength) * pred_neg

            if guidance_rescale > 0:
                x_0_pos = self._pred_to_xstart(x_t, t, pred_pos)
                x_0_cfg = self._pred_to_xstart(x_t, t, pred)
                std_pos = x_0_pos.std(dim=list(range(1, x_0_pos.ndim)), keepdim=True)
                std_cfg = x_0_cfg.std(dim=list(range(1, x_0_cfg.ndim)), keepdim=True)
                x_0_rescaled = x_0_cfg * (std_pos / std_cfg)
                x_0 = guidance_rescale * x_0_rescaled + (1 - guidance_rescale) * x_0_cfg
                pred = self._xstart_to_pred(x_t, t, x_0)

            alpha, beta, vel_normal_unit = _decompose_velocity(x_t.feats, pred.feats)
            self._anchor_ts.append(t)
            self._anchor_alpha.append(alpha)
            self._anchor_beta.append(beta)
            self._vel_normal_unit = vel_normal_unit
            self.real_calls += 1
            self._interval_call_count += 1
            return pred
        else:
            alpha_pred, beta_pred = _predict_alpha_beta(self._anchor_ts, self._anchor_alpha, self._anchor_beta, t)
            lat_norm = torch.norm(x_t.feats)
            pred_feats = alpha_pred * x_t.feats + beta_pred * lat_norm * self._vel_normal_unit
            self.estimate_calls += 1
            self._interval_call_count += 1
            return x_t.replace(pred_feats)
