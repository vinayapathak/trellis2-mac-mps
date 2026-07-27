"""
SCFM (ShortCut distillation for Flow Matching) -- Phase 1 of MDT_DIST_PLAN.md.

Reimplementation of "Shortcutting Pre-trained Flow Matching Diffusion Models is Almost Free
Lunch" (Cai et al., arXiv:2510.17858, NeurIPS 2025). No reference code was released (checked: the
paper's own project page link is a placeholder) -- every design decision below was cross-checked
directly against the paper's actual equations (read from the real PDF, not a web summary, to avoid
misquoted math), not guessed from the abstract.

What this implements: the paper's **vanilla** Algorithm 1 (single stopgrad EMA), not Algorithm 2 /
Appendix E's "final version" (dual fast/slow EMA) -- the paper itself frames the dual-EMA version
purely as a training-*speed* optimization ("negligible impact on the final converged results" --
Section 5.4), so it's a real, documented follow-up (see the module-level TODO at the bottom), not
something silently skipped.

The core mechanism, in the paper's own notation (equation numbers refer to the arXiv PDF):

  - Eq (1)/(2): x_t = (1-t) x_0 + t*z, v_t = z - x_0, t=0 clean / t=1 noise. This project's own
    `FlowMatchingTrainer.diffuse`/`get_v` already implement exactly this (with a `sigma_min` floor
    the real pretrained teacher was actually trained with) -- reused as-is via `self.diffuse`.
  - Eq (12): the windowed-interpolation distillation target,
        V(x_ti, ti) := [d_i/(d_i+d_{i+1})] * V(x_ti, ti) + [d_{i+1}/(d_i+d_{i+1})] * V(x_t{i+1}, t{i+1})
    where d_i = t_i - t_{i+1}. The paper explicitly flags this as "abuse of notation": the LHS is
    the *training target* assigned to (x_ti, ti); the RHS is two *raw* model evaluations at ti and
    t{i+1}, blended by how large the *next* interval (d_{i+1}, i.e. t{i+1} to t{i+2}) is relative to
    the current one. A third sampled point (t{i+2} here) is needed only for that interval's
    *length* -- the model is never evaluated there.
  - Eq (13): the loss, ℒ = (1/N)[Σ_{i≤k} (Vθ(x_ti,ti) - target_teacher)² + Σ_{i>k} (Vθ(x_ti,ti) -
    target_stopgrad)²]. Confirmed directly from the PDF (not the secondary web-summary, which
    initially mis-described index i) that i indexes **batch elements**, not timesteps: N is the
    real batch size, and k of the N samples get the frozen-teacher-built target, the rest get the
    EMA-stopgrad-built target -- both computed via the *same* Eq (12) formula, just substituting
    which model produces the two raw evaluations.
  - Eq (14): EMA update theta_minus = mu*theta_minus + (1-mu)*theta, mu=0.999 default.
  - Eq (17): timestep shift S_s(t) = s*t / (1 + (s-1)*t), s ~ U[2.5, 4.5]. **This is algebraically
    identical to this project's own `FlowEulerSampler.sample`'s `rescale_t` mechanism** (same
    `t_seq = rescale_t*t_seq/(1+(rescale_t-1)*t_seq)` formula) -- not a coincidence to paper over;
    the real production `shape_slat_flow_model_512` pipeline.json ships `rescale_t: 3.0`, which
    already sits inside the paper's own [2.5, 4.5] default range, so no override was needed to stay
    faithful to both the paper and this project's real teacher.
  - Algorithm 1 (Appendix A): for each of the N batch elements, sample three *consecutive* points
    (t1, t2, t3) from a shifted base grid L_n = shift(linspace(1, 0, n+1)); the first k elements use
    `skip=1` (adjacent points), the remaining N-k use a larger `skip` drawn from {2, 4, ..., n//4}.

Real adaptations made for this project (not blind ports -- each is a deliberate, documented choice):

  1. **n = 12, not an arbitrary finer virtual grid.** The paper's own Flux/SD3.5 experiments don't
     state a numeric `n`; papers in this space usually pick a moderately fine base discretization
     purely for constructing training targets. This project's actual subject, though, is whether
     distillation holds up *starting from an already-short, already-calibrated 12-step production
     schedule* (see MDT_DIST_PLAN.md's central research question) -- querying the real teacher at
     invented finer virtual timesteps it was never calibrated near would quietly dodge exactly the
     risk this project exists to test. `n=12` ties the windowing grid directly to the real
     `shape_slat_flow_model_512` production schedule (pipeline.json: steps=12). One real
     consequence: with n=12, the "coarse" skip set {2, 4, ..., n//4} collapses to just {2} (since
     4 > 12//4=3) -- a real, narrower version of the paper's mechanism, appropriately scaled down,
     not a bug.
  2. **SparseTensor.feats, not a dense image tensor.** `x_0`/noise/predictions are all
     `sp.SparseTensor` with a variable per-sample token count (`.layout`). Eq (12)'s per-sample
     interpolation weights (scalars) are broadcast to every active token row of that sample before
     being applied to `.feats` -- see `_broadcast_to_tokens`. Splitting the batch into the
     teacher-guided slice and the self-distill slice is done via `SparseTensor.__getitem__` (real
     sub-batch indexing, not a boolean mask on flattened feats, which would not respect per-sample
     token-count variability) and reassembled with `sp.sparse_cat`.
  3. **Real CFG applied when building targets, matching actual production inference, not a bare
     conditional-only network call.** `shape_slat_flow_model_512`'s own real sampler
     (`FlowEulerGuidanceIntervalSampler`, pipeline.json) uses guidance_strength=7.5,
     guidance_rescale=0.5, guidance_interval=(0.6, 1.0) -- i.e. the real teacher's production
     *behavior* already includes CFG+interval+rescale, and a target built from a plain,
     unguided forward pass would not be distilling the model this project actually cares about
     matching. `_cfg_velocity` below reimplements that exact mechanism (mirroring
     `ClassifierFreeGuidanceSamplerMixin`/`GuidanceIntervalSamplerMixin`/
     `FlowEulerSampler._pred_to_xstart`/`_xstart_to_pred`) rather than reusing those sampler
     classes directly, because they assume one shared scalar `t` for the whole batch (true during
     iterative sampling) -- SCFM training needs a *different* t per batch element within one
     forward pass, a real interface mismatch, not an oversight.

Per this project's established verification methodology (CLAUDE.md): this file is Phase 1 only --
implementation, cross-checked against the paper's actual equations. It has NOT yet been through
Phase 2 (synthetic gradient-flow/finite-loss check, then one real forward+backward pass on Phase
0's real encoded pilot data) -- do not treat this as verified working code yet.
"""
from typing import *
import copy
import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict

from ... import models
from ...modules import sparse as sp
from .sparse_flow_matching import ImageConditionedSparseFlowMatchingCFGTrainer


def _shifted_grid(n: int, shift: float) -> np.ndarray:
    """L_n := shift(linspace(1, 0, n+1)) -- Eq (17), identical to FlowEulerSampler's rescale_t."""
    t = np.linspace(1, 0, n + 1)
    return shift * t / (1 + (shift - 1) * t)


def _select_cond(cond, idxs: List[int]):
    """Index a batch-shaped conditioning value (Tensor / SparseTensor / VarLenTensor / list) by idxs."""
    if cond is None:
        return None
    if isinstance(cond, (sp.SparseTensor, sp.VarLenTensor)):
        return cond[idxs]
    if isinstance(cond, torch.Tensor):
        return cond[idxs]
    if isinstance(cond, list):
        return [cond[i] for i in idxs]
    raise TypeError(f"Unsupported cond type for indexing: {type(cond)}")


def _broadcast_to_tokens(per_sample: torch.Tensor, layout: List[slice], num_tokens: int) -> torch.Tensor:
    """
    Expand a [B] per-sample scalar (e.g. an Eq (12) interpolation weight) to a [num_tokens, 1]
    column aligned with a SparseTensor's flat `.feats` rows, via its `.layout` (per-sample slices).
    """
    out = per_sample.new_empty(num_tokens, 1)
    for i, sl in enumerate(layout):
        out[sl] = per_sample[i]
    return out


class SCFMDistillTrainer(ImageConditionedSparseFlowMatchingCFGTrainer):
    """
    SCFM distillation trainer for `shape_slat_flow_model_512` (or any `SLatFlowModel`-shaped
    sparse flow-matching model). See module docstring for the full derivation.

    `self.models['denoiser']` is the trainable **student**, following this project's existing
    checkpoint/EMA-export conventions exactly (so `train.py`, `BasicTrainer.save`/`load`, and the
    unrelated export-quality `ema_rate` mechanism all keep working unmodified). The frozen
    **teacher** (theta*) and the stopgrad **EMA copy** (theta-minus) are separate, non-trainable
    attributes (`self.teacher`, `self.stopgrad_model`) -- deliberately outside `self.models`, since
    neither should be gradient-updated, checkpointed via the normal per-name state_dict path, or
    export-EMA'd the way the trainable student is.

    Args:
        teacher_pretrained (str): HF path to the pretrained teacher checkpoint (e.g.
            "microsoft/TRELLIS.2-4B/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16"), passed to
            `trellis2.models.from_pretrained`. The student is a deep copy of this same model,
            trainable from the same initial weights, per MDT_DIST_PLAN.md's Phase 1 scope.
        n (int): base discretization grid size for L_n. Default 12, matching
            `shape_slat_flow_model_512`'s real production step count -- see module docstring
            point 1 for why this isn't an arbitrary finer virtual grid.
        k_over_n (float): teacher/self-distill mixing ratio in Eq (13). Paper default 0.4.
        shift_range (Tuple[float, float]): range to sample the Eq (17) shift `s` from every
            iteration. Paper default (2.5, 4.5); the real production `rescale_t=3.0` already
            falls inside this range.
        ema_decay (float): mu in Eq (14). Paper default 0.999.
        guidance_strength/guidance_rescale/guidance_interval: real production CFG settings used
            when evaluating the teacher/stopgrad models to build distillation targets (see module
            docstring point 3). Defaults match `shape_slat_flow_model_512`'s real pipeline.json.
        cache_dir (str): HF cache dir for `from_pretrained`, matching this project's
            `~/.cache/trellis2/huggingface` convention (see `scripts/download_weights.py`).
    """

    def __init__(
        self,
        *args,
        teacher_pretrained: str,
        n: int = 12,
        k_over_n: float = 0.4,
        shift_range: Tuple[float, float] = (2.5, 4.5),
        ema_decay: float = 0.999,
        guidance_strength: float = 7.5,
        guidance_rescale: float = 0.5,
        guidance_interval: Tuple[float, float] = (0.6, 1.0),
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        teacher = models.from_pretrained(teacher_pretrained, cache_dir=cache_dir).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        # Student: trainable copy initialized from the same weights (MDT_DIST_PLAN.md Phase 1).
        student = copy.deepcopy(teacher)
        for p in student.parameters():
            p.requires_grad_(True)

        models_dict = kwargs.pop('models', {})
        models_dict = {**models_dict, 'denoiser': student}

        super().__init__(*args, models=models_dict, **kwargs)

        self.teacher = teacher.to(self.device)
        self.stopgrad_model = copy.deepcopy(student).to(self.device).eval()
        for p in self.stopgrad_model.parameters():
            p.requires_grad_(False)

        self.n = n
        self.k_over_n = k_over_n
        self.shift_range = shift_range
        self.ema_decay = ema_decay
        self.guidance_strength = guidance_strength
        self.guidance_rescale = guidance_rescale
        self.guidance_interval = guidance_interval
        # {2, 4, ..., n//4} per Algorithm 1 -- with n=12 this is just {2}, a real, narrower
        # consequence of using the real production step count as the base grid (see module
        # docstring point 1), not an oversight.
        self._coarse_skips = [s for s in (2, 4, 8, 16, 32, 64) if s <= max(self.n // 4, 1)]
        if not self._coarse_skips:
            self._coarse_skips = [1]

    def update_ema(self):
        """
        Extends BasicTrainer's own export-quality EMA (unrelated, untouched) with the SCFM
        stopgrad update, Eq (14): theta_minus = mu*theta_minus + (1-mu)*theta. Called once per
        optimizer step via BasicTrainer's existing training loop -- no changes to `run_step`
        needed.
        """
        super().update_ema()
        with torch.no_grad():
            for p_stop, p_student in zip(
                self.stopgrad_model.parameters(),
                self.training_models['denoiser'].parameters(),
            ):
                p_stop.mul_(self.ema_decay).add_(p_student.detach(), alpha=1.0 - self.ema_decay)

    def _cfg_velocity(
        self,
        model: torch.nn.Module,
        x_t: sp.SparseTensor,
        t: torch.Tensor,
        cond,
        neg_cond,
        **kwargs
    ) -> sp.SparseTensor:
        """
        Real CFG + guidance-interval + guidance-rescale velocity evaluation with a **per-batch-
        element** `t` (module docstring point 3) -- mirrors
        `ClassifierFreeGuidanceSamplerMixin`/`GuidanceIntervalSamplerMixin`/
        `FlowEulerSampler._pred_to_xstart`/`_xstart_to_pred` exactly, since those sampler classes
        can't be reused as-is (they assume one shared scalar t for the whole batch).
        """
        t1000 = t * 1000
        in_interval = (t >= self.guidance_interval[0]) & (t <= self.guidance_interval[1])

        pred_pos = model(x_t, t1000, cond, **kwargs)
        if not bool(in_interval.any()) or self.guidance_strength == 1:
            return pred_pos

        pred_neg = model(x_t, t1000, neg_cond, **kwargs)
        gs = torch.where(in_interval, torch.full_like(t, self.guidance_strength), torch.ones_like(t))
        gs_tok = _broadcast_to_tokens(gs, x_t.layout, x_t.feats.shape[0])
        pred = gs_tok * pred_pos.feats + (1 - gs_tok) * pred_neg.feats
        pred = pred_pos.replace(pred)

        if self.guidance_rescale > 0:
            sigma_min = self.sigma_min
            t_tok = _broadcast_to_tokens(t, x_t.layout, x_t.feats.shape[0])
            def pred_to_xstart(v_feats):
                return (1 - sigma_min) * x_t.feats - (sigma_min + (1 - sigma_min) * t_tok) * v_feats
            def xstart_to_pred(x0_feats):
                return ((1 - sigma_min) * x_t.feats - x0_feats) / (sigma_min + (1 - sigma_min) * t_tok)
            x0_pos = pred_to_xstart(pred_pos.feats)
            x0_cfg = pred_to_xstart(pred.feats)
            # per-sample std, matching the sampler mixin's per-sample (not per-token) statistics
            gr_tok = _broadcast_to_tokens(
                torch.where(in_interval, torch.full_like(t, self.guidance_rescale), torch.zeros_like(t)),
                x_t.layout, x_t.feats.shape[0],
            )
            std_pos = torch.stack([x0_pos[sl].std() for sl in x_t.layout])
            std_cfg = torch.stack([x0_cfg[sl].std() for sl in x_t.layout])
            scale = _broadcast_to_tokens(std_pos / std_cfg.clamp_min(1e-8), x_t.layout, x_t.feats.shape[0])
            x0_rescaled = x0_cfg * scale
            x0_final = gr_tok * x0_rescaled + (1 - gr_tok) * x0_cfg
            pred = pred.replace(xstart_to_pred(x0_final))

        return pred

    def _sample_windows(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[int]]:
        """
        Algorithm 1, steps 3-8: sample a shift, build L_n, split the batch into k teacher-guided
        (skip=1) and N-k self-distill (skip in {2,...,n//4}) elements, and sample three
        consecutive grid points per element. Returns (t1, t2, t3) as [B] float arrays plus the
        teacher/self index lists.
        """
        shift = np.random.uniform(*self.shift_range)
        grid = _shifted_grid(self.n, shift)

        k = int(round(self.k_over_n * batch_size))
        k = min(max(k, 0), batch_size)
        perm = np.random.permutation(batch_size)
        teacher_idx = perm[:k].tolist()
        self_idx = perm[k:].tolist()

        t1 = np.empty(batch_size, dtype=np.float64)
        t2 = np.empty(batch_size, dtype=np.float64)
        t3 = np.empty(batch_size, dtype=np.float64)
        teacher_set = set(teacher_idx)
        for i in range(batch_size):
            skip = 1 if i in teacher_set else int(np.random.choice(self._coarse_skips))
            max_start = self.n - 2 * skip
            start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
            t1[i] = grid[start]
            t2[i] = grid[start + skip]
            t3[i] = grid[start + 2 * skip]

        return t1, t2, t3, teacher_idx, self_idx

    def training_losses(
        self,
        x_0: sp.SparseTensor,
        cond=None,
        neg_cond=None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        SCFM's Eq (13) loss, adapted for SparseTensor.feats. See module docstring for the full
        derivation; NOT yet run through Phase 2 verification.
        """
        assert neg_cond is not None, "neg_cond is required to build CFG-conditioned SCFM targets"
        B = x_0.shape[0]
        device = x_0.device
        noise = x_0.replace(torch.randn_like(x_0.feats))

        t1_np, t2_np, t3_np, teacher_idx, self_idx = self._sample_windows(B)
        t1 = torch.tensor(t1_np, device=device, dtype=torch.float32)
        t2 = torch.tensor(t2_np, device=device, dtype=torch.float32)
        t3 = torch.tensor(t3_np, device=device, dtype=torch.float32)

        x_t1 = self.diffuse(x_0, t1, noise=noise)
        x_t2 = self.diffuse(x_0, t2, noise=noise)

        # Student prediction -- the only branch that needs gradients.
        cond_train = self.get_cond(cond, neg_cond=neg_cond, **kwargs)
        pred_v1 = self.training_models['denoiser'](x_t1, t1 * 1000, cond_train, **kwargs)
        assert pred_v1.shape == x_0.shape

        # Eq (12) interpolation weights, d_i = t1-t2, d_{i+1} = t2-t3.
        d_i = t1 - t2
        d_ip1 = t2 - t3
        denom = (d_i + d_ip1).clamp_min(1e-8)
        w1 = d_i / denom
        w2 = d_ip1 / denom

        target_feats = torch.zeros_like(pred_v1.feats)
        with torch.no_grad():
            for idxs, model in ((teacher_idx, self.teacher), (self_idx, self.stopgrad_model)):
                if not idxs:
                    continue
                sub_x_t1 = x_t1[idxs]
                sub_x_t2 = x_t2[idxs]
                sub_t1 = t1[idxs]
                sub_t2 = t2[idxs]
                sub_cond = _select_cond(cond, idxs)
                sub_neg_cond = _select_cond(neg_cond, idxs)

                v1 = self._cfg_velocity(model, sub_x_t1, sub_t1, sub_cond, sub_neg_cond, **kwargs)
                v2 = self._cfg_velocity(model, sub_x_t2, sub_t2, sub_cond, sub_neg_cond, **kwargs)

                sub_w1 = _broadcast_to_tokens(w1[idxs], sub_x_t1.layout, sub_x_t1.feats.shape[0])
                sub_w2 = _broadcast_to_tokens(w2[idxs], sub_x_t1.layout, sub_x_t1.feats.shape[0])
                sub_target = sub_w1 * v1.feats + sub_w2 * v2.feats

                for local_i, global_i in enumerate(idxs):
                    target_feats[x_0.layout[global_i]] = sub_target[sub_x_t1.layout[local_i]]

        terms = edict()
        terms["mse"] = F.mse_loss(pred_v1.feats, target_feats)
        terms["loss"] = terms["mse"]
        terms["k_over_n_actual"] = len(teacher_idx) / B
        return terms, {}


# TODO (documented, not started -- see MDT_DIST_PLAN.md Phase 1):
#   - Algorithm 2 / Appendix E's dual fast/slow EMA (mu=0.99 "fast" + mu=0.999 "slow", Eq 22's
#     fast-slow target blend) -- the paper's own "final version", faster convergence, "negligible
#     impact on final converged results" per its own Section 5.4. Worth adding once Phase 3's pilot
#     shows the vanilla version's convergence speed is actually a bottleneck, not before.
#   - theta_minus (stopgrad_model) is not currently included in this trainer's checkpoint
#     save/resume path -- fine for a short Phase 3 pilot run, a real gap for any longer run that
#     needs to resume across sessions.
