"""
DINOv3's self-supervised training objective, implemented from the paper (Simeoni et al.,
arXiv:2508.10104): DINO (image-level self-distillation) + iBOT (masked patch-level
self-distillation) + KoLeo (feature-uniformity regularizer) + Gram anchoring (dense-feature
patch-consistency regularizer introduced specifically in v3 to fix long-training patch collapse).

See model.py's module docstring and this directory's README.md for what this is and is not --
these losses are real and correct against the paper's description, but running them to
convergence at Meta's actual scale (1.7B images, 256 GPUs) is not attempted or claimed here.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    """
    Image-level (CLS-token) self-distillation loss. Student output is compared against a
    centered, sharpened teacher target via cross-entropy in probability space; the teacher's
    center is an EMA of teacher outputs (Sinkhorn-Knopp centering per the paper is an alternative
    to the simpler EMA-mean centering from DINOv1/v2 -- both are valid within the "DINO loss"
    family the paper refers to; EMA-mean centering is implemented here as the well-established,
    simpler variant, noted explicitly rather than silently presented as the exact Sinkhorn variant).
    """

    def __init__(self, out_dim: int, student_temp: float = 0.1, teacher_temp: float = 0.04, center_momentum: float = 0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_out: torch.Tensor, teacher_out: torch.Tensor) -> torch.Tensor:
        student_logits = student_out / self.student_temp
        teacher_probs = F.softmax((teacher_out - self.center) / self.teacher_temp, dim=-1).detach()
        loss = torch.sum(-teacher_probs * F.log_softmax(student_logits, dim=-1), dim=-1).mean()
        self._update_center(teacher_out)
        return loss

    @torch.no_grad()
    def _update_center(self, teacher_out: torch.Tensor):
        batch_center = teacher_out.mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1 - self.center_momentum)


class IBOTLoss(nn.Module):
    """
    Patch-level masked self-distillation: for masked patch positions, the student (which saw the
    masked input) must predict the teacher's (unmasked-input) representation at those same
    positions. Same centering/sharpening mechanics as DINOLoss, applied per-patch instead of
    per-image, averaged only over masked positions.
    """

    def __init__(self, out_dim: int, student_temp: float = 0.1, teacher_temp: float = 0.04, center_momentum: float = 0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, 1, out_dim))

    def forward(self, student_patches: torch.Tensor, teacher_patches: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # student_patches, teacher_patches: [B, N, out_dim]; mask: [B, N] bool, True = masked position
        student_logits = student_patches / self.student_temp
        teacher_probs = F.softmax((teacher_patches - self.center) / self.teacher_temp, dim=-1).detach()
        per_token_loss = torch.sum(-teacher_probs * F.log_softmax(student_logits, dim=-1), dim=-1)  # [B, N]
        mask = mask.to(per_token_loss.dtype)
        denom = mask.sum().clamp(min=1.0)
        loss = (per_token_loss * mask).sum() / denom
        self._update_center(teacher_patches, mask)
        return loss

    @torch.no_grad()
    def _update_center(self, teacher_patches: torch.Tensor, mask: torch.Tensor):
        mask_ = mask.unsqueeze(-1)
        denom = mask_.sum().clamp(min=1.0)
        batch_center = (teacher_patches * mask_).sum(dim=(0, 1), keepdim=True) / denom
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1 - self.center_momentum)


class KoLeoLoss(nn.Module):
    """
    Kozachenko-Leonenko differential-entropy regularizer: encourages a batch of L2-normalized
    features to spread out uniformly, by pushing each feature away from its single nearest
    neighbor in the batch. Applied to CLS-token features in small (paper: 16-sample) sub-batches.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, dim=-1, p=2)
        with torch.no_grad():
            sim = features @ features.t()
            sim.fill_diagonal_(-2.0)  # exclude self-match
            nn_idx = sim.argmax(dim=1)
        nearest = features[nn_idx]
        dist = (features - nearest).norm(dim=-1).clamp(min=self.eps)
        return -torch.log(dist).mean()


class GramAnchoringLoss(nn.Module):
    """
    L_Gram = || X_S X_S^T - X_G X_G^T ||_F^2 over L2-normalized [P, d] patch-feature matrices,
    where X_G comes from a frozen "Gram teacher" checkpoint captured earlier in training (paper:
    ~100-200k iterations in), not the live EMA teacher -- this is the mechanism that fixes dense
    (patch-level) feature collapse observed after long training, per the paper's own ablation.
    """

    def forward(self, student_patches: torch.Tensor, gram_teacher_patches: torch.Tensor) -> torch.Tensor:
        # student_patches, gram_teacher_patches: [B, P, d]
        xs = F.normalize(student_patches, dim=-1, p=2)
        xg = F.normalize(gram_teacher_patches, dim=-1, p=2)
        gram_s = torch.bmm(xs, xs.transpose(1, 2))  # [B, P, P]
        gram_g = torch.bmm(xg, xg.transpose(1, 2))
        return ((gram_s - gram_g) ** 2).sum(dim=(1, 2)).mean()


class DINOv3PretrainLoss(nn.Module):
    """
    Combined objective. Pretraining phase: L_DINO + L_iBOT + 0.1 * L_KoLeo.
    Refinement phase (post ~1M iters in the paper): adds w_Gram * L_Gram, with configurable
    weights on the other terms (paper: w_D, w_DK reweighted in this phase).
    """

    def __init__(self, out_dim: int, koleo_weight: float = 0.1):
        super().__init__()
        self.dino = DINOLoss(out_dim)
        self.ibot = IBOTLoss(out_dim)
        self.koleo = KoLeoLoss()
        self.gram = GramAnchoringLoss()
        self.koleo_weight = koleo_weight

    def forward(
        self,
        student_cls: torch.Tensor,
        teacher_cls: torch.Tensor,
        student_patches: torch.Tensor,
        teacher_patches: torch.Tensor,
        ibot_mask: torch.Tensor,
        gram_teacher_patches: torch.Tensor | None = None,
        gram_weight: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        l_dino = self.dino(student_cls, teacher_cls)
        l_ibot = self.ibot(student_patches, teacher_patches, ibot_mask)
        l_koleo = self.koleo(student_cls)
        total = l_dino + l_ibot + self.koleo_weight * l_koleo

        out = {"dino": l_dino, "ibot": l_ibot, "koleo": l_koleo, "total": total}
        if gram_teacher_patches is not None and gram_weight > 0:
            l_gram = self.gram(student_patches, gram_teacher_patches)
            out["gram"] = l_gram
            out["total"] = total + gram_weight * l_gram
        return out
