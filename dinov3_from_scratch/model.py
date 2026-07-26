"""
DINOv3 ViT-L/16 architecture, implemented from scratch against the published paper
(Siméoni et al., "DINOv3", arXiv:2508.10104) rather than Meta's own (gated) checkpoint or
released source.

IMPORTANT, stated plainly (see this directory's README.md for the full account): this file
implements the *architecture* DINOv3 describes -- it does not, and cannot, reproduce the
*pretrained weights*, which come from self-distillation training on Meta's 1.7-billion-image
LVD-1689M dataset over a large GPU cluster. A freshly-initialized model built from this file
outputs architecturally-correct but semantically meaningless (randomly-initialized) features. It
is not a drop-in substitute for the real facebook/dinov3-vitl16-pretrain-lvd1689m checkpoint used
by trellis2-mac-mps's actual generation pipeline, and is not presented as one.

ViT-L/16 configuration (standard "Large" sizing, consistent with DINOv2's own published ViT-L and
the original ViT paper's convention -- the paper's own numbers are given for the 7B teacher and
state smaller variants "scale proportionally downward" without spelling out ViT-L's exact numbers,
so this uses the well-established standard for a ViT-L backbone):
  - patch_size = 16
  - embed_dim = 1024
  - depth = 24 transformer blocks
  - num_heads = 16 (head_dim = 64)
  - SwiGLU MLP with hidden_dim = ceil(embed_dim * 8/3 / 64) * 64 (matches published SwiGLU FFN
    parameter-matching convention: a 4x-equivalent standard MLP has 8*embed_dim^2 parameters;
    SwiGLU's three GEMMs (gate, up, down) at hidden_dim h have 3*h*embed_dim, so h = 8/3*embed_dim
    matches, rounded to a multiple of 64 for hardware-friendly tiling)
  - 4 register tokens + 1 CLS token, in addition to patch tokens
  - RoPE (rotary position embeddings) over 2D patch coordinates, normalized to [-1, 1], with the
    paper's "RoPE-box jittering" augmentation (random rescale of the coordinate box during
    training) -- implemented here as a configurable, disabled-by-default training-time augment
  - Pre-norm transformer blocks, LayerNorm (the paper does not indicate a departure from
    LayerNorm to RMSNorm, unlike some contemporary ViT variants)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DINOv3Config:
    img_size: int = 256
    patch_size: int = 16
    in_chans: int = 3
    embed_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    num_register_tokens: int = 4
    mlp_ratio: float = 8 / 3  # SwiGLU parameter-matched ratio, see module docstring
    mlp_hidden_multiple_of: int = 64
    rope_base: float = 100.0  # rotary frequency base over the normalized [-1,1] coordinate box
    rope_box_jitter_range: tuple[float, float] = (0.5, 2.0)  # paper's RoPE-box jittering, train-time only
    drop_path_rate: float = 0.0
    layerscale_init: float = 1e-5


def _swiglu_hidden_dim(embed_dim: int, ratio: float, multiple_of: int) -> int:
    hidden = int(embed_dim * ratio)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


class PatchEmbed(nn.Module):
    """Non-overlapping conv patchify, standard ViT convention."""

    def __init__(self, cfg: DINOv3Config):
        super().__init__()
        self.patch_size = cfg.patch_size
        self.proj = nn.Conv2d(cfg.in_chans, cfg.embed_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, (
            f"input size ({H},{W}) must be divisible by patch_size {self.patch_size}"
        )
        x = self.proj(x)  # [B, embed_dim, H/p, W/p]
        _, D, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]
        return x, Hp, Wp


class RotaryPositionEmbedding2D(nn.Module):
    """
    2D RoPE over patch coordinates normalized to [-1, 1], per the paper's description
    ("coordinates in a normalized [-1,1] box... applies a bias in the multi-head attention
    operation depending on relative position"). Standard axial-2D-RoPE construction: half the
    head dimension is rotated using the x-coordinate, half using the y-coordinate, each with the
    usual 1D rotary sinusoidal frequency bank.

    CLS and register tokens are not spatial and receive no rotation (identity rotation, i.e. a
    zero-angle pair prepended for them so they pass through attention unrotated) -- consistent
    with how RoPE is applied to non-patch tokens in other RoPE-ViT designs (e.g. EVA-02, ViT-22B).
    """

    def __init__(self, head_dim: int, base: float = 100.0):
        super().__init__()
        assert head_dim % 4 == 0, "head_dim must be divisible by 4 for axial 2D RoPE (2 axes x 2 for sin/cos pairing)"
        self.head_dim = head_dim
        self.base = base
        quarter = head_dim // 4
        freqs = 1.0 / (base ** (torch.arange(0, quarter, dtype=torch.float32) / quarter))
        self.register_buffer("freqs", freqs, persistent=False)

    def _angles_for_axis(self, coord: torch.Tensor) -> torch.Tensor:
        # coord: [N] in [-1, 1] -> angles: [N, quarter]
        return coord.unsqueeze(-1) * self.freqs.to(coord.dtype).to(coord.device) * math.pi

    def build(self, Hp: int, Wp: int, num_prefix_tokens: int, device, dtype, jitter_scale: float = 1.0):
        ys = (torch.arange(Hp, device=device, dtype=torch.float32) + 0.5) / Hp * 2 - 1
        xs = (torch.arange(Wp, device=device, dtype=torch.float32) + 0.5) / Wp * 2 - 1
        ys = ys * jitter_scale
        xs = xs * jitter_scale
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid_y = grid_y.reshape(-1)
        grid_x = grid_x.reshape(-1)

        ang_x = self._angles_for_axis(grid_x)  # [N, quarter]
        ang_y = self._angles_for_axis(grid_y)  # [N, quarter]
        angles = torch.cat([ang_x, ang_y], dim=-1)  # [N, head_dim/2]

        if num_prefix_tokens > 0:
            prefix_angles = torch.zeros(num_prefix_tokens, angles.shape[-1], device=device, dtype=angles.dtype)
            angles = torch.cat([prefix_angles, angles], dim=0)

        cos = torch.cos(angles).to(dtype)
        sin = torch.sin(angles).to(dtype)
        # duplicate each half so it can be applied to the full head_dim via the standard
        # rotate-half trick: [cos, cos] / [sin, sin] each of length head_dim/2 -> head_dim
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
        return cos, sin  # each [N_total, head_dim]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # q, k: [B, num_heads, N, head_dim]; cos, sin: [N, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


class SwiGLU(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Attention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, num_heads, N, head_dim]
        q, k = apply_rope(q, k, rope_cos, rope_sin)
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class Block(nn.Module):
    def __init__(self, cfg: DINOv3Config):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.embed_dim)
        self.attn = Attention(cfg.embed_dim, cfg.num_heads)
        self.ls1 = LayerScale(cfg.embed_dim, cfg.layerscale_init)
        self.norm2 = nn.LayerNorm(cfg.embed_dim)
        hidden = _swiglu_hidden_dim(cfg.embed_dim, cfg.mlp_ratio, cfg.mlp_hidden_multiple_of)
        self.mlp = SwiGLU(cfg.embed_dim, hidden)
        self.ls2 = LayerScale(cfg.embed_dim, cfg.layerscale_init)

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x), rope_cos, rope_sin))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DINOv3ViTL16(nn.Module):
    """
    From-scratch DINOv3 ViT-L/16 backbone. Forward pass returns patch features (post-final-norm),
    the CLS token, and the register tokens separately -- matching how trellis2-mac-mps's actual
    image_feature_extractor.py consumes the real HF model's output (patch tokens as the dense
    conditioning signal, CLS as a pooled global signal).
    """

    def __init__(self, cfg: DINOv3Config | None = None):
        super().__init__()
        self.cfg = cfg or DINOv3Config()
        cfg = self.cfg

        self.patch_embed = PatchEmbed(cfg)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, cfg.num_register_tokens, cfg.embed_dim))
        self.rope = RotaryPositionEmbedding2D(cfg.embed_dim // cfg.num_heads, base=cfg.rope_base)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(cfg.embed_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.register_tokens, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @property
    def num_prefix_tokens(self) -> int:
        return 1 + self.cfg.num_register_tokens  # CLS + registers

    def forward(self, x: torch.Tensor, rope_box_jitter: bool = False) -> dict[str, torch.Tensor]:
        B = x.shape[0]
        patches, Hp, Wp = self.patch_embed(x)  # [B, N, D]

        jitter_scale = 1.0
        if rope_box_jitter and self.training:
            lo, hi = self.cfg.rope_box_jitter_range
            jitter_scale = float(torch.empty(1).uniform_(lo, hi).item())

        cos, sin = self.rope.build(
            Hp, Wp, num_prefix_tokens=self.num_prefix_tokens,
            device=x.device, dtype=patches.dtype, jitter_scale=jitter_scale,
        )

        cls = self.cls_token.expand(B, -1, -1)
        registers = self.register_tokens.expand(B, -1, -1)
        tokens = torch.cat([cls, registers, patches], dim=1)  # [B, 1+R+N, D]

        for block in self.blocks:
            tokens = block(tokens, cos, sin)
        tokens = self.norm(tokens)

        cls_out = tokens[:, 0]
        register_out = tokens[:, 1:1 + self.cfg.num_register_tokens]
        patch_out = tokens[:, 1 + self.cfg.num_register_tokens:]

        return {
            "cls_token": cls_out,                     # [B, D]
            "register_tokens": register_out,           # [B, R, D]
            "patch_tokens": patch_out,                  # [B, N, D]
            "patch_grid_size": (Hp, Wp),
        }


def build_dinov3_vitl16(**overrides) -> DINOv3ViTL16:
    cfg = DINOv3Config(**overrides)
    return DINOv3ViTL16(cfg)
