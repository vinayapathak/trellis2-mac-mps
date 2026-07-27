"""
Phase 2 verification for SCFMDistillTrainer (MDT_DIST_PLAN.md): gradient flow to every trainable
student parameter, finite loss, no NaNs -- first on synthetic random data, then on one real
forward+backward pass using Phase 0's real encoded pilot data (/tmp/pilot_test) and this project's
real example conditioning image (assets/example_image/T.png).

Bypasses BasicTrainer's full __init__ (dataloader/optimizer/elastic/DDP plumbing -- pre-existing,
already-used machinery, not what this phase needs to verify) and directly exercises the new
research code (training_losses / _cfg_velocity / _sample_windows) via a bare, manually-wired
trainer object. Loads the real ~1.3B-param teacher/student/stopgrad and the real DINOv3
conditioning model -- run this only when you actually want to re-verify (heavy, not a fast unit
test; not part of tests/'s pytest suite for that reason).

Usage:
    SPARSE_BACKEND=pytorch ATTN_BACKEND=sdpa python scripts/verify_scfm_distill.py
"""
import os
os.environ.setdefault('SPARSE_BACKEND', 'pytorch')
os.environ.setdefault('ATTN_BACKEND', 'sdpa')

import copy
import glob
import sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from trellis2.trainers.flow_matching.scfm_distill import SCFMDistillTrainer
from trellis2.modules import sparse as sp
from trellis2.modules import image_feature_extractor as ife
import trellis2.models as models

DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
CACHE_DIR = os.path.expanduser('~/.cache/trellis2/huggingface')
TEACHER = 'microsoft/TRELLIS.2-4B/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16'
PILOT_SHAPE_LATENTS = '/tmp/pilot_test/shape_latents/shape_enc_next_dc_f16c32_fp16_512'
EXAMPLE_IMAGE = os.path.join(os.path.dirname(__file__), '..', 'assets', 'example_image', 'T.png')
IN_CHANNELS = 32  # slat_flow_img2shape_dit_1_3B_512's real config: in_channels=32


def build_trainer(teacher, student, stopgrad, k_over_n, p_uncond, image_cond_model=None):
    trainer = object.__new__(SCFMDistillTrainer)
    trainer.teacher = teacher
    trainer.stopgrad_model = stopgrad
    trainer.models = {'denoiser': student}
    trainer.training_models = {'denoiser': student}
    trainer.sigma_min = 1e-5
    trainer.n = 12
    trainer.k_over_n = k_over_n
    trainer.shift_range = (2.5, 4.5)
    trainer.ema_decay = 0.999
    trainer.guidance_strength = 7.5
    trainer.guidance_rescale = 0.5
    trainer.guidance_interval = (0.6, 1.0)
    trainer._coarse_skips = [2]
    trainer.p_uncond = p_uncond
    if image_cond_model is not None:
        trainer.image_cond_model = image_cond_model
    return trainer


def check_gradients(student, teacher, stopgrad, loss, label):
    assert torch.isfinite(loss), f'[{label}] Loss is not finite!'
    loss.backward()
    n_params = sum(1 for p in student.parameters() if p.requires_grad)
    n_with_grad = sum(1 for p in student.parameters() if p.requires_grad and p.grad is not None)
    n_nan = sum(
        1 for p in student.parameters()
        if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all()
    )
    teacher_grads = sum(1 for p in teacher.parameters() if p.grad is not None)
    stopgrad_grads = sum(1 for p in stopgrad.parameters() if p.grad is not None)
    print(f'[{label}] loss={loss.item():.6f} params={n_params} with_grad={n_with_grad} '
          f'nan_grad={n_nan} teacher_grads={teacher_grads} stopgrad_grads={stopgrad_grads}')
    assert n_with_grad == n_params, f'[{label}] Not all trainable params got gradients'
    assert n_nan == 0, f'[{label}] {n_nan} params have non-finite gradients'
    assert teacher_grads == 0, f'[{label}] Teacher should never receive gradients'
    assert stopgrad_grads == 0, f'[{label}] Stopgrad model should never receive gradients'
    student.zero_grad(set_to_none=True)


def synthetic_check(teacher, student, stopgrad):
    print('\n=== Synthetic check: gradient flow, finite loss, no NaNs ===')
    trainer = build_trainer(teacher, student, stopgrad, k_over_n=0.4, p_uncond=0.0)

    torch.manual_seed(0)
    B = 4
    tokens_per_sample = [37, 52, 19, 64]
    coords_list, feats_list = [], []
    for i, n_tok in enumerate(tokens_per_sample):
        coords = torch.randint(0, 32, (n_tok, 3), dtype=torch.int32)
        coords = torch.cat([torch.full((n_tok, 1), i, dtype=torch.int32), coords], dim=1)
        coords_list.append(coords)
        feats_list.append(torch.randn(n_tok, IN_CHANNELS))
    coords = torch.cat(coords_list, dim=0).to(DEVICE)
    feats = torch.cat(feats_list, dim=0).to(DEVICE)
    x_0 = sp.SparseTensor(feats=feats, coords=coords).to(DEVICE)

    cond_channels = 1024
    n_patches = 32 * 32
    fake_cond_feat = torch.randn(B, n_patches, cond_channels, device=DEVICE)
    trainer.encode_image = lambda image: fake_cond_feat  # stub -- see module docstring

    terms, _ = SCFMDistillTrainer.training_losses(trainer, x_0, cond='unused', neg_cond=None)
    check_gradients(student, teacher, stopgrad, terms['loss'], 'synthetic')
    print('Synthetic check PASSED.')


def real_data_check(teacher, student, stopgrad):
    print('\n=== Real-data check: real Phase 0 shape latents + real conditioning image ===')
    npz_files = sorted(glob.glob(os.path.join(PILOT_SHAPE_LATENTS, '*.npz')))
    assert len(npz_files) > 0, f"No real shape latents at {PILOT_SHAPE_LATENTS} -- run Phase 0 first"

    image_cond_model = ife.DinoV3FeatureExtractor(
        model_name='facebook/dinov3-vitl16-pretrain-lvd1689m', image_size=512,
    )
    image_cond_model.to(DEVICE)
    img = Image.open(EXAMPLE_IMAGE).convert('RGB')

    d = np.load(npz_files[0])
    coords = torch.from_numpy(d['coords'].astype(np.int32))
    coords = torch.cat([torch.zeros((coords.shape[0], 1), dtype=torch.int32), coords], dim=1).to(DEVICE)
    feats = torch.from_numpy(d['feats'].astype(np.float32)).to(DEVICE)
    assert torch.isfinite(feats).all(), "Real shape-latent feats contain non-finite values"
    x_0 = sp.SparseTensor(feats=feats, coords=coords).to(DEVICE)
    print(f'Real object: {os.path.basename(npz_files[0])}, {feats.shape[0]} tokens, '
          f'coords range [{coords[:,1:].min().item()}, {coords[:,1:].max().item()}]')

    # Single real object -> k/N rounds to different branches depending on k_over_n; exercise both
    # the self-distill (stopgrad) and teacher-guided branches explicitly on real data, since a
    # single real forward pass at the paper's default k/N=0.4 would only ever hit one of them.
    for k_over_n, label in [(0.0, 'real-data (self-distill branch)'), (1.0, 'real-data (teacher branch)')]:
        trainer = build_trainer(teacher, student, stopgrad, k_over_n=k_over_n, p_uncond=0.1,
                                 image_cond_model=image_cond_model)
        terms, _ = SCFMDistillTrainer.training_losses(trainer, x_0, cond=[img], neg_cond=None)
        assert terms['k_over_n_actual'] == k_over_n
        check_gradients(student, teacher, stopgrad, terms['loss'], label)

    print('Real-data check PASSED (both branches).')


def main():
    print('Loading real teacher/student/stopgrad (~1.3B params each)...')
    teacher = models.from_pretrained(TEACHER, cache_dir=CACHE_DIR).eval().to(DEVICE)
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = copy.deepcopy(teacher)
    for p in student.parameters():
        p.requires_grad_(True)
    stopgrad = copy.deepcopy(student).eval()
    for p in stopgrad.parameters():
        p.requires_grad_(False)

    synthetic_check(teacher, student, stopgrad)
    real_data_check(teacher, student, stopgrad)

    print('\nALL PHASE 2 CHECKS PASSED.')


if __name__ == '__main__':
    main()
