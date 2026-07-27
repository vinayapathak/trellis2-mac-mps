from typing import *
import torch
from PIL import Image

from ....utils import dist_utils
# Not written upstream -- this file used to define its own DinoV2FeatureExtractor/
# DinoV3FeatureExtractor with hardcoded .cuda() calls throughout (no CUDA on Apple Silicon,
# and never exercised on this Mac since nothing had run training before). trellis2/modules/
# image_feature_extractor.py already has a proper device-tracked version of the exact same
# classes (a `self._device` attribute, generic `.to(device)`), used successfully by the real,
# working inference pipeline (trellis2_image_to_3d.py/trellis2_texturing.py). Reusing that
# module instead of maintaining a second, drifting, never-fixed copy.
from ....modules import image_feature_extractor as _ife


class ImageConditionedMixin:
    """
    Mixin for image-conditioned models.
    
    Args:
        image_cond_model: The image conditioning model.
    """
    def __init__(self, *args, image_cond_model: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_cond_model_config = image_cond_model
        self.image_cond_model = None     # the model is init lazily
        
    def _init_image_cond_model(self):
        """
        Initialize the image conditioning model.
        """
        with dist_utils.local_master_first():
            self.image_cond_model = getattr(_ife, self.image_cond_model_config['name'])(**self.image_cond_model_config.get('args', {}))
            self.image_cond_model.to(self.device)

    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Encode the image.
        """
        if self.image_cond_model is None:
            self._init_image_cond_model()
        features = self.image_cond_model(image)
        return features
        
    def get_cond(self, cond, **kwargs):
        """
        Get the conditioning data.
        """
        cond = self.encode_image(cond)
        kwargs['neg_cond'] = torch.zeros_like(cond)
        cond = super().get_cond(cond, **kwargs)
        return cond
    
    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference.
        """
        cond = self.encode_image(cond)
        kwargs['neg_cond'] = torch.zeros_like(cond)
        cond = super().get_inference_cond(cond, **kwargs)
        return cond

    def vis_cond(self, cond, **kwargs):
        """
        Visualize the conditioning data.
        """
        return {'image': {'value': cond, 'type': 'image'}}
    

class MultiImageConditionedMixin:
    """
    Mixin for multiple-image-conditioned models.
    
    Args:
        image_cond_model: The image conditioning model.
    """
    def __init__(self, *args, image_cond_model: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_cond_model_config = image_cond_model
        self.image_cond_model = None     # the model is init lazily
        
    def _init_image_cond_model(self):
        """
        Initialize the image conditioning model.
        """
        with dist_utils.local_master_first():
            self.image_cond_model = getattr(_ife, self.image_cond_model_config['name'])(**self.image_cond_model_config.get('args', {}))
    
    @torch.no_grad()
    def encode_images(self, images: Union[List[torch.Tensor], List[List[Image.Image]]]) -> List[torch.Tensor]:
        """
        Encode the image.
        """
        if self.image_cond_model is None:
            self._init_image_cond_model()
        seqlen = [len(i) for i in images]
        images = torch.cat(images, dim=0) if isinstance(images[0], torch.Tensor) else sum(images, [])
        features = self.image_cond_model(images)
        features = torch.split(features, seqlen)
        features = [feature.reshape(-1, feature.shape[-1]) for feature in features]
        return features
        
    def get_cond(self, cond, **kwargs):
        """
        Get the conditioning data.
        """
        cond = self.encode_images(cond)
        kwargs['neg_cond'] = [
            torch.zeros_like(cond[0][:1, :]) for _ in range(len(cond))
        ]
        cond = super().get_cond(cond, **kwargs)
        return cond
    
    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference.
        """
        cond = self.encode_images(cond)
        kwargs['neg_cond'] = [
            torch.zeros_like(cond[0][:1, :]) for _ in range(len(cond))
        ]
        cond = super().get_inference_cond(cond, **kwargs)
        return cond

    def vis_cond(self, cond, **kwargs):
        """
        Visualize the conditioning data.
        """
        H, W = cond[0].shape[-2:]
        vis = []
        for images in cond:
            canvas = torch.zeros(3, H * 2, W * 2, device=images.device, dtype=images.dtype)
            for i, image in enumerate(images):
                if i == 4:
                    break
                kh = i // 2
                kw = i % 2
                canvas[:, kh*H:(kh+1)*H, kw*W:(kw+1)*W] = image
            vis.append(canvas)
        vis = torch.stack(vis)
        return {'image': {'value': vis, 'type': 'image'}}
