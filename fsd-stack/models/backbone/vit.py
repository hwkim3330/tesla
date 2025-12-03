"""
Vision Transformer (ViT) Backbone for Tesla FSD

ViT provides strong global context understanding, useful for V14+ architectures.
"""

import torch
import torch.nn as nn
from typing import List, Tuple
import timm


class ViTBackbone(nn.Module):
    """
    Vision Transformer backbone.

    Args:
        model_name: ViT variant
        pretrained: Use pretrained weights
        img_size: Input image size
        patch_size: Patch size
    """

    def __init__(
        self,
        model_name: str = 'vit_base_patch16_224',
        pretrained: bool = True,
        img_size: Tuple[int, int] = (480, 640),
        out_indices: Tuple[int, ...] = (3, 5, 7, 11),
    ):
        super().__init__()

        self.model_name = model_name
        self.out_indices = out_indices

        # Load ViT with feature extraction
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=img_size,
            num_classes=0,  # Remove classification head
        )

        # Get embedding dimension
        self.embed_dim = self.backbone.embed_dim

        # Feature dimensions for compatibility
        self.feature_dims = [self.embed_dim] * len(out_indices)

        # Hook to extract intermediate features
        self.features = {}
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to extract intermediate features."""
        def get_hook(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook

        for idx in self.out_indices:
            block = self.backbone.blocks[idx]
            block.register_forward_hook(get_hook(f'block_{idx}'))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-layer features."""
        B = x.shape[0]

        if x.dim() == 5:
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)

        # Forward pass (features collected via hooks)
        self.features = {}
        _ = self.backbone.forward_features(x)

        # Collect features
        outputs = []
        for idx in self.out_indices:
            feat = self.features[f'block_{idx}']
            # Reshape from (B, N_patches, D) to (B, D, H, W)
            if feat.dim() == 3:
                # Remove CLS token if present
                if hasattr(self.backbone, 'num_prefix_tokens'):
                    feat = feat[:, self.backbone.num_prefix_tokens:]

                # Calculate spatial dimensions
                h = w = int(feat.shape[1] ** 0.5)
                feat = feat.transpose(1, 2).reshape(-1, self.embed_dim, h, w)
            outputs.append(feat)

        return outputs

    @property
    def out_channels(self) -> List[int]:
        return self.feature_dims


if __name__ == '__main__':
    model = ViTBackbone('vit_base_patch16_224', pretrained=False, img_size=(224, 224))
    x = torch.randn(2, 3, 224, 224)
    features = model(x)
    print("ViT-Base features:")
    for i, f in enumerate(features):
        print(f"  Layer {model.out_indices[i]}: {f.shape}")
