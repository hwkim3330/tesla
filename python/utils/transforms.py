"""
Image transforms for Tesla FSD Vision

Handles preprocessing for the neural network:
- Normalization
- Augmentation
- Tensor conversion
"""

import torch
import numpy as np
from typing import Tuple, Optional


# ImageNet normalization (commonly used for pretrained backbones)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def normalize_image(
    image: np.ndarray,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD
) -> torch.Tensor:
    """
    Normalize image for neural network input

    Args:
        image: [H, W, 3] uint8 image (0-255)
        mean: Channel means
        std: Channel stds

    Returns:
        [3, H, W] normalized float tensor
    """
    # Convert to float
    img = image.astype(np.float32) / 255.0

    # Normalize
    img = (img - mean) / std

    # Convert to tensor [C, H, W]
    tensor = torch.from_numpy(img.transpose(2, 0, 1))

    return tensor


def denormalize_image(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD
) -> np.ndarray:
    """
    Denormalize tensor back to image

    Args:
        tensor: [3, H, W] or [B, 3, H, W] normalized tensor
        mean: Channel means
        std: Channel stds

    Returns:
        [H, W, 3] uint8 image
    """
    if tensor.dim() == 4:
        tensor = tensor[0]

    # Move to numpy
    img = tensor.detach().cpu().numpy()

    # Transpose to [H, W, C]
    img = img.transpose(1, 2, 0)

    # Denormalize
    img = img * std + mean

    # Clip and convert to uint8
    img = np.clip(img * 255, 0, 255).astype(np.uint8)

    return img


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    keep_aspect: bool = True
) -> np.ndarray:
    """
    Resize image to target size

    Args:
        image: Input image
        target_size: (height, width)
        keep_aspect: Whether to maintain aspect ratio

    Returns:
        Resized image
    """
    import cv2

    h, w = target_size

    if keep_aspect:
        # Calculate scale to fit
        scale = min(h / image.shape[0], w / image.shape[1])
        new_h = int(image.shape[0] * scale)
        new_w = int(image.shape[1] * scale)

        resized = cv2.resize(image, (new_w, new_h))

        # Pad to target size
        pad_h = (h - new_h) // 2
        pad_w = (w - new_w) // 2

        padded = np.zeros((h, w, 3), dtype=image.dtype)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        return padded
    else:
        return cv2.resize(image, (w, h))


def augment_image(
    image: np.ndarray,
    flip_prob: float = 0.5,
    brightness_range: Tuple[float, float] = (0.8, 1.2),
    contrast_range: Tuple[float, float] = (0.8, 1.2)
) -> np.ndarray:
    """
    Apply random augmentations to image

    Args:
        image: Input image
        flip_prob: Probability of horizontal flip
        brightness_range: Range for brightness adjustment
        contrast_range: Range for contrast adjustment

    Returns:
        Augmented image
    """
    aug_img = image.copy().astype(np.float32)

    # Random horizontal flip
    if np.random.random() < flip_prob:
        aug_img = aug_img[:, ::-1, :]

    # Random brightness
    brightness = np.random.uniform(*brightness_range)
    aug_img = aug_img * brightness

    # Random contrast
    contrast = np.random.uniform(*contrast_range)
    mean = aug_img.mean()
    aug_img = (aug_img - mean) * contrast + mean

    # Clip values
    aug_img = np.clip(aug_img, 0, 255).astype(np.uint8)

    return aug_img


class ImageTransform:
    """Image transformation pipeline"""

    def __init__(
        self,
        target_size: Optional[Tuple[int, int]] = None,
        normalize: bool = True,
        augment: bool = False
    ):
        self.target_size = target_size
        self.normalize = normalize
        self.augment = augment

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """Apply transformation pipeline"""
        img = image

        # Resize
        if self.target_size:
            img = resize_image(img, self.target_size)

        # Augment
        if self.augment:
            img = augment_image(img)

        # Normalize
        if self.normalize:
            img = normalize_image(img)
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1))

        return img
