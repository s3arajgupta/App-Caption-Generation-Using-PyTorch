"""Image transformations preserving aspect ratio with symmetric zero-padding."""

from typing import Tuple, Union
from PIL import Image
import numpy as np

class ResizePadTransform:
    """Resize image to fit within target square dimension while preserving aspect ratio,
    then symmetrically pad the remaining borders.
    """
    def __init__(self, target_size: int = 224, fill_color: Tuple[int, int, int] = (0, 0, 0)):
        self.target_size = target_size
        self.fill_color = fill_color

    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply resize with aspect ratio preservation and symmetric padding."""
        width, height = image.size
        aspect_ratio = width / height

        if width > height:
            new_width = self.target_size
            new_height = max(1, int(self.target_size / aspect_ratio))
        else:
            new_height = self.target_size
            new_width = max(1, int(self.target_size * aspect_ratio))

        # High quality bilinear/bicubic resampling
        resized = image.resize((new_width, new_height), Image.Resampling.BILINEAR)

        # Compute symmetric padding
        pad_width = self.target_size - new_width
        pad_height = self.target_size - new_height
        
        pad_left = pad_width // 2
        pad_top = pad_height // 2

        # Create target canvas and paste resized image centered
        padded = Image.new("RGB", (self.target_size, self.target_size), self.fill_color)
        padded.paste(resized, (pad_left, pad_top))
        return padded

def get_transforms(target_size: int = 224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Return composed torchvision transforms if available, otherwise return PIL pipeline."""
    try:
        from torchvision import transforms
        return transforms.Compose([
            ResizePadTransform(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ])
    except ImportError:
        return ResizePadTransform(target_size)

def preprocess_image_to_numpy(
    image: Image.Image,
    target_size: int = 224,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
) -> np.ndarray:
    """Preprocess PIL image into normalized float32 CHW array: (3, H, W)."""
    transformer = ResizePadTransform(target_size)
    padded = transformer(image.convert("RGB"))
    arr = np.array(padded, dtype=np.float32) / 255.0  # HWC in [0, 1]
    
    # Normalize with mean & std
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    
    # Transpose to CHW
    return np.transpose(arr, (2, 0, 1))
