import cv2
import numpy as np

TARGET_SIZE = 64

def preprocess_fake_3d(image_path):
    """
    Convert 2D image → Fake 3D volume (64,64,64,1)
    """
    
    # Load image (grayscale)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"Image not found: {image_path}")

    # Resize to 64x64
    img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE))

    # Normalize (0 to 1)
    img = img / 255.0

    # Create 3D volume by stacking
    volume = np.stack([img] * TARGET_SIZE, axis=0)   # (64,64,64)

    # Add channel dimension
    volume = volume[..., np.newaxis]                 # (64,64,64,1)

    return volume.astype(np.float32)