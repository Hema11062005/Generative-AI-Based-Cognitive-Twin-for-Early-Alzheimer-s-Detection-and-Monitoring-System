import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

TARGET_SHAPE = (64, 64, 64)

def load_nifti(path: str) -> np.ndarray:
    """Load .nii or .nii.gz, return float32 array."""
    img  = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    return data

def skull_strip_simple(volume: np.ndarray) -> np.ndarray:
    """
    Threshold-based pseudo skull-strip.
    For production, replace with HD-BET or FSL BET.
    """
    thresh = volume.mean() * 0.15
    mask   = volume > thresh
    return volume * mask

def resize_volume(volume: np.ndarray, target=TARGET_SHAPE) -> np.ndarray:
    factors = [t / s for t, s in zip(target, volume.shape)]
    return zoom(volume, factors, order=1)

def normalize(volume: np.ndarray) -> np.ndarray:
    v_min, v_max = volume.min(), volume.max()
    if v_max - v_min == 0:
        return volume
    return (volume - v_min) / (v_max - v_min)

def preprocess_nifti(path: str) -> np.ndarray:
    """Full pipeline → (1, 64, 64, 64, 1) tensor."""
    vol = load_nifti(path)
    vol = skull_strip_simple(vol)
    vol = resize_volume(vol)
    vol = normalize(vol)
    return vol.reshape(1, *TARGET_SHAPE, 1).astype(np.float32)

def extract_slice_for_display(volume_4d: np.ndarray, axis=1) -> np.ndarray:
    """Extract central slice for 2D Grad-CAM display."""
    vol = volume_4d[0, :, :, :, 0]
    mid = vol.shape[axis] // 2
    if axis == 0: return vol[mid, :, :]
    if axis == 1: return vol[:, mid, :]
    return vol[:, :, mid]