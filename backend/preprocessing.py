"""
Preprocessing utilities for BraTS MRI data.
Matches the exact preprocessing used during training.
"""
import numpy as np
import nibabel as nib
from typing import Tuple, List, Optional
import cv2
import gzip
import shutil
import tempfile
import os

# Image size must match training
IMG_SIZE = 128


def smart_load_nifti(file_path: str) -> np.ndarray:
    """
    Smart NIfTI loader that handles various orientations and formats.
    Handles cases where .nii.gz files are not actually gzipped.
    
    Args:
        file_path: Path to .nii or .nii.gz file
        
    Returns:
        3D numpy array with shape (H, W, D)
    """
    try:
        # First, try loading directly
        nii = nib.load(file_path)
        data = nii.get_fdata().astype(np.float32)
    except Exception as e:
        # If file claims to be .nii.gz but isn't actually gzipped
        if file_path.endswith('.nii.gz'):
            # Check if it's actually gzipped
            with open(file_path, 'rb') as f:
                magic = f.read(2)
            
            if magic != b'\x1f\x8b':  # Not a gzip file
                # Rename to .nii and try again
                temp_dir = tempfile.mkdtemp()
                temp_nii = os.path.join(temp_dir, 'temp.nii')
                shutil.copy(file_path, temp_nii)
                try:
                    nii = nib.load(temp_nii)
                    data = nii.get_fdata().astype(np.float32)
                finally:
                    shutil.rmtree(temp_dir, ignore_errors=True)
            else:
                raise e
        else:
            raise e
    
    # Ensure 3D
    if data.ndim == 4:
        data = data[:, :, :, 0]
    
    return data


def brats_normalize(img: np.ndarray) -> np.ndarray:
    """
    BraTS-style normalization using percentile clipping.
    MUST match training preprocessing exactly.
    
    Args:
        img: 2D slice or 3D volume
        
    Returns:
        Normalized array scaled to [0, 1]
    """
    img = img.astype(np.float32)
    mask = img > 0
    
    if mask.sum() == 0:
        return img
    
    # Percentile-based normalization (matches training)
    p1, p99 = np.percentile(img[mask], (1, 99))
    img = np.clip(img, p1, p99)
    
    # Scale to [0, 1]
    return (img - p1) / (p99 - p1 + 1e-6)


def load_and_preprocess_modalities(
    flair_path: str,
    t1_path: str,
    t1ce_path: str,
    t2_path: str,
    target_size: Tuple[int, int] = (IMG_SIZE, IMG_SIZE)
) -> Tuple[dict, Tuple[int, int, int]]:
    """
    Load all 4 MRI modalities.
    
    Returns:
        - Dictionary with modality volumes
        - Original spatial shape
    """
    modalities = {
        'flair': smart_load_nifti(flair_path),
        't1': smart_load_nifti(t1_path),
        't1ce': smart_load_nifti(t1ce_path),
        't2': smart_load_nifti(t2_path)
    }
    
    # Store original shape (H, W, D)
    original_shape = modalities['flair'].shape
    
    return modalities, original_shape


def resize_slice(slice_2d: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize a 2D slice to target size."""
    return cv2.resize(slice_2d, target_size, interpolation=cv2.INTER_LINEAR)


def resize_prediction(pred: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
    """Resize prediction back to original size using nearest neighbor."""
    return cv2.resize(pred, original_size, interpolation=cv2.INTER_NEAREST)


def prepare_slice_batch(
    volume_4d: np.ndarray,
    slice_idx: int,
    target_size: Tuple[int, int] = (240, 240)
) -> np.ndarray:
    """
    Prepare a single slice for inference.
    
    Args:
        volume_4d: 4D volume (4, H, W, D)
        slice_idx: Slice index along depth axis
        target_size: Target size for model input
        
    Returns:
        Batch-ready tensor (1, 4, H, W)
    """
    # Extract slice from all modalities: (4, H, W)
    slice_4ch = volume_4d[:, :, :, slice_idx]
    
    # Resize each channel
    resized_channels = []
    for ch in range(4):
        resized = resize_slice(slice_4ch[ch], target_size)
        resized_channels.append(resized)
    
    # Stack and add batch dimension: (1, 4, H, W)
    batch = np.stack(resized_channels, axis=0)[np.newaxis, ...]
    
    return batch.astype(np.float32)


def postprocess_prediction(
    pred_volume: np.ndarray,
    brain_mask: np.ndarray
) -> np.ndarray:
    """
    Post-process the prediction volume.
    
    Args:
        pred_volume: Raw prediction volume (H, W, D)
        brain_mask: Brain mask
        
    Returns:
        Processed prediction with class mapping
    """
    # Apply brain mask
    pred_volume[~brain_mask] = 0
    
    # Map class 3 to label 4 (BraTS convention)
    pred_volume[pred_volume == 3] = 4
    
    return pred_volume.astype(np.uint8)


def run_length_encode(mask: np.ndarray) -> str:
    """
    Run-length encode a binary mask.
    
    Args:
        mask: Binary mask (flattened or will be flattened)
        
    Returns:
        RLE string
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def get_rle_per_class(prediction: np.ndarray) -> dict:
    """
    Get RLE encoding for each tumor class.
    
    Args:
        prediction: 3D prediction volume
        
    Returns:
        Dictionary with RLE for each class
    """
    rle_dict = {}
    
    for class_id in [1, 2, 4]:
        mask = (prediction == class_id).astype(np.uint8)
        if mask.sum() > 0:
            rle_dict[f"class_{class_id}"] = run_length_encode(mask)
        else:
            rle_dict[f"class_{class_id}"] = ""
    
    return rle_dict
