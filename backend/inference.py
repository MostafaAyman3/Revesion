"""
Inference pipeline for Brain Tumor Segmentation.
Matches the exact inference logic used during training/validation.
"""
import torch
import numpy as np
from typing import Tuple, Dict, Any
import tempfile
import os
import cv2

from model import ResUNet2D, load_model
from preprocessing import (
    load_and_preprocess_modalities,
    brats_normalize,
    IMG_SIZE
)


class BraTSInferencePipeline:
    """Complete inference pipeline for BraTS segmentation."""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize the inference pipeline.
        
        Args:
            model_path: Path to trained model weights
            device: Device to use ("auto", "cuda", or "cpu")
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Loading model on {self.device}...")
        self.model = load_model(model_path, self.device)
        self.model.eval()
        print("Model loaded successfully!")
        
        # Must match training size
        self.img_size = IMG_SIZE
    
    def predict(
        self,
        flair_path: str,
        t1_path: str,
        t1ce_path: str,
        t2_path: str,
        return_rle: bool = True,
        return_t1ce: bool = True
    ) -> Dict[str, Any]:
        """
        Run inference on a patient's MRI scans.
        Matches the exact prediction logic from training.
        """
        # Load modalities
        modalities, original_shape = load_and_preprocess_modalities(
            flair_path, t1_path, t1ce_path, t2_path
        )
        
        # Get dimensions (H, W, D)
        H, W, D = original_shape
        
        # Initialize prediction volume
        pred_3d = np.zeros((H, W, D), dtype=np.uint8)
        
        # Slice-wise inference (matching training code exactly)
        self.model.eval()
        with torch.no_grad():
            for z in range(D):
                # Build slice stack for all modalities
                slice_stack = []
                for mod_name in ['flair', 't1', 't1ce', 't2']:
                    s = modalities[mod_name][:, :, z]
                    s = brats_normalize(s)
                    s = cv2.resize(s, (self.img_size, self.img_size))
                    slice_stack.append(s)
                
                # Stack: shape (H, W, 4) then transpose to (4, H, W)
                x = np.stack(slice_stack, axis=-1)
                x_tensor = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
                
                # Forward pass
                out = self.model(x_tensor)
                pred = torch.argmax(out, dim=1).squeeze(0).cpu().numpy()
                
                # Resize back to original size
                pred = cv2.resize(pred.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                pred_3d[:, :, z] = pred
        
        # Map class 3 to label 4 (BraTS convention)
        pred_3d[pred_3d == 3] = 4
        
        result = {
            "prediction": pred_3d,
            "original_shape": original_shape
        }
        
        # Store T1ce volume for brain shell visualization
        if return_t1ce:
            result["t1ce_volume"] = modalities['t1ce']
        
        if return_rle:
            result["rle"] = get_rle_per_class(pred_3d)
        
        return result
    
    def predict_from_bytes(
        self,
        flair_bytes: bytes,
        t1_bytes: bytes,
        t1ce_bytes: bytes,
        t2_bytes: bytes,
        return_rle: bool = True,
        return_t1ce: bool = True
    ) -> Dict[str, Any]:
        """
        Run inference from file bytes (for API usage).
        """
        # Save bytes to temporary files
        temp_files = []
        try:
            for name, data in [
                ("flair", flair_bytes),
                ("t1", t1_bytes),
                ("t1ce", t1ce_bytes),
                ("t2", t2_bytes)
            ]:
                temp_file = tempfile.NamedTemporaryFile(
                    suffix=".nii.gz", delete=False
                )
                temp_file.write(data)
                temp_file.close()
                temp_files.append(temp_file.name)
            
            # Run inference (with T1ce for brain visualization)
            result = self.predict(
                temp_files[0],  # flair
                temp_files[1],  # t1
                temp_files[2],  # t1ce
                temp_files[3],  # t2
                return_rle=return_rle,
                return_t1ce=return_t1ce
            )
            
            return result
            
        finally:
            # Cleanup temp files
            for f in temp_files:
                try:
                    os.unlink(f)
                except:
                    pass


def get_rle_per_class(prediction: np.ndarray) -> dict:
    """Get RLE encoding for each tumor class."""
    rle_dict = {}
    
    for class_id in [1, 2, 4]:
        mask = (prediction == class_id).astype(np.uint8)
        if mask.sum() > 0:
            rle_dict[f"class_{class_id}"] = run_length_encode(mask)
        else:
            rle_dict[f"class_{class_id}"] = ""
    
    return rle_dict


def run_length_encode(mask: np.ndarray) -> str:
    """Run-length encode a binary mask."""
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def generate_3d_visualization_data(
    prediction: np.ndarray, 
    t1ce_volume: np.ndarray = None
) -> Dict[str, Any]:
    """
    Generate mesh data for 3D visualization using marching cubes.
    Creates proper 3D surfaces like the reference visualization code.
    
    Args:
        prediction: 3D prediction volume
        t1ce_volume: T1ce volume for brain shell background (optional)
        
    Returns:
        Dictionary with mesh data (vertices, faces) for Plotly Mesh3d
    """
    from skimage import measure
    
    # Color mapping
    colors = {
        1: "red",      # Necrotic Core
        2: "green",    # Edema
        4: "gold"      # Enhancing Tumor
    }
    
    labels = {
        1: "Necrotic Core",
        2: "Edema",
        4: "Enhancing Tumor"
    }
    
    visualization_data = {
        "meshes": [],
        "brain_mesh": None
    }
    
    # Generate brain shell mesh from T1ce volume
    if t1ce_volume is not None:
        try:
            # Threshold for brain tissue
            brain_threshold = np.percentile(t1ce_volume[t1ce_volume > 0], 10) if np.sum(t1ce_volume > 0) > 0 else 10
            verts, faces, _, _ = measure.marching_cubes(
                t1ce_volume > brain_threshold, 
                step_size=3
            )
            visualization_data["brain_mesh"] = {
                "vertices": {
                    "x": verts[:, 0].tolist(),
                    "y": verts[:, 1].tolist(),
                    "z": verts[:, 2].tolist()
                },
                "faces": {
                    "i": faces[:, 0].tolist(),
                    "j": faces[:, 1].tolist(),
                    "k": faces[:, 2].tolist()
                },
                "color": "gray",
                "opacity": 0.1,
                "name": "Brain"
            }
        except Exception as e:
            print(f"Warning: Could not generate brain mesh: {e}")
    
    # Generate tumor class meshes
    for class_id in [1, 2, 4]:
        mask = (prediction == class_id)
        voxel_count = np.sum(mask)
        
        if voxel_count > 0:
            try:
                # Use marching cubes to create smooth 3D surface
                verts, faces, _, _ = measure.marching_cubes(
                    mask.astype(float), 
                    step_size=2
                )
                
                visualization_data["meshes"].append({
                    "class_id": class_id,
                    "label": labels[class_id],
                    "color": colors[class_id],
                    "opacity": 0.6,
                    "vertices": {
                        "x": verts[:, 0].tolist(),
                        "y": verts[:, 1].tolist(),
                        "z": verts[:, 2].tolist()
                    },
                    "faces": {
                        "i": faces[:, 0].tolist(),
                        "j": faces[:, 1].tolist(),
                        "k": faces[:, 2].tolist()
                    },
                    "voxel_count": int(voxel_count)
                })
            except Exception as e:
                print(f"Warning: Could not generate mesh for class {class_id}: {e}")
                # Fallback to scatter if mesh fails
                coords = np.where(mask)
                step = 2
                x = coords[0][::step].tolist()[:30000]
                y = coords[1][::step].tolist()[:30000]
                z = coords[2][::step].tolist()[:30000]
                visualization_data["meshes"].append({
                    "class_id": class_id,
                    "label": labels[class_id],
                    "color": colors[class_id],
                    "type": "scatter",  # Mark as scatter fallback
                    "x": x,
                    "y": y,
                    "z": z,
                    "voxel_count": int(voxel_count)
                })
    
    return visualization_data
