"""
ResUNet2D Model Architecture for Brain Tumor Segmentation (BraTS)
MUST match the exact architecture used during training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block matching training architecture."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c)
        )
        self.short = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1),
            nn.BatchNorm2d(out_c)
        ) if in_c != out_c else nn.Identity()
    
    def forward(self, x):
        return F.relu(self.conv(x) + self.short(x))


class ResUNet2D(nn.Module):
    """
    ResUNet2D for Brain Tumor Segmentation - matches training architecture.
    
    Input: 4 channels (FLAIR, T1, T1ce, T2)
    Output: 4 classes (background, necrotic core, edema, enhancing tumor)
    """
    def __init__(self, n_classes=4):
        super().__init__()
        # Encoder
        self.e1 = ResidualBlock(4, 32)
        self.e2 = ResidualBlock(32, 64)
        self.e3 = ResidualBlock(64, 128)
        
        # Bottleneck
        self.bot = ResidualBlock(128, 256)
        
        # Decoder
        self.d3 = ResidualBlock(256 + 128, 128)
        self.d2 = ResidualBlock(128 + 64, 64)
        self.d1 = ResidualBlock(64 + 32, 32)
        
        # Final
        self.final = nn.Conv2d(32, n_classes, 1)
        
        # Operations
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Encoder
        x1 = self.e1(x)
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))
        
        # Bottleneck
        b = self.bot(self.pool(x3))
        
        # Decoder with skip connections
        x = self.d3(torch.cat([self.up(b), x3], 1))
        x = self.d2(torch.cat([self.up(x), x2], 1))
        x = self.d1(torch.cat([self.up(x), x1], 1))
        
        return self.final(x)


def load_model(model_path: str, device: torch.device) -> ResUNet2D:
    """Load trained model weights."""
    model = ResUNet2D(n_classes=4)
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle different save formats
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    # Remove 'module.' prefix if saved with DataParallel
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    return model
