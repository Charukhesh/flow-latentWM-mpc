import os
import sys
import torch
import torch.nn as nn

# Add the cloned jepa_wms repo to the Python path
JEPA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../external/jepa_wms'))
if JEPA_PATH not in sys.path:
    sys.path.insert(0, JEPA_PATH)

# Import Meta's official builders
from src.models.ac_predictor import vit_ac_predictor

import torchvision.transforms.functional as TF

class VJepaEncoder(nn.Module):
    def __init__(self, checkpoint_path=None):
        super().__init__()
        
        print("Loading DINOv2 (ViT-Large) Foundation Model for Vision...")
        # Automatically downloads and loads Meta's official DINOv2 weights
        self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        
        # Freeze all weights - DINO already understands the world
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, image):
        """
        image shape: (B, 3, 256, 256) -> from our simulator
        """
        # Resize 256x256 to 224x224 for DINOv2
        if image.shape[-1] != 224:
            image = TF.resize(image, (224, 224), antialias=True)
            
        # Extract DINO patch tokens
        # DINO returns a dict. We want the spatial grid, ignoring the CLS token.
        # Output shape: (B, 256, 1024)
        features = self.encoder.forward_features(image)
        tokens = features['x_norm_patchtokens']
        
        return tokens

class VJepaPredictor(nn.Module):
    def __init__(self, action_dim=7, cond_dim=1024, checkpoint_path=None):
        super().__init__()
        
        # Initialize Action-Conditioned Predictor
        self.predictor = vit_ac_predictor(
            embed_dim=cond_dim, 
            action_dim=action_dim,
            predictor_embed_dim=1024,
            depth=12,
            num_heads=16,
            proprio_tokens=0,  # Disable proprioception
            num_frames=1, # Passing one frame at a time now
            tubelet_size=1, # Prevents 0x0 attention mask now
            img_size=256
        )
        
        # Freeze weights
        for param in self.predictor.parameters():
            param.requires_grad = False

    def forward(self, z_tokens, action):
        """
        z_tokens: (B, N_patches, cond_dim)
        action: (B, action_dim) OR (B, T, action_dim)
        """
        if action.ndim == 2:
            action = action.unsqueeze(1) # (B, 1, 7)
            
        # Predict next spatial tokens (states=None since proprio_tokens=0)
        # predictor returns: (x, action_features, proprio_features)
        z_next, _, _ = self.predictor(x=z_tokens, actions=action, states=None)
        return z_next

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        
        if 'predictor' in checkpoint:
            self.predictor.load_state_dict(checkpoint['predictor'], strict=False)
            print("Loaded Meta's V-JEPA Predictor weights successfully!")
        else:
            print("Error: Could not find predictor keys.")