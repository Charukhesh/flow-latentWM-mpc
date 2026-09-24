import os
import sys
import torch
import torch.nn as nn

# Add the cloned jepa_wms repo to the Python path
JEPA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../external/jepa_wms'))
if JEPA_PATH not in sys.path:
    sys.path.insert(0, JEPA_PATH)

# Import Meta's official builders
from src.models.vision_transformer_v2 import vit_large
from src.models.ac_predictor import vit_ac_predictor

class VJepaEncoder(nn.Module):
    def __init__(self, checkpoint_path=None):
        super().__init__()
        # Initialize ViT-Large (Patch Size 16, embedding dim 1024)
        self.encoder = vit_large(patch_size=16, img_size=256)
        
        # Freeze weights
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    def forward(self, image):
        # image shape: (B, 3, 224, 224)
        # Returns spatial tokens: (B, Num_Patches, 768)
        tokens = self.encoder(image)
        return tokens

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location='cpu')

        if 'encoder' in checkpoint:
            self.encoder.load_state_dict(checkpoint['encoder'], strict=False)
            print("Loaded V-JEPA Encoder weights.")
        else:
            print("Note: No encoder weights found in this checkpoint. (Using random weights for Encoder, this is expected if using DINO).")

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