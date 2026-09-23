import torch
import torch.nn as nn
from models.unet import ConditionalUnet1D 

class FlowMatchingPolicy(nn.Module):
    def __init__(self, action_dim, cond_dim, param_cfg):
        super().__init__()
        # 1. Network Backbone (Extracted from Honda)
        self.model = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=cond_dim,
            # Pass remaining U-Net params (down_dims, etc.) from your config
        )
        
    def compute_loss(self, x1, condition):
        """
        The Flow Matching Training Objective.
        x1: Expert trajectory shape (B, T, action_dim)
        condition: Latent visual state shape (B, cond_dim)
        """
        B, T, A = x1.shape
        device = x1.device
        
        # Sample random noise (x0) and time (t)
        x0 = torch.randn_like(x1)
        t = torch.rand((B,), device=device)
        
        # Reshape t for broadcasting: (B, 1, 1)
        t_expand = t.unsqueeze(-1).unsqueeze(-1)
        
        # 2. Flow Matching Math (sigma=0 straight line)
        xt = (1 - t_expand) * x0 + t_expand * x1
        ut = x1 - x0
        
        # Predict vector field
        vt = self.model(xt, t, global_cond=condition)
        
        # MSE Loss
        loss = torch.mean((vt - ut) ** 2)
        return loss

    @torch.no_grad()
    def sample(self, condition, num_proposals, horizon, action_dim, num_steps=16):
        """
        Euler Inference Loop for our "Generate & Verify" architecture.
        Generates `num_proposals` diverse trajectories for ONE current state.
        """
        device = condition.device
        
        # We need N proposals for the SAME visual condition. 
        # Repeat the condition vector N times.
        # condition shape: (1, cond_dim) -> (num_proposals, cond_dim)
        batched_cond = condition.repeat(num_proposals, 1)
        
        # Start from pure Gaussian noise
        x = torch.randn((num_proposals, horizon, action_dim), device=device)
        dt = 1.0 / num_steps
        
        # 3. Euler Integration
        for i in range(num_steps):
            # t needs to be shape (num_proposals,)
            t_val = i / num_steps
            t = torch.full((num_proposals,), t_val, device=device)
            
            # Predict velocity
            v = self.model(x, t, global_cond=batched_cond)
            
            # Euler step
            x = x + dt * v
            
        return x  # Shape: (num_proposals, horizon, action_dim)