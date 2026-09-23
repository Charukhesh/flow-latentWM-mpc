import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowLatentPlanner(nn.Module):
    def __init__(self, flow_model, visual_encoder, latent_predictor, cfg):
        super().__init__()
        self.flow_model = flow_model
        self.visual_encoder = visual_encoder
        self.latent_predictor = latent_predictor
        
        self.N = cfg.get('num_proposals', 16)
        self.H_gen = cfg.get('horizon_gen', 16)
        self.H_ver = cfg.get('horizon_ver', 5)
        self.action_dim = cfg.get('action_dim', 7)
        self.flow_steps = cfg.get('flow_inference_steps', 4)

    @torch.no_grad()
    def forward(self, current_image, goal_image):
        device = current_image.device
        
        # 1. Encode into Spatial Latent Tokens
        # z_tokens shape: (B, N_Patches, 768)
        z_tokens = self.visual_encoder(current_image)
        z_goal_tokens = self.visual_encoder(goal_image)
        
        # --- THE FIX: Squash spatial tokens into a 1D vector for Flow Matching ---
        z_flat = z_tokens.mean(dim=1) # Shape: (B, 768)
        
        # 2. Propose N Trajectories (Using z_flat)
        candidate_actions = self.flow_model.sample(
            condition=z_flat, 
            num_proposals=self.N, 
            horizon=self.H_gen, 
            action_dim=self.action_dim, 
            num_steps=self.flow_steps
        )
        
        # 3. Vectorized Latent Verification 
        # (Using full z_tokens grid so V-JEPA can simulate physics)
        z_hat = z_tokens.repeat(self.N, 1, 1)           # (N, N_Patches, 768)
        z_goal_batch = z_goal_tokens.repeat(self.N, 1, 1) # (N, N_Patches, 768)
        
        for j in range(self.H_ver):
            current_actions = candidate_actions[:, j, :] # (N, action_dim)
            z_hat = self.latent_predictor(z_hat, current_actions)
            
        # 4. Evaluate Cost Function (L1 Distance across all patches)
        # Flatten the spatial dimension to compute a single scalar cost per trajectory
        cost_diff = F.l1_loss(z_hat, z_goal_batch, reduction='none')
        costs = cost_diff.mean(dim=[1, 2]) # Average over patches and features -> (N,)
        
        # 5. Execute 
        best_idx = torch.argmin(costs)
        best_trajectory = candidate_actions[best_idx]
        
        return best_trajectory, best_idx, costs