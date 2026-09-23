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
        """
        Closed-loop MPC step. 
        Returns the optimal trajectory and the index of the winner.
        """
        device = current_image.device
        
        # 1. Encode into Latent Space
        # z_t shape: (1, cond_dim)
        z_t = self.visual_encoder(current_image)
        z_goal = self.visual_encoder(goal_image)
        
        # 2. Propose N Trajectories
        # candidate_actions shape: (N, H_gen, action_dim)
        candidate_actions = self.flow_model.sample(
            condition=z_t, 
            num_proposals=self.N, 
            horizon=self.H_gen, 
            action_dim=self.action_dim, 
            num_steps=self.flow_steps
        )
        
        # 3. Vectorized Latent Verification (Rollout)
        # Duplicate z_t and z_goal for parallel batch processing
        z_hat = z_t.repeat(self.N, 1)        # (N, cond_dim)
        z_goal_batch = z_goal.repeat(self.N, 1) # (N, cond_dim)
        
        for j in range(self.H_ver):
            # Extract the j-th action from all N proposals
            current_actions = candidate_actions[:, j, :] # (N, action_dim)
            
            # Predict next latent state for all N futures simultaneously
            z_hat = self.latent_predictor(z_hat, current_actions)
            
        # 4. Evaluate Cost Function (L1 Distance)
        # Calculate MAE across the cond_dim -> shape (N,)
        costs = F.l1_loss(z_hat, z_goal_batch, reduction='none').mean(dim=-1)
        
        # 5. Execute 
        best_idx = torch.argmin(costs)
        best_trajectory = candidate_actions[best_idx]
        
        return best_trajectory, best_idx, costs