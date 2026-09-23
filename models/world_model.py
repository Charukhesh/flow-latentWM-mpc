import torch
import torch.nn as nn

class MockVisualEncoder(nn.Module):
    """ Mimics V-JEPA's E_psi """
    def __init__(self, cond_dim=512):
        super().__init__()
        self.cond_dim = cond_dim

    def forward(self, image):
        # image: (B, C, H, W)
        B = image.shape[0]
        # Outputs a dummy physics-aware latent vector
        return torch.randn((B, self.cond_dim), device=image.device)


class MockLatentPredictor(nn.Module):
    """ Mimics V-JEPA's action-conditioned transition dynamics T_phi """
    def __init__(self, cond_dim=512, action_dim=7):
        super().__init__()
        self.cond_dim = cond_dim
        
        # A simple linear layer to simulate latent dynamics
        self.dynamics = nn.Linear(cond_dim + action_dim, cond_dim)

    def forward(self, z_t, action):
        # z_t: (B, cond_dim), action: (B, action_dim)
        # Concatenate state and action to predict next state
        x = torch.cat([z_t, action], dim=-1)
        z_next = z_t + 0.1 * self.dynamics(x) # Residual connection
        return z_next