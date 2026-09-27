import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.robomimic_dataset import RobomimicDataset
from models.flow_matching import FlowMatchingPolicy
from models.world_model import VJepaEncoder

class DummyEncoder(torch.nn.Module):
    def __init__(self, cond_dim=1024):
        super().__init__()
        self.cond_dim = cond_dim
        
    def forward(self, images):
        # Ignores the image and just outputs a zero-vector of the correct shape
        return torch.zeros((images.shape[0], self.cond_dim), device=images.device)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Hyperparams
    batch_size = 16
    epochs = 20
    action_dim = 7
    cond_dim = 1024
    horizon = 16
    lr = 1e-4

    # Dataset & DataLoader
    dataset_path = "data/lift/ph/robomimic/lift/ph/image.hdf5"
    dataset = RobomimicDataset(dataset_path, horizon=horizon)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Models
    print("Loading DINOv2 Vision Encoder...")
    visual_encoder = VJepaEncoder().to(device)
    flow_policy = FlowMatchingPolicy(action_dim=action_dim, cond_dim=cond_dim, param_cfg={}).to(device)
    
    # We only train the flow policy for now. (Encoder is theoretically frozen/pre-trained)
    optimizer = torch.optim.AdamW(flow_policy.parameters(), lr=lr, weight_decay=1e-6)
    
    # Training Loop
    print("\nStarting Training...")
    for epoch in range(epochs):
        flow_policy.train()
        epoch_loss = 0.0
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, actions in pbar:
            images = images.to(device)
            actions = actions.to(device)

            # Ensure images are (B, C, H, W)
            # Robomimic stores them as (B, H, W, 3). Permute to (B, 3, H, W).
            if images.shape[-1] == 3:
                images = images.permute(0, 3, 1, 2)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Encode Image (in reality, this would be V-JEPA - DINOv2)
            # images shape: (B, 3, 84, 84) -> z_t shape: (B, cond_dim)
            with torch.no_grad():
                z_tokens = visual_encoder(images)   # Shape: (B, 256, 1024)
                z_t = z_tokens.mean(dim=1)          # Squash to 1D: (B, 1024)
                
            # Compute ODE Loss
            # actions shape: (B, horizon, action_dim)
            loss = flow_policy.compute_loss(x1=actions, condition=z_t)
            
            # Backprop
            loss.backward()
            optimizer.step()

            torch.cuda.empty_cache() # Free up memory cache
            
            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
    print("\nTraining Complete! Saving checkpoint...")
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(flow_policy.state_dict(), "checkpoints/flow_policy_w_encoder.pth")
    print("Checkpoint saved to checkpoints/flow_policy_w_encoder.pth")

if __name__ == "__main__":
    train()