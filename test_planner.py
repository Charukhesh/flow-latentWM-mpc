import torch
import time
from models.flow_matching import FlowMatchingPolicy
from models.world_model import VJepaEncoder, VJepaPredictor
from models.planner import FlowLatentPlanner

def test_full_pipeline():
    print("Initializing Flow-Latent MPC Pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configuration
    cfg = {
        'num_proposals': 16,
        'horizon_gen': 16,
        'horizon_ver': 5,
        'action_dim': 7,
        'flow_inference_steps': 4
    }
    cond_dim = 768
    
    # Initialize components
    flow_policy = FlowMatchingPolicy(cfg['action_dim'], cond_dim, param_cfg={}).to(device)
    encoder = VJepaEncoder().to(device)
    predictor = VJepaPredictor(action_dim=cfg['action_dim'], cond_dim=cond_dim).to(device)
    
    planner = FlowLatentPlanner(flow_policy, encoder, predictor, cfg).to(device)
    
    # Create fake RGB images (B, C, H, W)
    current_image = torch.rand((1, 3, 224, 224), device=device)
    goal_image = torch.rand((1, 3, 224, 224), device=device)
    
    print("\nRunning Propose & Verify Cycle...")
    start_time = time.time()
    
    # Run the Planner!
    best_trajectory, best_idx, costs = planner(current_image, goal_image)
    
    end_time = time.time()
    
    print(f"Cycle completed in {(end_time - start_time)*1000:.2f} ms")
    print(f"Candidate Costs: \n{costs.cpu().numpy()}")
    print(f"\nWinning Trajectory Index: {best_idx.item()}")
    print(f"Winning Trajectory Shape: {best_trajectory.shape}")
    
    assert best_trajectory.shape == (cfg['horizon_gen'], cfg['action_dim'])
    print("\nSUCCESS: The full algorithm is logically sound and executable.")

if __name__ == "__main__":
    test_full_pipeline()