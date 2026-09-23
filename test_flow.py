import torch
from models.flow_matching import FlowMatchingPolicy

def test_flow_matching():
    print("Initializing Flow Matching Proposer...")
    
    # Dummy parameters matching Robomimic (e.g., Transport task)
    action_dim = 7       # 3D pos, 3D rot, 1D gripper
    cond_dim = 512       # Assuming a 512-dim latent vector from V-JEPA
    batch_size = 32
    horizon = 16         # H_gen
    num_proposals = 16   # N candidates for the World Model
    
    policy = FlowMatchingPolicy(action_dim, cond_dim, param_cfg={})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    
    # --- TEST 1: Training Loss ---
    print("\nTesting Training Step...")
    dummy_expert_actions = torch.randn((batch_size, horizon, action_dim), device=device)
    dummy_latent_states = torch.randn((batch_size, cond_dim), device=device)
    
    loss = policy.compute_loss(dummy_expert_actions, dummy_latent_states)
    print(f"Training Loss shape: {loss.shape} | Value: {loss.item():.4f}")
    assert loss.ndim == 0, "Loss should be a scalar!"
    
    # --- TEST 2: Multi-Proposal Inference ---
    print("\nTesting Multi-Proposal Inference...")
    single_latent_state = torch.randn((1, cond_dim), device=device)
    
    candidate_trajectories = policy.sample(
        condition=single_latent_state,
        num_proposals=num_proposals,
        horizon=horizon,
        action_dim=action_dim,
        num_steps=4 # Fast inference test
    )
    
    print(f"Output Trajectories Shape: {candidate_trajectories.shape}")
    assert candidate_trajectories.shape == (num_proposals, horizon, action_dim)
    print("SUCCESS: Engine is running perfectly.")

if __name__ == "__main__":
    test_flow_matching()