import os
import sys
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.flow_matching import FlowMatchingPolicy

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading Trained Flow Matching Policy...")
    
    # 1. Initialize Model
    action_dim = 7
    cond_dim = 512
    horizon = 16
    num_proposals = 16
    
    policy = FlowMatchingPolicy(action_dim=action_dim, cond_dim=cond_dim, param_cfg={}).to(device)
    
    # 2. Load Checkpoint
    checkpoint_path = "checkpoints/flow_policy_latest.pth"
    policy.load_state_dict(torch.load(checkpoint_path, map_location=device))
    policy.eval()
    
    # 3. Generate Trajectories
    print(f"Generating {num_proposals} trajectory proposals...")
    # Dummy visual condition (since we trained on blank images)
    dummy_latent = torch.randn((1, cond_dim), device=device)
    
    with torch.no_grad():
        trajectories = policy.sample(
            condition=dummy_latent,
            num_proposals=num_proposals,
            horizon=horizon,
            action_dim=action_dim,
            num_steps=16 # Let's use 16 Euler steps for a super smooth plot
        )
    
    # Move to CPU for plotting: Shape (16, 16, 7)
    trajectories = trajectories.cpu().numpy()
    
    # 4. Plot the Results
    print("Plotting results...")
    fig = plt.figure(figsize=(10, 5))
    
    # Plot X-axis movement over time
    ax1 = fig.add_subplot(1, 2, 1)
    for i in range(num_proposals):
        ax1.plot(trajectories[i, :, 0], alpha=0.6)
    ax1.set_title("Generated X-Axis Actions (Sine)")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("X Velocity/Position")
    
    # Plot Y-axis movement over time
    ax2 = fig.add_subplot(1, 2, 2)
    for i in range(num_proposals):
        ax2.plot(trajectories[i, :, 1], alpha=0.6)
    ax2.set_title("Generated Y-Axis Actions (Cosine)")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Y Velocity/Position")
    
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/trajectory_proposals.png")
    print("Saved plot to plots/trajectory_proposals.png")
    plt.show()

if __name__ == "__main__":
    evaluate()