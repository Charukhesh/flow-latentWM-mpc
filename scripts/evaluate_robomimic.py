import os
import sys
import time
import torch
import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
import imageio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from huggingface_hub import hf_hub_download
from models.flow_matching import FlowMatchingPolicy
from models.world_model import VJepaEncoder, VJepaPredictor
from models.planner import FlowLatentPlanner

def setup_planner(device):
    print("Loading Flow-Latent MPC Brain...")
    cfg = {
        'num_proposals': 16,
        'horizon_gen': 16,
        'horizon_ver': 5,
        'action_dim': 7,
        'flow_inference_steps': 4
    }
    cond_dim = 1024 
    
    # Loading Meta weights
    checkpoint_path = hf_hub_download(repo_id="facebook/jepa-wms", filename="jepa_wm_droid.pth.tar")
    
    # Flow Matching Proposer
    flow_policy = FlowMatchingPolicy(cfg['action_dim'], cond_dim, param_cfg={}).to(device)
    flow_checkpoint = "checkpoints/flow_policy_latest.pth"
    if os.path.exists(flow_checkpoint):
        flow_policy.load_state_dict(torch.load(flow_checkpoint, map_location=device))
        print("Loaded Sine-Wave Flow Matching weights!")
    
    # V-JEPA Verifier
    encoder = VJepaEncoder(checkpoint_path=checkpoint_path).to(device) 
    predictor = VJepaPredictor(action_dim=cfg['action_dim'], cond_dim=cond_dim, checkpoint_path=checkpoint_path).to(device)
    
    planner = FlowLatentPlanner(flow_policy, encoder, predictor, cfg).to(device)
    planner.eval()
    return planner

def run_simulation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    planner = setup_planner(device)
    
    print("\nStarting MuJoCo Simulator...")
    # Load Operational Space Control (OSC) for 7D actions (X,Y,Z, Roll,Pitch,Yaw, Gripper)
    controller_config = load_composite_controller_config(controller="BASIC")
    
    env = suite.make(
        env_name="Lift",               # Task: Lift the block
        robots="Panda",                # Robot: Franka Emika Panda
        controller_configs=controller_config,
        has_renderer=False,             # Pop up a window to watch the robot
        has_offscreen_renderer=True,   # Required to get camera images
        use_camera_obs=True,           
        camera_names="agentview",      # The main camera angle
        camera_heights=256,            # Match V-JEPA resolution
        camera_widths=256,
        control_freq=20,               # 20 Hz control loop
    )
    
    obs = env.reset()
    
    # Create a dummy goal image (In a real setup, this is an image of the block lifted)
    dummy_goal_image = torch.randn((1, 3, 256, 256), device=device)

    # Set up the Video Writer
    os.makedirs("videos", exist_ok=True)
    video_path = "videos/robot_dance.mp4"
    writer = imageio.get_writer(video_path, fps=20)
    print(f"\nSimulation Running! Saving video to {video_path}...")
    
    steps = 0
    while steps < 100:  # Let's run for 100 steps (5 seconds of video)
        img = obs["agentview_image"] 
        
        # Save the current frame to our video
        writer.append_data(img[::-1]) # MuJoCo images are flipped vertically by default
        
        # Prepare for neural network
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_tensor = img_tensor.to(device)
        
        start_time = time.time()
        
        # --- BONUS MEMORY FIX: Run in Half-Precision (Autocast) ---
        with torch.no_grad(), torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            best_trajectory, best_idx, _ = planner(img_tensor, dummy_goal_image)
            action = best_trajectory[0].float().cpu().numpy() 
            
        plan_time = (time.time() - start_time) * 1000
        
        # Execute action
        obs, reward, done, info = env.step(action)
        
        steps += 1
        print(f"Step {steps:03d}/100 | Plan Time: {plan_time:.1f} ms")

    # Close environments and save the video
    writer.close()
    env.close()
    print(f"Simulation finished. Check {video_path} to see your robot!")

if __name__ == "__main__":
    run_simulation()