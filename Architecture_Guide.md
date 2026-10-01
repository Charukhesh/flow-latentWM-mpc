# Flow-Latent MPC: Architecture & Codebase Guide (v3.0 - Production)

## The Core Philosophy: "Propose, Verify, & Smooth"
Standard imitation learning (Behavioral Cloning) uses a **reactive policy**: it looks at an image and blindly executes a trajectory. If the environment is complex, it compounds errors and crashes. 

Our architecture uses a **predictive, temporally-smoothed policy**: 
1. **The Proposer (Flow Matching):** Generates multiple *candidate* physical trajectories simultaneously using a learned ODE vector field.
2. **The Verifier (Latent World Model):** Simulates the physical consequences of all candidate trajectories into the future using foundation models.
3. **The Planner:** Evaluates the imagined futures, picks the trajectory that safely reaches the goal.
4. **The Executor (Action Chunking):** Executes a smooth chunk of actions sequentially to eliminate generative jitter.

## The Big 3 "Sim-to-Sim" Optimizations
To bridge the gap between pure mathematics and physical robot success, this codebase relies on three critical robotics optimizations:
1. **Action Normalization:** Neural networks struggle with mixed-scale outputs (e.g., tiny $0.01$ XYZ translations vs. massive $1.0$ gripper commands). We dynamically compute the min/max of the expert dataset, scale all training actions to `[-1, 1]`, and un-scale them during MuJoCo deployment.
2. **Action Chunking:** Generative models often produce high-frequency jitter at the very beginning of a trajectory. Instead of executing 1 step and replanning (which causes compounding drift), we execute $8$ steps of the planned trajectory sequentially before regenerating.
3. **Real Goal Conditioning:** The Latent World Model calculates distance based on a true goal state (an image of the robot successfully holding the block), rather than static noise.

## File-by-File Breakdown

### 1. `data/robomimic_dataset.py` (The Data Engine)
* **What it does:** Reads the `.hdf5` Robomimic expert demonstrations.
* **Key Features:** Chops continuous human trajectories into sliding windows of length $H_{gen} = 16$. Calculates and saves global `action_min` and `action_max` tensors for downstream normalization.

### 2. `models/unet.py` (The Mathematical Backbone)
* **What it is:** A pure PyTorch 1D Conditional U-Net, utilizing **FiLM** (Feature-wise Linear Modulation) to inject DINOv2 visual conditioning into 1D convolutional residual blocks. 
* **Its role:** Acts as the universal function approximator for the Flow Matching vector field.

### 3. `models/flow_matching.py` (The Proposer)
* **What it is:** The wrapper class defining the ODE Flow Matching mathematics.
* **Key Functions:**
  * `compute_loss()`: Defines the straight-line ODE. Interpolates $x_t = (1-t)x_0 + t x_1$ and trains the U-Net to predict the optimal flow velocity $u_t = x_1 - x_0$.
  * `sample()`: The Euler integration inference loop. It duplicates the current visual condition $N$ times to parallelize $N=16$ diverse trajectory proposals on the GPU.

### 4. `models/world_model.py` (The Dual-Engine Verifier)
* **What it is:** The perception and physics engine, powered by Meta's foundation models.
* **Key Components:**
  * `VJepaEncoder (DINOv2)`: Uses `dinov2_vitl14` to compress a `(3, 256, 256)` image into a rich 3D-aware spatial grid of 256 patches.
  * `VJepaPredictor (V-JEPA-2-AC)`: Loaded from the `jepa_wm_droid.pth.tar` checkpoint. Takes the spatial grid and a 7D robot action, and rolls physics forward in the latent space. `proprio_tokens` and `tubelet_size` are strictly modified for spatial robotic planning.

### 5. `models/planner.py` (The MPC Orchestrator)
* **What it is:** The brain connecting the Proposer and Verifier.
* **The Forward Pass:**
  1. **Encode:** Compresses `current_image` and `goal_image` into 256 spatial patches.
  2. **Squash:** Averages the 256 patches into a single 1D vector (`cond_dim=1024`) for Flow Matching.
  3. **Propose:** Flow Matcher outputs $N=16$ candidate 7D trajectories.
  4. **Verify:** Feeds the 16 trajectories and the full 256-patch spatial grid into V-JEPA. Simulates 16 different physical futures simultaneously.
  5. **Evaluate & Execute:** Calculates the L1 Distance to the goal, returning the safest/closest trajectory.

### 6. `scripts/evaluate_robomimic.py` (The Virtual Robot)
* **What it is:** A Headless MuJoCo simulation environment (`robosuite==1.4.1`).
* **Optimizations:** 
  * Fixes the OpenGL memory leak via headless rendering (`has_renderer=False`) to `.mp4`.
  * Runs the neural networks in `torch.float16` (`autocast`) for a massive inference speedup.
  * Implements Early Stopping (`if reward == 1.0: break`) to prevent Out-of-Distribution network panic upon task completion.

## The Tensor Flow Diagram (Deployment)

```text
Image (1, 3, 256, 256) ───[Resize(224) + DINOv2]──> z_tokens (1, 256, 1024)
                                                          │
                                         [z_tokens.mean(dim=1)] 
                                                          │
                                                          ▼
                                                  z_flat (1, 1024)
                                                          │
                                                   [Duplicate N=16]
                                                          │
                                                          ▼
noise (16, H_gen, 7) ───[Flow_Model(z_flat)]──> candidate_actions (16, H_gen, 7)
                                                          │
                                                          ▼
             [V-JEPA Predictor rolls out actions for H_ver steps on z_tokens]
                                                          │
                                                          ▼
                           z_hat_future (16, 256, 1024)
                                                          │
                                                          ▼
                     [L1_Loss against z_goal (1, 256, 1024)]
                                                          │
                                                          ▼
                                 costs (16,)
                                                          │
                                                          ▼
                               argmin() = Index (e.g., 4)
                                                          │
                                                          ▼
                     WINNING TRAJECTORY (16, 7)
                                                          │
                                                          ▼
                     [Un-normalize from [-1, 1] back to real limits]
                                                          │
                                                          ▼
                 [Extract first 8 steps (Action Chunking)] ---> Robot Motors
```