# Flow-Latent MPC: Architecture & Codebase Guide (v2.0)

## The Core Philosophy: "Propose & Verify"
Standard imitation learning (Behavioral Cloning) uses a **reactive policy**: it looks at an image and blindly executes a trajectory. If the environment is complex or perturbed, it crashes. 

Our architecture uses a **predictive policy**: 
1. **The Proposer (Flow Matching):** Generates multiple *candidate* physical trajectories simultaneously.
2. **The Verifier (Latent World Model):** Simulates the physical consequences of all candidate trajectories into the future.
3. **The Planner:** Evaluates the imagined futures, picks the trajectory that safely reaches the goal, and executes it.

## File-by-File Breakdown

### 1. `models/unet.py` (The Mathematical Backbone)
* **What it is:** A pure PyTorch implementation of a 1-Dimensional Conditional U-Net. 
* **How it works:** It takes a noisy trajectory sequence `(B, action_dim, T)` and a visual embedding `cond`, and uses **FiLM** (Feature-wise Linear Modulation) to inject the visual conditioning into 1D convolutional residual blocks. 
* **Its role:** It is the universal function approximator for the ODE vector field.

### 2. `models/flow_matching.py` (The Proposer)
* **What it is:** The wrapper class that defines the Flow Matching mathematics.
* **Key Functions:**
  * `compute_loss(x1, condition)`: Defines the straight-line ODE. It interpolates a point between noise and expert data ($x_t = (1-t)x_0 + t x_1$) and trains the U-Net to predict the velocity ($u_t = x_1 - x_0$).
  * `sample(...)`: The Euler integration inference loop. **Crucial detail:** It duplicates the current visual condition $N$ times to generate $N=16$ diverse trajectories in parallel on the GPU.

### 3. `models/world_model.py` (The Dual-Engine Verifier)
* **What it is:** The perception and physics engine, powered by Meta's foundation models.
* **Key Components:**
  * `VJepaEncoder (DINOv2)`: We use `dinov2_vitl14` to compress a `(3, 224, 224)` image into a rich 3D-aware spatial grid of 256 patches.
  * `VJepaPredictor (V-JEPA-2-AC)`: Loaded from the `jepa_wm_droid.pth.tar` checkpoint. It takes the spatial grid and a 7D robot action, and rolls physics forward in the latent space. We disable `proprio_tokens` and set `tubelet_size=1` to optimize for spatial (not video) planning.

### 4. `models/planner.py` (The Heart of the Paper)
* **What it is:** The Model Predictive Control (MPC) orchestrator.
* **The Forward Pass (Step-by-Step):**
  1. **Encode:** Compresses `current_image` and `goal_image` into 256 spatial patches.
  2. **Squash:** Averages the 256 patches into a single 1D vector (`cond_dim=1024`) for the Flow Matcher.
  3. **Propose:** Asks the Flow Matcher for $N=16$ candidate 7D trajectories.
  4. **Verify (Latent Rollout):** Feeds the 16 trajectories and the full 256-patch spatial grid into V-JEPA. Simulates 16 different physical futures simultaneously.
  5. **Evaluate Cost:** Calculates the L1 Distance between the 16 imagined futures and the $z_{goal}$.
  6. **Execute:** Returns the trajectory with the lowest distance cost.

### 5. `scripts/evaluate_robomimic.py` (The Virtual Robot)
* **What it is:** A Headless MuJoCo simulation environment (`robosuite==1.4.1`).
* **Optimizations:** Uses `torch.autocast(dtype=torch.float16)` and disabled OpenGL popup rendering (`has_renderer=False`) to prevent RAM overflow. Saves the output to a high-quality `.mp4` video.

## The Tensor Flow Diagram (Inference)
Use this map for debugging shape mismatches:

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
                               argmin() = Index (e.g., 7)
                                                          │
                                                          ▼
                     WINNING TRAJECTORY (16, 7) ---> Robot Motors
```
