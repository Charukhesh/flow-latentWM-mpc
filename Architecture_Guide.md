# Flow-Latent MPC: Architecture & Codebase Guide

## The Core Philosophy: "Propose & Verify"
Standard imitation learning (like the base Flow Matching paper) uses a **reactive policy**: it looks at an image and blindly executes a trajectory. If the environment is complex, it crashes. 
Our architecture uses a **predictive policy**: 
1. **The Proposer (Flow Matching):** Generates multiple *candidate* ways to move.
2. **The Verifier (Latent World Model):** Simulates the future physics of those candidates.
3. **The Planner:** Picks the trajectory that safely reaches the goal and executes it.

## File-by-File Breakdown

### 1. `models/unet.py` (The Mathematical Backbone)
* **What it is:** A pure PyTorch implementation of a 1-Dimensional Conditional U-Net. 
* **Where it came from:** Extracted directly from the Honda Flow Matching repository, completely stripped of bloated dependencies (`gym`, `cv2`, `torchcfm`).
* **How it works:** It takes a noisy trajectory sequence `(B, action_dim, T)` and a visual embedding `cond`, and uses **FiLM** (Feature-wise Linear Modulation) to inject the visual conditioning into 1D convolutional residual blocks. 
* **Its role in our paper:** It is simply a universal function approximator. It learns to predict the "velocity" required to push a noisy trajectory toward a clean, expert trajectory.

### 2. `models/flow_matching.py` (The Proposer)
* **What it is:** The wrapper class that defines the Flow Matching mathematics.
* **Why it exists:** To replace the massive `torchcfm` library with 20 lines of explicit math.
* **Key Functions:**
  * `compute_loss(x1, condition)`: Defines the straight-line ODE. It samples random noise ($x_0$), interpolates a point between noise and the expert data ($x_t = (1-t)x_0 + t x_1$), and calculates the target velocity ($u_t = x_1 - x_0$). It trains the U-Net to predict this $u_t$ using Mean Squared Error.
  * `sample(...)`: The Euler integration inference loop. **Crucial detail:** It is batched. We duplicate the current visual condition $N$ times to generate $N=16$ diverse trajectories in parallel on the GPU.

### 3. `models/world_model.py` (The Verifier)
* **What it is:** Currently a set of Mock PyTorch classes that output tensors of the exact shape we expect from a real Latent World Model (like V-JEPA 2-AC).
* **Key Components:**
  * `VisualEncoder ($E_\psi$)`: Compresses a massive `(3, 224, 224)` RGB image into a compact 1D latent vector ($z_t$). This vector mathematically encodes the physics and geometry of the scene.
  * `LatentPredictor ($T_\phi$)`: The "crystal ball." It takes the current latent state ($z_t$) and a 7D robot action ($a_t$), and predicts what the latent state will be one step in the future ($\hat{z}_{t+1}$). 

### 4. `models/planner.py` (The Heart of the Paper)
* **What it is:** The Model Predictive Control (MPC) orchestrator that connects the Proposer and the Verifier. 
* **The Forward Pass (Step-by-Step):**
  1. **Encode:** Compresses `current_image` and `goal_image` into latents ($z_t$, $z_{goal}$).
  2. **Propose:** Asks the `flow_model` for $N=16$ candidate 6D/7D trajectories ($H_{gen}$ steps long).
  3. **Verify (Latent Rollout):** Duplicates $z_t$ 16 times. Loops for $H_{ver}$ steps (e.g., 5 steps), feeding the 16 actions into the `LatentPredictor`. *This simulates 16 different physical futures simultaneously.*
  4. **Evaluate Cost:** Calculates the L1 Distance between the 16 imagined futures and the $z_{goal}$.
  5. **Execute:** Finds the trajectory with the lowest distance cost, discards the rest, and returns it to be sent to the robot's motors.

### 5. `test_*.py` (The Sanity Checks)
* **What they are:** Lightweight unit tests to guarantee tensor dimensions align without needing to load a massive 50GB dataset or launch a physics simulator.
* **Why keep them:** As you experiment with the model (e.g., changing from 7D actions to 10D actions, or swapping V-JEPA for LeWM), you run these tests first to ensure the pipeline hasn't broken.

## The Tensor Flow Diagram (Inference)
If you ever get confused about tensor shapes during debugging, refer to this map:

```text
Image (1, 3, 224, 224) ───[Encoder]──> z_t (1, 512)
                                          │
                                          ▼
                                     [Duplicate N=16]
                                          │
                                          ▼
noise (16, H_gen, 7) ───[Flow_Model(z_t)]──> candidate_actions (16, H_gen, 7)
                                          │
                                          ▼
             [LatentPredictor rolls out actions for H_ver steps]
                                          │
                                          ▼
                             z_hat_future (16, 512)
                                          │
                                          ▼
                   [L1_Loss against z_goal (1, 512)]
                                          │
                                          ▼
                                 costs (16,)
                                          │
                                          ▼
                               argmin() = Index 4
                                          │
                                          ▼
                     WINNING TRAJECTORY (16, 7) ---> Robot Motors
```