import h5py
import numpy as np
import os

def create_sine_hdf5(file_path):
    print(f"Creating Sine-Wave dataset at {file_path}...")
    
    with h5py.File(file_path, 'w') as f:
        data_group = f.create_group('data')
        
        # 10 demonstrations, 150 steps each
        num_demos = 10
        steps_per_demo = 150
        
        for i in range(num_demos):
            demo_group = data_group.create_group(f'demo_{i}')
            
            # 1. Fake Actions: A smooth physical motion (Circle / Sine wave)
            t = np.linspace(0, 4 * np.pi, steps_per_demo)
            actions = np.zeros((steps_per_demo, 7), dtype=np.float32)
            actions[:, 0] = np.sin(t)      # Robot moves back and forth on X axis
            actions[:, 1] = np.cos(t)      # Robot moves back and forth on Y axis
            actions[:, 6] = 1.0            # Gripper is closed
            demo_group.create_dataset('actions', data=actions)
            
            # 2. Fake Images (We just use blank images since we are testing action-learning)
            obs_group = demo_group.create_group('obs')
            images = np.zeros((steps_per_demo, 3, 84, 84), dtype=np.uint8)
            obs_group.create_dataset('agentview_image', data=images)
            
    print("Sine-Wave dataset created successfully!")

if __name__ == "__main__":
    os.makedirs(os.path.dirname("data/sine_image.hdf5"), exist_ok=True)
    create_sine_hdf5("data/sine_image.hdf5")