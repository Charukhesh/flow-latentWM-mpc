import h5py
import numpy as np
import os

def create_mock_hdf5(file_path):
    print(f"Creating mock dataset at {file_path}...")
    
    with h5py.File(file_path, 'w') as f:
        # Robomimic datasets store everything under a 'data' group
        data_group = f.create_group('data')
        
        # Create 5 fake human demonstrations
        num_demos = 5
        steps_per_demo = 100
        
        for i in range(num_demos):
            demo_group = data_group.create_group(f'demo_{i}')
            
            # 1. Fake Actions: shape (T, 7) -> continuous random numbers
            actions = np.random.randn(steps_per_demo, 7).astype(np.float32)
            demo_group.create_dataset('actions', data=actions)
            
            # 2. Fake Images: shape (T, 3, 84, 84) -> uint8 pixels (0-255)
            obs_group = demo_group.create_group('obs')
            images = np.random.randint(0, 256, size=(steps_per_demo, 3, 84, 84), dtype=np.uint8)
            obs_group.create_dataset('agentview_image', data=images)
            
    print("Mock dataset created successfully!")

if __name__ == "__main__":
    os.makedirs(os.path.dirname("data/mock_image.hdf5"), exist_ok=True)
    create_mock_hdf5("data/mock_image.hdf5")