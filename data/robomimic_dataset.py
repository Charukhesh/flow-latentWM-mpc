import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

class RobomimicDataset(Dataset):
    def __init__(self, dataset_path, horizon=16):
        super().__init__()
        self.horizon = horizon
        self.indices = []
        
        # Load dataset into memory (for small datasets) or keep file pointers
        self.data_dict = {'images': [], 'actions': []}
        
        print(f"Loading dataset from {dataset_path}...")
        with h5py.File(dataset_path, 'r') as f:
            demos = list(f['data'].keys())
            
            for demo_id in demos:
                demo = f[f'data/{demo_id}']
                
                # Extract actions (T, 7) and agentview images (T, 3, 84, 84)
                actions = np.array(demo['actions'])
                images = np.array(demo['obs/agentview_image'])
                
                num_steps = actions.shape[0]
                
                # Create sliding windows of length `horizon`
                # e.g., if demo is 100 steps, we get 100 - 16 + 1 = 85 chunks
                for i in range(num_steps - horizon + 1):
                    self.indices.append({
                        'demo_idx': len(self.data_dict['actions']),
                        'start_step': i
                    })
                
                self.data_dict['actions'].append(actions)
                self.data_dict['images'].append(images)
                
        print(f"Loaded {len(demos)} demonstrations.")
        print(f"Created {len(self.indices)} sliding window sequences of length {horizon}.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Find which demo and which step we are at
        idx_info = self.indices[idx]
        d_idx = idx_info['demo_idx']
        start = idx_info['start_step']
        end = start + self.horizon
        
        # We only need the FIRST image of the sequence as the condition (z_t)
        # Convert image to float and normalize to [0, 1]
        img = self.data_dict['images'][d_idx][start]
        img_tensor = torch.tensor(img, dtype=torch.float32) / 255.0
        
        # We need the FULL 16-step action sequence as the target (x_1)
        action_seq = self.data_dict['actions'][d_idx][start:end]
        action_tensor = torch.tensor(action_seq, dtype=torch.float32)
        
        return img_tensor, action_tensor