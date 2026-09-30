import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
import os

class RobomimicDataset(Dataset):
    def __init__(self, dataset_path, horizon=16):
        super().__init__()
        self.horizon = horizon
        self.indices = []
        self.data_dict = {'images': [], 'actions': []}
        
        print(f"Loading dataset from {dataset_path}...")
        with h5py.File(dataset_path, 'r') as f:
            demos = list(f['data'].keys())
            for demo_id in demos:
                demo = f[f'data/{demo_id}']
                actions = np.array(demo['actions'])
                images = np.array(demo['obs/agentview_image'])
                
                num_steps = actions.shape[0]
                for i in range(num_steps - horizon + 1):
                    self.indices.append({'demo_idx': len(self.data_dict['actions']), 'start_step': i})
                
                self.data_dict['actions'].append(actions)
                self.data_dict['images'].append(images)
                
        # Action Normalization
        all_actions = np.concatenate(self.data_dict['actions'], axis=0)
        self.action_min = torch.tensor(all_actions.min(axis=0), dtype=torch.float32)
        self.action_max = torch.tensor(all_actions.max(axis=0), dtype=torch.float32)
        
        # Prevent division by zero
        self.action_max = torch.where(self.action_max == self.action_min, self.action_max + 1e-6, self.action_max)
        
        # Save stats for the simulator to use!
        stats_path = "data/action_stats.pth"
        torch.save({'min': self.action_min, 'max': self.action_max}, stats_path)
        print(f"Saved Action Normalization Stats to {stats_path}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx_info = self.indices[idx]
        d_idx, start = idx_info['demo_idx'], idx_info['start_step']
        end = start + self.horizon
        
        # Image
        img = self.data_dict['images'][d_idx][start]
        img_tensor = torch.tensor(img, dtype=torch.float32) / 255.0
        if img_tensor.shape[-1] == 3:
            img_tensor = img_tensor.permute(2, 0, 1)
            
        # Target Actions
        action_seq = self.data_dict['actions'][d_idx][start:end]
        action_tensor = torch.tensor(action_seq, dtype=torch.float32)
        
        # Normalize to [-1, 1]
        action_tensor = 2.0 * (action_tensor - self.action_min) / (self.action_max - self.action_min) - 1.0
        
        return img_tensor, action_tensor