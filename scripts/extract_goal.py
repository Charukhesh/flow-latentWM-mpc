import h5py
import imageio.v2 as imageio
import os

def extract_goal_image():
    dataset_path = "data/lift/ph/robomimic/lift/ph/image.hdf5"
    
    print(f"Opening dataset: {dataset_path}")
    with h5py.File(dataset_path, 'r') as f:
        # Get the first human demonstration (e.g., 'demo_0')
        demo_id = list(f['data'].keys())[0]
        
        # Access the agentview camera images
        images = f[f'data/{demo_id}/obs/agentview_image']
        
        # Grab the very last frame of the demonstration (the success state!)
        # Robomimic stores images as (Time, Height, Width, Channels)
        goal_img = images[-1] 
        
    # Save it to disk
    save_path = "data/goal_image.png"
    imageio.imwrite(save_path, goal_img)
    print(f"SUCCESS! Goal image saved to {save_path}")

if __name__ == "__main__":
    extract_goal_image()