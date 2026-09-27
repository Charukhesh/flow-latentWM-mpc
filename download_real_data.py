import os
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError, HfHubHTTPError

def get_real_dataset():
    print("Hunting for the Real Robomimic Lift Dataset...")
    os.makedirs("data/lift/ph", exist_ok=True)
    
    # A list of known public repos and their file paths
    mirrors = [
        # Official Robomimic Dataset Repo (v1.4.1 format)
        {"repo": "robomimic/robomimic_datasets", "file": "lift/ph/image_v141.hdf5"},
        # Official Robomimic Dataset Repo (standard format)
        {"repo": "robomimic/robomimic_datasets", "file": "lift/ph/image.hdf5"},
        # A known public academic mirror
        {"repo": "ChaoyiPan/mip-dataset", "file": "robomimic/lift/ph/image.hdf5"}
    ]
    
    file_path = None
    
    for mirror in mirrors:
        print(f"\nTrying repository: {mirror['repo']}...")
        try:
            file_path = hf_hub_download(
                repo_id=mirror['repo'], 
                filename=mirror['file'],
                repo_type="dataset",
                local_dir="data/lift/ph"
            )
            print(f"SUCCESS! Dataset found and downloaded from {mirror['repo']}")
            break # Exit the loop, we got the file!
            
        except (EntryNotFoundError, RepositoryNotFoundError, HfHubHTTPError) as e:
            print(f"File not found in {mirror['repo']}. Moving to next mirror...")
            continue
            
    if file_path:
        print(f"\nYour real dataset is saved to: {file_path}")
        print("Update train_flow.py to point to this exact file path!")
    else:
        print("\nAll mirrors failed. Let me know and we will pull it from Columbia University's zip archives!")

if __name__ == "__main__":
    get_real_dataset()