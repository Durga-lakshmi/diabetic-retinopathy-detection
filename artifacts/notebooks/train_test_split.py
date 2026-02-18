import torch
import os
import pickle
from datasets.idrid.dataset import IDRID

def get_train_val_indices(cfg, save_path="artifacts/train_val_split.pkl"):
    """
    Creates a fixed train/val split once and saves it for reuse.
    """
    # If split already exists, load it
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            train_indices, val_indices = pickle.load(f)
        print(f"Loaded saved train/val split: Train={len(train_indices)}, Val={len(val_indices)}")
        return train_indices, val_indices

    # Otherwise, create the split
    full_dataset = IDRID(cfg, split='train')
    dataset_size = len(full_dataset)
    indices = torch.randperm(dataset_size).tolist()
    split_val = int(0.1 * dataset_size)

    val_indices = indices[:split_val]
    train_indices = indices[split_val:]

    # Save for future runs
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump((train_indices, val_indices), f)

    print(f"Created new train/val split: Train={len(train_indices)}, Val={len(val_indices)}")
    return train_indices, val_indices
