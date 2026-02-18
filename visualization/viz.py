import os
from collections import Counter

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Subset

# Function to visualize a tensor image 
def show_tensor_image(img_tensor, label=None, save_path='sample_image.png'):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    img_denorm = img_tensor * std + mean
    img_denorm = img_denorm.permute(1,2,0).numpy()
    img_denorm = np.clip(img_denorm, 0, 1)
    plt.imshow(img_denorm)
    if label is not None:
        plt.title(f"Label: {label.item()}")
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()


# Function to plot label distribution
def plot_label_distribution(dataset_or_subset, save_path='label_distribution.png', title='Label Distribution'):
    
    # get labels
    # labels = [subset.dataset.labels[i] for i in subset.indices]
    if isinstance(dataset_or_subset, Subset):
        # Subset of Dataset
        labels = [dataset_or_subset.dataset.labels[i] for i in dataset_or_subset.indices]
    else:
        # Dataset
        labels = dataset_or_subset.labels
    
    counter = Counter(labels)
    print(Counter(labels))

    # classes and counts
    classes = list(counter.keys())
    counts = list(counter.values())
    
    # plot
    plt.figure(figsize=(6,4))
    plt.bar(classes, counts, color='skyblue')
    plt.xticks(classes)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(title)
    
    
    if save_path:
        dir_name = os.path.dirname(save_path)
        if dir_name:  # avoid empty string
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved label distribution to: {save_path}")

    plt.close()