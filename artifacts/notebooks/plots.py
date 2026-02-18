import matplotlib.pyplot as plt
from collections import Counter
import torch
import os

def plot_and_save_class_distribution(dataset,save_path, title="Class Distribution"):
    """
    Count labels in a PyTorch dataset (dataset[i] -> (x, y)),
    plot a bar chart, and save it to a file.
    """
    # Make directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Collect labels
    labels = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        if torch.is_tensor(y):
            y = y.item()
        labels.append(int(y))

    counter = Counter(labels)
    classes = sorted(counter.keys())
    counts = [counter[c] for c in classes]

    # Plot
    plt.figure(figsize=(6, 4))
    plt.bar(classes, counts)
    plt.xticks(classes)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    # Save
    plt.savefig(save_path)
    plt.close()

    print(f"Saved class distribution plot to: {save_path}")
    print("Class counts:", counter)

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
    #labels = [subset.dataset.labels[i] for i in subset.indices]
    if isinstance(dataset_or_subset, Subset):
        # Subset of Dataset
        labels = [dataset_or_subset.dataset.labels[i] for i in dataset_or_subset.indices]
    else:
        # Dataset
        labels = dataset_or_subset.labels
    
    counter = Counter(labels)
    print(f'Bar Plot Numbers {Counter(labels)}')

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


from collections import Counter
from torch.utils.data import Subset

def print_split_distribution(name, dataset):
    # Case 1: Subset (train/val)
    if isinstance(dataset, Subset):
        labels = [dataset.dataset.labels[i] for i in dataset.indices]

    # Case 2: BalancedAugmentedDataset
    elif dataset.__class__.__name__ == "BalancedAugmentedDataset":
        labels = [label.item() for (_, label, _) in dataset.samples]

    # Case 3: Raw IDRID test set
    elif hasattr(dataset, "labels"):
        labels = dataset.labels

    else:
        raise TypeError(f"Unsupported dataset type for distribution: {type(dataset)}")

    counts = Counter(labels)

    print(f"\n{name} Class Distribution:")
    for cls, num in sorted(counts.items()):
        print(f"  Class {cls}: {num}")


import matplotlib.pyplot as plt
import torch

# ImageNet stats you used for normalization
mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3,1,1)
std  = torch.tensor([0.229, 0.224, 0.225]).reshape(3,1,1)

def show_sample(dataset, path, idx=0 ):
    img, label = dataset[idx]   # this is normalized tensor

    # UNNORMALIZE
    img = img * std + mean

    # Clamp to [0,1] to avoid numerical issues
    img = img.clamp(0,1)

    # Convert CHW -> HWC
    img_np = img.permute(1,2,0).cpu().numpy()

    plt.imshow(img_np)
    plt.title(f"Label: {label}")
    plt.axis("off")
    plt.savefig(path, dpi=200)
    print(f"Saved: {path}")

