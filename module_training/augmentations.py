import os
import random
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image



   
#========================================================
#--------------- more dataset operations-----------------
def check_distribution(dataset):
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(int(label))

    counter = Counter(labels)

    print("Class distribution:")
    for cls in sorted(counter.keys()):
        print(f"  Class {cls}: {counter[cls]} samples")

    return counter, labels



def get_balanced_dataset(dataset, target_ratio_up=1.0):

    print("\n•••••••••••• Upsampling Minority Classes  ••••••••••••")

    
    counter, labels = check_distribution(dataset)

    max_count = max(counter.values())
    target_count = int(max_count * target_ratio_up)

    # augmentation
    aug_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    ])

    class_indices = {cls: [] for cls in counter.keys()}
    for idx, l in enumerate(labels):
        class_indices[l].append(idx)

    augmented_subsets = []

    class AugmentedSubset(Dataset):
        def __init__(self, base_dataset, indices, n_to_add, transform):
            self.base_dataset = base_dataset
            self.indices = indices
            self.n_to_add = n_to_add
            self.transform = transform

        def __len__(self):
            return self.n_to_add

        def __getitem__(self, idx):
            i = random.choice(self.indices)
            img, label = self.base_dataset[i]
            img = transforms.ToPILImage()(img)
            img = self.transform(img)
            img = transforms.ToTensor()(img)
            return img, label

    # oversample all classes
    for cls, cnt in counter.items():
        if cnt < target_count:
            n_to_add = target_count - cnt
            augmented_subsets.append(
                AugmentedSubset(dataset, class_indices[cls], n_to_add, aug_transform)
            )
            print(f"  Upsample class {cls}: {cnt} → {target_count}")

    new_dataset = ConcatDataset([dataset] + augmented_subsets)

    print(f"Total dataset size after augmentation: {len(new_dataset)}")
    return new_dataset


def downsampling_majority_class(dataset, target_ratio_down=1.0):
    print("\n•••••••••••• Downsampling Majority Classes ••••••••••••")

    counter, labels = check_distribution(dataset)

    min_count = min(counter.values())
    target_count = int(min_count * target_ratio_down)

    class_indices = {cls: [] for cls in counter.keys()}
    for idx, l in enumerate(labels):
        class_indices[l].append(idx)

    new_indices = []
    random.seed(42)

    for cls, indices in class_indices.items():
        if len(indices) > target_count:
            selected = random.sample(indices, target_count)
            print(f"  Downsample class {cls}: {len(indices)} → {target_count}")
        else:
            selected = indices
        new_indices.extend(selected)

    random.shuffle(new_indices)
    new_dataset = Subset(dataset, new_indices)

    print(f"Total dataset size after downsampling: {len(new_dataset)}")
    return new_dataset


def get_balanced_dataset_fixed_classes(dataset, target_classes, target_ratio_up=1.0):
    """
    oversample specified classes to balance the dataset.
    target_classes: List[int] 例如 [1, 3]
    """

    print("\n•••••••••••• Fixed-Class Upsampling ••••••••••••")

    # check distribution
    counter, labels = check_distribution(dataset)

    max_count = max(counter.values())
    target_count = int(max_count * target_ratio_up)

    # augmentation
    aug_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    ])

    # build class indices
    class_indices = {cls: [] for cls in counter.keys()}
    for idx, l in enumerate(labels):
        class_indices[l].append(idx)

    augmented_subsets = []

    class AugmentedSubset(Dataset):
        def __init__(self, base_dataset, indices, n_to_add, transform):
            self.base_dataset = base_dataset
            self.indices = indices
            self.n_to_add = n_to_add
            self.transform = transform

        def __len__(self):
            return self.n_to_add

        def __getitem__(self, idx):
            i = random.choice(self.indices)
            img, label = self.base_dataset[i]

            img = transforms.ToPILImage()(img)
            img = self.transform(img)
            img = transforms.ToTensor()(img)

            return img, label

    # oversample target_classes
    for cls in target_classes:
        cnt = counter[cls]
        if cnt < target_count:
            n_to_add = target_count - cnt
            augmented_subsets.append(
                AugmentedSubset(dataset, class_indices[cls], n_to_add, aug_transform)
            )
            print(f"  Upsample class {cls}: {cnt} → {target_count}")
        else:
            print(f"  Class {cls} already ≥ target count ({cnt}), skip")

    new_dataset = ConcatDataset([dataset] + augmented_subsets)

    print(f"Total dataset size after augmentation: {len(new_dataset)}")
    return new_dataset


def build_weighted_dataloader(dataset, batch_size):
    """
    Build a DataLoader with WeightedRandomSampler to handle class imbalance.

    Args:
        train_dataset: The original training dataset (must implement __getitem__ and __len__).
        batch_size: Batch size for DataLoader.

    Returns:
        dataloader: DataLoader with WeightedRandomSampler.
    """

    # Get class counts
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(int(label))

    counter = Counter(labels)
    total_samples = len(dataset)

    # Calculate weights for each class
    class_weights = {cls: total_samples/count for cls, count in counter.items()}

    # Assign weight to each sample
    sample_weights = [class_weights[int(label)] for label in labels]

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=total_samples, replacement=True)
    print(f"Using WeightedRandomSampler with class weights: {class_weights}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=True
    )

    return dataloader



class AugmentedBalancedDatasetOnline(Dataset):
    """
    Wraps a base dataset:
      • Applies augmentations only to training samples
      • Oversamples minority class automatically
      • Keeps eval/test unmodified
    """

    def __init__(self, base_dataset, mode="train"):
        self.base = base_dataset
        self.mode = mode

        # ------------------------------
        # 1. Extract labels
        # ------------------------------
        self.labels = base_dataset.labels
        label_counts = Counter(self.labels)
        self.num_classes = len(label_counts)

        # ------------------------------
        # 2. Oversampling for training
        # ------------------------------
        if mode == "train":
            max_count = max(label_counts.values())

            # Create new indices list
            self.indices = []
            for cls, count in label_counts.items():
                cls_indices = [i for i, lab in enumerate(self.labels) if lab == cls]
                repeat_factor = max_count // count
                remainder = max_count % count

                # Repeat + random remainder
                self.indices.extend(cls_indices * repeat_factor)
                self.indices.extend(random.sample(cls_indices, remainder))
        else:
            # Eval/test should NOT be oversampled
            self.indices = list(range(len(self.base)))

        # ------------------------------
        # 3. Define augmentation pipeline (TRAIN ONLY)
        # ------------------------------
        if mode == "train":
            self.aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                )
            ])
        else:
            self.aug = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        img, label = self.base[base_idx]

        # Only training: apply augmentation AFTER base transform
        if self.mode == "train":
            img = self.aug(img)

        return img, label


def get_loader_and_dataset_online_augmentations(cfg, split, batch_size=128, num_workers=4, shuffle=None):
    """
    Returns:
        dataset  -> EYEPACS wrapped with augmentation/balancing if train
        loader   -> DataLoader for that dataset
    """

    # Base dataset (no aug, no balancing)
    base_ds = EYEPACS(cfg, split=split)

    # Wrap if TRAIN
    if split == "train":
        ds = AugmentedBalancedDatasetOnline(base_ds, mode="train")

        # Training should ALWAYS shuffle
        shuffle_flag = True

    # Eval / Test: NO augmentation, NO oversampling
    else:
        ds = AugmentedBalancedDatasetOnline(base_ds, mode=split)

        # Eval/Test MUST NOT shuffle unless user explicitly forces it
        shuffle_flag = False if shuffle is None else shuffle

    # Create loader
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=num_workers,
        pin_memory=True,
    )

    return ds, loader


    
class MixupCutMix:
    def __init__(self, mixup_alpha=0.4, cutmix_alpha=1.0, prob=1.0):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob

    def __call__(self, batch):
        imgs, labels = batch   # imgs: [B,C,H,W], labels: [B] or one-hot

        if random.random() > self.prob:
            return imgs, labels

        B = imgs.size(0)

        # random sample 
        indices = torch.randperm(B).to(imgs.device)
        img2 = imgs[indices]
        label2 = labels[indices]

        r = random.random()

        if r < 0.5:
            # ---------- Mixup ----------
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            imgs = lam * imgs + (1 - lam) * img2
            labels = lam * labels + (1 - lam) * label2
            return imgs, labels

        else:
            # ---------- CutMix ----------
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            B, C, H, W = imgs.shape

            cut_w = int(W * np.sqrt(1 - lam))
            cut_h = int(H * np.sqrt(1 - lam))

            cx = np.random.randint(W)
            cy = np.random.randint(H)

            x1 = np.clip(cx - cut_w // 2, 0, W)
            y1 = np.clip(cy - cut_h // 2, 0, H)
            x2 = np.clip(cx + cut_w // 2, 0, W)
            y2 = np.clip(cy + cut_h // 2, 0, H)

            imgs[:, :, y1:y2, x1:x2] = img2[:, :, y1:y2, x1:x2]

            lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
            labels = lam * labels + (1 - lam) * label2

            return imgs, labels