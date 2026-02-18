from .idrid.dataset import IDRID
from .idrid.graham_preprocessed_dataset import GrahamIDRID
from .eyepacs.dataset import EYEPACS
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset, WeightedRandomSampler
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
import random
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from collections import Counter
import math
from torch.utils.data import Dataset
import torch
import random
from torchvision import transforms
from collections import Counter



def get_dataset(cfg, indices=None, split='train'):

    # ----------------------------------------
    # 1. Build underlying dataset consistently
    # ----------------------------------------
    if cfg.dataset.name == 'IDRID':
        if split in ('train', 'eval'):
            # ALWAYS load the training dataset
            dataset = IDRID(cfg, split='train')
        elif split == 'test':
            dataset = IDRID(cfg, split='test')
        else:
            raise ValueError(f"Unknown split: {split}")

        # ----------------------------------------
        # 2. Subset logic becomes valid now
        # ----------------------------------------
        if split == 'train':
            if indices is None:
                raise RuntimeError("Train indices were not provided to get_dataset().")
            subset = Subset(dataset, indices)

        elif split == 'eval':
            if indices is None:
                raise RuntimeError("Eval indices were not provided.")
            subset = Subset(dataset, indices)

        elif split == 'test':
            subset = dataset  # test never uses subsets.

    # ----------------------------------------
    # EYEPACS unchanged
    # ----------------------------------------
    elif cfg.dataset.name == 'EYEPACS':
        dataset = EYEPACS(cfg, split)
        subset = dataset

    else:
        raise ValueError(f"Dataset '{cfg.dataset.name}' unknown")

    dataloader = DataLoader(
        subset,
        batch_size=cfg.dataset.batch_size,
        shuffle=(split == 'train'),
        pin_memory=True,
    )

    return subset, dataloader

def get_graham_idrid_dataset(cfg, indices=None, split='train'):

    # ----------------------------------------
    # 1. Build underlying dataset consistently
    # ----------------------------------------
    if cfg.dataset.name == 'IDRID':
        if split in ('train', 'eval'):
            # ALWAYS load the training dataset
            dataset = GrahamIDRID(cfg, split='train')
        elif split == 'test':
            dataset = GrahamIDRID(cfg, split='test')
        else:
            raise ValueError(f"Unknown split: {split}")

        # ----------------------------------------
        # 2. Subset logic becomes valid now
        # ----------------------------------------
        if split == 'train':
            if indices is None:
                raise RuntimeError("Train indices were not provided to get_dataset().")
            subset = Subset(dataset, indices)

        elif split == 'eval':
            if indices is None:
                raise RuntimeError("Eval indices were not provided.")
            subset = Subset(dataset, indices)

        elif split == 'test':
            subset = dataset  # test never uses subsets.

    # ----------------------------------------
    # EYEPACS unchanged
    # ----------------------------------------
    elif cfg.dataset.name == 'EYEPACS':
        dataset = EYEPACS(cfg, split)
        subset = dataset

    else:
        raise ValueError(f"Dataset '{cfg.dataset.name}' unknown")

    dataloader = DataLoader(
        subset,
        batch_size=cfg.dataset.batch_size,
        shuffle=(split == 'train'),
        pin_memory=True,
    )

    return subset, dataloader




def get_dataset_0(cfg, split='train'):
    #global val_indices, train_indices

    dataset = None
    if cfg.dataset.name == 'IDRID':
        dataset = IDRID(cfg, split)
        #dataset = IDRID_5(cfg, split) # for 5 classes classification
        # Split dataset into train and eval (remenber to change!!!!!)
        if split == 'train':  
            dataset_size = len(dataset)   
            # !! if k-fold cross-validation，split done in main_k_fold.py , don't split here, don't use below code    
            indices = list(range(dataset_size))
            split_val = int(0.1 * dataset_size) # 10% for validation
            ##print(f"Dataset size: {dataset_size}, Train size: {dataset_size - split_val}, Eval size: {split_val}")
            
            
            np.random.seed(42)
            np.random.shuffle(indices)
            val_indices = indices[:split_val]
            train_indices = indices[split_val:]

            train_subset = Subset(dataset, train_indices)
            shuffle = True

            val_subset = Subset(dataset, val_indices)
            shuffle = False


            
            train_loader = DataLoader(
                train_subset,
                batch_size=cfg.dataset.batch_size,
                shuffle=True,
                # num_workers=4,
                pin_memory=True
            )

            val_loader = DataLoader(
                val_subset,
                batch_size=cfg.dataset.batch_size,
                shuffle=False,      
                # num_workers=4,
               pin_memory=True
            )

            return train_subset, train_loader, val_subset, val_loader #k-fold -> dataset,dataloader because in main.py do split || no k-fold -> train_subset, train_loader, val_subset, val_loader
            #if k-fold cross-validation
            #dataloader = DataLoader(
            #    dataset,
            #    batch_size=cfg.dataset.batch_size,
            #    shuffle=True,
            #    # num_workers=4,
            #    pin_memory=True
            #)
            #return dataset, dataloader


        #elif split == 'eval':
        #    if val_indices is None:
        #        raise RuntimeError("val_indices not initialized. Call get_dataset(split='train') first.")
        #    subset = Subset(dataset, val_indices)
        #    shuffle = False

        elif split == 'test':
            # Use the full dataset for testing
            dataloader = DataLoader(
            dataset,
            batch_size=cfg.dataset.batch_size,
            shuffle=False,
            # num_workers=4,
            pin_memory=True
        )
            return dataset, dataloader
    
        else:
            raise ValueError(f"Unknown split: {split}")




    elif cfg.dataset.name == 'EYEPACS':
        # TODO:
        dataset = EYEPACS(cfg, split)
        
        #print(f"Dataset samples: {len(dataset)}")
        # Use the full dataset for the specified split
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.dataset.batch_size,
            shuffle=(split=='train'),
            # num_workers=4,
            pin_memory=True
        )


    else:
        raise ValueError(f'Dataset "{cfg.dataset.name}" unknown')





def get_dataset_fold(cfg, split='train'):
    #global val_indices, train_indices

    dataset = None
    if cfg.dataset.name == 'IDRID':
        dataset = IDRID(cfg, split)
        #dataset = IDRID_5(cfg, split) # for 5 classes classification
        # Split dataset into train and eval (remenber to change!!!!!)
        if split == 'train':  
            dataset_size = len(dataset)   
            # !! if k-fold cross-validation，split done in main_k_fold.py , don't split here, don't use below code    
            #indices = list(range(dataset_size))
            #split_val = int(0.1 * dataset_size) # 10% for validation
            ##print(f"Dataset size: {dataset_size}, Train size: {dataset_size - split_val}, Eval size: {split_val}")
            
            
            #np.random.seed(42)
            #np.random.shuffle(indices)
            #val_indices = indices[:split_val]
            #train_indices = indices[split_val:]

            #train_subset = Subset(dataset, train_indices)
            #shuffle = True

            #val_subset = Subset(dataset, val_indices)
            #shuffle = False

            #train_loader = DataLoader(
            #    train_subset,
            #    batch_size=cfg.dataset.batch_size,
            #    shuffle=True,
            #    # num_workers=4,
            #    pin_memory=True
            #)

            #val_loader = DataLoader(
            #    val_subset,
            #    batch_size=cfg.dataset.batch_size,
            #    shuffle=False,      
            #    # num_workers=4,
            #   pin_memory=True
            #)

            #return train_subset, train_loader, val_subset, val_loader #k-fold -> dataset,dataloader because in main.py do split || no k-fold -> train_subset, train_loader, val_subset, val_loader
            #if k-fold cross-validation
            dataloader = DataLoader(
                dataset,
                batch_size=cfg.dataset.batch_size,
                shuffle=True,
                # num_workers=4,
                pin_memory=True
            )
            return dataset, dataloader


        #elif split == 'eval':
        #    if val_indices is None:
        #        raise RuntimeError("val_indices not initialized. Call get_dataset(split='train') first.")
        #    subset = Subset(dataset, val_indices)
        #    shuffle = False

        elif split == 'test':
            # Use the full dataset for testing
            dataloader = DataLoader(
            dataset,
            batch_size=cfg.dataset.batch_size,
            shuffle=False,
            # num_workers=4,
            pin_memory=True
        )
            return dataset, dataloader
    
        else:
            raise ValueError(f"Unknown split: {split}")




    elif cfg.dataset.name == 'EYEPACS':
        # TODO:
        dataset = EYEPACS(cfg, split)
        
        #print(f"Dataset samples: {len(dataset)}")
        # Use the full dataset for the specified split
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.dataset.batch_size,
            shuffle=(split=='train'),
            # num_workers=4,
            pin_memory=True
        )
        
        if split != 'train':
            return dataset, dataloader
        else:
            dataset_5 = EYEPACS_5(cfg, split) # for 5 classes classification 
            dataloaders_5 = DataLoader(
                dataset_5,
                batch_size=cfg.dataset.batch_size,
                shuffle=(split=='train'),
                # num_workers=4,
                pin_memory=True
            )
            return dataset, dataloader, dataset_5, dataloaders_5  # --- IGNORE ---


    else:
        raise ValueError(f'Dataset "{cfg.dataset.name}" unknown')


class BalancedAugmentedDatasetOffline(Dataset):
    """
    Oversamples minority classes AND applies augmentation only to synthetic samples.
    """
    def __init__(self, base_dataset, img_size=256):
        self.base_dataset = base_dataset

        # Read all (image, label) pairs **as raw images**
        self.raw_samples = []
        labels = []
        for i in range(len(base_dataset)):
            img, label = base_dataset[i]
            self.raw_samples.append((img, label))
            labels.append(label.item())

        counts = Counter(labels)
        self.max_count = max(counts.values())

        # Group by class
        self.samples_by_class = {c: [] for c in counts.keys()}
        for img, label in self.raw_samples:
            self.samples_by_class[label.item()].append((img, label))

        # Augmentation transforms for minority classes
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])


        #self.to_tensor = transforms.ToTensor()

        # Build final list: (img, label, is_augmented_flag)
        self.samples = []
        for cls, samples in self.samples_by_class.items():
            n = len(samples)
            needed = self.max_count - n

            # original samples
            for img, label in samples:
                self.samples.append((img, label, False))

            # synthetic samples
            for i in range(needed):
                img, label = samples[i % n]
                self.samples.append((img, label, True))

        print(f"Original: {len(base_dataset)}, Balanced: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img, label, is_aug = self.samples[index]

        if is_aug:
            img = self.augment(img)    # no ToTensor() needed

        return img, label



def get_balanced_augmented_loader_offline(cfg, train_subset):
    aug_dataset = BalancedAugmentedDatasetOffline(train_subset)

    aug_loader = DataLoader(
        aug_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        pin_memory=True
    )
    return aug_dataset, aug_loader

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


class SMOTEDataset(Dataset):
    def __init__(self, X, y): #, transform=None
        self.X = X
        self.labels = y
        #self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        img = Image.fromarray(img)

        label = self.labels[idx]

        #if self.transform:
        #    img = self.transform(img)

        return img, label