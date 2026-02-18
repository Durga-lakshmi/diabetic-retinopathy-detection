import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms


# ------------------------------------------------------------
# Benjamin Graham Preprocessing
# ------------------------------------------------------------
def graham_preprocess(pil_img, scale=300):
    """
    Apply Benjamin Graham preprocessing:
    - normalize eyeball radius to ~300px
    - subtract local average color (Gaussian)
    - boost contrast
    - crop center region (remove dark edges)
    """
    img = np.array(pil_img)

    # 1. Determine eyeball radius
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    radius = int(min(center[0], center[1], w - center[0], h - center[1]))

    # 2. Scale so radius becomes 300
    scale_factor = scale / max(radius, 1)
    img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)

    # 3. Subtract local average (Graham trick)
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=scale / 30)
    img = cv2.addWeighted(img, 4, blur, -4, 128)

    # 4. Clip borders (90% crop)
    h, w = img.shape[:2]
    crop_h = int(h * 0.9)
    crop_w = int(w * 0.9)
    y1 = (h - crop_h) // 2
    x1 = (w - crop_w) // 2
    img = img[y1:y1 + crop_h, x1:x1 + crop_w]

    return Image.fromarray(img)


# ------------------------------------------------------------
# IDRID Dataset Class
# ------------------------------------------------------------
class GrahamIDRID(Dataset):
    def __init__(self, cfg, split):
        super().__init__()

        data_root = cfg.dataset.path

        if split == "train":
            self.img_dir = os.path.join(data_root, cfg.dataset.images.train)
            self.label_path = os.path.join(data_root, cfg.dataset.labels.train)
        else:
            self.img_dir = os.path.join(data_root, cfg.dataset.images.test)
            self.label_path = os.path.join(data_root, cfg.dataset.labels.test)

        # Load CSV
        df = pd.read_csv(self.label_path)
        df.columns = df.columns.str.strip()

        # Target columns: 'Image name', 'Retinopathy grade'
        self.img_paths = []
        self.labels = []

        for _, row in df.iterrows():
            img_name = row["Image name"].strip()
            label = row["Retinopathy grade"]

            # Convert to binary classification (0/1)
            # binary = 0 if label in [0, 1] else 1

            # Find image file
            img_file = self._find_image(img_name)
            if img_file:
                self.img_paths.append(img_file)
                self.labels.append(int(label))
                # self.labels.append(binary)

        # Image transforms AFTER Graham preprocessing
        image_size = cfg.get("image_size", 256)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=cfg.dataset.mean,
                std=cfg.dataset.std
            ),
        ])

    # ------------------------------------------------------------
    # File finder (JPG/PNG/TIFF)
    # ------------------------------------------------------------
    def _find_image(self, base):
        exts = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
        for ext in exts:
            p = os.path.join(self.img_dir, base + ext)
            if os.path.exists(p):
                return p
        return None

    # ------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")

        # Apply Graham preprocessing BEFORE transforms
        img = graham_preprocess(img)

        img = self.transform(img)
        label = torch.tensor(label, dtype=torch.long)

        return img, label

    # ------------------------------------------------------------
    # __len__
    # ------------------------------------------------------------
    def __len__(self):
        return len(self.img_paths)