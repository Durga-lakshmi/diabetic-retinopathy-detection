import torch
import numpy as np
from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image, ImageFilter
from torchvision import transforms
#from ..graham_transform import GrahamTransform




class IDRID(Dataset):
    # ------------------------------------------------------------
    # Helper function to find image file with various extensions
    # ------------------------------------------------------------
    def find_image_file(self, img_dir, img_name):
        possible_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
        for ext in possible_extensions:
            img_path = os.path.join(img_dir, img_name + ext)
            if os.path.exists(img_path):
                return img_path
        return None

    # ------------------------------------------------------------
    # Initialize the dataset
    # ------------------------------------------------------------
    def __init__(self, cfg, split):
        super().__init__()
        # 1. read config
        data_root = cfg.dataset.path # (?
        # TODO:
        
        #dont change!
        if cfg.name !="default":
            self.mean = cfg.get('mean', [0.485, 0.456, 0.406])
            self.std = cfg.get('std', [0.229, 0.224, 0.225])
            self.num_classes=cfg.problem.num_classes

            # 2. define transforms 
            image_size = cfg.get('image_size', 256) # (?
            self.transform = transforms.Compose([
            GrahamTransform(scale=300),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(), # PIL Image -> Tensor (C, H, W) in [0.0, 1.0]
            transforms.Normalize(
                mean=self.mean, 
                std=self.std
            )
        ])
        else:
            self.mean= cfg.dataset.mean
            self.std= cfg.dataset.std
            self.num_classes= cfg.model.num_classes
            
            image_size = cfg.get('image_size', 256) # (?
            self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(), # PIL Image -> Tensor (C, H, W) in [0.0, 1.0]
            transforms.Normalize(
                mean=cfg.dataset.mean, # need to be defined in cfg 'mean' and 'std'
                std=cfg.dataset.mean)
            
        ])



        
        if split == 'train' :
            self.img_dir = os.path.join(data_root, cfg.dataset.images.train) #maybe need to be changed
            self.label_dir = os.path.join(data_root, cfg.dataset.labels.train)
        elif split == 'test':
            self.img_dir = os.path.join(data_root, cfg.dataset.images.test)
            self.label_dir = os.path.join(data_root, cfg.dataset.labels.test)
        
            


        
        
        

        # 3. read CSV path/labels
        df = pd.read_csv(self.label_dir)
        #print("Total CSV entries:", len(df))
        df.columns = df.columns.str.strip()  # delete space
        
        self.img_paths = []
        self.labels = []
        
        # CSV has columns 'Image name', 'Retinopathy grade', 'Risk of macular edema'
        # assume 'RG' is the target label
        missing = []
        for _, row in df.iterrows():
            img_name = row['Image name'].strip()
            label = row['Retinopathy grade'] 
            label_RME = row['Risk of macular edema'] 
            # try to find image file
            img_path = self.find_image_file(self.img_dir, img_name)
            if img_path is not None:
                self.img_paths.append(img_path)
                if self.num_classes == 1:
                    binary_label = 0 if label in [0, 1] else 1
                    self.labels.append(binary_label)
                #print(f"Image: {img_name}, Label: {binary_label}") # for visualization
                # NRDR[0]:0,1  / RDR[1]:2-4
                else:
                    self.labels.append(label)#self.labels.append(label) # for 5-class
                
            else:
                missing.append(img_name)

        #print(f"Missing images count: {len(missing)}")
        #print("Examples:", missing[:10])
        print("[DEBUG] Unique labels:", sorted(list(set(self.labels))))
        print(f"[DEBUG] Loaded {len(self.img_paths)} images from {self.img_dir}")
                

    # ------------------------------------------------------------
    # Get item by index
    # ------------------------------------------------------------  
    def __getitem__(self, index):
        # TODO: Load image and label for index
        img_path = self.img_paths[index]
        label = self.labels[index]
        #print(f"Index {index}: Image = {os.path.basename(img_path)}, Label = {label}")

        # 1. Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return torch.zeros((3, 256, 256)), torch.tensor(-1, dtype=torch.long) # return placeholder

        image_tensor = self.transform(image)
        label_tensor = torch.tensor(label, dtype=torch.long)

        #return image, label # (?)
        return image_tensor, label_tensor #, image # (?)


    # ------------------------------------------------------------
    # Get length of dataset
    # ------------------------------------------------------------
    def __len__(self):
        # TODO: Return number of samples
        #print(f"Dataset length: {len(self.img_paths)}")
        return len(self.img_paths)
        # return 1


class GrahamTransform(object): 
    def __init__(self, scale=300):
        self.scale = scale

    def __call__(self, img: Image.Image):
        img = img.convert("RGB")

        # 1. resize
        img_np = np.array(img).astype(np.float32)
        x = img_np[img_np.shape[0]//2, :, :].sum(axis=1)
        r = (x > x.mean()/10).sum() / 2
        s = self.scale / r
        img = img.resize((int(img.width*s), int(img.height*s)), Image.LANCZOS)

        # 2. enhance contrast
        img_np = np.array(img).astype(np.float32)
        blur = np.array(img.filter(ImageFilter.GaussianBlur(radius=self.scale/30))).astype(np.float32)
        enhanced = np.clip(4*img_np - 4*blur + 128, 0, 255)

        # 3. circular crop
        h, w, _ = enhanced.shape
        yy, xx = np.mgrid[:h, :w]
        center = np.array([h/2, w/2])
        radius = self.scale * 0.9
        mask = ((xx - center[1])**2 + (yy - center[0])**2) <= radius**2
        mask = np.repeat(mask[:, :, None], 3, axis=2)
        enhanced = enhanced * mask + 128*(1 - mask)

        # 4. to PIL Image
        enhanced = Image.fromarray(enhanced.astype(np.uint8))
        return enhanced

