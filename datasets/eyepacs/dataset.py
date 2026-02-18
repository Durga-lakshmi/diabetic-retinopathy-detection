import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image
from torchvision import transforms



class EYEPACS(Dataset):
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
        if split == 'train':
            self.img_dir = os.path.join(data_root, cfg.dataset.images.train) #maybe need to be changed
            self.label_dir = os.path.join(data_root, cfg.dataset.labels.train)
        elif split == 'eval':
            self.img_dir = os.path.join(data_root, cfg.dataset.images.val)
            self.label_dir = os.path.join(data_root, cfg.dataset.labels.val)
        elif split == 'test':
            self.img_dir = os.path.join(data_root, cfg.dataset.images.test)
            self.label_dir = os.path.join(data_root, cfg.dataset.labels.test)
        
        
        # 2. define transforms 
        image_size = cfg.get('image_size', 256) # (?
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(), # PIL Image -> Tensor (C, H, W) in [0.0, 1.0]
            transforms.Normalize(
                mean=cfg.get('mean', [0.485, 0.456, 0.406]), # need to be defined in cfg 'mean' and 'std'
                std=cfg.get('std', [0.229, 0.224, 0.225])
            )
        ])

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
            img_name = row['filename'].strip()
            label = row['label'] 
            # try to find image file
            img_path = self.find_image_file(self.img_dir, img_name)
            if img_path is not None:
                self.img_paths.append(img_path)

                if cfg.model.num_classes == 1:
                    binary_label = 0 if label in [0, 1] else 1
                    self.labels.append(binary_label)
                #print(f"Image: {img_name}, Label: {binary_label}") # for visualization
                # NRDR[0]:0,1  / RDR[1]:2-4
                else:
                    self.labels.append(label)#self.labels.append(label) # for 5-class


                # binary_label = 0 if label in [0, 1] else 1
                # #self.labels.append(binary_label)/
                #self.labels.append(int(label))
        
                # NRDR[0]:0,1  / RDR[1]:2-4
                #self.labels.append(label)
                #print(f"Image: {img_name}, Label: {binary_label}") # for visualization
            else:
                missing.append(img_name)
        print("Unique labels in dataset:", set(self.labels))

        #print(f"Missing images count: {len(missing)}")
        #print("Examples:", missing[:10])
                

    # ------------------------------------------------------------
    # Get item by index for binary classification
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
        return image_tensor, label_tensor # (?)


    # ------------------------------------------------------------
    # Get length of dataset
    # ------------------------------------------------------------
    def __len__(self):
        # TODO: Return number of samples
        #print(f"Dataset length: {len(self.img_paths)}")
        return len(self.img_paths)
        # return 1


