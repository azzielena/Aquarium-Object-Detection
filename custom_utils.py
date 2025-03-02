import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import v2
from torchvision import tv_tensors


class CustomDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transforms=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transforms = transforms
        
        # List of images (jpg, jpeg, png)
        self.image_files = [
            f for f in os.listdir(images_dir) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # load original image
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_filename)
        image = Image.open(img_path).convert("RGB")

        w, h = image.size
        
        # Load YOLO annotations
        label_path = os.path.join(
            self.labels_dir, 
            os.path.splitext(img_filename)[0] + ".txt"
        )
        boxes_list = []
        labels_list = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                classe, cx, cy, bw, bh = parts
                classe = int(classe)
                cx = float(cx) * w  # x-center in absolute coordinates
                cy = float(cy) * h  # y-center in absolute coordinates
                bw = float(bw) * w  # box width in absolute coordinates
                bh = float(bh) * h  # box height in absolute coordinates
                xmin = cx - bw/2
                ymin = cy - bh/2
                xmax = cx + bw/2
                ymax = cy + bh/2
                boxes_list.append([xmin, ymin, xmax, ymax])
                # Faster R-CNN reserves label 0 for the background
                labels_list.append(classe + 1)
        
        # If there are no bounding boxes, create empty tensors
        if len(boxes_list) == 0:
            boxes_tensor = torch.empty((0, 4), dtype=torch.float32)
            labels_tensor = torch.empty((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.as_tensor(boxes_list, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels_list, dtype=torch.int64)
        
        area = (boxes_tensor[:, 3] - boxes_tensor[:, 1]) * (boxes_tensor[:, 2] - boxes_tensor[:, 0])
        iscrowd = torch.zeros((boxes_tensor.shape[0],), dtype=torch.int64)

        # Target Dictionary (Faster R-CNN compatible)
        target = {}
        target["boxes"] = boxes_tensor
        target["labels"] = labels_tensor
        target["image_id"] = torch.tensor([idx])
        target["area"] = area
        target["iscrowd"] = iscrowd

        # it converts bounding boxes to tv_tensors.BoundingBoxes
        if target["boxes"].numel() > 0:  
            target["boxes"] = tv_tensors.BoundingBoxes(
                target["boxes"], 
                format="xyxy", 
                canvas_size=(h, w)  
            )
        else:
            target["boxes"] = tv_tensors.BoundingBoxes(
                torch.empty((0, 4), dtype=torch.float32),
                format="xyxy", 
                canvas_size=(h, w)
            )

        # Create the sample for the transforms v2
        sample = {
            "image": image,               
            "boxes": target["boxes"],     
            "labels": target["labels"]
        }

        # Apply the transformations
        if self.transforms:
            sample = self.transforms(sample)

        # Update image and transformed boxes/labels
        image_transformed = sample["image"]
        target["boxes"] = sample["boxes"]
        target["labels"] = sample["labels"]

        return image_transformed, target

    def collate_fn(self, batch): 
        return tuple(zip(*batch))
    

# -------------------------------------------------------------
# TRANSFORMATIONS (v2)
# -------------------------------------------------------------

transformsTrain = v2.Compose([
    v2.ToImage(),
    v2.RandomPhotometricDistort(p=0.5), 
    v2.RandomZoomOut(fill={tv_tensors.Image: (123, 117, 104), "others": 0}),
    #v2.RandomIoUCrop(),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.SanitizeBoundingBoxes(),
    v2.Resize((768, 768)),
    v2.ToDtype(torch.float32, scale=True)
])

transformsTest = v2.Compose([
    v2.ToImage(),
    v2.Resize((768, 768)),
    v2.ToDtype(torch.float32, scale=True)
])
