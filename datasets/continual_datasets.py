import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os
from torch.utils.data import Dataset
from PIL import Image

class JSONDataset(Dataset):
    dataset_root_map = {
        "continual_ad": "/datasets/MegaInspection/megainspection",
        "mvtec_anomaly_detection": "/datasets/MegaInspection/non_megainspection/MVTec",
        "VisA_20220922": "/datasets/MegaInspection/non_megainspection/VisA",
        "Real-IAD-512": "/datasets/MegaInspection/non_megainspection/Real-IAD",
        "VIADUCT": "/datasets/MegaInspection/non_megainspection/VIADUCT",
        "BTAD": "/datasets/MegaInspection/non_megainspection/BTAD",
        "MPDD": "/datasets/MegaInspection/non_megainspection/MPDD"
    }

    json_path_map = {
        "meta_continual_ad_test_total": "continual_ad",
        "meta_mvtec": "mvtec_anomaly_detection",
        "meta_visa": "VisA_20220922"
    }

    def resolve_path(self, relative_path, zero_shot_category=None):
        if zero_shot_category:
            data_root = self.json_path_map[zero_shot_category]
            root = self.dataset_root_map.get(data_root, "")
            if not relative_path:
                return None
            if os.path.isabs(relative_path):
                return relative_path
            return os.path.normpath(os.path.join(root, relative_path))
            
        else:
            if not relative_path:
                return None
            if os.path.isabs(relative_path):
                return relative_path
            parts = relative_path.split("/", 1)
            if len(parts) != 2:
                return None
            prefix, sub_path = parts
            root = self.dataset_root_map.get(prefix, "")
            return os.path.normpath(os.path.join(root, sub_path))
    
    def __init__(self, json_data, transform=None, mask_transform=None, train=True, zero_shot_category=None):
        self.samples = []
        for class_data in json_data.values():
            for sample in class_data:
                sample["img_path"] = self.resolve_path(sample["img_path"], zero_shot_category)
                anomaly = sample.get("anomaly", 0)
                
                if train:
                    if anomaly != 0:
                        continue
                else:
                    sample["mask_path"] = self.resolve_path(sample["mask_path"], zero_shot_category) if sample.get("mask_path") else ""

                self.samples.append(sample)
        self.transform = transform
        self.mask_transform = mask_transform
        self.imagesize = (3, *transform.transforms[0].size) if transform else None

    def __getitem__(self, index):
        item = self.samples[index]
        img = Image.open(item["img_path"]).convert("RGB")
        mask_path = item.get("mask_path", "")
        if self.transform:
            img = self.transform(img)
        mask = None
        if mask_path and mask_path != "" and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
            mask = self.mask_transform(mask)
        else:
            # Dummy mask (통과 마스크, 전부 0으로): 이상 없는 픽셀
            mask = torch.zeros((1, *self.imagesize[1:])).float()

        output = {
            "image": img,
            "mask": mask,
            "cls_name": item["cls_name"],
            "is_anomaly": item["anomaly"],
            "image_path": item["img_path"],
            "mask_path": item["mask_path"]
        }
        return output

    def __len__(self):
        return len(self.samples)


def prepare_loader_from_json_by_chunk(
    json_data,
    image_size=336,
    batch_size=32,
    num_workers=4,
    train=True,
    zero_shot_category=None
):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        
    
    dataset = JSONDataset(
        json_data=json_data, transform=transform, 
        mask_transform=mask_transform, train=train, 
        zero_shot_category=zero_shot_category)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )