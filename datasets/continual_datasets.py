import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os
from torch.utils.data import Dataset
from PIL import Image

class JSONContinualDataset(Dataset):
    dataset_root_map = {
        "continual_ad": "/datasets/MegaInspection/megainspection",
        "mvtec_anomaly_detection": "/datasets/MegaInspection/non_megainspection/MVTec",
        "VisA_20220922": "/datasets/MegaInspection/non_megainspection/VisA",
        "Real-IAD-512": "/datasets/MegaInspection/non_megainspection/Real-IAD",
        "VIADUCT": "/datasets/MegaInspection/non_megainspection/VIADUCT",
        "BTAD": "/datasets/MegaInspection/non_megainspection/BTAD",
        "MPDD": "/datasets/MegaInspection/non_megainspection/MPDD"
    }

    def __init__(self, json_data, transform=None, name="json_dataset"):
        self.samples = []
        self.name = name
        for class_data in json_data.values():
            for sample in class_data:
                sample["img_path"] = self.resolve_path(sample["img_path"])
                if sample["mask_path"]:
                    sample["mask_path"] = self.resolve_path(sample["mask_path"])
                self.samples.append(sample)
        self.transform = transform
        self.imagesize = (3, *transform.transforms[0].size) if transform else None

    def resolve_path(self, relative_path):
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

    def __getitem__(self, index):
        item = self.samples[index]
        img = Image.open(item["img_path"]).convert("RGB")
        mask_path = item["mask_path"]
        if self.transform:
            img = self.transform(img)
        mask = None
        if mask_path and os.path.exists(mask_path):
            try:
                mask = Image.open(mask_path)
                if self.transform:
                    mask = self.transform(mask)
            except Exception as e:
                print(f"[!] Failed to open mask at {mask_path}: {e}")
                mask = torch.zeros((1, *self.imagesize[1:])).float()
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