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
        "continual_ad": "/data/datasets/MegaInspection/megainspection/continual_ad",
        "mvtec": "/data/datasets/MegaInspection/non_megainspection/mvtec",
        "visa": "/data/datasets/MegaInspection/non_megainspection/visa",
        "realiad": "/data/datasets/MegaInspection/non_megainspection/realiad",
        "viaduct": "/data/datasets/MegaInspection/non_megainspection/viaduct",
        "btad": "/data/datasets/MegaInspection/non_megainspection/btad",
        "mpdd": "/data/datasets/MegaInspection/non_megainspection/mpdd"
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
        if os.path.isabs(relative_path):
            return relative_path
        parts = relative_path.split("/", 1)
        if len(parts) != 2:
            return relative_path
        prefix, sub_path = parts
        root = self.dataset_root_map.get(prefix, "")
        return os.path.normpath(os.path.join(root, sub_path))

    def __getitem__(self, index):
        item = self.samples[index]
        img = Image.open(item["img_path"]).convert("RGB")
        mask_path = item["mask_path"]
        if self.transform:
            img = self.transform(img)
            if mask_path is not None:
                mask = Image.open(mask_path)
                mask = self.transform(mask)
            else:
                # Create dummy mask if missing (all ones: pass-through mask)
                mask = torch.zeros((1, *self.imagesize[1:]), dtype=torch.uint8)

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