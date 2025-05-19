import os
import argparse
import re
import numpy as np
from tqdm import tqdm
import pandas as pd
import json
import backbones
import utils

import torch
import torch
import torch.nn.functional as F
import gc

from simplenet import SimpleNet
from datasets.continual_datasets import prepare_loader_from_json_by_chunk


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--backbone_name", "-b", type=str, default="wideresnet50")
    parser.add_argument("--layers_to_extract_from", "-le", type=str, nargs='+', default=["layer2", "layer3"])
    parser.add_argument("--pretrain_embed_dimension", type=int, default=1536)
    parser.add_argument("--target_embed_dimension", type=int, default=1536)
    parser.add_argument("--patchsize", type=int, default=3)
    parser.add_argument("--embedding_size", type=int, default=256)
    parser.add_argument("--meta_epochs", type=int, default=1)
    parser.add_argument("--aed_meta_epochs", type=int, default=1)
    parser.add_argument("--gan_epochs", type=int, default=1)
    parser.add_argument("--dsc_layers", type=int, default=2)
    parser.add_argument("--dsc_hidden", type=int, default=1024)
    parser.add_argument("--noise_std", type=float, default=0.015)
    parser.add_argument("--dsc_margin", type=float, default=0.5)
    parser.add_argument("--dsc_lr", type=float, default=0.0002)
    parser.add_argument("--auto_noise", type=float, default=0)
    parser.add_argument("--train_backbone", action="store_true", default=False)
    parser.add_argument("--cos_lr", action="store_true")
    parser.add_argument("--pre_proj", type=int, default=1)
    parser.add_argument("--proj_layer_type", type=int, default=0)
    parser.add_argument("--mix_noise", type=int, default=1)
    
    parser.add_argument("--image_size", type=int, default=336)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--score_dir", type=str, default="/workspace/MegaInspection/SimpleNet/scores")
    
    parser.add_argument("--json_path", type=str, default="meta_mvtec", help="Name of the task JSON file (e.g., 5classes_tasks)")
    parser.add_argument('--scenario', default='scenario_1', type=str)
    parser.add_argument('--case', default='5_classes_tasks', type=str)

    # Parse arguments
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    args = parse_args()
    
    model_weight_path = "/workspace/MegaInspection/SimpleNet/outputs"
     
    json_path = os.path.join("/workspace/meta_files", f"{args.json_path}.json")
    with open(json_path, 'r') as f:
        data_dict = json.load(f)
    data_dict = data_dict['test']

    weights_list = os.listdir(os.path.join(model_weight_path, args.scenario, args.case))
    weights_list = [weight for weight in weights_list if weight.endswith("pth")]
    weights_list = sorted(weights_list, key=lambda x: int(x.split(".")[0][4:]))
    last_model = weights_list[-1]
    last_model_num = int(last_model.split(".")[0][4:])
    
    pretrained_path = os.path.join(model_weight_path, 
                                        args.scenario, 
                                        args.case, 
                                        last_model)
    final_score_path = os.path.join(args.score_dir,
                                    args.scenario, 
                                    args.case, 
                                    args.json_path,
                                    f"model{last_model_num}.json")

    args.final_score_path = final_score_path
    model = SimpleNet("cuda")

    backbone_seed = None
    backbone_name = args.backbone_name
    if ".seed-" in backbone_name:
        backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(
            backbone_name.split("-")[-1]
        )
    backbone = backbones.load(backbone_name)
    backbone.name, backbone.seed = backbone_name, backbone_seed
    
    device = utils.set_torch_device("0")
    
    model.load(
            backbone=backbone,
            layers_to_extract_from=args.layers_to_extract_from,
            device=device,
            input_shape=(3, 336, 336),
            pretrain_embed_dimension=args.pretrain_embed_dimension,
            target_embed_dimension=args.target_embed_dimension,
            patchsize=args.patchsize,
            embedding_size=args.embedding_size,
            meta_epochs=args.meta_epochs,
            aed_meta_epochs=args.aed_meta_epochs,
            gan_epochs=args.gan_epochs,
            noise_std=args.noise_std,
            dsc_layers=args.dsc_layers,
            dsc_hidden=args.dsc_hidden,
            dsc_margin=args.dsc_margin,
            dsc_lr=args.dsc_lr,
            auto_noise=args.auto_noise,
            train_backbone=args.train_backbone,
            cos_lr=args.cos_lr,
            pre_proj=args.pre_proj,
            proj_layer_type=args.proj_layer_type,
            mix_noise=args.mix_noise,
        )
    model.load_checkpoint(pretrained_path)
    
    os.makedirs(os.path.dirname(final_score_path), exist_ok=True)
    
    for i, (cls_name, samples) in enumerate(data_dict.items()):
        len_samples = len(samples)
        print(f"[{i+1}/{len(data_dict)}] {cls_name}")
        print("length of samples:", len_samples)
        print()
        if len_samples > 1500:
            print(f"Sample size {len_samples} is larger than 1500, passing...")
            continue
        
        if os.path.exists(final_score_path):
            with open(final_score_path, 'r') as f:
                cumm_score = json.load(f)
            if cls_name in cumm_score:
                print(f"json_path: {json_path}")
                print(f"Already evaluated {cls_name} class")
                continue
        else:
            cumm_score = {}
        
        sub_data_dict = {}
        sub_data_dict[cls_name] = samples
        
        test_loader = prepare_loader_from_json_by_chunk(
            sub_data_dict,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train=False,
            zero_shot_category=None,
        )        
        
        with torch.no_grad():
            results = model.test_custom(test_loader)
        
        cumm_score[cls_name] = results
        
        with open(final_score_path, 'w') as f:
            json.dump(cumm_score, f, indent=4)
        
        del test_loader
        torch.cuda.empty_cache()
        gc.collect()