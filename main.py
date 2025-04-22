# ------------------------------------------------------------------
# SimpleNet: A Simple Network for Image Anomaly Detection and Localization (https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_SimpleNet_A_Simple_Network_for_Image_Anomaly_Detection_and_Localization_CVPR_2023_paper.pdf)
# Github source: https://github.com/DonaldRR/SimpleNet
# Licensed under the MIT License [see LICENSE for details]
# The script is based on the code of PatchCore (https://github.com/amazon-science/patchcore-inspection)
# ------------------------------------------------------------------

import logging
import os
import sys
import json
import random
import re

import click
import numpy as np
import torch

sys.path.append("src")
import backbones
import common
import metrics
import simplenet 
import utils
from sklearn.model_selection import train_test_split

from datasets.continual_datasets import JSONDataset
import torchvision.transforms as transforms

LOGGER = logging.getLogger(__name__)

_DATASETS = {
    "mvtec": ["datasets.mvtec", "MVTecDataset"],
}


@click.group(chain=True)
@click.option("--results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_group", type=str, default="group")
@click.option("--log_project", type=str, default="project")
@click.option("--run_name", type=str, default="test")
@click.option("--test", is_flag=True)
@click.option("--save_segmentation_images", is_flag=True, default=False, show_default=True)
@click.option("--json_path", type=str, default="", help="Path to class/task json file for continual learning.")
@click.option("--task_id", type=int, default=0, help="Specify which task to load.")
def main(**kwargs):
    pass


@main.result_callback()
def run(
    methods,
    results_path,
    gpu,
    seed,
    log_group,
    log_project,
    run_name,
    test,
    save_segmentation_images,
    json_path,
    task_id,
):
    methods = {key: item for (key, item) in methods}
    
    
    if task_id == 0:
        phase = "base"
    else:
        phase = "continual"
    
    if json_path.endswith("except_mvtec_visa"):
        scenario = "scenario_2"
    elif json_path.endswith("except_continual_ad"):
        scenario = "scenario_3"
    else:
        scenario = "scenario_1"
    
    if phase == "base":
        output_dir = f"/workspace/MegaInspection/SimpleNet/outputs/{scenario}/base"
    elif phase == "continual":
        num_classes_per_task = int(re.match(r'\d+', json_path).group())
        num_classes_per_task = num_classes_per_task
        output_dir = f"/workspace/MegaInspection/SimpleNet/outputs/{scenario}/{num_classes_per_task}classes_tasks"
    
    pid = os.getpid()
    list_of_dataloaders = methods["get_dataloaders"](seed)

    device = utils.set_torch_device(gpu)

    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        LOGGER.info(
            "Evaluating dataset ({}/{})...".format(
                dataloader_count + 1,
                len(list_of_dataloaders),
            )
        )

        utils.fix_seeds(seed, device)


        imagesize = dataloaders["training"].dataset.imagesize
        simplenet_list = methods["get_simplenet"](imagesize, device)

        for i, SimpleNet in enumerate(simplenet_list):
            
            if phase == "base":
            
                # # Load base checkpoint if provided
                # if base_checkpoint is not None:
                #     checkpoint_path = os.path.join(base_checkpoint, f"{dataset_name}_{i}.pth")
                #     if os.path.exists(checkpoint_path):
                #         LOGGER.info(f"Loading checkpoint from {checkpoint_path}")
                #         SimpleNet.load_checkpoint(checkpoint_path)
                #     else:
                #         LOGGER.warning(f"Checkpoint path not found: {checkpoint_path}")

                # torch.cuda.empty_cache()
                if SimpleNet.backbone.seed is not None:
                    utils.fix_seeds(SimpleNet.backbone.seed, device)
                LOGGER.info(
                    "Training models ({}/{})".format(i + 1, len(simplenet_list))
                )
                # torch.cuda.empty_cache()
                SimpleNet.set_model_dir(output_dir)

                if not test:
                    _ = SimpleNet.train(dataloaders["training"], None)
                else:
                    # BUG: the following line is not using. Set test with True by default.
                    # i_auroc, p_auroc, pro_auroc =  SimpleNet.test(dataloaders["training"], dataloaders["testing"], save_segmentation_images)
                    print("Warning: Pls set test with true by default")

                # Save checkpoint
                save_path = os.path.join(output_dir, "base.pth")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                SimpleNet.save(save_path)
                LOGGER.info(f"[BASE] Traning end. Saved checkpoint to {save_path}")
            
            elif phase == "continual":
                
                pretrained_path = (
                    f"/workspace/MegaInspection/SimpleNet/outputs/{scenario}/base/base.pth" if task_id == 1
                    else os.path.join(output_dir, f"task{task_id - 1}.pth")
                )
                SimpleNet.load_checkpoint(pretrained_path)
                
                if SimpleNet.backbone.seed is not None:
                    utils.fix_seeds(SimpleNet.backbone.seed, device)
                LOGGER.info(
                    "Training models ({}/{})".format(i + 1, len(simplenet_list))
                )
                
                SimpleNet.set_model_dir(output_dir)

                if not test:
                    _ = SimpleNet.train(dataloaders["training"], None)
                else:
                    # BUG: the following line is not using. Set test with True by default.
                    # i_auroc, p_auroc, pro_auroc =  SimpleNet.test(dataloaders["training"], dataloaders["testing"], save_segmentation_images)
                    print("Warning: Pls set test with true by default")
                
                # Save checkpoint
                save_path = os.path.join(output_dir, f"task{task_id}.pth")
                os.makedirs(output_dir, exist_ok=True)
                SimpleNet.save(save_path)
                LOGGER.info(f"[CONTINUAL] Traning end. Saved checkpoint to {save_path}")


        LOGGER.info("\n\n-----\n")

    # # Store all results and mean scores to a csv-file.
    # result_metric_names = list(result_collect[-1].keys())[1:]
    # result_dataset_names = [results["dataset_name"] for results in result_collect]
    # result_scores = [list(results.values())[1:] for results in result_collect]
    # utils.compute_and_store_final_results(
    #     run_save_path,
    #     result_scores,
    #     column_names=result_metric_names,
    #     row_names=result_dataset_names,
    # )


@main.command("net")
@click.option("--backbone_names", "-b", type=str, multiple=True, default=["wideresnet50"])
@click.option("--layers_to_extract_from", "-le", type=str, multiple=True, default=["layer2", "layer3"])
@click.option("--pretrain_embed_dimension", type=int, default=1536)
@click.option("--target_embed_dimension", type=int, default=1536)
@click.option("--patchsize", type=int, default=3)
@click.option("--embedding_size", type=int, default=256)
@click.option("--meta_epochs", type=int, default=1)
@click.option("--aed_meta_epochs", type=int, default=1)
@click.option("--gan_epochs", type=int, default=1)
@click.option("--dsc_layers", type=int, default=2)
@click.option("--dsc_hidden", type=int, default=1024)
@click.option("--noise_std", type=float, default=0.015)
@click.option("--dsc_margin", type=float, default=0.5)
@click.option("--dsc_lr", type=float, default=0.0002)
@click.option("--auto_noise", type=float, default=0)
@click.option("--train_backbone", is_flag=True)
@click.option("--cos_lr", is_flag=True)
@click.option("--pre_proj", type=int, default=1)
@click.option("--proj_layer_type", type=int, default=0)
@click.option("--mix_noise", type=int, default=1)
def net(
    backbone_names,
    layers_to_extract_from,
    pretrain_embed_dimension,
    target_embed_dimension,
    patchsize,
    embedding_size,
    meta_epochs,
    aed_meta_epochs,
    gan_epochs,
    noise_std,
    dsc_layers, 
    dsc_hidden,
    dsc_margin,
    dsc_lr,
    auto_noise,
    train_backbone,
    cos_lr,
    pre_proj,
    proj_layer_type,
    mix_noise,
):
    backbone_names = list(backbone_names) if not isinstance(backbone_names, list) else backbone_names
    if len(backbone_names) > 1:
        layers_to_extract_from_coll = [[] for _ in range(len(backbone_names))]
        for layer in layers_to_extract_from:
            idx = int(layer.split(".")[0])
            layer = ".".join(layer.split(".")[1:])
            layers_to_extract_from_coll[idx].append(layer)
    else:
        layers_to_extract_from_coll = [layers_to_extract_from]

    def get_simplenet(input_shape, device):
        simplenets = []
        for backbone_name, layers_to_extract_from in zip(
            backbone_names, layers_to_extract_from_coll
        ):
            backbone_seed = None
            if ".seed-" in backbone_name:
                backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(
                    backbone_name.split("-")[-1]
                )
            backbone = backbones.load(backbone_name)
            backbone.name, backbone.seed = backbone_name, backbone_seed

            simplenet_inst = simplenet.SimpleNet(device)
            simplenet_inst.load(
                backbone=backbone,
                layers_to_extract_from=layers_to_extract_from,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=pretrain_embed_dimension,
                target_embed_dimension=target_embed_dimension,
                patchsize=patchsize,
                embedding_size=embedding_size,
                meta_epochs=meta_epochs,
                aed_meta_epochs=aed_meta_epochs,
                gan_epochs=gan_epochs,
                noise_std=noise_std,
                dsc_layers=dsc_layers,
                dsc_hidden=dsc_hidden,
                dsc_margin=dsc_margin,
                dsc_lr=dsc_lr,
                auto_noise=auto_noise,
                train_backbone=train_backbone,
                cos_lr=cos_lr,
                pre_proj=pre_proj,
                proj_layer_type=proj_layer_type,
                mix_noise=mix_noise,
            )
            simplenets.append(simplenet_inst)
        return simplenets

    return ("get_simplenet", get_simplenet)


@main.command("dataset")
@click.argument("name", type=str, required=False)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False), required=False)
@click.option("--subdatasets", "-d", multiple=True, type=str, required=False, default=[])
@click.option("--train_val_split", type=float, default=1, show_default=True)
@click.option("--batch_size", default=2, type=int, show_default=True)
@click.option("--num_workers", default=2, type=int, show_default=True)
@click.option("--resize", default=336, type=int, show_default=True)
@click.option("--imagesize", default=336, type=int, show_default=True)
@click.option("--rotate_degrees", default=0, type=int)
@click.option("--translate", default=0, type=float)
@click.option("--scale", default=0.0, type=float)
@click.option("--brightness", default=0.0, type=float)
@click.option("--contrast", default=0.0, type=float)
@click.option("--saturation", default=0.0, type=float)
@click.option("--gray", default=0.0, type=float)
@click.option("--hflip", default=0.0, type=float)
@click.option("--vflip", default=0.0, type=float)
@click.option("--augment", is_flag=True)
@click.option("--json_path", type=str, default="", help="Path to class/task json file for continual learning.")
@click.option("--task_id", type=int, default=0, help="Specify which task to load.")
def dataset(
    name,
    data_path,
    subdatasets,
    train_val_split,
    batch_size,
    resize,
    imagesize,
    num_workers,
    rotate_degrees,
    translate,
    scale,
    brightness,
    contrast,
    saturation,
    gray,
    hflip,
    vflip,
    augment,
    json_path,
    task_id,
):
    if json_path is None and (name is None or data_path is None):
        raise ValueError("Either --json_path or both positional arguments (name, data_path) must be provided.")
    
    dataloaders = []
    def get_dataloaders(seed):
        
        json_path_ = os.path.join("/workspace/meta_files", f"{json_path}.json")
        
        with open(json_path_, 'r') as f:
            full_data = json.load(f)

        if task_id == 0:
            train_data = full_data["train"]
        else:
            task_key = f"task_{task_id}"
            train_data = full_data[task_key]["train"]
            

        transform = transforms.Compose([
            transforms.Resize((imagesize, imagesize)),
            transforms.ToTensor(),
        ])

        train_dataset = JSONDataset(train_data, transform=transform, train=True)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            prefetch_factor=2,
            pin_memory=True,
        )
        
        dataloaders.append({
            "training": train_dataloader,
            "validation": None,
            "testing": None,
        })

        LOGGER.info(f"[JSON Mode] Loaded {len(train_dataset)} train samples.")
        return dataloaders
        
    return ("get_dataloaders", get_dataloaders)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()
