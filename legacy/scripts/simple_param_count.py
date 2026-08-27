#!/usr/bin/env python3
"""
Simple Parameter Counter for Water Segmentation Model.
"""
import torch
import sys
import os

# Add the project root to Python path
sys.path.append('/data1/huantao/workspace/project/flood_seg/Mask2Former')

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config

from mask2former import add_maskformer2_config
from train_net_custom import Trainer


def count_params_simple(model):
    """Simple parameter counting."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
    print(f"Trainable Percentage: {(trainable_params/total_params)*100:.2f}%")
    
    # Model size
    model_size_mb = (total_params * 4) / (1024 * 1024)  # float32
    print(f"Model Size: {model_size_mb:.2f} MB")
    
    return total_params, trainable_params


def main():
    # Configuration
    model_path = "/data1/huantao/workspace/project/flood_seg/Mask2Former/checkpoints/mask2former_2000_normal_water_model_final.pth"
    config_path = "/data1/huantao/workspace/project/flood_seg/Mask2Former/configs/custom/maskformer2_custom_R50_bs16_160k.yaml"
    
    print(f"Loading model from: {model_path}")
    print(f"Using config: {config_path}")
    
    # Load model
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_path)
    cfg.freeze()
    
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model).resume_or_load(model_path, resume=False)
    
    print("\n" + "="*50)
    print("PARAMETER COUNT")
    print("="*50)
    
    total_params, trainable_params = count_params_simple(model)
    
    print("="*50)


if __name__ == "__main__":
    main()
