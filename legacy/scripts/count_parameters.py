# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Parameter Count Script for Pretrained Water Segmentation Model.

This script calculates the total number of parameters in your model.
"""
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import os
import torch
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from collections import defaultdict

# MaskFormer
from mask2former import add_maskformer2_config

# Import custom dataset registration
from mask2former.data.datasets.register_custom_dataset import register_all_custom_dataset

# Import the trainer class
from train_net_custom import Trainer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg


def count_parameters(model):
    """
    Count parameters in the model and provide detailed breakdown.
    """
    total_params = 0
    trainable_params = 0
    param_breakdown = defaultdict(int)
    layer_breakdown = {}
    
    print("="*80)
    print("PARAMETER COUNT BREAKDOWN")
    print("="*80)
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        if param.requires_grad:
            trainable_params += param_count
        
        # Categorize by layer type
        layer_type = name.split('.')[0] if '.' in name else name
        param_breakdown[layer_type] += param_count
        
        # Store detailed info for major components
        if any(component in name.lower() for component in ['backbone', 'sem_seg_head', 'mask_former']):
            if layer_type not in layer_breakdown:
                layer_breakdown[layer_type] = []
            layer_breakdown[layer_type].append((name, param_count, param.requires_grad))
    
    # Print overall statistics
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
    print(f"Trainable Percentage: {(trainable_params/total_params)*100:.2f}%")
    
    print(f"\n{'='*80}")
    print("PARAMETER BREAKDOWN BY COMPONENT")
    print(f"{'='*80}")
    
    # Sort by parameter count
    sorted_components = sorted(param_breakdown.items(), key=lambda x: x[1], reverse=True)
    
    for component, count in sorted_components:
        percentage = (count / total_params) * 100
        print(f"{component:30} {count:>12,} ({percentage:>6.2f}%)")
    
    print(f"\n{'='*80}")
    print("DETAILED BREAKDOWN BY MAJOR COMPONENTS")
    print(f"{'='*80}")
    
    # Detailed breakdown for major components
    for component, layers in layer_breakdown.items():
        if component in ['backbone', 'sem_seg_head', 'mask_former']:
            print(f"\n{component.upper()}:")
            print("-" * 50)
            
            component_total = sum(count for _, count, _ in layers)
            for name, count, trainable in layers:
                status = "Trainable" if trainable else "Frozen"
                percentage = (count / component_total) * 100
                print(f"  {name:40} {count:>10,} ({percentage:>5.1f}%) [{status}]")
            
            print(f"  {'Total:':40} {component_total:>10,}")
    
    # Model size estimation
    print(f"\n{'='*80}")
    print("MODEL SIZE ESTIMATION")
    print(f"{'='*80}")
    
    # Assuming float32 (4 bytes per parameter)
    model_size_mb = (total_params * 4) / (1024 * 1024)
    print(f"Model size (float32): {model_size_mb:.2f} MB")
    
    # Assuming float16 (2 bytes per parameter)
    model_size_mb_fp16 = (total_params * 2) / (1024 * 1024)
    print(f"Model size (float16): {model_size_mb_fp16:.2f} MB")
    
    # Memory usage estimation (rough)
    print(f"\nRough GPU memory usage:")
    print(f"  Model parameters: {model_size_mb:.2f} MB")
    print(f"  Gradients (training): {model_size_mb:.2f} MB")
    print(f"  Optimizer states (AdamW): {model_size_mb * 2:.2f} MB")
    print(f"  Total training memory: {model_size_mb * 4:.2f} MB")
    
    print(f"{'='*80}")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params,
        'model_size_mb': model_size_mb,
        'param_breakdown': dict(param_breakdown)
    }


def main(args):
    # Register main dataset
    register_all_custom_dataset("/data1/huantao/workspace/project/flood_seg/dataset/my_waterdataset_tmp")
    
    cfg = setup(args)

    # Load model
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        args.model_weights, resume=False
    )
    
    print(f"Model loaded from: {args.model_weights}")
    print(f"Config file: {args.config_file}")
    
    # Count parameters
    results = count_parameters(model)
    
    # Save results to file if requested
    if args.output_file:
        import json
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--model-weights", 
                       default="/data1/huantao/workspace/project/flood_seg/Mask2Former/checkpoints/mask2former_2000_normal_water_model_final.pth",
                       help="Path to the trained model weights")
    parser.add_argument("--output-file", 
                       help="Path to save parameter count results (optional)")
    args = parser.parse_args()
    
    if not args.config_file:
        args.config_file = "/data1/huantao/workspace/project/flood_seg/Mask2Former/configs/custom/maskformer2_custom_R50_bs16_160k.yaml"
    
    print("Command Line Args:", args)
    args.num_gpus = 1
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
