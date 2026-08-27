# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Test Script for Pretrained Water Segmentation Model.

This script loads a pretrained model and evaluates it on all test datasets.
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
from detectron2.evaluation import verify_results
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

# MaskFormer
from mask2former import add_maskformer2_config

# Import custom dataset registration
from mask2former.data.datasets.register_custom_dataset import register_all_custom_dataset, register_additional_test_datasets

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


def main(args):
    # Register all datasets
    register_all_custom_dataset("/data1/huantao/workspace/project/flood_seg/dataset/my_challenging_flood_huantao")
    register_additional_test_datasets()
    
    cfg = setup(args)

    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        args.model_weights, resume=False
    )
    
    # Evaluate on all test datasets
    res = Trainer.test(cfg, model)
    
    if comm.is_main_process():
        verify_results(cfg, res)
        
        # Print detailed results for each dataset
        print("\n" + "="*80)
        print("MODEL EVALUATION RESULTS")
        print("="*80)
        print(f"Model: {args.model_weights}")
        print("="*80)
        
        datasets = ["custom_sem_seg_val", "custom_sem_seg_edge", "custom_sem_seg_flood", "custom_sem_seg_normal"]
        dataset_names = ["Main Validation", "Edge Cases", "Flood Images", "Normal Images"]
        
        for dataset, name in zip(datasets, dataset_names):
            if dataset in res:
                sem_seg_results = res[dataset]['sem_seg']
                print(f"\n{name} ({dataset}):")
                
                # Handle mIoU
                mIoU = sem_seg_results.get('mIoU')
                if mIoU is not None:
                    print(f"  mIoU: {mIoU:.4f}")
                else:
                    print(f"  mIoU: N/A")
                
                # Handle Pixel Accuracy
                pixel_acc = sem_seg_results.get('Pixel Acc')
                if pixel_acc is not None:
                    print(f"  Pixel Accuracy: {pixel_acc:.4f}")
                else:
                    print(f"  Pixel Accuracy: N/A")
                
                # Print per-class results if available
                if 'IoU' in sem_seg_results:
                    ious = sem_seg_results['IoU']
                    print(f"  Background IoU: {ious[0]:.4f}")
                    print(f"  Water IoU: {ious[1]:.4f}")
                
                # Print additional metrics if available
                mean_acc = sem_seg_results.get('Mean Acc')
                if mean_acc is not None:
                    print(f"  Mean Accuracy: {mean_acc:.4f}")
                
                freqw_acc = sem_seg_results.get('FreqW Acc')
                if freqw_acc is not None:
                    print(f"  Frequency Weighted Acc: {freqw_acc:.4f}")
            else:
                print(f"\n{name} ({dataset}): No results available")
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        # Calculate average mIoU across all datasets
        valid_results = []
        for dataset in datasets:
            if dataset in res and 'sem_seg' in res[dataset]:
                mIoU = res[dataset]['sem_seg'].get('mIoU')
                if mIoU is not None:
                    valid_results.append(mIoU)
        
        if valid_results:
            avg_mIoU = sum(valid_results) / len(valid_results)
            print(f"Average mIoU across all datasets: {avg_mIoU:.4f}")
            print(f"Number of datasets evaluated: {len(valid_results)}")
        
        print("="*80)
    
    return res


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--model-weights", 
                       default="/data1/huantao/workspace/project/flood_seg/Mask2Former/checkpoints/mask2former_setting2_model_final.pth",
                       help="Path to the trained model weights")
    args = parser.parse_args()
    
    if not args.config_file:
        args.config_file = "/data1/huantao/workspace/project/flood_seg/Mask2Former/configs/custom/maskformer2_custom_multi_test.yaml"
    
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
