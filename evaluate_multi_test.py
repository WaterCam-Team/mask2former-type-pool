# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Evaluation Script for Custom Water Segmentation Dataset.

This script evaluates a trained model on multiple test datasets.
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
        
        # Print results for each dataset
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        datasets = ["custom_sem_seg_normal_flood", "custom_sem_seg_challenging_flood",
                    "custom_sem_seg_challenging_water"]
        dataset_names = ["normal flood", "challenging flood", "challenging water"]
        
        for dataset, name in zip(datasets, dataset_names):
            if dataset in res:
                print(f"\n{name} ({dataset}):")
                print(f"  mIoU: {res[dataset]['sem_seg'].get('mIoU', 'N/A'):.4f}")
                print(f"  Pixel Accuracy: {res[dataset]['sem_seg'].get('pACC', float('nan')):.4f}")
            else:
                print(f"\n{name} ({dataset}): No results available")
        
        print("\n" + "="*50)
    
    return res


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--model-weights", required=True, help="Path to the trained model weights")
    args = parser.parse_args()
    
    if not args.config_file:
        args.config_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "configs/custom/maskformer2_mta_clip_custom_multi_test_attribute.yaml")
    
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
