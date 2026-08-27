# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
FPS Calculation Script for Pretrained Water Segmentation Model.

This script measures the inference speed (FPS) of your model.
"""
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import os
import time
import torch
import numpy as np
from PIL import Image
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from detectron2.data import transforms as T
from detectron2.structures import ImageList

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


def preprocess_image(image_path, cfg):
    """
    Preprocess image for inference.
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    
    # Get image dimensions
    original_height, original_width = image.shape[:2]
    
    # Apply transforms
    transform_gen = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], 
        cfg.INPUT.MAX_SIZE_TEST
    )
    
    transform = transform_gen.get_transform(image)
    image = transform.apply_image(image)
    
    # Convert to tensor and normalize
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    
    # Normalize
    pixel_mean = torch.tensor(cfg.MODEL.PIXEL_MEAN).view(3, 1, 1)
    pixel_std = torch.tensor(cfg.MODEL.PIXEL_STD).view(3, 1, 1)
    image = (image - pixel_mean) / pixel_std
    
    # Don't add batch dimension here - MaskFormer will handle it
    return image, (original_height, original_width)


def measure_fps(model, image_path, cfg, num_warmup=10, num_iterations=10):
    """
    Measure FPS for a single image.
    """
    model.eval()
    
    # Preprocess image
    input_tensor, original_size = preprocess_image(image_path, cfg)
    
    # Move to GPU if available
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Format input for MaskFormer model (expects list of dicts)
    batched_inputs = [{"image": input_tensor}]
    
    # Warmup runs
    # print(f"Running {num_warmup} warmup iterations...")
    # with torch.no_grad():
    #     for _ in range(num_warmup):
    #         _ = model(batched_inputs)
    
    # Synchronize GPU
    # if torch.cuda.is_available():
    #     torch.cuda.synchronize()
    
    # Measure inference time
    print(f"Running {num_iterations} iterations for FPS measurement...")
    times = []
    
    with torch.no_grad():
        for i in range(num_iterations):
            start_time = time.time()
            
            # Forward pass
            outputs = model(batched_inputs)
            
            # Synchronize GPU
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append(end_time - start_time)
            
            if (i + 1) % 20 == 0:
                print(f"Completed {i + 1}/{num_iterations} iterations")
    
    # Calculate statistics
    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    fps = 1.0 / mean_time
    
    return {
        'fps': fps,
        'mean_time': mean_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'times': times
    }


def main(args):
    # Register main dataset
    # register_all_custom_dataset("/data1/huantao/workspace/project/flood_seg/dataset/my_challenging_flood_huantao")
    
    cfg = setup(args)

    # Load model
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        args.model_weights, resume=False
    )
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    model = model.to(device)
    
    print(f"Model loaded on device: {device}")
    print(f"Model weights: {args.model_weights}")
    
    # Test on multiple images if available
    test_images = []
    
    # Look for test images in the dataset
    # img_dir = "/data1/huantao/workspace/project/flood_seg/dataset/my_normal_flood_huantao/img_dir/val"
    img_dir = args.image_path
    if os.path.exists(img_dir):
        for file in os.listdir(img_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_images.append(os.path.join(img_dir, file))
                # if len(test_images) >= 5:  # Limit to 5 images for testing
                #     break
    
    if not test_images:
        print("No test images found. Please provide an image path with --image-path")
        return
    
    print(f"Found {len(test_images)} test images")
    
    # Measure FPS for each image
    all_results = []
    
    for i, image_path in enumerate(test_images):
        print(f"\n{'='*60}")
        print(f"Testing image {i+1}/{len(test_images)}: {os.path.basename(image_path)}")
        print(f"{'='*60}")
        
        try:
            results = measure_fps(model, image_path, cfg, 
                                num_warmup=args.num_warmup, 
                                num_iterations=args.num_iterations)
            all_results.append(results)
            
            print(f"\nResults for {os.path.basename(image_path)}:")
            print(f"  FPS: {results['fps']:.2f}")
            print(f"  Mean inference time: {results['mean_time']*1000:.2f} ms")
            print(f"  Std deviation: {results['std_time']*1000:.2f} ms")
            print(f"  Min time: {results['min_time']*1000:.2f} ms")
            print(f"  Max time: {results['max_time']*1000:.2f} ms")
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    # Calculate average FPS across all images
    if all_results:
        avg_fps = np.mean([r['fps'] for r in all_results])
        avg_time = np.mean([r['mean_time'] for r in all_results])
        
        print(f"\n{'='*60}")
        print("OVERALL RESULTS")
        print(f"{'='*60}")
        print(f"Average FPS across {len(all_results)} images: {avg_fps:.2f}")
        print(f"Average inference time: {avg_time*1000:.2f} ms")
        print(f"Device: {device}")
        print(f"Model: {args.model_weights}")
        print(f"{'='*60}")


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--model-weights", required=True,
                       help="Path to the trained model weights")
    parser.add_argument("--image_path", required=True,
                       help="Directory of images to run on (no annotations needed)")
    parser.add_argument("--num-warmup", 
                       type=int, default=10,
                       help="Number of warmup iterations")
    parser.add_argument("--num-iterations", 
                       type=int, default=1,
                       help="Number of iterations for FPS measurement")
    args = parser.parse_args()
    
    if not args.config_file:
        args.config_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "configs/custom/maskformer2_mta_clip_custom_deploy_attribute.yaml")
    
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
