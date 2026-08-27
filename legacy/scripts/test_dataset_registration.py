#!/usr/bin/env python3
"""
Test script to verify that all datasets are properly registered.
"""
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from mask2former.data.datasets.register_custom_dataset import register_all_custom_dataset, register_additional_test_datasets

def test_dataset_registration():
    """Test if all datasets are properly registered."""
    
    # Register all datasets
    register_all_custom_dataset("/data1/huantao/workspace/project/flood_seg/dataset/my_waterdataset_challenge_val")
    register_additional_test_datasets()
    
    # List of expected datasets
    expected_datasets = [
        "custom_sem_seg_train",
        "custom_sem_seg_val", 
        "custom_sem_seg_edge",
        "custom_sem_seg_flood",
        "custom_sem_seg_normal"
    ]
    
    print("="*60)
    print("DATASET REGISTRATION TEST")
    print("="*60)
    
    all_registered = True
    
    for dataset_name in expected_datasets:
        try:
            dataset = DatasetCatalog.get(dataset_name)
            metadata = MetadataCatalog.get(dataset_name)
            
            print(f"✓ {dataset_name}:")
            print(f"  - Samples: {len(dataset)}")
            print(f"  - Image root: {metadata.get('image_root', 'N/A')}")
            print(f"  - Sem seg root: {metadata.get('sem_seg_root', 'N/A')}")
            print(f"  - Evaluator type: {metadata.get('evaluator_type', 'N/A')}")
            print()
            
        except Exception as e:
            print(f"✗ {dataset_name}: FAILED - {str(e)}")
            all_registered = False
    
    print("="*60)
    if all_registered:
        print("✓ ALL DATASETS SUCCESSFULLY REGISTERED!")
    else:
        print("✗ SOME DATASETS FAILED TO REGISTER!")
    print("="*60)
    
    return all_registered

if __name__ == "__main__":
    test_dataset_registration()
