# Copyright (c) Facebook, Inc. and its affiliates.
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

# Define your custom dataset classes
# Replace these with your actual class names
CUSTOM_DATASET_CATEGORIES = [
    {"name": "background", "id": 0, "trainId": 0},
    {"name": "water", "id": 1, "trainId": 1},
]

def _get_custom_dataset_meta():
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing, so all ids are shifted by 1.
    ret = {
        "stuff_classes": [k["name"] for k in CUSTOM_DATASET_CATEGORIES],
        "stuff_dataset_id_to_contiguous_id": {k["id"]: k["trainId"] for k in CUSTOM_DATASET_CATEGORIES},
        "stuff_colors": [(k["trainId"] * 7 + 1) % 255 for k in CUSTOM_DATASET_CATEGORIES],
        "class_ignore": [0],  # background class to ignore
    }
    return ret

def register_custom_dataset_sem_seg(name, metadata, image_root, sem_seg_root):
    """
    Register a dataset for semantic segmentation.
    
    Args:
        name (str): the name that identifies a dataset, e.g. "custom_sem_seg_train".
        metadata (dict): extra metadata associated with this dataset.
        image_root (str): directory which contains all the images.
        sem_seg_root (str): directory which contains semantic segmentation annotations.
    """
    DatasetCatalog.register(
        name,
        lambda: load_sem_seg(sem_seg_root, image_root, gt_ext="png", image_ext="jpg"),
    )
    MetadataCatalog.get(name).set(
        sem_seg_root=sem_seg_root,
        image_root=image_root,
        evaluator_type="sem_seg",
        ignore_label=255,
        **metadata,
    )

def register_all_custom_dataset(root="/data1/huantao/workspace/project/flood_seg/dataset/my_waterdataset_challenge_val"):
    """
    Register all custom dataset splits.
    
    Args:
        root (str): path to the dataset root directory.
    """
    metadata = _get_custom_dataset_meta()
    
    # Register training dataset
    register_custom_dataset_sem_seg(
        "custom_sem_seg_train",
        metadata,
        os.path.join(root, "img_dir", "train"),  # Images are in img_dir/train
        os.path.join(root, "ann_dir", "train"),  # Annotations are in ann_dir/train
    )
    
    # Register validation dataset
    register_custom_dataset_sem_seg(
        "custom_sem_seg_val",
        metadata,
        os.path.join(root, "img_dir", "val"),    # Images are in img_dir/val
        os.path.join(root, "ann_dir", "val"),    # Annotations are in ann_dir/val
    )

def register_additional_test_datasets():
    """
    Register additional test datasets for evaluation.
    """
    metadata = _get_custom_dataset_meta()
    
    # Register flood test dataset
    # register_custom_dataset_sem_seg(
    #     "custom_sem_seg_all_flood",
    #     metadata,
    #     # os.path.join("/data1/huantao/workspace/project/flood_seg/dataset/my_flood_all_new_915", "img_dir", "val"),
    #     # os.path.join("/data1/huantao/workspace/project/flood_seg/dataset/my_flood_all_new_915", "ann_dir", "val"),
    #     # os.path.join("/data1/huantao/workspace/project/flood_seg/dataset_setting2/all_flood_setting2", "img_dir", "val"),
    #     # os.path.join("/data1/huantao/workspace/project/flood_seg/dataset_setting2/all_flood_setting2", "ann_dir", "val"),
    #     # os.path.join("/data1/huantao/workspace/project/flood_seg/dataset_setting2/all_flood_setting2_more", "img_dir", "val"),
    #     # os.path.join("/data1/huantao/workspace/project/flood_seg/dataset_setting2/all_flood_setting2_more", "ann_dir", "val"),
    #     os.path.join("/data1/huantao/workspace/project/flood_seg/dataset_unified/all_flood_setting2_new", "img_dir", "val"),
    #     os.path.join("/data1/huantao/workspace/project/flood_seg/dataset_unified/all_flood_setting2_new", "ann_dir", "val"),
    # )
    
    register_custom_dataset_sem_seg(
        "custom_sem_seg_normal_flood",
        metadata,
        # os.path.join("/data1/huantao/workspace/project/flood_seg/dataset/my_normal_flood_huantao", "img_dir", "val"),
        # os.path.join("/data1/huantao/workspace/project/flood_seg/dataset/my_normal_flood_huantao", "ann_dir", "val"),
        # os.path.join("/data1/huantao/workspace/project/flood_seg/dataset_unified/my_normal_flood_huantao_setting2_new", "img_dir", "val"),
        # os.path.join("/data1/huantao/workspace/project/flood_seg/dataset_unified/my_normal_flood_huantao_setting2_new", "ann_dir", "val"),
        os.path.join("/data1/huantao/workspace/project/flood_seg/dataset_w_urban_flood/my_normal_flood_huantao_w_urban_flood", "img_dir", "val"),
        os.path.join("/data1/huantao/workspace/project/flood_seg/dataset_w_urban_flood/my_normal_flood_huantao_w_urban_flood", "ann_dir", "val"),
    )
    
    register_custom_dataset_sem_seg(
        "custom_sem_seg_challenging_flood",
        metadata,
        # os.path.join("/data1/huantao/workspace/project/flood_seg/dataset/my_challenging_flood_huantao", "img_dir", "val"),
        # os.path.join("/data1/huantao/workspace/project/flood_seg/dataset/my_challenging_flood_huantao", "ann_dir", "val"),
        # os.path.join("/data1/huantao/workspace/project/flood_seg/dataset_setting2/my_challenging_flood_huantao_setting2", "img_dir", "val"),
        # os.path.join("/data1/huantao/workspace/project/flood_seg/dataset_setting2/my_challenging_flood_huantao_setting2", "ann_dir", "val"),
        # os.path.join("/data1/huantao/workspace/project/flood_seg/dataset_setting2/my_challenging_flood_huantao_setting2_more", "img_dir", "val"),
        # os.path.join("/data1/huantao/workspace/project/flood_seg/dataset_setting2/my_challenging_flood_huantao_setting2_more", "ann_dir", "val"),
        # os.path.join("/data1/huantao/workspace/project/flood_seg/dataset_unified/my_challenging_flood_huantao_setting2_new", "img_dir", "val"),
        # os.path.join("/data1/huantao/workspace/project/flood_seg/dataset_unified/my_challenging_flood_huantao_setting2_new", "ann_dir", "val"),
        os.path.join("/data1/huantao/workspace/project/flood_seg/dataset_w_urban_flood/my_challenging_flood_huantao_w_urban_flood_new", "img_dir", "val"),
        os.path.join("/data1/huantao/workspace/project/flood_seg/dataset_w_urban_flood/my_challenging_flood_huantao_w_urban_flood_new", "ann_dir", "val"),
    )
    
    register_custom_dataset_sem_seg(
        "custom_sem_seg_challenging_water",
        metadata,
        # os.path.join("/data1/huantao/workspace/project/flood_seg/dataset/my_waterdataset_val_edge", "img_dir", "val"),
        # os.path.join("/data1/huantao/workspace/project/flood_seg/dataset/my_waterdataset_val_edge", "ann_dir", "val"),
        os.path.join("/data1/huantao/workspace/project/flood_seg/dataset_setting2/my_waterdataset_val_edge_setting2", "img_dir", "val"),
        os.path.join("/data1/huantao/workspace/project/flood_seg/dataset_setting2/my_waterdataset_val_edge_setting2", "ann_dir", "val"),
    )
    
    # Register normal test dataset
    # register_custom_dataset_sem_seg(
    #     "custom_sem_seg_normal",
    #     metadata,
    #     os.path.join("/data1/huantao/workspace/project/flood_seg/dataset_unified/my_waterdataset_tmp_less", "img_dir", "val"),
    #     os.path.join("/data1/huantao/workspace/project/flood_seg/dataset_unified/my_waterdataset_tmp_less", "ann_dir", "val"),
    # )
    
if __name__ == "__main__":
    # Test the dataset registration
    register_all_custom_dataset()
    print("Custom dataset registered successfully!")
    
    # Test loading a sample
    from detectron2.data import DatasetCatalog
    train_data = DatasetCatalog.get("custom_sem_seg_train")
    print(f"Training samples: {len(train_data)}")
    if len(train_data) > 0:
        print(f"Sample data: {train_data[0]}")
