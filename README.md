# Mask2Former Type Pool

Binary water segmentation (`background`, `water`) on Mask2Former + detectron2.

The CLIP text branch is used **only during training**. At inference it is skipped
entirely, so the deployed model is just Mask2Former: **44M parameters, 176 MB**.

## 1. Get the checkpoint

Use the **setting2** weights — they perform better than the setting1.

Download **`mask2former_type_pool_setting2_8ctx_2cls_deploy.pth`** (176 MB):

https://drive.google.com/file/d/17uA902mzzhmhJP9bwsQKhmf50vPl8ao3/view?usp=sharing

The text branch is already removed — nothing to convert, just run it.

```
md5: f10fe41f078d9d3e471c346275759453
```

<details>
<summary>Full training checkpoint (1.4 GB) — only needed for fine-tuning</summary>

**`mask2former_type_pool_setting2_8ctx_2cls_model_final.pth`**
https://drive.google.com/file/d/1HLGfPkzwLZ7WKe9BlD93jEfQQqYUU1o2/view?usp=sharing
</details>

## 2. Run

This needs the labelled test sets (`img_dir/` + `ann_dir/`). Their paths are
**hardcoded** in `mask2former/data/datasets/register_custom_dataset.py` — edit
`register_additional_test_datasets()` to point at your local copies first.

```bash
python evaluate_multi_test.py \
    --config-file configs/custom/maskformer2_mta_clip_custom_deploy_attribute.yaml \
    --model-weights mask2former_type_pool_setting2_8ctx_2cls_deploy.pth
```

`..._deploy_attribute.yaml` builds the model **without** the text branch (no CLIP,
no prompt learner). This is the config to use on an edge device — it needs neither
`open_clip` nor the CLIP weights.

---

`legacy/` holds unused code kept for reference.
