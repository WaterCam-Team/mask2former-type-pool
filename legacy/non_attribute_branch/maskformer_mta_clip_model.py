# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple
import json
import logging
import os
import torch
from torch import nn
from torch.nn import functional as F
import random
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

# from .modeling.criterion import SetCriterion
# from .modeling.criterion_mta_clip import SetCriterion
from .modeling.criterion_mta_clip_multi_layer import SetCriterion
from .modeling.matcher import HungarianMatcher

CHALLENGING_TYPES = ['transparent', 'shallow', 'reflection', 'glare', 'dark', 'muddy', 'rainy', 'blurry']
# CHALLENGING_TYPES = ['transparent', 'reflective', 'glare', 'dark', 'muddy', 'rainy']

@META_ARCH_REGISTRY.register()
class MaskFormerMtaCLIP(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        K: int,
        # K_water: int,
        # K_bg: int,
        align_loss_weight: float = 1.0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image
        
        # Store alignment loss weight
        #### added by Huantao
        self._align_loss_weight = align_loss_weight
        self.K = K
        # self.K_water = K_water
        # self.K_bg = K_bg
        # self.K_per_cls = [K_bg, K_water]
        
        # Only needed by the (currently disabled) fixed-prompt learner during
        # training. Absent on deployment machines, so do not hard-fail on it.
        fixed_prompt_path  = "/data1/huantao/workspace/project/flood_seg/dataset_w_urban_flood/water_scene_descriptions.json"
        if os.path.exists(fixed_prompt_path):
            with open(fixed_prompt_path, "r") as f:
                self.fixed_prompt_dict = json.load(f)
        else:
            logging.getLogger(__name__).warning(
                f"fixed prompt file not found, using empty dict: {fixed_prompt_path}")
            self.fixed_prompt_dict = {}
        ###############
        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
            
        if cfg.MODEL.MASK_FORMER.USE_ONE_TO_MANY:
            # weight for the final layer one-to-many loss
            weight_dict.update({
                "loss_ce_1toM": 0.7,
                "loss_mask_1toM": 1.7,
                "loss_dice_1toM": 1.7,
            })
        losses = ["labels", "masks"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )
        
        # Store alignment loss weight for use in forward pass
        align_loss_weight = getattr(cfg.MODEL.MASK_FORMER, 'ALIGN_LOSS_WEIGHT', 1.0)

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "align_loss_weight": align_loss_weight,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            'K': cfg.MODEL.MASK_FORMER.NUM_PROMPTS_PER_CLASS
            # 'K_water': cfg.MODEL.MASK_FORMER.NUM_PROMPTS_WATER,
            # 'K_bg': cfg.MODEL.MASK_FORMER.NUM_PROMPTS_BG
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        difficulty = []
        fixed_prompts = []
        # 0: easy  1:hard
        # for x in batched_inputs:
        #     img_name = x['file_name'].split('/')[-1].split('_')
        #     if 'challenging' in img_name:
        #         difficulty.append(1) 
        #     else:
        #         difficulty.append(0)
        # difficulty = torch.tensor(difficulty, device=self.device)
        # 0: normal 1: transparent 2:shallow 3:reflection 4:glare 5:dark 6:muddy 7:rainy 8:blurry
        if self.training:
            for x in batched_inputs:
                img_name = x['file_name'].split('/')[-1].split('_')
                if 'challenging' in img_name:
                    challenging_type = img_name[-1].split('.')[0]
                    ### less challenging types ###
                    # if challenging_type == 'shallow':
                    #     challenging_type = 'transparent'
                    # elif challenging_type == 'blurry':
                    #     challenging_type = 'rainy'
                    # elif challenging_type == 'reflection':
                    #     challenging_type = 'reflective'
                    ######################
                    type_idx = CHALLENGING_TYPES.index(challenging_type)+1
                    difficulty.append(type_idx) 
                else:
                    difficulty.append(0)
                img_name_full = x['file_name'].split('/')[-1]
                fixed_prompts.append(self.fixed_prompt_dict.get(img_name_full, ""))
            difficulty = torch.tensor(difficulty, device=self.device)
        else:
            difficulty = None
            fixed_prompts = None
        
        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features, self.training, difficulty, fixed_prompts)

        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            # bipartite matching-based loss
            losses, mask_to_gt_cls = self.criterion(outputs, targets)
            
            align_loss = self.masktext_alignment_loss(outputs['text_clip'], outputs['mask_clip'], mask_to_gt_cls)
            # align_loss = self.masktext_alignment_loss_mixneg(outputs['text_clip'], outputs['mask_clip'], mask_to_gt_cls, self.K)
            # align_loss = self.masktext_alignment_loss_scene_type(outputs['text_clip'], outputs['mask_clip'], mask_to_gt_cls, difficulty)
            # align_loss = self.masktext_alignment_loss_scene_type_siglip(outputs['text_clip'], outputs['mask_clip'], mask_to_gt_cls, difficulty)
            # align_loss = self.masktext_alignment_loss_mixnef_difficulty_binary(outputs['text_clip'], outputs['mask_clip'], mask_to_gt_cls, difficulty)
            # align_loss = self.masktext_alignment_loss_mixnef_difficulty_binary_multiple_chall(outputs['text_clip'], outputs['mask_clip'], mask_to_gt_cls, difficulty)
            # align_loss = self.masktext_alignment_loss_mixnef_difficulty(outputs['text_clip'], outputs['mask_clip'], mask_to_gt_cls, self.K, difficulty)
            # align_loss = self.masktext_alignment_loss_mixneg_multi_layer(outputs['text_clip_layers'], outputs['mask_clip_layers'], mask_to_gt_cls, self.K)
            # align_loss = self.masktext_alignment_loss_mixneg_diffK(outputs['text_clip'], outputs['mask_clip'], mask_to_gt_cls, self.K_per_cls)
            # align_loss = self.masktext_alignment_loss_sepneg(outputs['text_clip'], outputs['mask_clip'], mask_to_gt_cls, (self.K, self.K), temperature=0.07)
            align_loss_weight = getattr(self, '_align_loss_weight', 1.0)
            losses['mta_clip_contrastive_loss'] = align_loss * align_loss_weight
            
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                elif k == 'mta_clip_contrastive_loss':
                    # Keep alignment loss even if not in weight_dict (use weight already applied)
                    pass
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r
                
                # instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["instances"] = instance_r

            return processed_results

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets
    
    def masktext_alignment_loss(self, text_clip, mask_clip, mask_to_gt_cls, temperature=0.07):
        """
        Simplified for 2 fixed text prompts.
        Args:
            text_clip: (B, 2, D) projected CLIP embeddings for 'water' and 'non-water'
            mask_clip: (B, N, D) projected mask queries
            mask_to_gt_cls: (B, N) ground-truth class index for each mask query (0 or 1, -1 ignore)
            temperature: softmax temperature
        Returns:
            scalar contrastive loss
        """
        B, N, D = mask_clip.shape

        # normalize
        mask_norm = F.normalize(mask_clip, dim=-1)
        text_norm = F.normalize(text_clip, dim=-1)

        losses = []
        # loss_text2mask = []
        for b in range(B):
            gt_classes = mask_to_gt_cls[b]       # (N,)
            valid_mask = gt_classes >= 0
            if valid_mask.sum() == 0:
                continue

            mask_feats = mask_norm[b, valid_mask]  # (n_valid, D)
            gt_classes_valid = gt_classes[valid_mask]  # (n_valid,)

            # compute cosine similarity with 2 text embeddings
            sim = mask_feats @ text_norm[b].T   # (n_valid, 2)

            # cross-entropy loss
            loss_b = F.cross_entropy(sim / temperature, gt_classes_valid)
            losses.append(loss_b)
            
            # 2) TEXT → MASK (CLIP-style)
            # -------------------------
            # text_feats = text_norm[b]  # (2, D)

            # for cls in [0, 1]:
            #     cls_mask = gt_classes == cls
            #     if cls_mask.sum() == 0:
            #         continue

            #     # aggregate positive masks for this class
            #     pos_mask_feats = mask_norm[b, cls_mask]   # (n_pos, D)
            #     pos_mask_feat = pos_mask_feats.mean(dim=0, keepdim=True)  # (1, D)

            #     # similarity against both class texts
            #     sim_t2m = pos_mask_feat @ text_feats.T    # (1, 2)
            #     target = torch.tensor([cls], device=sim_t2m.device)

            #     loss_t2m = F.cross_entropy(sim_t2m / temperature, target)
            #     loss_text2mask.append(loss_t2m)

        if len(losses) == 0:
            return torch.tensor(0.0, device=mask_clip.device, requires_grad=True)
        # loss_m2t = torch.stack(losses).mean()
        # loss_t2m = torch.stack(loss_text2mask).mean() if len(loss_text2mask) > 0 else 0.0
        # return 0.5 * (loss_m2t + loss_t2m)
        return torch.stack(losses).mean()
    
    
    def masktext_alignment_loss_mixneg(self, text_clip, mask_clip, mask_to_gt_cls, K, temperature=0.07):
        B, N, D = mask_clip.shape
        _, CK, _ = text_clip.shape
        C = CK // K
        # normalize
        mask_norm = F.normalize(mask_clip, dim=-1)
        text_norm = F.normalize(text_clip, dim=-1)
        # text_norm = text_norm.view(B, C, K, D)
        losses = []
        # loss_text2mask = []
        ######## random #####
        for b in range(B):
            gt_classes = mask_to_gt_cls[b]       # (N,)
            valid_mask = gt_classes >= 0
            # ---- NEW: ignore background ----
            water_mask = gt_classes == 1
            valid_mask = valid_mask & water_mask
            # --------------------------------
            if valid_mask.sum() == 0:
                continue

            mask_feats = mask_norm[b, valid_mask]  # (n_valid, D)
            gt_classes_valid = gt_classes[valid_mask]  # (n_valid,)
            # compute cosine similarity with 2 text embeddings
            sim = mask_feats @ text_norm[b].T   # (n_valid, 2)

            # cross-entropy loss
            pos_indices = gt_classes_valid * K + torch.randint(
            low=0, high=K, size=gt_classes_valid.shape, device=gt_classes.device)

            loss_b = F.cross_entropy(
                sim / temperature,
                pos_indices)
            losses.append(loss_b)
        
        ### average ####
        # text_norm = text_norm.view(B, C, K, D)
        # text_norm = text_norm.mean(dim=2)
        # for b in range(B):
        #     gt_classes = mask_to_gt_cls[b]       # (N,)
        #     valid_mask = gt_classes >= 0
        #     if valid_mask.sum() == 0:
        #         continue

        #     mask_feats = mask_norm[b, valid_mask]  # (n_valid, D)
        #     gt_classes_valid = gt_classes[valid_mask]  # (n_valid,)

        #     # compute cosine similarity with 2 text embeddings
            
        #     sim = mask_feats @ text_norm[b].T   # (n_valid, 2)

        #     # cross-entropy loss
        #     loss_b = F.cross_entropy(
        #         sim / temperature,
        #         gt_classes_valid)
        #     losses.append(loss_b)
        
        #### highest similar ####
        # text_norm = text_norm.view(B, C, K, D)
        # for b in range(B):
        #     for i in range(N):
        #         gt = mask_to_gt_cls[b, i].item()
        #         if gt < 0:
        #             continue

        #         m = mask_norm[b, i]                    # (D,)
        #         t_gt = text_norm[b, gt]                # (K, D)

        #         # 🔹 select hardest positive
        #         sim_pos_all = self.cosine_sim(m[None], t_gt).squeeze(0)  # (K,)
        #         k_pos = sim_pos_all.argmax()
        #         s_pos = sim_pos_all[k_pos]

        #         # 🔹 negatives: all other prompts
        #         negs = []
        #         for c in range(C):
        #             for k in range(K):
        #                 if not (c == gt and k == k_pos):
        #                     negs.append(text_norm[b, c, k])

        #         negs = torch.stack(negs, dim=0)  # (C*K-1, D)
        #         s_neg = self.cosine_sim(m[None], negs).squeeze(0)

        #         logits = torch.cat([s_pos.unsqueeze(0), s_neg]) / temperature
        #         labels = torch.zeros(1, dtype=torch.long, device=logits.device)

        #         loss = F.cross_entropy(logits.unsqueeze(0), labels)
        #         losses.append(loss)
            
        if len(losses) == 0:
            return torch.tensor(0.0, device=mask_clip.device, requires_grad=True)
        return torch.stack(losses).mean()
    
    def masktext_alignment_loss_scene_type(self,
        text_clip,          # (C, D)  -> 10 prompts
        mask_clip,          # (B, Q, D)
        mask_to_gt_cls,     # (B, Q)  0=background 1=water
        difficulty,         # (B,) scene type: 0~8
        temperature=0.07):

        B, Q, D = mask_clip.shape
        C = text_clip.shape[0]

        losses = []
        # normalize
        mask_norm = F.normalize(mask_clip, dim=-1)
        text_norm = F.normalize(text_clip, dim=-1)

        for b in range(B):
            gt_classes = mask_to_gt_cls[b]  # (Q,)
            valid_mask = gt_classes >= 0
            if valid_mask.sum() == 0:
                continue

            mask_feats = mask_norm[b, valid_mask]     # (N, D)
            gt_cls_valid = gt_classes[valid_mask]     # (N,)

            scene_type = difficulty[b]                # 0~8
            pos_indices = []
            for cls in gt_cls_valid:
                if cls == 0:   # background
                    pos_idx = 0
                else:          # water
                    pos_idx = difficulty[b] + 1

                pos_indices.append(pos_idx)

            pos_indices = torch.tensor(
                pos_indices, device=mask_clip.device)

            # similarity with all prompts
            sim = mask_feats @ text_norm[b].T  # (N, C)
            loss_b = F.cross_entropy(sim / temperature, pos_indices)
            losses.append(loss_b)

        if len(losses) == 0:
            return torch.tensor(
                0.0, device=mask_clip.device, requires_grad=True)
        return torch.stack(losses).mean()
    
    def masktext_alignment_loss_scene_type_siglip(self,
        text_clip,          # (C, D)  -> 10 prompts
        mask_clip,          # (B, Q, D)
        mask_to_gt_cls,     # (B, Q)  0=background 1=water
        difficulty,         # (B,) scene type: 0~8
        temperature=0.07):

        B, Q, D = mask_clip.shape
        _, P, _ = text_clip.shape

        losses = []
        mask_norm = F.normalize(mask_clip, dim=-1)
        text_norm = F.normalize(text_clip, dim=-1)

        for b in range(B):
            gt_classes = mask_to_gt_cls[b]  # (Q,)
            valid_mask = gt_classes >= 0
            if valid_mask.sum() == 0:
                continue

            mask_feats = mask_norm[b, valid_mask]     # (N, D)
            text_feats = text_norm[b] 
            gt_cls_valid = gt_classes[valid_mask]     # (N,)
            
            pos_indices = []
            for cls in gt_cls_valid:
                if cls == 0:   # background
                    pos_idx = 0
                else:          # water
                    pos_idx = difficulty[b] + 1

                pos_indices.append(pos_idx)

            pos_indices = torch.tensor(
                pos_indices, device=mask_clip.device)

            # similarity with all prompts
            sim = mask_feats @ text_norm[b].T / temperature # (N, C)
            labels = -torch.ones(mask_feats.shape[0], P, device=mask_clip.device)
            labels[torch.arange(mask_feats.shape[0]), pos_indices] = 1.0
            loss_b = -F.logsigmoid(labels * sim).mean()
            losses.append(loss_b)

        if len(losses) == 0:
            return torch.tensor(
                0.0, device=mask_clip.device, requires_grad=True)
        return torch.stack(losses).mean()
    
    def masktext_alignment_loss_mixnef_difficulty_binary_multiple_chall(self,
        text_clip,         # (B, T, D)
        mask_clip,         # (B, Q, D)
        mask_to_gt_cls,    # (B, Q)
        difficulty,
        temperature=0.07,
        topk=2):

        B, N, D = mask_clip.shape
        losses = []
        # normalize
        mask_norm = F.normalize(mask_clip, dim=-1)
        text_norm = F.normalize(text_clip, dim=-1)
        for b in range(B):
            gt_classes = mask_to_gt_cls[b]
            valid_mask = gt_classes >= 0
            if valid_mask.sum() == 0:
                continue

            mask_feats = mask_norm[b, valid_mask]
            gt_classes_valid = gt_classes[valid_mask]

            sim = mask_feats @ text_norm[b].T  # (N, 6)

            # build logits
            bg_logit = sim[:, 0:1]
            normal_logit = sim[:, 1:2]

            chall_sim = sim[:, 2:6]
            topk_val, _ = torch.topk(chall_sim, k=2, dim=1)
            chall_logit = topk_val.mean(dim=1, keepdim=True)

            final_logits = torch.cat(
                [bg_logit, normal_logit, chall_logit], dim=1)  # (N, 3)

            # targets
            targets = []
            for cls in gt_classes_valid:
                if cls == 0:
                    targets.append(0)
                else:
                    if difficulty[b] == 0:
                        targets.append(1)
                    else:
                        targets.append(2)

            targets = torch.tensor(targets, device=mask_clip.device)
            loss_b = F.cross_entropy(final_logits / temperature, targets)
            losses.append(loss_b)

        if len(losses) == 0:
            return torch.tensor(0.0, device=mask_clip.device, requires_grad=True)

        return torch.stack(losses).mean()
    
    def masktext_alignment_loss_mixnef_difficulty_binary(self,
        text_clip,         # (B, T, D)
        mask_clip,         # (B, Q, D)
        mask_to_gt_cls,    # (B, Q)
        difficulty,
        temperature=0.07):
        
        B, N, D = mask_clip.shape
        losses = []

        # normalize
        mask_norm = F.normalize(mask_clip, dim=-1)
        text_norm = F.normalize(text_clip, dim=-1)

        for b in range(B):
            gt_classes = mask_to_gt_cls[b]  # (N,)
            valid_mask = gt_classes >= 0
            if valid_mask.sum() == 0:
                continue

            mask_feats = mask_norm[b, valid_mask]       # (n_valid, D)
            gt_classes_valid = gt_classes[valid_mask]   # (n_valid,)

            pos_indices = []

            for cls in gt_classes_valid:
                if cls == 0:   # background
                    pos_idx = 0
                else:          # water
                    if difficulty[b] == 0:
                        pos_idx = 1  # normal water
                    else:
                        pos_idx = 2  # challenging water

                pos_indices.append(pos_idx)

            pos_indices = torch.tensor(
                pos_indices, device=mask_clip.device)
            # similarity with all prompts
            sim = mask_feats @ text_norm[b].T 

            # cross-entropy loss
            loss_b = F.cross_entropy(sim / temperature, pos_indices)
            losses.append(loss_b)

        if len(losses) == 0:
            return torch.tensor(0.0, device=mask_clip.device, requires_grad=True)
        return torch.stack(losses).mean()
    
    def masktext_alignment_loss_mixnef_difficulty(self,
        text_clip,         # (B, T, D)
        mask_clip,         # (B, Q, D)
        mask_to_gt_cls,    # (B, Q)
        K,                 # total prompts per class
        difficulty,
        hard_prompt_idx=[0,1,2],
        easy_prompt_idx=[3,4],
        temperature=0.07):
        
        B, N, D = mask_clip.shape
        _, CK, _ = text_clip.shape
        C = CK // K
        losses = []

        # normalize
        mask_norm = F.normalize(mask_clip, dim=-1)
        text_norm = F.normalize(text_clip, dim=-1)

        for b in range(B):
            gt_classes = mask_to_gt_cls[b]  # (N,)
            valid_mask = gt_classes >= 0
            if valid_mask.sum() == 0:
                continue

            mask_feats = mask_norm[b, valid_mask]       # (n_valid, D)
            gt_classes_valid = gt_classes[valid_mask]   # (n_valid,)

            # select prompt subset based on difficulty
            if difficulty[b] == 1:  # hard
                subset_idx = hard_prompt_idx
            else:  # easy
                subset_idx = easy_prompt_idx

            # randomly select a prompt index **per mask query**
            # rand_idx = torch.randint(0, len(subset_idx), size=gt_classes_valid.shape, device=text_norm.device)
            # pos_indices = gt_classes_valid * K + torch.tensor([subset_idx[i] for i in rand_idx], device=text_norm.device)

            # compute similarity with all text prompts
            sim = mask_feats @ text_norm[b].T  # (n_valid, C*K)
            ### high sim ###
            pos_indices = []
            for i, cls in enumerate(gt_classes_valid):
                cls = cls.item()
                # choose only the prompts in the difficulty subset
                cls_subset_idx = [cls*K + idx for idx in subset_idx]
                sim_cls_subset = sim[i, cls_subset_idx]   # (len(subset_idx),)
                best_k = torch.argmax(sim_cls_subset)    # select prompt with highest similarity
                pos_indices.append(cls_subset_idx[best_k])
            pos_indices = torch.tensor(pos_indices, device=mask_clip.device)
            ###################

            # cross-entropy loss
            loss_b = F.cross_entropy(sim / temperature, pos_indices)
            losses.append(loss_b)

        if len(losses) == 0:
            return torch.tensor(0.0, device=mask_clip.device, requires_grad=True)
        return torch.stack(losses).mean()

    
    def masktext_alignment_loss_mixneg_multi_layer(self, text_clip_layers, mask_clip_layers, mask_to_gt_cls, K, temperature=0.07):
        num_layers = len(mask_clip_layers)
        B, N, D = mask_clip_layers[-1].shape
        CK = text_clip_layers[-1].shape[1]
        C = CK // K

        losses = []
        for b in range(B):
            gt_classes = mask_to_gt_cls[b]  # (N,)
            valid_mask = gt_classes >= 0

            if valid_mask.sum() == 0:
                continue

            gt_classes_valid = gt_classes[valid_mask]   # (n_valid,)

            # -------- sample prompt index ONCE --------
            rand_prompt = torch.randint(
                low=0,
                high=K,
                size=gt_classes_valid.shape,
                device=gt_classes.device)

            pos_indices = gt_classes_valid * K + rand_prompt  # (n_valid,)
            layer_losses = []

            for l in range(num_layers):

                mask_clip = mask_clip_layers[l]
                text_clip = text_clip_layers[l]

                mask_norm = F.normalize(mask_clip, dim=-1)
                text_norm = F.normalize(text_clip, dim=-1)

                mask_feats = mask_norm[b, valid_mask]  # (n_valid, D)

                sim = mask_feats @ text_norm[b].T  # (n_valid, C*K)

                loss_layer = F.cross_entropy(
                    sim / temperature,
                    pos_indices
                )

                layer_losses.append(loss_layer)

            losses.append(torch.stack(layer_losses).mean())

        if len(losses) == 0:
            return torch.tensor(
                0.0,
                device=mask_clip_layers[0].device,
                requires_grad=True
            )

        return torch.stack(losses).mean()
    
    def masktext_alignment_loss_mixneg_diffK(self,
        text_clip, mask_clip, mask_to_gt_cls,
        K_per_class, temperature=0.07):
        B, N, D = mask_clip.shape
        device = mask_clip.device

        class_offsets = torch.tensor(
            [0, K_per_class[0]],
            device=device)

        mask_clip = F.normalize(mask_clip, dim=-1)
        text_clip = F.normalize(text_clip, dim=-1)

        losses = []

        for b in range(B):
            gt = mask_to_gt_cls[b]          # (N,)
            valid = gt >= 0
            if valid.sum() == 0:
                continue

            m_feats = mask_clip[b, valid]  # (n_valid, D)
            gt_valid = gt[valid]

            # Compute cosine similarity (already normalized, so dot product gives similarity in [-1, 1])
            similarity = m_feats @ text_clip[b].T  # (n_valid, sum_K)
            logits = similarity / temperature  # (n_valid, sum_K)

            labels = []
            for g in gt_valid:
                labels.append(
                    self.sample_prompt_index(
                        g, K_per_class, class_offsets
                    )
                )
            labels = torch.cat(labels).to(device)

            loss = F.cross_entropy(logits, labels)
            losses.append(loss)
        final_loss = torch.stack(losses).mean()
        return final_loss

    def masktext_alignment_loss_sepneg(self, text_clip, mask_clip, mask_to_gt_cls, K_per_class=(3,5), temperature=0.07):
        B, N, D = mask_clip.shape
        device = mask_clip.device

        class_offsets = torch.tensor(
            [0, K_per_class[0]],
            device=device
        )

        mask_clip = F.normalize(mask_clip, dim=-1)
        text_clip = F.normalize(text_clip, dim=-1)

        losses = []

        for b in range(B):
            gt = mask_to_gt_cls[b]
            valid = gt >= 0
            if valid.sum() == 0:
                continue

            m_feats = mask_clip[b, valid]
            gt_valid = gt[valid]

            batch_loss = 0
            for i, g in enumerate(gt_valid):
                pos_idx = self.sample_prompt_index(
                    g, K_per_class, class_offsets
                )

                # negatives = prompts of other classes only
                neg_indices = []
                for cls in range(len(K_per_class)):
                    if cls != g:
                        start = class_offsets[cls]
                        neg_indices.extend(
                            range(start, start + K_per_class[cls])
                        )

                indices = torch.tensor(
                    [pos_idx] + neg_indices,
                    device=device
                )

                logits = (
                    m_feats[i:i+1] @ text_clip[b, indices].T
                ) / temperature

                labels = torch.zeros(1, dtype=torch.long, device=device)
                batch_loss += F.cross_entropy(logits, labels)

            losses.append(batch_loss / len(gt_valid))

        return torch.stack(losses).mean()

    
    
    def sample_prompt_index(self, gt_cls, K_per_class, class_offsets):
        """
        gt_cls: scalar tensor ∈ {0,1}
        returns prompt index ∈ [0, sum_K)
        """
        k = torch.randint(
            0, K_per_class[gt_cls],
            (1,), device=gt_cls.device
        )
        return class_offsets[gt_cls] + k

    def cosine_sim(self, a, b):
        return torch.matmul(a, b.transpose(-1, -2))
    
    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // self.sem_seg_head.num_classes
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result
