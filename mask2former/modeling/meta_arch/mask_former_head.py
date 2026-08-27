# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
import torch

from ..transformer_decoder.maskformer_transformer_decoder import build_transformer_decoder
from ..pixel_decoder.fpn import build_pixel_decoder


@SEM_SEG_HEADS_REGISTRY.register()
class MaskFormerHead(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "sem_seg_head" in k and not k.startswith(prefix + "predictor"):
                    newk = k.replace(prefix, prefix + "pixel_decoder.")
                    # logger.debug(f"{k} ==> {newk}")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        pixel_decoder: nn.Module,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        # extra parameters
        transformer_predictor: nn.Module,
        transformer_in_feature: str,
        use_visual: bool = False,
        use_text: bool = False,
        use_yoloe: bool = False,
        cfg=None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        feature_strides = [v.stride for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor
        self.transformer_in_feature = transformer_in_feature

        self.num_classes = num_classes
        self.use_visual = use_visual
        self.use_text = use_text
        self.use_yoloe = use_yoloe
        
        # Text feature alignment components (YoLoE-style)
        if self.use_text and self.use_yoloe:
            mask_dim = pixel_decoder.mask_dim
            clip_dim = 512  # CLIP text embedding dimension
            
            # RepRTAHead: transforms visual features to text space
            # Assuming it's a small MLP/conv to match dimensions
            self.reprta_head = nn.Sequential(
                nn.Conv2d(mask_dim, mask_dim, kernel_size=1),
                nn.BatchNorm2d(mask_dim),
                nn.GELU(),
                nn.Conv2d(mask_dim, clip_dim, kernel_size=1),
            )
            
            self.visual_proj = nn.Sequential(
                nn.Linear(256, 512),  # project to CLIP dim
                nn.LayerNorm(512),
            )
            
            # Load pre-computed text embedding for "water"
            # Default path, can be overridden
            text_feat_path = None
            # if cfg is not None:
            #     text_feat_path = getattr(cfg.MODEL.SEM_SEG_HEAD, 'TEXT_FEAT_PATH', None)
            #     # If empty string, treat as None
            #     if text_feat_path == "":
            #         text_feat_path = None
            # if text_feat_path is None:
            text_feat_path = "/data1/huantao/workspace/project/flood_seg/SegFormer/text_feat/mean_water_feat.pt"
            
            logger = logging.getLogger(__name__)
            logger.info(f"Loading text embedding from: {text_feat_path}")
                
            e_water = torch.load(text_feat_path, map_location='cpu')
            e_water = e_water.to(torch.float32)     
            logger.info(f"Successfully loaded text embedding with shape {e_water.shape}")
            self.register_buffer("E_WATER", e_water)
        
        # Visual prototype alignment components
        if self.use_visual:
            # Get mask_dim from pixel_decoder
            mask_dim = pixel_decoder.mask_dim
            
            # Approach: Similarity-based classification (not feature alignment)
            # We maintain a "water prototype" learned from water pixels via EMA
            # For each pixel, we compute cosine similarity to this prototype
            # - Water pixels should have high similarity (close to prototype)
            # - Background pixels should have low similarity (far from prototype)
            
            self.logit_scale = nn.Parameter(torch.tensor(10.0))  # temperature for cosine logits
            self.register_buffer('p_water', torch.zeros(mask_dim))  # EMA visual prototype for water
            self.ema_m = 0.99  # EMA momentum
            self.water_id = 1  # class ID for water
            
            # Background head: learned classifier for background (complements the prototype)
            # This gives flexibility - bg_head learns what's "not water" implicitly
            self.bg_head = nn.Conv2d(mask_dim, 1, kernel_size=1)

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        # figure out in_channels to transformer predictor
        if cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "transformer_encoder":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        elif cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "pixel_embedding":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        elif cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "multi_scale_pixel_decoder" or \
                cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == 'multi_scale_mta_clip_decoder' or \
                cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == 'multi_scale_mta_clip_attri_decoder':  # for maskformer2
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        elif cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "res5_attribute":
            transformer_predictor_in_channels = input_shape['res5'].channels
        else:
            transformer_predictor_in_channels = input_shape[cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE].channels

        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "pixel_decoder": build_pixel_decoder(cfg, input_shape),
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            "transformer_in_feature": cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE,
            "transformer_predictor": build_transformer_decoder(
                cfg,
                transformer_predictor_in_channels,
                mask_classification=True,
            ),
            "use_visual": getattr(cfg.MODEL.SEM_SEG_HEAD, "USE_VISUAL", False),
            "use_text": getattr(cfg.MODEL.SEM_SEG_HEAD, "USE_TEXT", False),
            "use_yoloe": getattr(cfg.MODEL.SEM_SEG_HEAD, "USE_YOLOE", False),
            "cfg": cfg,
        }

    def forward(self, features, is_training, difficulty, fixed_prompts, mask=None):
        return self.layers(features, is_training, difficulty, fixed_prompts, mask)

    def layers(self, features, is_training, difficulty, fixed_prompts, mask=None):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features)
        
        # Apply visual prototype alignment if enabled
        if self.use_visual:
            # Normalize mask features
            Fv = mask_features / (mask_features.norm(dim=1, keepdim=True) + 1e-6)  # [B,C,H,W]
            
            # Normalize visual prototype
            p = self.p_water / (self.p_water.norm() + 1e-6)  # [C]
            
            # Compute water logit = cosine similarity with water prototype
            # This measures "how water-like is each pixel?" (range: -1 to 1)
            # - Water pixels: high similarity (close to prototype) -> positive values
            # - Background pixels: low similarity (far from prototype) -> negative/neutral values
            # Note: This is similarity-based classification, NOT feature alignment
            logit_water = (Fv * p.view(1, -1, 1, 1)).sum(1, keepdim=True)  # [B,1,H,W], range [-1, 1]
            logit_water = self.logit_scale.clamp(5., 20.) * logit_water  # Scale to useful range
            
            # Background logit: Learned classifier (learns what's "not water" implicitly)
            # This complements the prototype approach by learning background patterns directly
            logit_bg = self.bg_head(mask_features)  # [B,1,H,W]
            
            # Store visual similarity logits for potential use in loss or post-processing
            self._visual_logits = torch.cat([logit_bg, logit_water], dim=1)  # [B,2,H,W]
        
        if self.transformer_in_feature == "multi_scale_pixel_decoder":
            predictions = self.predictor(multi_scale_features, mask_features, mask)
        elif self.transformer_in_feature == "multi_scale_mta_clip_decoder":
            predictions = self.predictor(multi_scale_features, mask_features,is_training, difficulty, fixed_prompts, mask)
        elif self.transformer_in_feature == "multi_scale_mta_clip_attri_decoder":
            predictions = self.predictor(multi_scale_features, mask_features,is_training, difficulty, fixed_prompts, mask)
        else:
            if self.transformer_in_feature == "transformer_encoder":
                assert (
                    transformer_encoder_features is not None
                ), "Please use the TransformerEncoderPixelDecoder."
                predictions = self.predictor(transformer_encoder_features, mask_features, mask)
            elif self.transformer_in_feature == "pixel_embedding":
                predictions = self.predictor(mask_features, mask_features, mask)
            elif self.transformer_in_feature == "res5_attribute":
                predictions = self.predictor(features['res5'], mask_features, is_training, mask)
            else:
                predictions = self.predictor(features[self.transformer_in_feature], mask_features, mask)
        
        # If using visual alignment, we can modify predictions or return additional info
        if self.use_visual:
            # Optionally return mask_features for potential EMA update during training
            predictions['mask_features'] = mask_features
            predictions['visual_logits'] = self._visual_logits
        
        # Store mask_features for text alignment loss computation
        if self.use_text and self.use_yoloe:
            predictions['mask_features'] = mask_features
            cls_prob = predictions['pred_logits'].softmax(-1)
            water_prob = cls_prob[..., 1] 
            topk = 3  # or 1
            _, top_idx = torch.topk(water_prob, topk, dim=1)
            query_feats = predictions['output_query_feature'].permute(1, 0, 2)
            selected_feats = []
            for b in range(water_prob.shape[0]):
                selected_feats.append(query_feats[b, top_idx[b]])  # (topk, D)
            selected_feats = torch.stack(selected_feats)
            v_query = selected_feats.mean(dim=1)
            predictions['water_query_feat'] = v_query
        
        return predictions
    
    def masked_pooled_alignment_loss(self, Fv, text_proj_emb, gt_mask, eps=1e-6):
        """
        YoLoE-style masked pooled alignment loss.
        
        Args:
            Fv: (B, D, H, W) normalized visual features from reprta_head
            text_proj_emb: (D,) or (B, D) normalized text embedding
            gt_mask: (B, H, W) float {0,1} - ground truth mask for water class
            eps: small value for numerical stability
            
        Returns:
            scalar MSE loss on pooled cosine similarity
        """
        B, D, H, W = Fv.shape
        device = Fv.device
        
        # Prepare text emb
        if text_proj_emb.dim() == 1:
            text = text_proj_emb.view(1, D).expand(B, -1)
        else:
            text = text_proj_emb
        
        # Ensure gt_mask matches feature resolution and is binary
        if gt_mask.shape[-2:] != (H, W):
            gt_mask = F.interpolate(gt_mask.float().unsqueeze(1), size=(H, W), mode="nearest").squeeze(1)
        mask = (gt_mask > 0.5).to(torch.float32)  # (B, H, W)
        
        # Compute area (number of positive pixels)
        area = mask.unsqueeze(1).sum(dim=(2, 3))  # (B, 1)
        area_safe = torch.where(area == 0, torch.ones_like(area), area)
        
        # Pool features within mask: average pool over positive pixels
        # Fv: (B, D, H, W), mask: (B, H, W) -> expand to (B, 1, H, W)
        v_pool = (Fv * mask.unsqueeze(1)).sum(dim=(2, 3)) / (area_safe + eps)  # (B, D)
        
        # Fallback to global avg if no positive pixels
        no_pos = (area.squeeze(-1) == 0)
        if no_pos.any():
            v_global = Fv.view(B, D, -1).mean(-1)  # Global average
            v_pool[no_pos] = v_global[no_pos]
        
        # Normalize
        v_pool = v_pool / (v_pool.norm(dim=1, keepdim=True) + eps)
        text = text / (text.norm(dim=1, keepdim=True) + eps)
        
        # Compute cosine similarity
        sims = (v_pool * text).sum(dim=1)  # (B,)
        target = torch.ones_like(sims, device=device)  # Target is perfect alignment (similarity = 1)
        
        # Optional weighting by area ratio
        area_ratio = (area.squeeze(-1) / (H * W)).clamp(min=0.05)
        
        # MSE loss weighted by area
        loss = ((sims - target) ** 2 * area_ratio).mean()
        return loss
    
    def update_visual_prototype(self, mask_features, gt_masks, water_id=1):
        """
        Update the visual prototype using EMA during training.
        
        Args:
            mask_features: [B, C, H, W] - mask features from the forward pass
            gt_masks: [B, H, W] or [B, num_instances, H, W] - ground truth masks
            water_id: int - class ID for water in the ground truth
        
        This method should be called during training to update the visual prototype.
        """
        if not self.use_visual:
            return
        
        # Normalize mask features
        Fv = mask_features / (mask_features.norm(dim=1, keepdim=True) + 1e-6)  # [B,C,H,W]
        
        # Handle different GT mask formats
        if gt_masks.dim() == 3:  # [B, H, W] - semantic segmentation format
            water_mask = (gt_masks == water_id).float()  # [B, H, W]
        elif gt_masks.dim() == 4:  # [B, num_instances, H, W] - instance segmentation format
            # Check if any instance has water_id class
            water_mask = (gt_masks == water_id).any(dim=1).float()  # [B, H, W]
        else:
            return  # Unsupported format
        
        # Aggregate features from water regions
        # [B, C, H, W] * [B, 1, H, W] -> [B, C, H, W]
        water_features = Fv * water_mask.unsqueeze(1)
        
        # Average pool over spatial dimensions, weighted by water mask
        # Sum over H, W: [B, C, H, W] -> [B, C]
        feature_sum = water_features.sum(dim=[2, 3])  # [B, C]
        mask_sum = water_mask.sum(dim=[1, 2])  # [B]
        
        # Avoid division by zero
        mask_sum = mask_sum.clamp(min=1e-6)
        
        # Compute prototype per image: [B, C]
        prototypes = feature_sum / mask_sum.view(-1, 1)  # [B, C]
        
        # Average over batch: [C]
        batch_prototype = prototypes.mean(dim=0)  # [C]
        
        # Normalize before EMA update
        if batch_prototype.norm() > 1e-6:
            batch_prototype = batch_prototype / batch_prototype.norm()
            
            # EMA update
            with torch.no_grad():
                self.p_water.data = self.ema_m * self.p_water.data + (1 - self.ema_m) * batch_prototype
