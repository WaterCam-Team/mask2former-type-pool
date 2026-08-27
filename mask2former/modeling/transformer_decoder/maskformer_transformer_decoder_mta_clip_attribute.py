# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from .position_encoding import PositionEmbeddingSine
from .transformer import Transformer
from .maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY
from mask2former.challenging_types import CHALLENGING_TYPES
import open_clip
import clip 

def build_transformer_decoder(cfg, in_channels, mask_classification=True):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME
    return TRANSFORMER_DECODER_REGISTRY.get(name)(cfg, in_channels, mask_classification)


@TRANSFORMER_DECODER_REGISTRY.register()
class StandardTransformerDecoder_attri(nn.Module):
    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dropout: float,
        dim_feedforward: int,
        enc_layers: int,
        dec_layers: int,
        pre_norm: bool,
        deep_supervision: bool,
        mask_dim: int,
        enforce_input_project: bool,
        class_names: list = None,
        clip_text_dim: int,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dropout: dropout in Transformer
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            deep_supervision: whether to add supervision to every decoder layers
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        transformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=pre_norm,
            return_intermediate_dec=deep_supervision,
        )

        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model

        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        if in_channels != hidden_dim or enforce_input_project:
            self.input_proj = Conv2d(in_channels, hidden_dim, kernel_size=1)
            weight_init.c2_xavier_fill(self.input_proj)
        else:
            self.input_proj = nn.Sequential()
        self.aux_loss = deep_supervision

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        
        ######################
        self.to_mask_space = nn.Linear(clip_text_dim, hidden_dim)
        self.mask_to_clip = nn.Linear(hidden_dim, clip_text_dim)
        self.clip_model, self.clip_tokenizer = self.build_clip()
        self.prompt_learner = Half_PromptLearner_type(
            class_names=class_names,
            num_classes=num_classes,
            clip_model=self.clip_model,
            clip_tokenizer=self.clip_tokenizer)

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification

        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dropout"] = cfg.MODEL.MASK_FORMER.DROPOUT
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        ret["enc_layers"] = cfg.MODEL.MASK_FORMER.ENC_LAYERS
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["deep_supervision"] = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["class_names"] = cfg.MODEL.SEM_SEG_HEAD.CLASS_NAMES
        ret['clip_text_dim'] = cfg.MODEL.MASK_FORMER.CLIP_TEXT_DIM

        return ret
    
    def build_clip(self):
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="openai")
        tokenizer = open_clip.get_tokenizer("ViT-B-16")
        clip_model = clip_model.cuda().eval()
        return clip_model, tokenizer

    def forward(self, x, mask_features, is_training, mask=None):
        if is_training:
            B = mask_features.shape[0]
            #### prompt learner ####
            # clip_text_embeddings = self.prompt_learner(mask_features)
            clip_text_embeddings, psudo_label = self.prompt_learner(mask_features)
            # clip_text_embeddings, tokenized_prompts = self.half_prompt_learner(mask_features)
            # clip_text_embeddings = self.text_encoder(clip_text_embeddings, tokenized_prompts)
            text_tokens = self.to_mask_space(clip_text_embeddings) 
            # text_tokens = text_tokens.unsqueeze(1).repeat(1, B, 1)
            output = torch.cat([text_tokens, self.query_embed.weight], dim=0)
            T = text_tokens.shape[0]
        else:
            output = self.query_embed.weight 
            text_tokens = None
            psudo_label = None
            T = 0
        if mask is not None:
            mask = F.interpolate(mask[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        pos = self.pe_layer(x, mask)

        src = x
        # query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1) 
        # if is_training:
        #     output = torch.cat([text_tokens, query_embed], dim=0)  # (T+Q, B, hidden_dim)
        # else:
        #     output = query_embed
        # hs, memory = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos)
        hs, memory = self.transformer(self.input_proj(src), mask, output, pos)
        hs_mask = hs[:, :, T:, :] 

        if self.mask_classification:
            outputs_class = self.class_embed(hs_mask)
            out = {"pred_logits": outputs_class[-1]}
        else:
            out = {}

        if self.aux_loss:
            # [l, bs, queries, embed]
            mask_embed = self.mask_embed(hs_mask)
            outputs_seg_masks = torch.einsum("lbqc,bchw->lbqhw", mask_embed, mask_features)
            out["pred_masks"] = outputs_seg_masks[-1]
            out["aux_outputs"] = self._set_aux_loss(
                outputs_class if self.mask_classification else None, outputs_seg_masks
            )
        else:
            # FIXME h_boxes takes the last one computed, keep this in mind
            # [bs, queries, embed]
            mask_embed = self.mask_embed(hs_mask[-1])
            outputs_seg_masks = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
            out["pred_masks"] = outputs_seg_masks
        
        if is_training:
            hs_last = hs[-1]  # (B, T+Q, hidden_dim)
            query_clip = self.mask_to_clip(hs_last)                              # (B, T+Q, clip_dim)
            text_clip = query_clip[:, :T] + clip_text_embeddings.unsqueeze(0).repeat(B, 1, 1)  # (B, T, clip_dim)
            mask_clip = query_clip[:, T:]                                         # (B, Q, clip_dim)
            out["text_clip"] = text_clip
            out["mask_clip"] = mask_clip
            out["clip_text_embeddings"] = clip_text_embeddings
        else:
            out["text_clip"] = None
            out["mask_clip"] = None
            
        out['psudo_label'] = psudo_label    
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class Half_PromptLearner_type(nn.Module):
    def __init__(
        self,
        class_names,
        num_classes: int,
        context_len: int = 8,
        clip_model=None,
        clip_tokenizer=None,
        device='cuda'
    ):
        super().__init__()

        self.class_names = class_names
        self.types = ['normal'] + CHALLENGING_TYPES
        # self.C = num_classes
        self.context_len = context_len
        self.device = device

        assert clip_model is not None
        assert clip_tokenizer is not None

        self.clip_model = clip_model
        self.clip_tokenizer = clip_tokenizer

        clip_ctx_dim = clip_model.token_embedding.weight.shape[1]
        self.clip_ctx_dim = clip_ctx_dim
        self.clip_out_dim = clip_model.text_projection.shape[1]

        # index_to_replace = self.types.index("dark")
        # self.types[index_to_replace] = "nighttime" 
        # prefix replaced by context tokens
        self.full_class_names = []
        for cls in class_names:
            if cls == "water":
                self.full_class_names.extend([f"{t} water" for t in self.types])
            else:
                self.full_class_names.append(cls)
        # index_to_replace = self.full_class_names.index("dark water")
        # self.full_class_names[index_to_replace] = 'water in dark scene'
        tokens = self.clip_tokenizer(self.full_class_names[1:]).cuda()  
        with torch.no_grad():
            embedding_type = self.clip_model.encode_text(tokens)
        
        self.C = len(self.full_class_names)
        # learnable context tokens (C, context_len, D)
        self.prompt_ctx = nn.Parameter(torch.randn(self.C, context_len, clip_ctx_dim) * 0.02)
        self.prefix = " ".join(["X"] * context_len) + " "

        texts = [self.prefix + name for name in self.full_class_names]
        token_ids = self.clip_tokenizer(texts).to(self.device)

        with torch.no_grad():
            embedding = self.clip_model.token_embedding(token_ids)

        self.register_buffer("token_ids", token_ids)
        self.register_buffer("classname_emb", embedding)
        self.register_buffer("water_type_emb", embedding_type)
        self.proj = torch.nn.Conv2d(256, 512, kernel_size=1)

    def forward(self, pixel_feat):
        pixel_feat = self.proj(pixel_feat)
        # pixel_feat = F.normalize(pixel_feat, dim=1)
        pixel_global = pixel_feat.mean(dim=[2, 3]) 
        pixel_global = F.normalize(pixel_global, dim=-1)
        #### select one #####
        sim = torch.einsum('bc,nc->bn', pixel_global, self.water_type_emb)
        # weights = F.softmax(sim / 0.07, dim=-1)
        # selected = torch.einsum('bn,nd->bd', weights, self.water_type_emb)
        bs = sim.shape[0]
        best_sim, psudo_label = sim.max(dim=1)
        
        token_ids = self.token_ids
        classname_emb = self.classname_emb
        B_cls, L, D = classname_emb.shape
        outputs = []

        for ci in range(self.C):
            ctx = self.prompt_ctx[ci]  # (context_len, D)
            cls_emb = classname_emb[ci].clone()
            cls_emb[1:1+self.context_len] = ctx
            x = cls_emb + self.clip_model.positional_embedding
            x = x.unsqueeze(1)
            x = self.clip_model.transformer(x)
            x = x.squeeze(1)
            x = self.clip_model.ln_final(x)

            eos_idx = token_ids[ci].argmax().item()
            context_tokens = x[1:1+self.context_len]
            context_pooled = context_tokens.mean(dim=0)

            cls_tokens = x[1+self.context_len:eos_idx]
            cls_pooled = cls_tokens.mean(dim=0)

            text_feat = 0.7 * context_pooled + 0.3 * cls_pooled
            text_feat = text_feat @ self.clip_model.text_projection

            outputs.append(text_feat)
        outputs = torch.stack(outputs, dim=0)
        return outputs, psudo_label
