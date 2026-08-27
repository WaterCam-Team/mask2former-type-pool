# mta_clip_modules.py
# Implements PromptLearner, MaskTextDecoder, and Mask-to-Text contrastive loss
# Minimal, readable, and easily integrable with Mask2Former / SegFormer.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional
from .position_encoding import PositionEmbeddingSine
from detectron2.config import configurable
from detectron2.layers import Conv2d
import fvcore.nn.weight_init as weight_init
import open_clip
import clip 
import json
from .maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY
from mask2former.challenging_types import CHALLENGING_TYPES
from detectron2.utils.events import get_event_storage, has_event_storage

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     K=3,
                     is_training=True):
        if is_training:
            B, dim = query_pos.shape[1], query_pos.shape[2]
            # text_pos = torch.zeros(2*K, B, dim, device=query_pos.device)
            text_pos = torch.zeros(K, B, dim, device=query_pos.device)
            query_pos = torch.cat([text_pos, query_pos], dim=0)
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                K=3,
                is_training=True):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos, K, is_training)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     K=3,
                     is_training=True):
        if is_training:
            B, dim = query_pos.shape[1], query_pos.shape[2]
            # text_pos = torch.zeros(2*K, B, dim, device=query_pos.device)
            text_pos = torch.zeros(K, B, dim, device=query_pos.device)
            query_pos = torch.cat([text_pos, query_pos], dim=0)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                K=3,
                is_training=True):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos, K, is_training)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


    



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





class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
# ---------------------------
# Mask-Text Decoder
# ---------------------------
@TRANSFORMER_DECODER_REGISTRY.register()
class MaskTextDecoder_attri(nn.Module):
    @configurable
    def __init__(self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        clip_text_dim: int,
        num_prompts_per_class: int,
        # num_prompts_water: int,
        # num_prompts_bg: int,
        clip_model_name: str = "ViT-B/16",
        class_names: list = None,
        deploy: bool = False,
    ):
        super().__init__()

        # Inference-only build: the text branch is unused when is_training=False,
        # so skip constructing it entirely (see cfg.MODEL.MASK_FORMER.DEPLOY).
        self.deploy = deploy
        self.mask_classification = mask_classification
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.num_layers = dec_layers
        self.num_heads = nheads
        self.K = num_prompts_per_class
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.num_feature_levels = 3
        # 1. keep positional encodings
        self.pe_layer = PositionEmbeddingSine(hidden_dim // 2, normalize=True)

        # 2. learnable mask queries (same as original Mask2Former)
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        if not deploy:
            self.clip_model, self.clip_tokenizer = self.build_clip()
        if not deploy:
            self.prompt_learner = Half_PromptLearner_type(
                class_names=class_names,
                num_classes=num_classes,
                clip_model=self.clip_model,
                clip_tokenizer=self.clip_tokenizer)
        
        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        # 5. NEW: project to CLIP for contrastive loss
        if not deploy:
            self.to_mask_space = nn.Linear(clip_text_dim, hidden_dim)
            self.mask_to_clip = nn.Linear(hidden_dim, clip_text_dim)
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
 
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # original mask classifier still works
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes+1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        
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
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        
        # NEW: MTA-CLIP specific additions
        ret["clip_model_name"] = cfg.MODEL.MASK_FORMER.CLIP_MODEL_NAME
        ret["num_prompts_per_class"] = cfg.MODEL.MASK_FORMER.NUM_PROMPTS_PER_CLASS
        # ret["num_prompts_water"] = cfg.MODEL.MASK_FORMER.NUM_PROMPTS_WATER
        # ret["num_prompts_bg"] = cfg.MODEL.MASK_FORMER.NUM_PROMPTS_BG
        ret["class_names"] = cfg.MODEL.SEM_SEG_HEAD.CLASS_NAMES
        ret['clip_text_dim'] = cfg.MODEL.MASK_FORMER.CLIP_TEXT_DIM
        ret["deploy"] = cfg.MODEL.MASK_FORMER.DEPLOY
        return ret
    def build_clip(self):
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="openai")
        tokenizer = open_clip.get_tokenizer("ViT-B-16")
        clip_model = clip_model.cuda().eval()
        return clip_model, tokenizer
    
    def forward(self,x, mask_features, is_training, difficulty, fixed_prompts, mask=None):
        if is_training and self.deploy:
            raise RuntimeError(
                "MaskTextDecoder_attri was built with MODEL.MASK_FORMER.DEPLOY=True "
                "(text branch omitted) and cannot be trained. Set DEPLOY=False.")
        if is_training:
            B = mask_features.shape[0]
            #### prompt learner ####
            clip_text_embeddings, psudo_label = self.prompt_learner(mask_features)
            text_tokens = self.to_mask_space(clip_text_embeddings) 
            text_tokens = text_tokens.unsqueeze(1).repeat(1, B, 1)
            T = text_tokens.shape[0]
        else:
            text_tokens = None
            psudo_label = None
            T = 0

        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        mask_queries = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)  # (B, Q, hidden_dim)

        # 4. Combine
        if is_training:
            output = torch.cat([text_tokens, mask_queries], dim=0)  # (B, T+Q, hidden_dim)
        else:
            output = mask_queries
            
        predictions_class = []
        predictions_mask = [] 
        # text_clip_layers = []
        # mask_clip_layers = []
        #### multi-scale s=alignment ####
        # if self.training:
        #     query_clip = self.mask_to_clip(output.transpose(0,1))

        #     text_clip = query_clip[:, :T] + clip_text_embeddings.unsqueeze(0)#.repeat(B,1,1)
        #     mask_clip = query_clip[:, T:]

        #     text_clip_layers.append(text_clip)
        #     mask_clip_layers.append(mask_clip)
        ##########################
        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_class[:,T:])
        predictions_mask.append(outputs_mask[:,T:])

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed,
                # K = self.K_water + self.K_bg,
                # K = self.K*2,
                K = self.K,
                is_training=is_training
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed,
                # K = self.K_water + self.K_bg,
                # K = self.K*2,
                K = self.K,
                is_training=is_training)
            
            # FFN
            output = self.transformer_ffn_layers[i](output)

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class[:,T:])
            predictions_mask.append(outputs_mask[:,T:])
            ### multi-scale alignment ####
            # if self.training:
            #     query_clip = self.mask_to_clip(output.transpose(0,1))

            #     text_clip = query_clip[:, :T] + clip_text_embeddings.unsqueeze(0)#.repeat(B,1,1)
            #     mask_clip = query_clip[:, T:]

            #     text_clip_layers.append(text_clip)
            #     mask_clip_layers.append(mask_clip)
            ############################

        assert len(predictions_class) == self.num_layers + 1
        #### only align last layer #####
        if self.training:
            query_clip = self.mask_to_clip(output.transpose(0,1))  
            # MTA-CLIP: Add query features to preserve original CLIP textual space
            # This ensures the text embeddings maintain their semantic meaning from CLIP
            text_clip = query_clip[:, :T] + clip_text_embeddings.unsqueeze(0).repeat(B, 1, 1)    # (B, T, C)
            # text_clip = query_clip[:, :T] + clip_text_embeddings.transpose(0,1)
            # text_clip = clip_text_embeddings.unsqueeze(0).repeat(B, 1, 1)
            mask_clip = query_clip[:, T:] 
        else:
            text_clip = None
            mask_clip = None
        #     scale = None
        ##################################
            
        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            ),
            ### added by Huantao ##
            'output_query_feature': output,
            # 'text_clip_layers': text_clip_layers,
            # 'mask_clip_layers': mask_clip_layers
            'text_clip': text_clip,
            'mask_clip': mask_clip,
            'psudo_label': psudo_label
            # 'scale': scale
        }
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask
    
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

