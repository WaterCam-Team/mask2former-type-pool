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
from transformers import SiglipTextModel, AutoTokenizer
from .maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY
from mask2former.maskformer_mta_clip_model import CHALLENGING_TYPES

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

class Fixed_PromptLearner(nn.Module):
    def __init__(
        self,
        clip_model=None,
        clip_tokenizer=None,
        device='cuda'
    ):
        super().__init__()
        
        self.device = device
        assert clip_model is not None, "clip_model required"
        assert clip_tokenizer is not None, "clip_tokenizer required"

        self.clip_model = clip_model
        self.clip_tokenizer = clip_tokenizer

        # Freeze CLIP text encoder
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, fixed_prompts):
        bg_prompts = []
        water_prompts = []

        for text in fixed_prompts:
            # split into two sentences
            sentences = text.strip().split(". ")

            if len(sentences) >= 2:
                res1 = sentences[0]
                res2 = sentences[-1]
            else:
                # fallback
                res1 = text
                res2 = text

            # add period back if removed
            if not res1.endswith("."):
                res1 += "."

            bg_prompts.append(res1)
            water_prompts.append(res2)

        # tokenize
        bg_tokens = self.clip_tokenizer(bg_prompts).to(self.device)

        water_tokens = self.clip_tokenizer(water_prompts).to(self.device)

        # encode
        with torch.no_grad():
            bg_features = self.clip_model.encode_text(bg_tokens)
            water_features = self.clip_model.encode_text(water_tokens)
        text_features = torch.stack([bg_features, water_features], dim=0)
        return text_features.to(self.device)  
        

class PromptLearner(nn.Module):
    """
    CoOp-style prompt learner for CLIP:
    - Learns context tokens (continuous embeddings)
    - Builds textual prompts combining learned context + class names
    - Tokenizes using CLIP tokenizer
    - Feeds into frozen CLIP text encoder
    - Returns CLIP text embeddings (C*K, clip_dim)
    """

    def __init__(
        self,
        class_names,
        num_classes: int,
        context_len: int = 8,
        K: int = 3,
        # K_water: int = 5,
        # K_bg: int = 3,
        clip_model=None,
        clip_tokenizer=None,
        device='cuda'
    ):
        super().__init__()
        self.class_names = class_names
        self.C = num_classes
        self.K = K
        # self.K_water = K_water
        # self.K_bg = K_bg
        # self.K_per_cls = [K_bg, K_water]
        self.context_len = context_len
        self.device = device

        assert clip_model is not None, "clip_model required"
        assert clip_tokenizer is not None, "clip_tokenizer required"

        self.clip_model = clip_model
        self.clip_tokenizer = clip_tokenizer

        # CLIP dimensions
        self.clip_ctx_dim = clip_model.token_embedding.weight.shape[1]
        self.clip_out_dim = clip_model.text_projection.shape[1]

        # Learnable continuous context tokens
        # shape (C, K, context_len, clip_ctx_dim)
        self.prompt_ctx = nn.Parameter(
            torch.randn(num_classes, K, context_len, self.clip_ctx_dim) * 0.02)
        # self.prompt_ctx = nn.ParameterList([
        #     nn.Parameter(torch.randn(Ki, context_len, self.clip_ctx_dim) * 0.02)
        #     for Ki in self.K_per_cls])
        # self.prompt_ctx = nn.ParameterList([
        #     nn.Parameter(torch.randn(Ki, context_len, self.clip_ctx_dim))
        #     for Ki in self.K_per_cls])
        # for p in self.prompt_ctx:
        #     nn.init.trunc_normal_(p)
        # Template: simple
        self.prefix = "a photo of "
        # self.prefix = 'a scene containing '
        # self.prefix = " ".join(["X"] * context_len) + " "
        
        ####### class name will be the only hard token
        # self.prefix = ""
        # # Pre-compute class token lengths
        # with torch.no_grad():
        #     tokenized = clip_tokenizer(class_names)
        #     self.class_token_lens = [
        #         (t != 0).sum().item() - 2  # exclude SOS & EOS
        #         for t in tokenized]

    def _embed_classnames(self):
        """Tokenize raw class words (no context added)."""
        # texts = [self.prefix + name for name in self.class_names]  # C strings
        # texts = [f"{self.prefix} {name}" for name in self.class_names]
        texts = [self.prefix + name + "." for name in self.class_names]
        token_ids = self.clip_tokenizer(texts).to(self.device)
        with torch.no_grad():
            emb = self.clip_model.token_embedding(token_ids)  # (C, L, ctx_dim)
        return token_ids, emb  # raw CLIP embeddings

    def forward(self):
        token_ids, classname_emb = self._embed_classnames()
        B_cls, L, D = classname_emb.shape  # (C, L, ctx_dim)

        outputs = []
        for ci in range(self.C):
            # Ki = self.K_per_cls[ci]
            Ki = self.K
            ### keep class name ###
            # cls_token_len = self.class_token_lens[ci]
            ################
            for ki in range(Ki):
                ctx = self.prompt_ctx[ci][ki]  # (context_len, D)

                # ctx_len = self.context_len
                cls_emb = classname_emb[ci].clone()  # (L, D)

                # max_replace = min(ctx_len, cls_emb.shape[0] - 1)
                # assert max_replace + cls_token_len < L, \
                #     "Context too long for CLIP max token length"
                # cls_emb[1:1+max_replace] = ctx[:max_replace]
                cls_emb[1 : 1 + self.context_len] = ctx

                x = cls_emb + self.clip_model.positional_embedding  # (L, D)
                
                x = x.unsqueeze(1)
                x = self.clip_model.transformer(x)
                x = x.squeeze(1)
                x = self.clip_model.ln_final(x)

                eos_idx = token_ids[ci].argmax().item()
                # text_feat = x[eos_idx] @ self.clip_model.text_projection
                context_tokens = x[1:1+self.context_len, :]
                context_pooled = context_tokens.mean(dim=0)
                cls_tokens = x[1+self.context_len:eos_idx]
                cls_pooled = cls_tokens.mean(dim=0)
                text_feat = 0.7 * context_pooled + 0.3 * cls_pooled
                text_feat = text_feat @ self.clip_model.text_projection
                text_feat = context_pooled @ self.clip_model.text_projection
                outputs.append(text_feat)

        # for ci in range(self.C):
        #     # if ci == 0:
        #     #     Ki = 1
        #     # else:
        #     Ki = self.K
        #     for ki in range(Ki):
        #         ctx = self.prompt_ctx[ci][ki]  # (context_len, D)

        #         ctx_len = self.context_len
        #         cls_emb = classname_emb[ci].clone()  # (L, D)

        #         max_replace = min(ctx_len, cls_emb.shape[0] - 1)
        #         cls_emb[1:1+max_replace] = ctx[:max_replace]

        #         x = cls_emb + self.clip_model.positional_embedding  # (L, D)
        #         x = x.unsqueeze(1)
        #         x = self.clip_model.transformer(x)
        #         x = x.squeeze(1)
        #         x = self.clip_model.ln_final(x)

        #         eos_idx = token_ids[ci].argmax().item()
        #         text_feat = x[eos_idx] @ self.clip_model.text_projection

        #         outputs.append(text_feat)
        
        final = torch.stack(outputs, dim=0)
        #scale = self.clip_model.logit_scale.exp()
        return final.to(self.device)#, scale

class CoOpPromptLearner(nn.Module):
    def __init__(
        self,
        classnames,
        clip_model,
        K_per_cls,
        n_ctx=8,
        ctx_init=None,
        device="cuda"
    ):
        super().__init__()

        self.classnames = classnames
        self.n_cls = len(classnames)
        self.n_ctx = n_ctx
        self.K_per_cls = K_per_cls
        self.device = device

        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        self.ctx = nn.ParameterList()

        for K in K_per_cls:
            if ctx_init:
                ctx_init = ctx_init.replace("_", " ")
                tokens = clip.tokenize(ctx_init).to(device)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(tokens).type(dtype)
                init_ctx = embedding[0, 1:1+n_ctx, :]
                ctx = init_ctx.unsqueeze(0).repeat(K, 1, 1)
            else:
                ctx = torch.empty(K, n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx, std=0.02)

            self.ctx.append(nn.Parameter(ctx))

        # --------------------------------------------------
        # Tokenize class names
        # --------------------------------------------------
        prompts = [f"{' '.join(['X'] * n_ctx)} {name}" for name in classnames]
        # prompts = [' '.join(['X'] * n_ctx) for _ in range(len(classnames))]
        self.tokenized = torch.cat([clip.tokenize(p) for p in prompts]).to(device)

        with torch.no_grad():
            embedding = clip_model.token_embedding(self.tokenized).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])   # SOS
        self.register_buffer("token_suffix", embedding[:, 1+n_ctx:, :])  # class + EOS
        self.register_buffer("tokenized_prompts", self.tokenized)

    def forward(self):
        prompt_list = []
        tokenized_list = []
        class_ids = []

        for cls_id, K in enumerate(self.K_per_cls):
            prefix = self.token_prefix[cls_id:cls_id+1].repeat(K, 1, 1)
            suffix = self.token_suffix[cls_id:cls_id+1].repeat(K, 1, 1)
            ctx = self.ctx[cls_id]

            prompts = torch.cat([prefix, ctx, suffix], dim=1)
            prompt_list.append(prompts)

            tokenized_list.append(
                self.tokenized_prompts[cls_id:cls_id+1].repeat(K, 1)
            )

            class_ids.extend([cls_id] * K)

        prompts = torch.cat(prompt_list, dim=0)
        tokenized = torch.cat(tokenized_list, dim=0)
        class_ids = torch.tensor(class_ids, device=prompts.device)

        return prompts, tokenized
    
class Difficulty_CoOpPromptLearner_siglip(nn.Module):
    def __init__(
        self,
        classnames,
        n_ctx=8,
        ctx_init=None,
        model_name="google/siglip2-base-patch16-224",
        device="cuda"
    ):
        super().__init__()

        self.classnames = classnames
        self.types = ['normal'] + CHALLENGING_TYPES
        self.n_ctx = n_ctx
        self.device = device

        # ── Load SigLIP2 ──────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_encoder = SiglipTextModel.from_pretrained(model_name).to(device)
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        ctx_dim = self.text_encoder.config.hidden_size

        # ── Build full class name list ────────────────────────────────────
        self.full_class_names = []
        for cls in classnames:
            if cls == "water":
                self.full_class_names.extend([f"{t} water" for t in self.types])
            else:
                self.full_class_names.append(cls)
        self.C = len(self.full_class_names)

        # ── Learnable context tokens (C, n_ctx, D) ────────────────────────
        self.ctx = nn.Parameter(torch.randn(self.C, n_ctx, ctx_dim) * 0.02)

        # ── Tokenize and cache embeddings ─────────────────────────────────
        prefix = " ".join(["X"] * n_ctx) + " "
        prompts = [prefix + name for name in self.full_class_names]

        tokenized = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            max_length=64,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            embedding = self.text_encoder.text_model.embeddings.token_embedding(
                tokenized["input_ids"]
            )  # (C, L, D)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])
        self.register_buffer("token_ids", tokenized["input_ids"])

        pad_token_id = self.tokenizer.pad_token_id
        attention_mask = (tokenized["input_ids"] != pad_token_id).long()
        self.register_buffer("attention_mask", attention_mask)

    def forward(self):
        # (C, L, D): [BOS] [ctx x n_ctx] [class tokens] [EOS] [PAD...]
        prompts = torch.cat([self.token_prefix, self.ctx, self.token_suffix], dim=1)

        # add position embeddings
        pos_ids = self.text_encoder.text_model.embeddings.position_ids
        pos_emb = self.text_encoder.text_model.embeddings.position_embedding(pos_ids)
        x = prompts + pos_emb  # (C, L, D)

        # run through encoder layers
        out = self.text_encoder.text_model.encoder(
            inputs_embeds=x,
        )
        x = out.last_hidden_state  # (C, L, D)
        x = self.text_encoder.text_model.final_layer_norm(x)

        # pool at EOS position for each class
        eos_token_id = self.tokenizer.eos_token_id
        text_feats = []
        for ci in range(self.C):
            eos_idx = (self.token_ids[ci] == eos_token_id).nonzero(as_tuple=True)[0][0].item()
            text_feats.append(x[ci, eos_idx])

        text_feats = torch.stack(text_feats, dim=0)   # (C, D)

        return text_feats

class Difficulty_CoOpPromptLearner(nn.Module):
    def __init__(
        self,
        classnames,
        clip_model,
        n_ctx=8,
        ctx_init=None,
        device="cuda"
    ):
        super().__init__()

        self.classnames = classnames
        self.types = ['normal'] + CHALLENGING_TYPES
        self.n_cls = len(classnames)
        self.n_ctx = n_ctx
        self.device = device

        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # index_to_replace = self.types.index("dark")
        # self.types[index_to_replace] = "nighttime" 
        self.full_class_names = []
        for cls in classnames:
            if cls == "water":
                self.full_class_names.extend([f"{t} water" for t in self.types])
            else:
                self.full_class_names.append(cls)
        self.C = len(self.full_class_names)
        self.ctx = nn.Parameter(torch.randn(self.C, n_ctx, ctx_dim) * 0.02)
        # --------------------------------------------------
        # Tokenize class names
        # --------------------------------------------------
        prompts = [f"{' '.join(['X'] * n_ctx)} {name}" for name in self.full_class_names]
        # prompts = [' '.join(['X'] * n_ctx) for _ in range(len(classnames))]
        self.tokenized = torch.cat([clip.tokenize(p) for p in prompts]).to(device)

        with torch.no_grad():
            embedding = clip_model.token_embedding(self.tokenized).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])   # SOS
        self.register_buffer("token_suffix", embedding[:, 1+n_ctx:, :])  # class + EOS
        self.register_buffer("tokenized_prompts", self.tokenized)

    def forward(self):
    
        prompts = torch.cat([self.token_prefix, self.ctx, self.token_suffix], dim=1)
        tokenized = self.tokenized_prompts
        return prompts, tokenized
    
class Difficulty_PromptLearner_siglip(nn.Module):

    def __init__(
        self,
        class_names,
        context_len: int = 8,
        model_name: str = "google/siglip2-base-patch16-224",
        device='cuda'
    ):
        super().__init__()

        self.class_names = class_names
        self.types = ['normal'] + CHALLENGING_TYPES
        self.context_len = context_len
        self.device = device

        # ── Load SigLIP2 text encoder ─────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_encoder = SiglipTextModel.from_pretrained(model_name).to(device)
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        ctx_dim = self.text_encoder.config.hidden_size  # e.g. 768

        # ── Build full class name list (same logic as before) ─────────────
        self.full_class_names = []
        for cls in class_names:
            if cls == "water":
                self.full_class_names.extend([f"{t} water" for t in self.types])
            else:
                self.full_class_names.append(cls)
        # idx = self.full_class_names.index("dark water")
        # self.full_class_names[idx] = 'water in dark scene'

        self.C = len(self.full_class_names)

        # ── Learnable context tokens (C, context_len, D) ──────────────────
        self.prompt_ctx = nn.Parameter(torch.randn(self.C, context_len, ctx_dim) * 0.02)

        # ── Pre-tokenize and cache embeddings ─────────────────────────────
        # use placeholder prefix so we know where to inject ctx later
        prefix = " ".join(["X"] * context_len) + " "
        texts = [prefix + name for name in self.full_class_names]

        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            max_length=64,
            truncation=True,
        ).to(device)
        attention_mask = (tokens["input_ids"] != 0).long()
        
        with torch.no_grad():
            token_emb = self.text_encoder.text_model.embeddings.token_embedding(
                tokens["input_ids"]
            )  # (C, L, D)

        self.register_buffer("token_ids", tokens["input_ids"])
        self.register_buffer("attention_mask", attention_mask)
        self.register_buffer("classname_emb", token_emb)

    def forward(self):
        outputs = []

        for ci in range(self.C):
            # base token embeddings for this class
            cls_emb = self.classname_emb[ci].clone()  # (L, D)
            # inject learnable ctx at positions 1..context_len (skip BOS)
            cls_emb[1:1 + self.context_len] = self.prompt_ctx[ci]
            # add position embeddings
            pos_ids = self.text_encoder.text_model.embeddings.position_ids
            pos_emb = self.text_encoder.text_model.embeddings.position_embedding(pos_ids)
            x = cls_emb + pos_emb  # (L, D)

            # run transformer
            attn_mask = self.attention_mask[ci].unsqueeze(0)  # (1, L)
            attn_mask_4d = attn_mask[:, None, None, :].float()        # (1, 1, 1, L)
            attn_mask_4d = attn_mask_4d.expand(1, 1, 64, 64)          # (1, 1, L, L)
            attn_mask_4d = (1.0 - attn_mask_4d) * -10000.0           # additive mask
            attn_mask_4d = attn_mask_4d.to(x.dtype)
            out = self.text_encoder.text_model.encoder(
                inputs_embeds=x,
                # attention_mask=attn_mask.bool(),
                attention_mask=attn_mask_4d,
            )
            x = out.last_hidden_state.squeeze(0)  # (L, D)
            x = self.text_encoder.text_model.final_layer_norm(x)

            # pool: same strategy as before
            # EOS position = last non-pad token
            eos_idx = self.attention_mask[ci].sum().item() - 1
            context_tokens = x[1:1 + self.context_len]
            context_pooled = context_tokens.mean(dim=0)

            cls_tokens = x[1 + self.context_len:eos_idx]
            cls_pooled = cls_tokens.mean(dim=0) if cls_tokens.shape[0] > 0 else context_pooled

            text_feat = 0.5 * context_pooled + 0.5 * cls_pooled
            # SigLIP2 has no text_projection — embeddings are already in the joint space
            outputs.append(text_feat)

        return torch.stack(outputs, dim=0)  # (C, D)

class Difficulty_PromptLearner(nn.Module):

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
        index_to_replace = self.full_class_names.index("dark water")
        self.full_class_names[index_to_replace] = 'water in dark scene'
        
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

    def forward(self):
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

            text_feat = 0.5 * context_pooled + 0.5 * cls_pooled
            text_feat = text_feat @ self.clip_model.text_projection

            outputs.append(text_feat)

        outputs = torch.stack(outputs, dim=0)

        return outputs
    
class Binary_Difficulty_PromptLearner(nn.Module):

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
        self.types = ['normal','challenging','challenging','challenging','challenging','challenging','challenging'] 
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

    def forward(self):
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

            text_feat = 0.5 * context_pooled + 0.5 * cls_pooled
            text_feat = text_feat @ self.clip_model.text_projection

            outputs.append(text_feat)

        outputs = torch.stack(outputs, dim=0)
        return outputs

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        pos = self.positional_embedding.type(self.dtype)
        pos = pos + 1e-4 * torch.randn_like(pos)
        x = prompts + pos
        # x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        # EOS token (last non-zero)
        eos_idx = tokenized_prompts.argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), eos_idx] @ self.text_projection

        return x

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
class MaskTextDecoder(nn.Module):
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
        # self.K_water = num_prompts_water
        # self.K_bg = num_prompts_bg
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
        #### fixed prompt ####
        # self.fixed_prompt_learner = Fixed_PromptLearner(clip_model=self.clip_model,clip_tokenizer=self.clip_tokenizer)
        ######################
        # 3. NEW: text prompt learner (CoOp)
        if not deploy:
            self.prompt_learner = PromptLearner(
                class_names=class_names,
                num_classes=num_classes,
                K=num_prompts_per_class,
                # K_water=num_prompts_water,
                # K_bg=num_prompts_bg,
                clip_model=self.clip_model,
                clip_tokenizer=self.clip_tokenizer)
        ### difficulty prompt learner#####
        # self.prompt_learner = Difficulty_PromptLearner(
        #     class_names=class_names,
        #     num_classes=num_classes,
        #     clip_model=self.clip_model,
        #     clip_tokenizer=self.clip_tokenizer)
        # self.prompt_learner = Difficulty_PromptLearner_siglip(class_names=class_names)
        # self.prompt_learner = Binary_Difficulty_PromptLearner(
        #     class_names=class_names,
        #     num_classes=num_classes,
        #     clip_model=self.clip_model,
        #     clip_tokenizer=self.clip_tokenizer)
        ######################
        # self.device = "cuda"
        # self.clip_model, _ = clip.load("ViT-B/16", device=self.device)
        # self.clip_model = self.clip_model.float()
        # self.clip_model.eval()
        # self.prompt_learner =  CoOpPromptLearner(
        #     classnames=class_names,
        #     clip_model=self.clip_model,
        #     K_per_cls=[num_prompts_per_class, num_prompts_per_class],
        #     n_ctx=8)
        # self.prompt_learner =  Difficulty_CoOpPromptLearner_siglip(classnames=class_names)
        # self.prompt_learner =  Difficulty_CoOpPromptLearner(
        #     classnames=class_names,
        #     clip_model=self.clip_model,
        #     K_per_cls=[num_prompts_per_class, num_prompts_per_class],
        #     n_ctx=8,
        #     n_diff_ctx=2)
        # self.text_encoder = TextEncoder(self.clip_model)
        ##### difficulty coop prompt learner #####
        # self.device = "cuda"
        # self.clip_model, _ = clip.load("ViT-B/16", device=self.device)
        # self.clip_model = self.clip_model.float()
        # self.clip_model.eval()
        # self.prompt_learner =  Difficulty_CoOpPromptLearner(
        #     classnames=class_names,
        #     clip_model=self.clip_model,
        #     n_ctx=8)
        # self.text_encoder = TextEncoder(self.clip_model)
        #################################
        # 4. NEW: MaskTextDecoder layers (instead of Mask2Former’s cross-attn stack)
        # self.layers = nn.ModuleList([
        #     CrossSelfLayer(d_model=hidden_dim, nhead=nheads)
        #     for _ in range(dec_layers)
        # ])
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
                "MaskTextDecoder was built with MODEL.MASK_FORMER.DEPLOY=True "
                "(text branch omitted) and cannot be trained. Set DEPLOY=False.")
        if is_training:
            B = mask_features.shape[0]
            # #### hard code the text embeddings for water and non-water ####
            # water_text_feat_path = "/data1/huantao/workspace/project/flood_seg/SegFormer/text_feat/mean_water_feat.pt"
            # # water_text_feat_path = '/data1/huantao/workspace/project/flood_seg/SegFormer/text_feat/mean_water_flood_feat.pt'
            # non_water_text_feat_path = "/data1/huantao/workspace/project/flood_seg/SegFormer/text_feat/non_water_feat.pt"
            # clip_water_text_embeddings = torch.load(water_text_feat_path, map_location='cuda',weights_only=True)
            # clip_non_water_text_embeddings = torch.load(non_water_text_feat_path, map_location='cuda',weights_only=True)
            # clip_text_embeddings = torch.stack([clip_non_water_text_embeddings, clip_water_text_embeddings], dim=0)
            # text_tokens = self.to_mask_space(clip_text_embeddings)  # (C*K, hidden_dim)
            # text_tokens = text_tokens.unsqueeze(1).repeat(1,B, 1)       # (B, T, hidden_dim)
            # T = text_tokens.shape[0]
            #### prompt learner ####
            clip_text_embeddings = self.prompt_learner()
            # clip_text_embeddings, tokenized_prompts = self.prompt_learner()
            # clip_text_embeddings, tokenized_prompts = self.prompt_learner(difficulty)
            # clip_text_embeddings = self.text_encoder(clip_text_embeddings, tokenized_prompts)
            # text_feat = text_feat.unsqueeze(0).expand(n, -1, -1)
            # text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            # clip_text_embeddings = self.fixed_prompt_learner(fixed_prompts)
            # clip_text_embeddings = self.fixed_prompt_learner(['The background of water. A photo of water.']).squeeze()
            text_tokens = self.to_mask_space(clip_text_embeddings) 
            text_tokens = text_tokens.unsqueeze(1).repeat(1, B, 1)
            T = text_tokens.shape[0]
        else:
            text_tokens = None
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
                K = self.K*2,
                # K = self.K,
                is_training=is_training
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed,
                # K = self.K_water + self.K_bg,
                K = self.K*2,
                # K = self.K,
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


# ---------------------------
# Mask-Text Contrastive Loss
# ---------------------------
class MaskTextContrastiveLoss(nn.Module):
    def __init__(self, temp: float = 0.07, reduction='mean', mix_neg: bool = True):
        """
        Args:
            temp: softmax temperature
            mix_neg: whether to use MixNeg (all other text tokens are negatives) or SeparateNeg (only other classes)
        """
        super().__init__()
        self.temp = temp
        self.reduction = reduction
        self.mix_neg = mix_neg

    def forward(self, mask_clip: torch.Tensor, text_clip: torch.Tensor, mask_to_gt_cls: torch.Tensor, K: int):
        """
        Args:
            mask_clip: (B, N, D) projected mask tokens in CLIP space
            text_clip: (B, C*K, D) projected text tokens in CLIP space
            mask_to_gt_cls: (B, N) ground-truth class index for each mask token (values in [0, C-1] or -1 for ignore)
            K: number of prompts per class
        Returns:
            loss scalar
        Notes:
            This computes, for each mask_i, the text token among K prompts of its gt class with max similarity as positive.
            Negatives are either all other text tokens (MixNeg) or only other classes' text tokens (SeparateNeg).
        """
        B, N, D = mask_clip.shape
        _, CK, _ = text_clip.shape
        C = CK // K
        device = mask_clip.device

        # normalize
        mask_norm = F.normalize(mask_clip, dim=-1)  # (B, N, D)
        text_norm = F.normalize(text_clip, dim=-1)  # (B, CK, D)

        # compute similarity: (B, N, CK)
        sim = torch.einsum('bnd,bkd->bnk', mask_norm, text_norm)  # (B, N, CK)

        # For each mask, find positive index among K prompts of its class: choose max over K
        # Build expected positive indices mask
        # mask_to_gt_cls: (B, N) with class idx in [0,C-1], -1 ignore
        losses = []
        for b in range(B):
            sim_b = sim[b]  # (N, CK)
            gt_b = mask_to_gt_cls[b]  # (N,)
            valid_mask = gt_b >= 0
            if valid_mask.sum() == 0:
                continue
            sim_valid = sim_b[valid_mask]  # (n_valid, CK)
            gt_valid = gt_b[valid_mask].long()  # (n_valid,)

            # for each gt, indices of its K prompts
            # compute per-sample positive index among CK: choose k with max similarity
            pos_indices = []
            for i, g in enumerate(gt_valid):
                start = g * K
                end = start + K
                k_slice = sim_valid[i, start:end]  # (K,)
                k_idx = torch.argmax(k_slice).item()
                pos_idx = start + k_idx
                pos_indices.append(pos_idx)
            pos_indices = torch.tensor(pos_indices, device=device, dtype=torch.long)  # (n_valid,)

            # compute logits for cross entropy: treat pos as target among CK
            logits = sim_valid / self.temp  # (n_valid, CK)
            target = pos_indices  # (n_valid,)
            loss_b = F.cross_entropy(logits, target, reduction='mean')
            losses.append(loss_b)
        if len(losses) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        loss = torch.stack(losses).mean()
        return loss


# ---------------------------
# Simple integration helper
# ---------------------------
def example_integration_step(mask_tokens, pixel_features, clip_text_embeddings, mask_gt_class_for_each_token, prompt_learner: PromptLearner,
                             decoder: MaskTextDecoder, contrastive_loss_module: MaskTextContrastiveLoss, K=3):
    """
    mask_tokens: (B, N, d_model)
    pixel_features: (B, S, d_model)
    clip_text_embeddings: (C*K, clip_text_dim) or (B, C*K, clip_text_dim)
    mask_gt_class_for_each_token: (B, N) integers in [0, C-1] or -1
    """
    # 1) project prompts -> text embeddings if using PromptLearner projection path
    # Here assume clip_text_embeddings is given; if not, get via prompt_learner.forward_projected()

    # 2) forward decoder
    updated_mask_tokens, masks_clip_list, text_clip_space = decoder(mask_tokens, pixel_features, text_tokens_proj=clip_text_embeddings)

    # 3) compute contrastive loss (use final layer masks or average)
    # choose last layer mask projection
    mask_clip_final = masks_clip_list[-1]  # (B, N, D_clipproj)
    # text_clip_space is (B, C*K, D_clipproj)
    loss_sim = contrastive_loss_module(mask_clip_final, text_clip_space, mask_gt_class_for_each_token, K)

    # 4) other mask losses (mask classification, dice, mask BCE) handled by segmentation head as usual

    return updated_mask_tokens, loss_sim
