"""Unused prompt-learner variants. The active model uses Half_PromptLearner_type.

Removed from mask2former/modeling/transformer_decoder/mta_clip_modules_attribute.py during cleanup; kept for reference.
Not imported anywhere.
"""

# ---- Fixed_PromptLearner ----
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


# ---- Half_Fixed_PromptLearner ----
class Half_Fixed_PromptLearner(nn.Module):
    def __init__(
        self,
        clip_model=None,
        clip_tokenizer=None,
        attri_path = None,
        bg_path = None,
        device='cuda'
    ):
        super().__init__()
        
        self.device = device
        assert clip_model is not None, "clip_model required"
        assert clip_tokenizer is not None, "clip_tokenizer required"
        self.clip_model = clip_model
        self.clip_tokenizer = clip_tokenizer
        
        self.clip_attri_text_embeddings = torch.load(attri_path, map_location=self.device,weights_only=True)
        self.clip_bg_text_embeddings = torch.load(bg_path, map_location=self.device,weights_only=True)
        self.all_attri_prompts = json.load(open("/data1/huantao/workspace/project/flood_seg/Mask2Former/attribute_text_feat/water_attribute_prompts.json"))
        self.proj = torch.nn.Conv2d(256, 512, kernel_size=1)

    def forward(self, pixel_feat):
        pixel_feat = self.proj(pixel_feat)
        pixel_feat = F.normalize(pixel_feat, dim=1)
        pixel_global = pixel_feat.mean(dim=[2, 3]) 
        sim = torch.einsum('bc,akc->bak', pixel_global, self.clip_attri_text_embeddings)
        # sim = torch.einsum('bchw,akc->bakhw', pixel_feat, self.clip_attri_text_embeddings)
        best_sim, best_idx = sim.max(dim=2)
        bs, num_attr = best_idx.shape
        
        selected_prompts = []
        for b in range(bs):
            prompts_per_sample = []
            for a in range(num_attr):
                idx = best_idx[b, a].item()
                prompt = self.all_attri_prompts[a][idx]
                prompts_per_sample.append(prompt)

            # ---- Step 2: combine with fixed prefix ----
            full_prompt = "A photo of water" + "".join(prompts_per_sample)
            selected_prompts.append(full_prompt)

        # ---- Step 3: tokenize ----
        tokenized = self.clip_tokenizer(selected_prompts).to(self.device)
        # ---- Step 4: encode ----
        text_features = self.clip_model.encode_text(tokenized)
        text_features = F.normalize(text_features, dim=-1)
        bg_feat = self.clip_bg_text_embeddings
        output = torch.cat([text_features, bg_feat], dim=0)

        return output


# ---- Half_CoOpPromptLearner ----
class Half_CoOpPromptLearner(nn.Module):
    def __init__(
        self,
        classnames,
        clip_model,
        n_ctx=8,
        attri_path = None,
        bg_path = None,
        ctx_init=None,
        device="cuda",
    ):
        super().__init__()
        
        self.classnames = classnames
        # self.classnames[0] = "background of water"
        self.n_cls = len(classnames)
        self.n_ctx = n_ctx
        self.device = device

        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.clip_model = clip_model
        self.max_len = 77

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            tokens = clip.tokenize(ctx_init).to(device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokens).type(dtype)
            init_ctx = embedding[0, 1:1+n_ctx, :]
            ctx = init_ctx.unsqueeze(0).repeat(K, 1, 1)
        else:
            # ctx = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            ctx = torch.empty(self.n_cls, n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx, std=0.02)

            self.ctx = nn.Parameter(ctx)

        # prompts = [f"{' '.join(['X'] * n_ctx)} {name}" for name in classnames]
        # # prompts = [' '.join(['X'] * n_ctx) for _ in range(len(classnames))]
        # self.tokenized = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        
        self.clip_attri_text_embeddings = torch.load(attri_path, map_location=self.device,weights_only=True)
        self.clip_bg_text_embeddings = torch.load(bg_path, map_location=self.device,weights_only=True)
        self.all_attri_prompts = json.load(open("/data1/huantao/workspace/project/flood_seg/Mask2Former/attribute_text_feat/water_attribute_prompts.json"))
        self.proj = torch.nn.Conv2d(256, 512, kernel_size=1)

    def forward(self, pixel_feat):
        pixel_feat = self.proj(pixel_feat)
        pixel_feat = F.normalize(pixel_feat, dim=1)
        pixel_global = pixel_feat.mean(dim=[2, 3]) 
        #### each attribute selects one ###
        # sim = torch.einsum('bc,akc->bak', pixel_global, self.clip_attri_text_embeddings)
        # # sim = torch.einsum('bchw,akc->bakhw', pixel_feat, self.clip_attri_text_embeddings)
        # best_sim, best_idx = sim.max(dim=2)
        # bs, num_attr = best_idx.shape
        # fg_prompts = []
        # for b in range(bs):
        #     attr_list = []
        #     for a in range(num_attr):
        #         idx = best_idx[b, a].item()
        #         attr_list.append(self.all_attri_prompts[a][idx])

        #     full_prompt = ' '.join(['X'] * self.n_ctx) + " water" + "".join(attr_list)
        #     fg_prompts.append(full_prompt)
        #### select two in total #####
        _,num_prompts_per_attr,C = self.clip_attri_text_embeddings.shape
        text_embeddings_all = self.clip_attri_text_embeddings.view(-1,C)
        sim = torch.einsum('bc,nc->bn', pixel_global, text_embeddings_all)
        top2_sim, top2_idx = sim.topk(2, dim=1)  # (bs, 2)
        bs = sim.shape[0]
        fg_prompts = []
        for b in range(bs):
            attr_list = []
            for k in range(2):
                flat_idx = top2_idx[b, k].item()
                attr_idx = flat_idx // num_prompts_per_attr
                prompt_idx = flat_idx % num_prompts_per_attr
                attr_list.append(self.all_attri_prompts[attr_idx][prompt_idx])
            full_prompt = ' '.join(['X'] * self.n_ctx) + " water " + " ".join(attr_list)
            fg_prompts.append(full_prompt)

        # bg_prompts = ["The background of water"]
        ### bg is not fixed ###
        bg_prompts = [' '.join(['X'] * self.n_ctx) + ", the background of water"]
        ###########
        fg_tokenized = clip.tokenize(fg_prompts).to(self.device)
        bg_tokenized = clip.tokenize(bg_prompts).to(self.device)

        with torch.no_grad():
            fg_embedding = self.clip_model.token_embedding(fg_tokenized).type(self.clip_model.dtype)
            bg_embedding = self.clip_model.token_embedding(bg_tokenized).type(self.clip_model.dtype)

        # --- FG: replace placeholder tokens with learnable ctx ---
        fg_prefix = fg_embedding[:, :1, :]           # (bs, 1, dim)  SOS
        fg_suffix = fg_embedding[:, 1+self.n_ctx:, :] # (bs, L, dim)  class tokens + EOS
        # ctx = self.ctx.unsqueeze(0).expand(bs, -1, -1) # (bs, n_ctx, dim)
        ctx_fg = self.ctx[0].unsqueeze(0).expand(bs, -1, -1) 
        # fg_prompts_emb = torch.cat([fg_prefix, ctx, fg_suffix], dim=1)  # (bs, 77, dim)
        fg_prompts_emb = torch.cat([fg_prefix, ctx_fg, fg_suffix], dim=1)

        # --- BG: use the fixed embedding as-is, no ctx injection ---
        # bg_prompts_emb = bg_embedding  # (1, 77, dim)
        bg_prefix = bg_embedding[:, :1, :]              # (1, 1, dim)
        bg_suffix = bg_embedding[:, 1+self.n_ctx:, :]   # (1, L, dim)
        ctx_bg = self.ctx[1].unsqueeze(0)   
        bg_prompts_emb = torch.cat([bg_prefix, ctx_bg, bg_suffix], dim=1)           

        # --- Combine: [fg_0, fg_1, ..., fg_bs-1, bg] ---
        prompts = torch.cat([fg_prompts_emb, bg_prompts_emb], dim=0)  # (bs+1, 77, dim)
        tokenized = torch.cat([fg_tokenized, bg_tokenized], dim=0)     # (bs+1, 77)

        return prompts, tokenized


# ---- Half_CoOpPromptLearner_type ----
class Half_CoOpPromptLearner_type(nn.Module):
    def __init__(
        self,
        classnames,
        clip_model,
        n_ctx=8,
        ctx_init=None,
        device="cuda",
    ):
        super().__init__()
        
        self.classnames = classnames
        self.n_cls = len(classnames)
        self.n_ctx = n_ctx
        self.device = device

        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.clip_model = clip_model
        self.max_len = 77
        
        self.types = ['normal'] + CHALLENGING_TYPES
        self.types_with_water = [t + ' water' for t in self.types]
        type_tokens = clip.tokenize(self.types_with_water).to(device)
        with torch.no_grad():
            self.embedding_type = clip_model.encode_text(type_tokens).type(dtype)
        
        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            tokens = clip.tokenize(ctx_init).to(device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokens).type(dtype)
            init_ctx = embedding[0, 1:1+n_ctx, :]
            ctx = init_ctx.unsqueeze(0).repeat(K, 1, 1)
        else:
            # ctx = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            ctx = torch.empty(self.n_cls, n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx, std=0.02)

            self.ctx = nn.Parameter(ctx)
        self.proj = torch.nn.Conv2d(256, 512, kernel_size=1)

    def forward(self, pixel_feat):
        pixel_feat = self.proj(pixel_feat)
        pixel_feat = F.normalize(pixel_feat, dim=1)
        pixel_global = pixel_feat.mean(dim=[2, 3]) 
        #### select one #####
        sim = torch.einsum('bc,nc->bn', pixel_global, self.embedding_type)
        bs = sim.shape[0]
        best_sim, best_idx = sim.max(dim=1)
        fg_prompts = []
        for b in range(bs):
            idx = best_idx[b].item()
            full_prompt = ' '.join(['X'] * self.n_ctx) + " " + self.types_with_water[idx]
            fg_prompts.append(full_prompt)

        # bg_prompts = ["The background of water"]
        ### bg is not fixed ###
        bg_prompts = [' '.join(['X'] * self.n_ctx) + ", the background of water"]
        ###########
        fg_tokenized = clip.tokenize(fg_prompts).to(self.device)
        bg_tokenized = clip.tokenize(bg_prompts).to(self.device)

        with torch.no_grad():
            fg_embedding = self.clip_model.token_embedding(fg_tokenized).type(self.clip_model.dtype)
            bg_embedding = self.clip_model.token_embedding(bg_tokenized).type(self.clip_model.dtype)

        # --- FG: replace placeholder tokens with learnable ctx ---
        fg_prefix = fg_embedding[:, :1, :]           # (bs, 1, dim)  SOS
        fg_suffix = fg_embedding[:, 1+self.n_ctx:, :] # (bs, L, dim)  class tokens + EOS
        # ctx = self.ctx.unsqueeze(0).expand(bs, -1, -1) # (bs, n_ctx, dim)
        ctx_fg = self.ctx[0].unsqueeze(0).expand(bs, -1, -1) 
        # fg_prompts_emb = torch.cat([fg_prefix, ctx, fg_suffix], dim=1)  # (bs, 77, dim)
        fg_prompts_emb = torch.cat([fg_prefix, ctx_fg, fg_suffix], dim=1)

        # --- BG: use the fixed embedding as-is, no ctx injection ---
        # bg_prompts_emb = bg_embedding  # (1, 77, dim)
        bg_prefix = bg_embedding[:, :1, :]              # (1, 1, dim)
        bg_suffix = bg_embedding[:, 1+self.n_ctx:, :]   # (1, L, dim)
        ctx_bg = self.ctx[1].unsqueeze(0)   
        bg_prompts_emb = torch.cat([bg_prefix, ctx_bg, bg_suffix], dim=1)           

        # --- Combine: [fg_0, fg_1, ..., fg_bs-1, bg] ---
        prompts = torch.cat([fg_prompts_emb, bg_prompts_emb], dim=0)  # (bs+1, 77, dim)
        tokenized = torch.cat([fg_tokenized, bg_tokenized], dim=0)     # (bs+1, 77)

        return prompts, tokenized


# ---- Half_PromptLearner_type_iteration ----
class Half_PromptLearner_type_iteration(nn.Module):
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
                self.full_class_names.append(cls)
                self.full_class_names.extend([f"{t} water" for t in self.types])
            else:
                self.full_class_names.append(cls)
        # index_to_replace = self.full_class_names.index("dark water")
        # self.full_class_names[index_to_replace] = 'water in dark scene'
        tokens = self.clip_tokenizer(self.full_class_names[2:]).cuda()  
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
        pixel_feat = F.normalize(pixel_feat, dim=1)
        pixel_global = pixel_feat.mean(dim=[2, 3]) 
        #### select one #####
        sim = torch.einsum('bc,nc->bn', pixel_global, self.water_type_emb)
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


# ---- PromptLearner ----
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


# ---- CoOpPromptLearner ----
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


# ---- TextEncoder ----
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


