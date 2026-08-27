"""Unused mask-text alignment losses. The active model uses masktext_alignment_loss_scene_type.

Removed from mask2former/maskformer_mta_clip_model_attribute.py during cleanup; kept for reference.
Not imported anywhere.
"""

# ---- MaskFormerMtaCLIP_attri.masktext_alignment_loss ----
    def masktext_alignment_loss(self, text_clip, mask_clip, mask_to_gt_cls, temperature=0.07):
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


# ---- MaskFormerMtaCLIP_attri.masktext_alignment_loss_mixneg ----
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


# ---- MaskFormerMtaCLIP_attri.masktext_alignment_loss_scene_type_iter ----
    def masktext_alignment_loss_scene_type_iter(self,
        text_clip,          # (C, D)  -> 10 prompts
        mask_clip,          # (B, Q, D)
        mask_to_gt_cls,     # (B, Q)  0=background 1=water
        difficulty,         # (B,) scene type: 0~8
        temperature=0.07):
        if has_event_storage():
            current_iter = get_event_storage().iter
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

            pos_indices = []
            for cls in gt_cls_valid:
                if cls == 0:   # background
                    pos_idx = 0
                else:          # water
                    if current_iter > 60:
                        pos_idx = difficulty[b] + 2
                    else:
                        pos_idx = 1
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


# ---- MaskFormerMtaCLIP_attri.masktext_alignment_loss_fixed_attribute_pool ----
    def masktext_alignment_loss_fixed_attribute_pool(self,
        text_clip,          # (C, D)  -> 10 prompts
        mask_clip,          # (B, Q, D)
        mask_to_gt_cls,     # (B, Q)  0=background 1=water
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

            pos_indices = []
            for cls in gt_cls_valid:
                if cls == 0:   # background
                    pos_idx = B
                else:          # water
                    pos_idx = b

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


# ---- MaskFormerMtaCLIP_attri.sample_prompt_index ----
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


