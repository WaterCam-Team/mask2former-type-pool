"""
Strip the training-only text branch from an MTA-CLIP checkpoint.

At inference the decoder runs with is_training=False, which bypasses the prompt
learner, CLIP, to_mask_space and mask_to_clip entirely (text_tokens=None, T=0).
Those modules therefore contribute nothing to the prediction and can be removed
from the checkpoint without changing a single output pixel.

The frozen CLIP weights are stored *twice* (once under the decoder, once under
the prompt learner, since they are the same module object referenced from two
places), which is why MTA-CLIP checkpoints are ~1.4GB versus ~44M real params.

Pair the stripped checkpoint with a config that sets
MODEL.MASK_FORMER.DEPLOY: True so the text modules are never built.

Usage:
    python strip_text_branch.py <in.pth> <out.pth>
"""
import argparse
import os
from collections import OrderedDict

import torch

# Substrings identifying training-only parameters.
TEXT_BRANCH_KEYS = (
    ".clip_model.",       # frozen CLIP (stored twice)
    ".prompt_learner.",   # learned prompt context + its CLIP copy
    ".to_mask_space.",    # text_embed -> hidden_dim, used only in the training branch
    ".mask_to_clip.",     # hidden_dim -> clip_dim, used only for the contrastive loss
)


def is_text_branch(key):
    # Leading "." so we match module boundaries rather than bare substrings.
    return any(pat in "." + key for pat in TEXT_BRANCH_KEYS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="input checkpoint (.pth)")
    ap.add_argument("dst", help="output checkpoint (.pth)")
    ap.add_argument("--keep-trainer-state", action="store_true",
                    help="keep optimizer/AMP/iteration state so the result can still "
                         "resume training (default: drop it, inference only)")
    args = ap.parse_args()

    ckpt = torch.load(args.src, map_location="cpu")
    # detectron2 checkpoints wrap weights under "model"; raw state_dicts do not.
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    kept, dropped = OrderedDict(), []
    for k, v in sd.items():
        (dropped.append(k) if is_text_branch(k) else kept.update({k: v}))

    # CRITICAL: state_dict carries a `_metadata` attribute holding each module's
    # `version`. MaskFormerHead._load_from_state_dict treats a missing version as
    # "old format" and rewrites every non-predictor sem_seg_head key to
    # sem_seg_head.pixel_decoder.*, so the pixel decoder silently loads nothing
    # and the model outputs garbage. Entries for dropped modules are never
    # consulted (lookup is per existing module), so carry the whole thing over.
    meta = getattr(sd, "_metadata", None)
    if meta is not None:
        kept._metadata = meta
    else:
        print("WARNING: source state_dict has no _metadata; "
              "the result may trigger legacy key conversion on load")

    def count(d):
        vals = d.values() if isinstance(d, dict) else (sd[k] for k in d)
        return sum(v.numel() for v in vals if hasattr(v, "numel"))

    n_keep, n_drop = count(kept), count(dropped)
    print(f"kept    {len(kept):5d} tensors  {n_keep/1e6:8.1f}M params")
    print(f"dropped {len(dropped):5d} tensors  {n_drop/1e6:8.1f}M params")

    # Preserve the detectron2 wrapper. "trainer" holds optimizer/AMP state, which
    # is large and useless for inference, so drop it unless asked otherwise.
    if isinstance(ckpt, dict) and "model" in ckpt:
        skip = {"model"} if args.keep_trainer_state else {"model", "trainer", "iteration"}
        out = {k: v for k, v in ckpt.items() if k not in skip}
        out["model"] = kept
        for k in sorted(set(ckpt) - set(out) - {"model"}):
            print(f"dropped top-level '{k}' (training state)")
    else:
        out = kept

    torch.save(out, args.dst)
    src_mb = os.path.getsize(args.src) / 1e6
    dst_mb = os.path.getsize(args.dst) / 1e6
    print(f"\n{args.src}  {src_mb:.0f} MB\n{args.dst}  {dst_mb:.0f} MB"
          f"   ({100 * (1 - dst_mb / src_mb):.0f}% smaller)")


if __name__ == "__main__":
    main()
