import torch, clip
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _ = clip.load("ViT-B/16", device=device)

color = [
    ", with a clear blue-green transparent color",         # normal
    ", with a brown turbid and opaque color",              # muddy
    ", with a bright overexposed washed-out color",        # glare
    ", with a dark and low-brightness shadowed color",     # shadowed
    ", with a grey rain-darkened color tone",              # rainy
]

texture = [
    ", with a smooth and gently rippled texture",          # normal
    ", with a thick sediment-laden turbid texture",        # muddy
    ", with a mirror-like smooth specular texture",        # reflective
    ", with a heavily disturbed raindrop-pitted texture",  # rainy
    ", with an irregular churned and foamy texture",       # turbulent
]

surface = [
    ", showing a calm and flat water surface",             # normal
    ", showing a mirror-like sky reflection on its surface",   # reflective
    ", showing no visible underwater structure on its surface", # muddy
    ", showing visible underwater substrate on its surface",    # transparent
    ", showing violent turbulent white patches on its surface", # foamy
]

boundary = [
    ", with a clearly defined water-land boundary",        # normal
    ", with a blurred and poorly defined water edge",      # blurry
    ", with an indistinct and muddy diffuse boundary",     # muddy
    ", with a faint and barely distinguishable water edge", # transparent
    ", with a chaotic and turbulence-disrupted boundary",  # foamy
]

# feat_list = []
# with torch.no_grad():
#     tokens_color = clip.tokenize(color).to(device)
#     feat_color = clip_model.encode_text(tokens_color)            # [K,512]
#     feat_color = feat_color / feat_color.norm(dim=-1, keepdim=True)          # normalize each
#     feat_list.append(feat_color)
    
#     tokens_texture = clip.tokenize(texture).to(device)
#     feat_texture = clip_model.encode_text(tokens_texture)            # [K,512]
#     feat_texture = feat_texture / feat_texture.norm(dim=-1, keepdim=True)
#     feat_list.append(feat_texture) 
    
#     tokens_surface = clip.tokenize(surface).to(device)
#     feat_surface = clip_model.encode_text(tokens_surface)            # [K,512]
#     feat_surface = feat_surface / feat_surface.norm(dim=-1, keepdim=True) 
#     feat_list.append(feat_surface)
    
#     tokens_boundary = clip.tokenize(boundary).to(device)
#     feat_boundary = clip_model.encode_text(tokens_boundary)            # [K,512]
#     feat_boundary = feat_boundary / feat_boundary.norm(dim=-1, keepdim=True) 
#     feat_list.append(feat_boundary)
    
# attribute_feat = torch.stack(feat_list)
# torch.save(attribute_feat.float().cpu(), "/data1/huantao/workspace/project/flood_seg/Mask2Former/attribute_text_feat/normalized_feat_4attri_5each.pt")

# bg = 'The background of water'
# with torch.no_grad():
#     tokens_bg = clip.tokenize([bg]).to(device)
#     feat_bg = clip_model.encode_text(tokens_bg)            # [1,512]
#     feat_bg = feat_bg / feat_bg.norm(dim=-1, keepdim=True)          # normalize each
# torch.save(feat_bg.float().cpu(), "/data1/huantao/workspace/project/flood_seg/Mask2Former/attribute_text_feat/normalized_feat_bg.pt")

# water = 'A photo of water'
# with torch.no_grad():
#     tokens_water = clip.tokenize([water]).to(device)
#     feat_water = clip_model.encode_text(tokens_water)            # [1,512]
#     feat_water = feat_water / feat_water.norm(dim=-1, keepdim=True)          # normalize each
# torch.save(feat_water.float().cpu(), "/data1/huantao/workspace/project/flood_seg/Mask2Former/attribute_text_feat/normalized_feat_water.pt")


data = [
    [  # color
        ", with a clear blue-green transparent color",
        ", with a brown turbid and opaque color",
        ", with a bright overexposed washed-out color",
        ", with a dark and low-brightness shadowed color",
        ", with a grey rain-darkened color tone"
    ],
    [  # texture
        ", with a smooth and gently rippled texture",
        ", with a thick sediment-laden turbid texture",
        ", with a mirror-like smooth specular texture",
        ", with a heavily disturbed raindrop-pitted texture",
        ", with an irregular churned and foamy texture"
    ],
    [  # surface
        ", showing a calm and flat water surface",
        ", showing a mirror-like sky reflection on its surface",
        ", showing no visible underwater structure on its surface",
        ", showing visible underwater substrate on its surface",
        ", showing violent turbulent white patches on its surface"
    ],
    [  # boundary
        ", with a clearly defined water-land boundary",
        ", with a blurred and poorly defined water edge",
        ", with an indistinct and muddy diffuse boundary",
        ", with a faint and barely distinguishable water edge",
        ", with a chaotic and turbulence-disrupted boundary"
    ]
]

with open("/data1/huantao/workspace/project/flood_seg/Mask2Former/attribute_text_feat/water_attribute_prompts.json", "w") as f:
    json.dump(data, f, indent=4)