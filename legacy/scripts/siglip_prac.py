import torch
from transformers import SiglipTextModel, AutoTokenizer

model_name = "google/siglip2-base-patch16-224"

tokenizer = AutoTokenizer.from_pretrained(model_name)
text_encoder = SiglipTextModel.from_pretrained(model_name)
text_encoder.eval()

texts = ["a photo of a flooded street", "a photo of a dry road"]

tokens = tokenizer(
    texts,
    return_tensors="pt",
    padding="max_length",
    max_length=64,
    truncation=True,
)

with torch.no_grad():
    output = text_encoder(**tokens)

# pooler_output is the sentence-level embedding
embeddings = output.pooler_output  # (2, 768)
embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

# cosine similarity between the two texts
sim = embeddings[0] @ embeddings[1]
print(f"Embeddings shape: {embeddings.shape}")
print(f"Cosine similarity: {sim.item():.4f}")