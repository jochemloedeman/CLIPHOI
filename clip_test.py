import os
import clip
import torch
from pathlib import Path
# Load the model
from hico_dataset import HICODataset

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Download the dataset
hico = HICODataset(Path(__file__).parent / 'hico_20150920', train=False)

# Prepare the inputs
image, class_id = hico[3637]
image_input = preprocess(image).unsqueeze(0).to(device)
text_inputs = torch.cat([clip.tokenize(f"a person {hoi.verb_ing} {hoi.article} {hoi.noun}") for hoi in hico.hoi_classes]).to(device)
hois = [f"a person {hoi.verb_ing} {hoi.article} {hoi.noun}" for hoi in hico.hoi_classes]

# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(5)

# Print the result
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{hico.hoi_classes[index]:>16s}: {100 * value.item():.2f}%")