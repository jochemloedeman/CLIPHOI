import clip
import torch
from pathlib import Path

from torchvision import transforms

from datasets.hico_dataset import HICODataset

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Download the dataset
hico = HICODataset(Path(__file__).parent / 'hico_20150920', train=False, transform=transforms.Compose([transforms.ToTensor()]))

# Prepare the inputs
image, target = hico[46]
for hoi in hico.hoi_classes:
    print(hoi.hoi_phrase)
positive_indices = hico.get_positive_indices(target)
image_input = preprocess(image).unsqueeze(0).to(device)
text_inputs = torch.cat([clip.tokenize(f"a person {hoi.hoi_phrase}") for hoi in hico.hoi_classes]).to(device)
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
    print(f"{hico.hoi_classes[index].hoi_phrase}: {100 * value.item():.2f}%")
print("\nGround truths:\n")
for index in positive_indices:
    print(hico.hoi_classes[index].hoi_phrase)