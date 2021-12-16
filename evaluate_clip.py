import clip
import torch

from hico_evaluator import HICOEvaluator

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)
evaluator = HICOEvaluator(model, device, preprocess, clip.tokenize,
                          known_object=False, exclude_no_interaction=True)
evaluator.evaluate()
evaluator.export_aps_per_hoi()
print(evaluator.final_map.item())
