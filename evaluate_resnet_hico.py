from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.hico_dataset import HICODataset, get_testing_transforms
from evaluators.model_evaluator import MultiLabelModelEvaluator
from metrics.hicomap import HICOmAP
from models.resnet_hico_classifier import HICOResNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'
hico_test_set = HICODataset(Path('data/hico_20150920'),
                            train=False,
                            transform=get_testing_transforms(),
                            exclude_no_interaction=True)

hico_test_loader = DataLoader(hico_test_set, batch_size=1, shuffle=False)
saved_models_folder = 'saved_models'
model = HICOResNet().to(device)
model.load_state_dict(torch.load(f"{saved_models_folder}/final_from_scratch_model.pt"))

map_metric = HICOmAP(hico_test_set)

evaluator = MultiLabelModelEvaluator(model, hico_test_loader, map_metric, device)
map = evaluator.evaluate()
print(map)
