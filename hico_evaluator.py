import json
import clip
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from hico_dataset import HICODataset
from pathlib import Path

from hicomap import HICOmAP


class CLIPEvaluator(object):

    def __init__(self, dataset, metric, device, batch_size):
        self.device = device
        self.dataset = dataset
        self.metric = metric
        self.batch_size = batch_size
        self.model, self.preprocessor = clip.load('ViT-B/32', self.device)
        self.final_metric = None

    def evaluate(self):
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

        for images, target in tqdm(dataloader):
            text_inputs = torch.cat([clip.tokenize(f"a person {hoi.hoi_phrase}") for hoi in
                                     self.dataset.hoi_classes]).to(self.device)
            with torch.no_grad():
                logits_per_image, logits_per_text = self.model(images.to(self.device), text_inputs)
                probs = logits_per_image.softmax(dim=-1)
                self.metric.update(probs, target.to(self.device))

        self.final_metric = self.metric.compute()
