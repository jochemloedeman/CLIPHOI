import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from hico_dataset import HICODataset
from pathlib import Path

from hicomap import HICOmAP


class HICOEvaluator(object):

    def __init__(self, model, device, image_prepper=None, text_prepper=None,
                 train_set=False, known_object=False, exclude_no_interaction=False):
        self.model = model
        self.image_prepper = image_prepper
        self.text_prepper = text_prepper
        self.device = device
        self.train_set = train_set
        self.known_object = known_object
        self.exclude_no_interaction = exclude_no_interaction
        self.hico_dataset = HICODataset(Path(__file__).parent / 'hico_20150920',
                                        train=self.train_set,
                                        transform=self.image_prepper,
                                        exclude_no_interaction=self.exclude_no_interaction)

        self.hico_map = HICOmAP(hoi_classes=self.hico_dataset.hoi_classes,
                                known_object_mode=self.known_object,
                                exclude_no_interaction=self.exclude_no_interaction).to(self.device)
        self.final_map = None

    def evaluate(self):
        hico_loader = DataLoader(self.hico_dataset, batch_size=512, shuffle=False)

        for images, target in tqdm(hico_loader):
            text_inputs = torch.cat([self.text_prepper(f"a person {hoi.hoi_phrase}") for hoi in
                                     self.hico_dataset.hoi_classes]).to(
                self.device)
            with torch.no_grad():
                logits_per_image, logits_per_text = self.model(images.to(self.device), text_inputs)
                probs = logits_per_image.softmax(dim=-1)
                self.hico_map.update(probs, target.to(self.device))

        self.final_map = self.hico_map.compute()

    def export_aps_per_hoi(self):
        with open('ap_per_hoi.json', 'w') as f:
            json.dump(self.hico_map.get_ap_per_hoi(), f, indent=2)
