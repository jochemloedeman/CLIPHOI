import pathlib
from typing import Tuple, List, Dict

import scipy.io
import torch

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from hoi import HOI


class HICODataset(Dataset):
    """
    Subclass of PyTorch Dataset that provides an interface to the HICO dataset
    for Human-Object Interaction Recognition.

    Args:
        hico_root_dir:
            Path to the root directory of the HICO dataset, as
            downloaded from http://www-personal.umich.edu/~ywchao/hico/
        transform:
            Torchvision transformations to be applied to the raw images
        train:
            Choose whether the training set should be used instead of the test set
        exclude_no_interaction:
            Exclude the trivial interactions (no interaction)
        """

    annot_file = 'anno.mat'
    plural_nouns = ('scissors', 'skis')
    no_interaction_verb = 'no_interaction'

    def __init__(self, hico_root_dir: pathlib.Path, transform: transforms.Compose = None,
                 train: bool = True, exclude_no_interaction: bool = False) -> None:

        self.hico_root_dir = hico_root_dir
        self.transform = transform if transform is not None else transforms.Compose([transforms.ToTensor()])
        self.train = train
        self.exclude_no_interaction = exclude_no_interaction

        self.subset = 'train' if self.train else 'test'
        self.image_dir = self.hico_root_dir / f'images/{self.subset}2015'
        self.annotation_dict = scipy.io.loadmat(str(self.hico_root_dir / self.annot_file))
        self.hoi_classes, self.interaction_class_indices = self._create_hoi_classes(
            self.annotation_dict['list_action'])

        image_filenames = self.annotation_dict[f'list_{self.subset}']
        hoi_targets = torch.from_numpy(self.annotation_dict[f'anno_{self.subset}'])[self.interaction_class_indices, :]
        annotated_image_indices = self._find_annotated_images(hoi_targets)

        self.image_filenames = image_filenames[annotated_image_indices.numpy()]
        self.hoi_targets = hoi_targets[:, annotated_image_indices]

    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        image_filename = self.image_filenames[index].item().item()
        image = Image.open(Path(self.image_dir) / image_filename)
        target = self.hoi_targets[:, index]

        return self.transform(image), target

    def _create_hoi_classes(self, classes) -> Tuple[List[HOI], torch.LongTensor]:
        """Converts the supplied list of HOIs in the HICO dataset to a list of HOI objects. It takes into account
        whether the 'no interaction' classes should be excluded.
        """
        hoi_classes = []
        interaction_indices = []
        for hoi_index, entry in enumerate(classes):
            noun = entry.item()[0].item()
            verb = entry.item()[1].item()
            verb_ing = entry.item()[2].item()
            synonyms = entry.item()[3].tolist()
            definition = entry.item()[4].tolist()
            noun_is_plural = noun in self.plural_nouns

            if verb != self.no_interaction_verb or not self.exclude_no_interaction:
                hoi_classes.append(HOI(noun, verb, verb_ing, synonyms, definition, noun_is_plural))
                interaction_indices.append(hoi_index)

        return hoi_classes, torch.LongTensor(interaction_indices)

    @staticmethod
    def _find_annotated_images(hoi_targets: torch.Tensor) -> torch.LongTensor:
        """Return the indices images for which at least one positive annotation is available"""
        return torch.LongTensor(torch.unique(torch.nonzero(hoi_targets == 1, as_tuple=True)[1]))

    def get_hoi_statistics(self) -> Dict[str, int]:
        """Calculate the frequency of each HOI in the selected subset (train or test)"""
        targets_without_nan = torch.nan_to_num(self.hoi_targets, nan=0.0)
        hoi_counts = torch.sum(torch.where(targets_without_nan == 1, 1, 0), dim=1).tolist()
        hoi_phrases = [hoi.hoi_phrase for hoi in self.hoi_classes]
        count_per_hoi = dict(zip(hoi_phrases, hoi_counts))
        sorted_count_per_hoi = {k: v for k, v in sorted(count_per_hoi.items(), key=lambda item: item[1], reverse=True)}
        return sorted_count_per_hoi
