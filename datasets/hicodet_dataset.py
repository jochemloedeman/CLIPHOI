import pathlib
from pathlib import Path
from typing import Tuple, List, Union

import scipy.io
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
from hoi.hoi import HOI


class HICODETDataset(Dataset):
    annot_file = 'anno.mat'
    annot_bbox_file = 'anno_bbox.mat'
    plural_nouns = ('scissors', 'skis')
    no_interaction_verb = 'no_interaction'

    def __init__(self, hico_root_dir: pathlib.Path, transform: transforms.Compose = None,
                 train: bool = True, exclude_no_interaction: bool = False,
                 save_predictions: bool = False) -> None:
        self.hico_root_dir = hico_root_dir
        self.transform = transform if transform is not None else transforms.Compose([])
        self.train = train
        self.exclude_no_interaction = exclude_no_interaction
        self.save_predictions = save_predictions

        self.subset = 'train' if self.train else 'test'
        self.image_dir = self.hico_root_dir / f'images/{self.subset}2015'
        self.annotation_dict = scipy.io.loadmat(str(self.hico_root_dir / self.annot_file))
        self.annotation_bbox_dict = scipy.io.loadmat(str(self.hico_root_dir / self.annot_bbox_file))
        self.hoi_classes, self.interaction_class_indices = self._create_hoi_classes(
            self.annotation_dict['list_action'])
        self.bbox_annotations = self._process_bbox_annots()

    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, index) -> Union[Tuple[Image.Image, torch.Tensor, torch.Tensor],
                                          Tuple[torch.Tensor, torch.Tensor]]:

        image_filename = self.image_filenames[index].item().item()
        image = Image.open(Path(self.image_dir) / image_filename).convert('RGB')
        target = self.hoi_targets[:, index]

        if self.save_predictions:
            return image, self.transform(image), target
        else:
            return self.transform(image), target

    def _process_bbox_annots(self):
        annotations = self.annotation_bbox_dict[f"bbox_{self.subset}"].squeeze().tolist()
        for annot in annotations:
            annot = annot[2][0].tolist()
            for sub_annot in annot:
                hoi_index = sub_annot[0]
                
                print()
            print()
        print()

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


if __name__ == '__main__':
    hicodet_path = Path(__file__).cwd().parent / 'data/hico_20160224_det'
    hicodet_dataset = HICODETDataset(hicodet_path, train=False)
    print()
