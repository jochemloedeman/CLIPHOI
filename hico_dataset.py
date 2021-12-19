import scipy.io
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

from hoi import HOI


class HICODataset(Dataset):
    train_image_dir = 'images/train2015'
    test_image_dir = 'images/test2015'
    annot_file = 'anno.mat'
    plural_nouns = ('scissors', 'skis')
    no_interaction_verb = 'no_interaction'

    def __init__(self, hico_root_dir, transform=None, train=True, exclude_no_interaction=False):
        self.hico_root_dir = hico_root_dir
        self.train = train
        self.transform = transform
        self.exclude_no_interaction = exclude_no_interaction
        self.annotation_dict = scipy.io.loadmat(self.hico_root_dir / self.annot_file)
        self.hoi_classes, self.interaction_class_indices = self.__convert_hoi_classes(
            self.annotation_dict['list_action'])

        if self.train:
            self.image_dir = self.hico_root_dir / self.train_image_dir
            image_filenames = self.annotation_dict['list_train']
            hoi_targets = torch.from_numpy(self.annotation_dict['anno_train'])[self.interaction_class_indices, :]
        else:
            self.image_dir = self.hico_root_dir / self.test_image_dir
            image_filenames = self.annotation_dict['list_test']
            hoi_targets = torch.from_numpy(self.annotation_dict['anno_test'])[self.interaction_class_indices, :]

        annotated_image_indices = self.__find_annotated_images(hoi_targets)
        self.image_filenames = image_filenames[annotated_image_indices.numpy()]
        self.hoi_targets = hoi_targets[:, annotated_image_indices]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image_filename = self.image_filenames[index].item().item()
        image = Image.open(Path(self.image_dir) / image_filename)
        target = self.hoi_targets[:, index]
        image = self.transform(image) if self.transform is not None else image
        return image, target

    # def preprocess

    def __convert_hoi_classes(self, classes):
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
    def __find_annotated_images(hoi_targets):
        return torch.LongTensor(torch.unique(torch.nonzero(hoi_targets == 1, as_tuple=True)[1]))

    def get_hoi_statistics(self):
        targets_without_nan = torch.nan_to_num(self.hoi_targets, nan=0.0)
        hoi_counts = torch.sum(torch.where(targets_without_nan == 1, 1, 0), dim=1).tolist()
        hoi_phrases = [hoi.hoi_phrase for hoi in self.hoi_classes]
        count_per_hoi = dict(zip(hoi_phrases, hoi_counts))
        sorted_count_per_hoi = {k: v for k, v in sorted(count_per_hoi.items(), key=lambda item: item[1], reverse=True)}
        return sorted_count_per_hoi

if __name__ == '__main__':
    current_folder = Path(__file__).parent
    dataset = HICODataset(hico_root_dir=current_folder / 'hico_20150920')
    print(len(dataset))
    print(dataset[2])
