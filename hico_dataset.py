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
    annot_format = ['nname', 'vname', 'vname_ing', 'syn', 'def', 'synset', 'add_def']

    def __init__(self, hico_root_dir, train=True):
        self.hico_root_dir = hico_root_dir
        self.train = train
        self.annotation_dict = scipy.io.loadmat(self.hico_root_dir / self.annot_file)
        self.hoi_classes = self.__convert_hoi_classes(self.annotation_dict['list_action'])

        if self.train:
            self.image_dir = self.hico_root_dir / self.train_image_dir
            self.image_filenames = self.annotation_dict['list_train']
            self.hoi_targets = torch.from_numpy(self.annotation_dict['anno_train'])
        else:
            self.image_dir = self.hico_root_dir / self.test_image_dir
            self.image_filenames = self.annotation_dict['list_test']
            self.hoi_targets = torch.from_numpy(self.annotation_dict['anno_test'])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image_filename = self.image_filenames[index].item().item()
        image = Image.open(Path(self.image_dir) / image_filename)
        target = self.hoi_targets[:, index]

        return image, target

    def __convert_hoi_classes(self, classes):
        hoi_classes = []
        for entry in classes:
            noun = entry.item()[0].item()
            verb = entry.item()[1].item()
            verb_ing = entry.item()[2].item()
            synonyms = entry.item()[3].tolist()
            definition = entry.item()[4].tolist()
            hoi_classes.append(HOI(noun, verb, verb_ing, synonyms, definition))
        return hoi_classes


if __name__ == '__main__':
    current_folder = Path(__file__).parent
    dataset = HICODataset(hico_root_dir=current_folder/'hico_20150920')
    print(len(dataset))
    print(dataset[2])
