import argparse

import torchvision.models as models
from torch.nn import BCEWithLogitsLoss
from pathlib import Path

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from datasets.hico_dataset import HICODataset, get_training_transforms
from models.resnet_hico_classifier import HICOResNet
from trainers.multilabel_trainer import ModelTrainer

parser = argparse.ArgumentParser()

parser.add_argument('--ko', default=False, type=bool,
                    help='Evaluate in known object mode')
parser.add_argument('--exclude_no_interaction', default=True, type=bool,
                    help='Exclude no interaction classes')
parser.add_argument('--train', default=False, type=bool,
                    help='Use the train set for evaluation')
parser.add_argument('--batch_size', default=16, type=int,
                    help='Batch size for the evaluation')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='Batch size for the evaluation')

args = parser.parse_args()

model = HICOResNet()

hico_training_set = HICODataset(Path(__file__).parent / 'data' / 'hico_20150920',
                                train=True,
                                transform=get_training_transforms(),
                                exclude_no_interaction=args.exclude_no_interaction)
hico_test_set = HICODataset(Path(__file__).parent / 'data' / 'hico_20150920',
                            train=False,
                            transform=get_training_transforms(),
                            exclude_no_interaction=args.exclude_no_interaction)

device = 'cuda'
optimizer = Adam(model.parameters(), lr=args.lr)
training_loader = DataLoader(hico_training_set, batch_size=args.batch_size)
validation_loader = DataLoader(hico_test_set, batch_size=args.batch_size)
loss_module = BCEWithLogitsLoss()
writer = SummaryWriter()
num_epochs = 4

trainer = ModelTrainer(model=model, device=device, optimizer=optimizer, training_loader=training_loader,
                       validation_loader=validation_loader, loss_module=loss_module, writer=writer,
                       num_epochs=num_epochs)

trainer.train_model()
