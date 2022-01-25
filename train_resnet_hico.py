from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.utils.data import DataLoader

from datasets.hico_dataset import HICODataset, get_training_transforms, get_testing_transforms
from models.resnet_hico_classifier import HICOResNet
from trainers.multilabel_trainer import ModelTrainer

model = HICOResNet(pretrained=True)

hico_training_set = HICODataset(Path('data/hico_20150920'),
                                train=True,
                                transform=get_training_transforms(),
                                exclude_no_interaction=True)

hico_test_set = HICODataset(Path('data/hico_20150920'),
                            train=False,
                            transform=get_testing_transforms(),
                            exclude_no_interaction=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

optimizer = Adam(model.model.fc.parameters())

training_loader = DataLoader(hico_training_set, 32)
test_loader = DataLoader(hico_test_set, 1)

loss_module = CrossEntropyLoss()

writer = SummaryWriter()

num_epochs = 6

experiment_name = "from_scratch"

trainer = ModelTrainer(model=model, device=device, optimizer=optimizer,
                       training_loader=training_loader, loss_module=loss_module,
                       writer=writer, num_epochs=num_epochs,
                       experiment_name=experiment_name, validation_loader=test_loader)

trainer.train_model()
