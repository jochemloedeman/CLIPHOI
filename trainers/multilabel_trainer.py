from datetime import datetime

import torch
from tqdm import tqdm
from torch.utils import data, tensorboard


class ModelTrainer(object):

    def __init__(self, model: torch.nn.Module, device: str, optimizer: torch.optim.Optimizer,
                 training_loader: data.DataLoader, validation_loader: data.DataLoader,
                 loss_module: torch.nn.Module, writer: tensorboard.SummaryWriter, num_epochs: int) -> None:

        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.loss_module = loss_module
        self.writer = writer
        self.num_epochs = num_epochs

    def train_model(self) -> None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.model.train().to(self.device)
        best_val_loss = 1_000_000.

        # Training loop
        for epoch in tqdm(range(self.num_epochs)):
            print('\nEPOCH {}:'.format(epoch + 1))
            avg_batch_loss = self._train_one_epoch(epoch)

            self.model.eval()
            avg_validation_loss = self._validate_model()
            print('LOSS train {} valid {}'.format(avg_batch_loss, avg_validation_loss))

            self.writer.add_scalars('Training vs. Validation Loss',
                                    {'Training': avg_batch_loss, 'Validation': avg_validation_loss},
                                    epoch + 1)

            self.writer.flush()

            if avg_validation_loss < best_val_loss:
                best_val_loss = avg_validation_loss
                model_path = 'saved_models/model_{}_{}'.format(timestamp, epoch)
                torch.save(self.model.state_dict(), model_path)

    def _validate_model(self) -> float:
        running_val_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for i, val_data in enumerate(self.validation_loader):
                val_inputs, val_labels = val_data[0].to(self.device), val_data[1].to(self.device)
                val_outputs = self.model(val_inputs)
                val_loss = self.loss_module(val_outputs, val_labels)
                running_val_loss += val_loss

        avg_val_loss = running_val_loss / (i + 1)
        return avg_val_loss

    def _train_one_epoch(self, epoch_index) -> float:
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        self.model.train()
        for i, data in enumerate(self.training_loader):
            # Every data instance is an input + label pair
            inputs, labels = data[0].to(self.device), data[1].to(self.device)

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(inputs)

            # Compute the loss and its gradients
            loss = self.loss_module(outputs, labels.float())
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 200 == 199:
                last_loss = running_loss / 200  # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(self.training_loader) + i + 1
                self.writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss
