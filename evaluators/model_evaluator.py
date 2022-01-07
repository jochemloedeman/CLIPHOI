import torch
import torchmetrics
from torch.utils.data import DataLoader
from tqdm import tqdm


class MultiLabelModelEvaluator(object):
    def __init__(self, model: torch.nn.Module, dataset: torch.utils.data.Dataset,
                 metric: torchmetrics.Metric, device: str, batch_size: int) -> None:
        """
        Class that is intended for evaluating a multilabel classifier on a PyTorch Dataset,
        with a given TorchMetrics metric.

        Args:
            model:
                A PyTorch nn Module
            dataset:
                A PyTorch Dataset
            metric:
                A TorchMetrics metric
            device:
                Set the device on which the evaluation is executed
            batch_size:
                Set the batch size for the evaluation procedure
        """
        self.device = device
        self.dataset = dataset
        self.metric = metric
        self.batch_size = batch_size
        self.model = model
        self.final_metric = None

    def evaluate(self) -> None:
        """Iterates through a DataLoader that is created from the given Dataset.
        In each iteration, the images are passed through the CLIP image encoder. The resulting embeddings are then
        compared to the HOI class embeddings to obtain logits per image. A distribution over the HOI classes is
        computed using a softmax.
        """
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            for images, target in tqdm(dataloader):
                logits = self.model(images)
                probs = torch.sigmoid(logits)
                self.metric.update(probs, target.to(self.device))

        self.final_metric = self.metric.compute()
