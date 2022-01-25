import torch
import torchmetrics
from torch.utils.data import DataLoader
from tqdm import tqdm


class MultiLabelModelEvaluator:
    def __init__(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                 metric: torchmetrics.Metric, device: str) -> None:
        """
        Class that is intended for evaluating a multilabel classifier on a PyTorch Dataset,
        with a given TorchMetrics metric.

        Args:
            model:
                A PyTorch nn Module
            dataloader:
                A PyTorch DataLoader
            metric:
                A TorchMetrics metric
            device:
                Set the device on which the evaluation is executed
        """
        self.device = device
        self.dataloader = dataloader
        self.metric = metric
        self.model = model
        self.final_metric = None

    def evaluate(self) -> float:
        """Iterates through a DataLoader that is created from the given Dataset.
        In each iteration, the images are passed through the CLIP image encoder. The resulting embeddings are then
        compared to the HOI class embeddings to obtain logits per image. A distribution over the HOI classes is
        computed using a softmax.
        """
        self.model.eval()
        with torch.no_grad():
            for pair in tqdm(self.dataloader):
                images, target = pair[0].to(self.device), pair[1].to(self.device)
                logits = self.model(images)
                probs = torch.sigmoid(logits)
                self.metric.update(probs, target.to(self.device))

        self.final_metric = self.metric.compute()
        return self.final_metric.item()
