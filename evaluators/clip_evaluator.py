import clip
import torch
import torchmetrics
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import Dataset


class CLIPEvaluator(object):
    def __init__(self, dataset: Dataset, metric: torchmetrics.Metric, device: str, batch_size: int) -> None:
        """
        Class that is intended for evaluating OpenAI's CLIP (https://github.com/openai/CLIP) on a PyTorch Dataset,
        with a given TorchMetrics metric

        Args:
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
        self.model, self.preprocessor = clip.load('ViT-B/32', self.device)
        self.final_metric = None

    def evaluate(self) -> None:
        """Iterates through a DataLoader that is created from the given Dataset.
        In each iteration, the images are passed through the CLIP image encoder. The resulting embeddings are then
        compared to the HOI class embeddings to obtain logits per image. A distribution over the HOI classes is
        computed using a softmax.
        """
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

        text_inputs = torch.cat([clip.tokenize(f"a photo of a person {hoi.hoi_phrase}") for hoi in
                                 self.dataset.hoi_classes]).to(self.device)

        with torch.no_grad():
            for images, target in tqdm(dataloader):
                logits_per_image, logits_per_text = self.model(images.to(self.device), text_inputs)
                probs = logits_per_image.softmax(dim=-1)
                self.metric.update(probs, target.to(self.device))

        self.final_metric = self.metric.compute()

    @staticmethod
    def _center_logits(logits):
        mean_logit = torch.mean(logits, dim=1, keepdim=True)
        centered_logits = logits - mean_logit
        return centered_logits

