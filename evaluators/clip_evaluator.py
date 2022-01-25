import math
from time import time

import clip
import torch
import torchmetrics
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, FiveCrop, Lambda
from tqdm import tqdm
from torch.utils.data import Dataset


class CLIPEvaluator(object):
    def __init__(self, dataset: Dataset, metric: torchmetrics.Metric,
                 backbone: str, device: str, batch_size: int, ir_threshold: float) -> None:
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
        self.model, self.preprocessor = clip.load(backbone, self.device)
        self.dataset = dataset
        self.metric = metric
        self.batch_size = batch_size
        self.ir_threshold = ir_threshold
        self.final_metric = None

    def evaluate(self) -> None:
        """Iterates through a DataLoader that is created from the given Dataset.
        In each iteration, the images are passed through the CLIP image encoder. The resulting embeddings are then
        compared to the HOI class embeddings to obtain logits per image. A distribution over the HOI classes is
        computed using a softmax.
        """
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

        text_inputs = torch.cat([clip.tokenize(f"a person {hoi.hoi_phrase}") for hoi in
                                 self.dataset.hoi_classes]).to(self.device)

        with torch.no_grad():
            for images, target in tqdm(dataloader):
                # logits_per_image, logits_per_text = self.model.forward(images.to(self.device), text_inputs)
                # probs = logits_per_image.softmax(dim=-1)

                image_features = self.model.encode_image(images.to(self.device))
                text_features = self.model.encode_text(text_inputs)

                # normalized features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                logit_scale = self.model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                probs = torch.softmax(logits_per_image, dim=-1)
                # probs = to_probabilities_by_distance(logits_per_image, max_offset=0.03)
                # print(end_time - start_time)
                self.metric.update(probs, target.to(self.device))

        self.final_metric = self.metric.compute()

    @torch.no_grad()
    def evaluate_in_batches(self, batch_size) -> None:
        """Iterates through a DataLoader that is created from the given Dataset.
        In each iteration, the images are passed through the CLIP image encoder. The resulting embeddings are then
        compared to the HOI class embeddings to obtain logits per image. A distribution over the HOI classes is
        computed using a softmax.
        """
        dataloader = DataLoader(self.dataset, batch_size=batch_size,
                                shuffle=False, collate_fn=custom_collate)

        text_inputs = torch.cat([clip.tokenize(f"a person {hoi.hoi_phrase}") for hoi in
                                 self.dataset.hoi_classes]).to(self.device)
        text_features = self.model.encode_text(text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        for batch in tqdm(dataloader):
            for image, target in batch:

                image_features = self.model.encode_image(image.to(self.device))
                # normalized features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                logit_scale = self.model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                probs = torch.softmax(logits_per_image, dim=-1)
                # probs = to_probabilities_by_distance(logits_per_image, max_offset=0.03)
                # print(end_time - start_time)
                self.metric.update(probs, target.to(self.device))

        self.final_metric = self.metric.compute()


def center_logits(logits):
    mean_logit = torch.mean(logits, dim=1, keepdim=True)
    centered_logits = logits - mean_logit
    return centered_logits


def to_probabilities_by_threshold(logits, prob_threshold):
    # sort the original logits
    sorted_logits, sorted_indices = torch.sort(logits, dim=1, descending=True)
    # obtain original probabilities
    probabilities = logits.softmax(dim=-1)
    max_probs = torch.max(probabilities, dim=1).values
    # obtain indices corresponding to all but the largest logits
    sorted_indices = sorted_indices[:, 1:]
    # and corresponding probabilities
    new_probabilities = torch.gather(logits, dim=-1, index=sorted_indices).softmax(dim=-1)

    # obtain largest probabilities
    top_prob = new_probabilities[:, 0]
    # top_prob = torch.max(new_probabilities, dim=1)
    assert torch.equal(top_prob, torch.max(new_probabilities, dim=1).values)

    # get batch indices for which the largest prob exceeds the threshold
    exceeds_threshold = top_prob > prob_threshold

    if torch.any(exceeds_threshold):
        for batch_index in torch.nonzero(exceeds_threshold):
            # get indices, top prob and new probs corresponding to index\
            sample_logits = logits[batch_index].squeeze()
            sample_indices = sorted_indices[batch_index].squeeze()
            sample_probs = new_probabilities[batch_index].squeeze()

            while sample_probs[0] > prob_threshold:
                probabilities[batch_index, sample_indices[0]] = sample_probs[0]
                sample_indices = sample_indices[1:]
                sample_probs = sample_logits[sample_indices].softmax(dim=-1)

    return probabilities


def to_probabilities_by_distance(logits, max_offset):
    batch_size, nr_of_classes = logits.shape
    sorted_logits, sorted_indices = torch.sort(logits, dim=1, descending=True)
    probabilities = logits.softmax(dim=-1)
    top_logit = sorted_logits[:, 0].unsqueeze(dim=-1)
    fractional_offset = torch.div(top_logit, sorted_logits) - 1

    for batch_index in range(batch_size):

        sample_logits = logits[batch_index].squeeze()
        sample_indices = sorted_indices[batch_index]

        nn_mask = fractional_offset[batch_index] < max_offset
        candidate_positives = sample_indices[nn_mask]
        candidate_negatives = sample_indices[~nn_mask]

        if len(candidate_positives) > 1:
            for class_index in candidate_positives:
                included_indices = torch.cat((candidate_negatives, class_index.unsqueeze(dim=0)))
                increased_probability = torch.max(sample_logits[included_indices].softmax(dim=-1))
                probabilities[batch_index, class_index] = increased_probability

    return probabilities


def custom_collate(batch):
    # data = [item[0] for item in batch]
    # target = [item[1] for item in batch]
    # return [data, target]
    batch = [(image.unsqueeze(0), target.unsqueeze(0)) for image, target in batch]
    return batch


def interpolate_pos_encoding(x, w, h, pos_embed, patch_size):
    npatch = x.shape[1] - 1
    N = pos_embed.shape[1] - 1
    if npatch == N and w == h:
        return pos_embed
    class_pos_embed = pos_embed[:, 0]
    patch_pos_embed = pos_embed[:, 1:]
    dim = x.shape[-1]
    w0 = w // patch_size
    h0 = h // patch_size
    # we add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8
    w0, h0 = w0 + 0.1, h0 + 0.1
    patch_pos_embed = interpolate(
        patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
        scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
        mode='bicubic',
    )
    assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


if __name__ == '__main__':
    N = 196 + 1
    random_image_embeds = torch.rand(8, N, 768)
    w = 300
    h = 200
    dim = random_image_embeds.shape[-1]
    pos_embed = torch.rand(1, N, dim)
    patch_size = 16
    interpolated_pos_embed = interpolate_pos_encoding(random_image_embeds, w, h, pos_embed, patch_size)
    print()
