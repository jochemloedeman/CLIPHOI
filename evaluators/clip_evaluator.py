import functools
import math
import pickle
from collections import Counter
from pathlib import Path
from pprint import pprint

import clip
import torch
import torchmetrics

from torch.nn.functional import interpolate
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import Dataset

from datasets.hico_dataset import preprocess_targets_for_loss
from util.hico_probe import Prediction, prob_based_renorm, prob_based_tensor, logit_based_tensor


class CLIPEvaluator(object):

    def __init__(self, dataset: Dataset, metric: torchmetrics.Metric,
                 backbone: str, device: str, batch_size: int,
                 prob_fn: str, center_crop: bool, five_crop: bool,
                 ir_threshold: float, save_predictions: bool,
                 include_image: bool) -> None:
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
        self.prob_fn = prob_fn
        self.center_crop = center_crop
        self.five_crop = five_crop
        self.ir_threshold = ir_threshold
        self.save_predictions = save_predictions
        self.include_image = include_image
        self.final_metric = None

        self.prob_fn_matcher = {
            "softmax": functools.partial(torch.softmax, dim=-1),
            "sigmoid": torch.sigmoid,
            "prob_based_greater": functools.partial(prob_based_tensor, mode="prob_based_greater",
                                                    threshold=self.ir_threshold),
            "prob_based_smaller": functools.partial(prob_based_tensor, mode="smaller",
                                                    threshold=self.ir_threshold),
            "logit_based": functools.partial(logit_based_tensor, threshold=self.ir_threshold),
            "ir_softmax": None
        }

    def evaluate(self) -> None:
        if self.save_predictions:
            self._evaluate_with_saving()
        else:
            self._evaluate_map()

    @torch.no_grad()
    def _evaluate_map(self) -> None:
        """Iterates through a DataLoader that is created from the given Dataset.
        In each iteration, the images are passed through the CLIP image encoder. The resulting embeddings are then
        compared to the HOI class embeddings to obtain logits per image. A distribution over the HOI classes is
        computed using a softmax.
        """
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        text_inputs = torch.cat([clip.tokenize(f"a person {hoi.hoi_phrase}") for hoi in
                                 self.dataset.hoi_classes]).to(self.device)

        text_features = self.model.encode_text(text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        prob_fn = self.prob_fn_matcher[self.prob_fn]

        logit_statistics = Counter()
        for images, target in tqdm(dataloader):

            if self.five_crop:
                batch_size, nr_of_crops, channels, height, width = images.size()
                image_features = self.model.encode_image(images.to(self.device).view(-1, channels, height, width))
                image_features = image_features.view(batch_size, nr_of_crops, -1).mean(1)
            else:
                image_features = self.model.encode_image(images.to(self.device))

            # normalized features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logit_scale = self.model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            update_logit_statistics(logit_statistics, logits_per_image, target)
            probs = prob_fn(logits_per_image)
            self.metric.update(probs, target.to(self.device))

        self.final_metric = self.metric.compute()
        logit_statistics = {position: logit_statistics[position] / len(self.dataset) for position in
                            logit_statistics.keys()}
        pprint(logit_statistics)

    @torch.no_grad()
    def _evaluate_with_saving(self) -> None:
        """Iterates through a DataLoader that is created from the given Dataset.
        In each iteration, the images are passed through the CLIP image encoder. The resulting embeddings are then
        compared to the HOI class embeddings to obtain logits per image. A distribution over the HOI classes is
        computed using a softmax.
        """
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size,
                                shuffle=False, collate_fn=custom_collate)

        text_inputs = torch.cat([clip.tokenize(f"a person {hoi.hoi_phrase}") for hoi in
                                 self.dataset.hoi_classes]).to(self.device)

        text_features = self.model.encode_text(text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        prob_fn = self.prob_fn_matcher[self.prob_fn]

        predictions = []
        for batch_index, batch in enumerate(tqdm(dataloader)):
            for pil_image, image_tensor, target in batch:

                binary_target = preprocess_targets_for_loss(target)

                if self.five_crop:
                    batch_size, nr_of_crops, channels, height, width = image_tensor.size()
                    image_features = self.model.encode_image(
                        image_tensor.to(self.device).view(-1, channels, height, width))
                    image_features = image_features.view(1, nr_of_crops, -1).mean(1)
                else:
                    image_features = self.model.encode_image(image_tensor.to(self.device))

                # normalized features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                logit_scale = self.model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()

                probs = prob_fn(logits_per_image)

                predictions.append(
                    Prediction(logits_dict=self.dataset.create_hoi_dict(logits_per_image),
                               raw_target=target,
                               target_string=self.dataset.to_positive_classes(binary_target),
                               probabilities=self.dataset.create_hoi_dict(probs),
                               binary_target=binary_target,
                               image=pil_image if self.include_image else None)
                )

                self.metric.update(probs, target.to(self.device))

        save_predictions(predictions, Path(__file__).parents[1] / "predictions.pkl")
        self.final_metric = self.metric.compute()


def center_logits(logits):
    mean_logit = torch.mean(logits, dim=1, keepdim=True)
    centered_logits = logits - mean_logit
    return centered_logits


def update_logit_statistics(counter, logits, target):
    processed_target = preprocess_targets_for_loss(target).squeeze()
    sorted_indices = torch.argsort(logits, descending=True).cpu()
    sorted_target = processed_target[sorted_indices].squeeze()
    for i in range(15):
        if sorted_target[i] == 1.:
            counter[str(i)] += 1


def save_predictions(predictions, path):
    pickle.dump(predictions, open(path, 'wb'))


def to_probabilities_by_threshold(logits, prob_threshold):
    # sort the original logits
    sorted_logits, sorted_indices = torch.sort(logits, dim=1, descending=True)
    # obtain original probabilities
    probabilities = logits.softmax(dim=-1)
    # obtain indices corresponding to all but the largest logits
    sorted_indices = sorted_indices[:, 1:]
    # and corresponding probabilities
    new_probabilities = torch.gather(logits, dim=-1, index=sorted_indices).softmax(dim=-1)

    # obtain largest probabilities
    top_prob = new_probabilities[:, 0]
    # top_prob = torch.max(new_probabilities, dim=1)
    assert torch.equal(top_prob, torch.max(new_probabilities, dim=1).values)

    # get batch indices for which the largest prob exceeds the threshold
    exceeds_threshold = top_prob < prob_threshold

    if torch.any(exceeds_threshold):
        for batch_index in torch.nonzero(exceeds_threshold):
            # get indices, top prob and new probs corresponding to index
            sample_logits = logits[batch_index].squeeze()
            sample_indices = sorted_indices[batch_index].squeeze()
            sample_probs = new_probabilities[batch_index].squeeze()

            while sample_probs[0] < prob_threshold:
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
    batch = [(image, image_tensor.unsqueeze(0), target.unsqueeze(0)) for image, image_tensor, target in batch]
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
