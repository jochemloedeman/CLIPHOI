import json
import torch

from typing import Tuple, Union, Dict
from torchmetrics import Metric
from torchmetrics.functional import retrieval_average_precision
from torchmetrics.utilities.data import get_group_indexes
from hico_dataset import HICODataset


class HICOmAP(Metric):
    """
    Subclass of the TorchMetrics base class. Implements mean average precision, as intended by the
    creators/authors of the HICO dataset (http://www-personal.umich.edu/~ywchao/hico/).

    Args:
        hico_dataset:
            Used to extract class information and statistics
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step.
        known_object_mode:
            Evaluate in known-object mode.
            Only images annotated as negatives for a how are taken into account in the mAP calculation
    """

    def __init__(self, hico_dataset: HICODataset, dist_sync_on_step: bool = False,
                 known_object_mode: bool = False) -> None:

        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.hico_dataset = hico_dataset
        self.known_object_mode = known_object_mode

        self.add_state("indexes", default=[], dist_reduce_fx=None)
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("target", default=[], dist_reduce_fx=None)

        self._aps = None
        self._mean_ap = None

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Add preds and targets of a batch to accumulators.

        For each update, indexes of equal shapes are added that identify the preds and targets to the corresponding
        HOI classes
        """
        self.indexes.append(torch.arange(preds.shape[1]).repeat(preds.shape[0], 1))
        self.preds.append(preds)
        self.target.append(target)

        assert preds.shape == target.shape

    def compute(self) -> torch.Tensor:
        """Computes the mean average precision from a Tensor containing the AP's per HOI"""
        ap_per_hoi = self._compute_ap_per_hoi()
        mean_ap = ap_per_hoi.mean()
        self._mean_ap = mean_ap.item()
        return mean_ap

    def _compute_ap_per_hoi(self) -> torch.Tensor:
        """First flatten the accumulated preds, targets and indexes. Then determine the image-hoi pairs that should be
        included in the AP calculations and calculate the AP per HOI-group.
        """
        indexes = torch.flatten(torch.cat(self.indexes, dim=0))
        preds = torch.flatten(torch.cat(self.preds, dim=0))
        target = torch.flatten(torch.cat(self.target, dim=0))

        included_pairs = self._get_included_pairs(target)
        indexes, preds, target = (indexes[included_pairs], preds[included_pairs], target[included_pairs])
        target = torch.where(target == 1, 1, 0)
        aps = []
        groups = get_group_indexes(indexes)

        for group in groups:
            mini_preds = preds[group]
            mini_target = target[group]

            aps.append(retrieval_average_precision(mini_preds, mini_target))

        aps = torch.stack([ap.to(preds) for ap in aps])
        self._aps = aps

        return aps

    def _get_included_pairs(self, target: torch.Tensor) -> Tuple[torch.Tensor]:
        """Calculate indices of included image-hoi pairs based on which evaluation mode is chosen"""
        if self.known_object_mode:
            mask = (target != 0) & (~torch.isnan(target))
        else:
            mask = target != 0

        return torch.nonzero(mask, as_tuple=True)

    def _get_ap_per_hoi(self, include_statistics: bool = True) -> Union[Dict[str, float], Dict[str, Tuple[float, int]]]:
        """Combines the APs per HOI with the corresponding HOI class phrase"""
        hoi_phrases = [hoi.hoi_phrase for hoi in self.hico_dataset.hoi_classes]
        ap_per_hoi = dict(zip(hoi_phrases, self._aps))

        if include_statistics:
            statistics = self.hico_dataset.get_hoi_statistics()
            ap_per_hoi = {hoi: (ap, statistics[hoi]) for hoi, ap in ap_per_hoi}

        sorted_ap_per_hoi = {key: value.item() for key, value in
                             sorted(ap_per_hoi.items(), key=lambda item: item[1], reverse=True)}

        return sorted_ap_per_hoi

    def export_to_json(self, filename: str) -> None:
        json_list = [{'map': self._mean_ap}, self._get_ap_per_hoi(include_statistics=True)]
        with open(f'{filename}.json', 'w') as f:
            json.dump(json_list, f, indent=4)
