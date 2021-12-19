import json
import torch
from torchmetrics import Metric
from torchmetrics.functional import retrieval_average_precision
from torchmetrics.utilities.data import get_group_indexes


class HICOmAP(Metric):

    def __init__(self, hico_dataset, dist_sync_on_step=False,
                 known_object_mode=False, exclude_no_interaction=False):

        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.hico_dataset = hico_dataset
        self.known_object_mode = known_object_mode
        self.exclude_no_interaction = exclude_no_interaction

        self.add_state("indexes", default=[], dist_reduce_fx=None)
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("target", default=[], dist_reduce_fx=None)

        self._aps = None
        self._mean_ap = None

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.indexes.append(torch.arange(preds.shape[1]).repeat(preds.shape[0]))
        self.preds.append(preds)
        self.target.append(target)

        assert preds.shape == target.shape

    def compute(self):
        ap_per_hoi = self._compute_ap_per_hoi()
        mean_ap = torch.stack([x for x in ap_per_hoi]).mean() if ap_per_hoi else torch.tensor(0.0)
        self._mean_ap = mean_ap.item()
        return mean_ap

    def _compute_ap_per_hoi(self):
        indexes = torch.cat(self.indexes, dim=0)
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

        self._aps = [ap.item() for ap in aps]

        return aps

    def _get_included_pairs(self, target):
        if self.known_object_mode:
            mask = (target != 0) & (~torch.isnan(target))
        else:
            mask = target != 0

        return torch.nonzero(mask, as_tuple=True)

    def _get_ap_per_hoi(self, include_statistics=True):
        hoi_phrases = [hoi.hoi_phrase for hoi in self.hico_dataset.hoi_classes]
        ap_per_hoi = dict(zip(hoi_phrases, self._aps))
        if include_statistics:
            statistics = self.hico_dataset.get_hoi_statistics()
            sorted_ap_per_hoi = {key: (value, statistics[key]) for key, value in sorted(ap_per_hoi.items(), key=lambda item: item[1], reverse=True)}
        else:
            sorted_ap_per_hoi = {k: v for k, v in sorted(ap_per_hoi.items(), key=lambda item: item[1], reverse=True)}
        return sorted_ap_per_hoi

    def export_to_json(self, filename):
        json_list = [{'map': self._mean_ap}, self._get_ap_per_hoi(include_statistics=True)]
        with open(f'{filename}.json', 'w') as f:
            json.dump(json_list, f, indent=4)
