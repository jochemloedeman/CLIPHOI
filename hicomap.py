import torch
from torchmetrics import Metric
from torchmetrics.functional import retrieval_average_precision
from torchmetrics.utilities.data import get_group_indexes


class HICOmAP(Metric):

    def __init__(self, hoi_classes, dist_sync_on_step=False,
                 known_object_mode=False, exclude_no_interaction=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.hoi_classes = hoi_classes
        self.known_object_mode = known_object_mode
        self.exclude_no_interaction = exclude_no_interaction
        self.add_state("indexes", default=[], dist_reduce_fx=None)
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("target", default=[], dist_reduce_fx=None)
        self.aps = None

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # update metric states
        self.indexes.append(torch.arange(preds.shape[1]).repeat(preds.shape[0]))
        self.preds.append(preds)
        self.target.append(target)

        assert preds.shape == target.shape

    def compute(self):
        # compute final result
        ap_per_hoi = self.compute_ap_per_hoi()
        return torch.stack([x for x in ap_per_hoi]).mean() if ap_per_hoi else torch.tensor(0.0)

    def compute_ap_per_hoi(self):
        # compute final result
        indexes = torch.cat(self.indexes, dim=0)
        preds = torch.flatten(torch.cat(self.preds, dim=0))
        target = torch.flatten(torch.cat(self.target, dim=0))

        included_pairs = self.get_included_pairs(target)
        indexes, preds, target = (indexes[included_pairs], preds[included_pairs], target[included_pairs])
        target = torch.where(target == 1, 1, 0)
        aps = []
        groups = get_group_indexes(indexes)

        for group in groups:
            mini_preds = preds[group]
            mini_target = target[group]

            aps.append(retrieval_average_precision(mini_preds, mini_target))

        self.aps = [ap.item() for ap in aps]

        return aps

    def get_included_pairs(self, target):
        if self.known_object_mode:
            mask = (target != 0) & (~torch.isnan(target))
        else:
            mask = target != 0

        return torch.nonzero(mask, as_tuple=True)

    def get_ap_per_hoi(self):
        hoi_phrases = [hoi.hoi_phrase for hoi in self.hoi_classes]
        ap_per_hoi = dict(zip(hoi_phrases, self.aps))
        sorted_ap_per_hoi = {k: v for k, v in sorted(ap_per_hoi.items(), key=lambda item: item[1], reverse=True)}
        return sorted_ap_per_hoi


if __name__ == '__main__':
    map = HICOmAP(known_object_mode=True)
    target_1 = torch.tensor([1.0, 0.0, float('nan')])
    print(map.get_included_pairs(target_1))
