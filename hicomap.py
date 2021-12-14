import torch
from torchmetrics import Metric
from torchmetrics.functional import retrieval_average_precision
from torchmetrics import RetrievalMAP
from torchmetrics.utilities.data import get_group_indexes


class HICOmAP(Metric):
    hico_target_size = 600

    def __init__(self, dist_sync_on_step=False, known_object_mode=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.known_object_mode = known_object_mode
        self.add_state("indexes", default=[], dist_reduce_fx=None)
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("target", default=[], dist_reduce_fx=None)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # update metric states
        self.indexes.append(torch.arange(self.hico_target_size))
        self.preds.append(preds)
        self.target.append(target)

        assert preds.shape == target.shape

    def compute(self):
        # compute final result
        indexes = torch.cat(self.indexes, dim=0)
        preds = torch.cat(self.preds, dim=0)
        target = torch.cat(self.target, dim=0)

        included_pairs = self.get_included_pairs(target)
        indexes, preds, target = (indexes[included_pairs], preds[included_pairs], target[included_pairs])

        res = []
        groups = get_group_indexes(indexes)

        for group in groups:
            mini_preds = preds[group]
            mini_target = target[group]
            res.append(self.retrieval_average_precision(mini_preds, mini_target))

        return torch.stack([x.to(preds) for x in res]).mean() if res else torch.tensor(0.0).to(preds)

    def get_included_pairs(self, target):
        if self.known_object_mode:
            mask = (target != 0) & (~torch.isnan(target))
        else:
            mask = target != 0

        return torch.nonzero(mask, as_tuple=True)


if __name__ == '__main__':
    map = HICOmAP(known_object_mode=True)
    target_1 = torch.tensor([1.0, 0.0, float('nan')])
    print(map.get_included_pairs(target_1))
