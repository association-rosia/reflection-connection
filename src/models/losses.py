import torch
import torch.nn.functional as F
from torch import nn


class DINOLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ps, pt):
        return -(pt * torch.log(ps)).sum(dim=1).mean()


class iBOTLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ps, pt, bool_masked_pos):
        # Apply the mask to select only the relevant entries for loss calculation
        ps_masked = torch.masked_select(ps, bool_masked_pos.unsqueeze(-1)).view(-1, ps.size(-1))
        pt_masked = torch.masked_select(pt, bool_masked_pos.unsqueeze(-1)).view(-1, pt.size(-1))

        loss = -(pt_masked * torch.log(ps_masked)).sum(dim=1).mean()

        return loss


def make_triplet_criterion(wandb_config):
    if wandb_config['criterion'] == 'TMWDL-Euclidean':
        return nn.TripletMarginLoss(swap=True)
    elif wandb_config['criterion'] == 'TMWDL-Cosine':
        return nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y),
            swap=True
        )
    else:
        return nn.TripletMarginLoss(swap=True)
