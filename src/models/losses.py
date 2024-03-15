import torch
import torch.nn.functional as F
from torch import nn


class DINOLoss(nn.Module):
    def __init__(self, wandb_config):
        super().__init__()
        self.wandb_config = wandb_config
        self.register_buffer('center', torch.zeros(1, self.wandb_config['num_prototypes']))
        self.updated = True
        self.len_teacher_logits = None
        self.async_batch_center = None

    def forward(self, ps, pt):
        return -(pt * torch.log(ps)).sum(dim=1).mean()

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_logits, teacher_temp=0.07):
        self.apply_center_update()

        return F.softmax((teacher_logits - self.center) / teacher_temp, dim=-1)

    @torch.no_grad()
    def update_center(self, teacher_logits):
        self.reduce_center_update(teacher_logits)

    @torch.no_grad()
    def reduce_center_update(self, teacher_logits):
        self.updated = False
        self.len_teacher_logits = len(teacher_logits)
        self.async_batch_center = torch.sum(teacher_logits, dim=0, keepdim=True)

    @torch.no_grad()
    def apply_center_update(self):
        if self.updated is False:
            center_momentum = self.wandb_config['center_momentum']
            _t = self.async_batch_center / self.len_teacher_logits
            self.center = self.center * center_momentum + _t * (1 - center_momentum)
            self.updated = True


class iBOTLoss(nn.Module):
    def __init__(self, wandb_config):
        super().__init__()
        self.wandb_config = wandb_config
        self.register_buffer('center', torch.zeros(1, 1, self.wandb_config['num_prototypes']))
        self.updated = True
        self.len_teacher_logits = None
        self.async_batch_center = None

    def forward(self, ps, pt, bool_masked_pos):
        bool_masked_pos = torch.cat([torch.tensor([False], device=bool_masked_pos.device), bool_masked_pos], dim=0)
        ps_masked = torch.masked_select(ps, bool_masked_pos.unsqueeze(-1)).view(-1, ps.size(-1))
        pt_masked = torch.masked_select(pt, bool_masked_pos.unsqueeze(-1)).view(-1, pt.size(-1))

        loss = -(pt_masked * torch.log(ps_masked)).sum(dim=1).mean()

        return loss

    def softmax_center_teacher(self, teacher_logits, teacher_temp=0.07):
        self.apply_center_update()

        return F.softmax((teacher_logits - self.center) / teacher_temp, dim=-1)

    @torch.no_grad()
    def update_center(self, teacher_logits):
        self.reduce_center_update(teacher_logits)

    @torch.no_grad()
    def reduce_center_update(self, teacher_logits):
        self.updated = False
        self.len_teacher_logits = len(teacher_logits)
        self.async_batch_center = torch.sum(teacher_logits.mean(1), dim=0, keepdim=True)

    @torch.no_grad()
    def apply_center_update(self):
        if self.updated is False:
            center_momentum = self.wandb_config['center_momentum']
            _t = self.async_batch_center / self.len_teacher_logits
            self.center = self.center * center_momentum + _t * (1 - center_momentum)
            self.updated = True


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
