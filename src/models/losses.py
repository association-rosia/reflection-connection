import torch
import torch.nn.functional as F
from torch import nn


class DINOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer('center', torch.zeros(1, out_dim))
        self.updated = True
        self.len_teacher_output = None
        self.async_batch_center = None

    def forward(self, ps, pt):
        return -(pt * torch.log(ps)).sum(dim=1).mean()

    @torch.no_grad()
    def update_center(self, teacher_output):
        self.reduce_center_update(teacher_output)

    @torch.no_grad()
    def reduce_center_update(self, teacher_output):
        self.updated = False
        self.len_teacher_output = len(teacher_output)
        self.async_batch_center = torch.sum(teacher_output, dim=0, keepdim=True)

    @torch.no_grad()
    def apply_center_update(self):
        if self.updated is False:
            _t = self.async_batch_center / self.len_teacher_output
            self.center = self.center * self.center_momentum + _t * (1 - self.center_momentum)
            self.updated = True


class iBOTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer('center', torch.zeros(1, 1, patch_out_dim))
        self.updated = True
        self.len_teacher_patch_tokens = None
        self.async_batch_center = None

    def forward(self, ps, pt, bool_masked_pos):
        ps_masked = torch.masked_select(ps, bool_masked_pos.unsqueeze(-1)).view(-1, ps.size(-1))
        pt_masked = torch.masked_select(pt, bool_masked_pos.unsqueeze(-1)).view(-1, pt.size(-1))

        loss = -(pt_masked * torch.log(ps_masked)).sum(dim=1).mean()

        return loss

    def softmax_center_teacher(self, teacher_patch_tokens, teacher_temp):
        self.apply_center_update()

        return F.softmax((teacher_patch_tokens - self.center) / teacher_temp, dim=-1)

    @torch.no_grad()
    def update_center(self, teacher_patch_tokens):
        self.reduce_center_update(teacher_patch_tokens)

    @torch.no_grad()
    def reduce_center_update(self, teacher_patch_tokens):
        self.updated = False
        self.len_teacher_patch_tokens = len(teacher_patch_tokens)
        self.async_batch_center = torch.sum(teacher_patch_tokens.mean(1), dim=0, keepdim=True)

    @torch.no_grad()
    def apply_center_update(self):
        if self.updated is False:
            _t = self.async_batch_center / self.len_teacher_patch_tokens
            self.center = self.center * self.center_momentum + _t * (1 - self.center_momentum)
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
