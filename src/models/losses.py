import sys

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

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = input * torch.log(target)
        loss = - loss.sum(dim=1).mean()

        return loss

    @torch.no_grad()
    def softmax_center(self, teacher_logits, teacher_temp=0.07):
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

    # def forward(self, input: torch.Tensor, target: torch.Tensor, bool_masked_pos: torch.Tensor) -> torch.Tensor:
    #     loss = 0
    #     batch_size, num_patches, _ = input.shape
    #
    #     for b in range(batch_size):
    #         b_bool_masked_pos = bool_masked_pos[b]
    #         false_tensor = torch.tensor([0]).bool().to(b_bool_masked_pos.device)
    #         b_bool_masked_pos = torch.cat([false_tensor, b_bool_masked_pos])
    #
    #         loss_patch = 0
    #         for p in range(num_patches):
    #             if b_bool_masked_pos[p].item():
    #                 loss_patch += input[b, p, :] * torch.log(target[b, p, :])
    #
    #         loss += loss_patch / b_bool_masked_pos.sum()
    #
    #     loss = - (loss / batch_size).sum()
    #
    #     return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, bool_masked_pos: torch.Tensor) -> torch.Tensor:
        batch_size, num_patches, _ = input.shape

        # Assuming bool_masked_pos is of shape [batch_size, num_patches]
        false_tensor = torch.tensor([False], dtype=torch.bool, device=bool_masked_pos.device)
        # Expand false_tensor to match bool_masked_pos's batch size, making it [batch_size, 1]
        expanded_false_tensor = false_tensor.unsqueeze(0).expand(batch_size, -1)
        # Now concatenate along the patches dimension (dimension 1)
        bool_masked_pos = torch.cat([expanded_false_tensor, bool_masked_pos], dim=1)

        # Use broadcasting to apply bool_masked_pos to input and target
        masked_input = input * bool_masked_pos.unsqueeze(-1)
        masked_target = target * bool_masked_pos.unsqueeze(-1)

        # Compute loss for all patches in a vectorized manner
        loss_patch = masked_input * torch.log(masked_target + 1e-8)
        loss_patch_sum = loss_patch.sum(dim=2)  # Sum over the last dimension

        # Compute the loss per batch item
        loss_per_batch = loss_patch_sum.sum(dim=1) / bool_masked_pos.sum(dim=1)

        # Compute the final loss
        loss = -loss_per_batch.mean()

        return loss

    def softmax_center(self, teacher_logits, teacher_temp=0.07):
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
