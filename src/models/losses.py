import torch
import torch.nn.functional as F
from torch import nn


def make_triplet_criterion(wandb_config):
    if wandb_config['criterion'] == 'TMWDL-Euclidean':
        return nn.TripletMarginLoss(swap=True)
    elif wandb_config['criterion'] == 'TMWDL-Cosine':
        return nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y),
            swap=True)
    else:
        return nn.TripletMarginLoss(swap=True)


class DINOLoss(nn.Module):
    def __init__(self, wandb_config):
        super().__init__()
        self.wandb_config = wandb_config
        self.register_buffer('center', torch.zeros(1, self.wandb_config['num_prototypes']))
        self.updated = True
        self.len_teacher_logits = None
        self.async_batch_center = None

    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        return - (teacher * F.log_softmax(student, dim=-1)).sum(dim=1).mean()

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

    def forward(self, student: torch.Tensor, teacher: torch.Tensor, bool_masked_pos: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(teacher[:, 1:] * F.log_softmax(student[:, 1:], dim=-1), dim=-1)
        loss = torch.sum(loss * bool_masked_pos.float(), dim=-1) / bool_masked_pos.sum(dim=-1).clamp(min=1.0)

        return - loss.mean()

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


class KoLeoLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.pairwise_distance = nn.PairwiseDistance(p=2, eps=1e-8)

    @staticmethod
    def _pairwise_nearest_neighbors(features):
        dot_products = torch.mm(features, features.t())
        n = features.size(0)
        dot_products.view(-1)[::n + 1] = -1
        _, nearest_neighbors = torch.max(dot_products, dim=1)

        return nearest_neighbors

    def forward(self, features, eps=1e-8):
        with torch.cuda.amp.autocast(enabled=False):
            normalized_features = F.normalize(features, p=2, dim=-1, eps=eps)
            nearest_neighbors = self._pairwise_nearest_neighbors(normalized_features)
            distances = self.pairwise_distance(normalized_features, normalized_features[nearest_neighbors])
            loss = - torch.log(distances + eps).mean()

        return loss
