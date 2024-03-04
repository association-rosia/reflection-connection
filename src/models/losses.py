from torch import nn
import torch.nn.functional as F

def make_triplet_criterion(wandb_config):
    if wandb_config['criterion'] == 'TMWDL-Euclidean':
        return nn.TripletMarginLoss(swap=True)
    elif wandb_config['criterion'] == 'TMWDL-Cosine':
        return nn.TripletMarginWithDistanceLoss(
                distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y),
                swap=True,
                # cosine_similarity va de -1 à 1
                margin=2
                )
    else: 
        return nn.TripletMarginLoss(swap=True)