import warnings

warnings.filterwarnings('ignore')

import os

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Any

import torch
from torchvision.models import VisionTransformer, vit_b_16, vit_l_16

import src.data.datasets.triplet_dataset as td
from src.models.losses import make_triplet_criterion
from src import utils


class RefConLightning(pl.LightningModule):
    def __init__(
            self,
            config: dict,
            wandb_config: dict,
            model: VisionTransformer,
            *args: Any,
            **kwargs: Any
    ):
        super(RefConLightning, self).__init__()
        self.config = config
        self.wandb_config = wandb_config
        self.model = model
        self.criterion = make_triplet_criterion(self.wandb_config)

    def forward(self, anchors, positives, negatives):
        anchors_embed = self.model(anchors)
        positives_embed = self.model(positives)
        negatives_embed = self.model(negatives)
        loss = self.criterion(anchors_embed, positives_embed, negatives_embed)

        return loss

    def training_step(self, batch):
        loss = self.forward(*batch)
        self.log('train/loss', loss, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch):
        loss = self.forward(*batch)
        self.log('val/loss', loss, on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(params=self.model.parameters(), lr=self.wandb_config['lr'])
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True),
            'monitor': 'val/loss',
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    def train_dataloader(self):
        dataset = td.make_train_triplet_dataset(self.config, self.wandb_config)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.wandb_config['batch_size'],
            num_workers=self.config['archi']['num_workers'],
            drop_last=True,
            shuffle=True,
        )

        return dataloader

    def val_dataloader(self):
        dataset = td.make_val_triplet_dataset(self.config, self.wandb_config)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.wandb_config['batch_size'],
            num_workers=self.config['archi']['num_workers'],
            drop_last=True,
            shuffle=False,
        )

        return dataloader


def get_model(wandb_config) -> VisionTransformer:
    model_id = wandb_config['model_id']

    if 'ViT_B_16' in model_id:
        model = vit_b_16(pretrained=False)
    elif 'ViT_L_16' in model_id:
        model = vit_l_16(pretrained=False)
    else:
        ValueError(f'Unknown model_id: {model_id}')

    weights_path = os.path.join('models', f'{model_id}.pth')
    weights = torch.load(weights_path)
    model.load_state_dict(weights)

    return model


def _debug():
    config = utils.get_config()
    wandb_config = utils.init_wandb('vit.yml')
    model = get_model(wandb_config)

    kwargs = {
        'config': config,
        'wandb_config': wandb_config,
        'model': model,
    }

    lightning = RefConLightning(**kwargs)

    return


if __name__ == '__main__':
    _debug()
