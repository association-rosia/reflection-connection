import warnings

warnings.filterwarnings('ignore')

import os

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Any

from transformers import ViTModel
import src.models.pretrain.lightning as pretrain_l

import torch

import src.data.datasets.triplet_dataset as td
from src.models.losses import make_triplet_criterion
from src import utils


class RefConLightning(pl.LightningModule):
    def __init__(
            self,
            config: dict,
            wandb_config: dict,
            model: ViTModel,
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


def get_model(config, wandb_config) -> ViTModel:
    kwargs = {'config': config, 'wandb_config': utils.init_wandb('pretrain.yml')}
    path_checkpoint = os.path.join(config['path']['models'], f'{wandb_config["checkpoint"]}.ckpt')
    lightning = pretrain_l.RefConLightning.load_from_checkpoint(path_checkpoint, **kwargs)

    return model


def _debug():
    config = utils.get_config()
    wandb_config = utils.init_wandb('vit_transformers.yml')
    model = get_model(config, wandb_config)

    kwargs = {
        'config': config,
        'wandb_config': wandb_config,
        'model': model,
    }

    lightning = RefConLightning(**kwargs)

    return


if __name__ == '__main__':
    _debug()
