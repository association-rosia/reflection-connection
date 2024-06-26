import warnings

warnings.filterwarnings('ignore')

import os

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Any

import torch
from torchvision.models import vit_b_16, vit_l_16
from src.models.modules import RefConTorchvisionViT

import src.data.datasets.triplet as triplet_d
from src.models.losses import make_triplet_criterion
from src import utils


class RefConLightning(pl.LightningModule):
    def __init__(
            self,
            config: dict,
            wandb_config: dict,
            model: RefConTorchvisionViT,
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
        self.log('train/loss', loss, on_epoch=True)

        return loss

    def validation_step(self, batch):
        loss = self.forward(*batch)
        self.log('val/loss', loss, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(params=self.model.parameters(), lr=self.wandb_config['lr'])
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True),
            'monitor': 'val/loss',
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    def train_dataloader(self):
        dataset = triplet_d.make_train_triplet_dataset(self.config, self.wandb_config)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.wandb_config['batch_size'],
            num_workers=self.config['archi']['num_workers'],
            drop_last=True,
            shuffle=True,
        )

        return dataloader

    def val_dataloader(self):
        dataset = triplet_d.make_val_triplet_dataset(self.config, self.wandb_config)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.wandb_config['batch_size'],
            num_workers=self.config['archi']['num_workers'],
            drop_last=True,
            shuffle=False,
        )

        return dataloader


def get_model(config, wandb_config):
    model_id = wandb_config['model_id']

    if 'ViT_B_16' in model_id:
        vit = vit_b_16(pretrained=False)
    elif 'ViT_L_16' in model_id:
        vit = vit_l_16(pretrained=False)
    else:
        ValueError(f'Unknown model_id: {model_id}')

    weights_path = os.path.join('models', f'{model_id}.pth')
    weights = torch.load(weights_path)
    vit.load_state_dict(weights)

    return vit


def _debug():
    config = utils.get_config()
    # Specify the ID of the model you wish to use.
    wandb_id = 'nszfciym'
    # Specify the Name of the model you wish to use.
    wandb_name = 'key-lime-pie-110'

    # If you are using a WandB account to record the runs, use the code below. wandb_run = utils.get_run(wandb_id)
    # Otherwise, specify the name and ID of the model and choose the corresponding configuration file for training
    # the model. wandb_run = utils.RunDemo('fine_tuning/vit.yml', id=wandb_id, name=wandb_name,
    # sub_config='torchvision')
    wandb_run = utils.get_run(wandb_id)
    wandb_config = wandb_run.config
    # wandb_config = utils.init_wandb('training/vit.yml', 'torchvision')
    model = get_model(wandb_config)

    kwargs = {
        'config': config,
        'wandb_config': wandb_config,
        'model': model,
    }

    lightning = RefConLightning(**kwargs)
    path_checkpoint = os.path.join(config['path']['models'], f'{wandb_run.name}-{wandb_run.id}.ckpt')
    path_checkpoint = utils.get_notebooks_path(path_checkpoint)
    lightning = RefConLightning.load_from_checkpoint(path_checkpoint, **kwargs)

    return


if __name__ == '__main__':
    _debug()
