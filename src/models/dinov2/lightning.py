from torch.optim import AdamW
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Any

from transformers import Dinov2Model

import src.data.datasets.triplet_dataset as td
from src.models.losses import make_triplet_criterion
from src import utils


class RefConLightning(pl.LightningModule):
    def __init__(
        self,
        config: dict,
        wandb_config: dict,
        model: Dinov2Model,
        *args: Any,
        **kargs: Any
        ):
        super(RefConLightning, self).__init__()
        self.config = config
        self.wandb_config = wandb_config
        self.model = model

        self.criterion = make_triplet_criterion(self.wandb_config)
    
    def forward(self, anchors, positives, negatives):
        anchors_embed = self.model(pixel_values=anchors)['pooler_output']
        positives_embed = self.model(pixel_values=positives)['pooler_output']
        negatives_embed = self.model(pixel_values=negatives)['pooler_output']
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
        return AdamW(params=self.model.parameters(), lr=self.wandb_config['lr'])

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


def get_model(wandb_config) -> Dinov2Model:
    model = Dinov2Model.from_pretrained(
        pretrained_model_name_or_path=wandb_config['model_id'],
        ignore_mismatched_sizes=True
    )

    return model


def _debug():
    config = utils.get_config()
    wandb_config = utils.init_wandb('dinov2.yml')
    model = get_model(wandb_config)

    kargs = {
        'config': config,
        'wandb_config': wandb_config,
        'model': model,
    }

    lightning = RefConLightning(**kargs)

    return 


if __name__ == '__main__':
    _debug()